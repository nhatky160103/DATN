import torch
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import cloudinary.uploader
import requests
from pathlib import Path
import time
import os
from glob import glob


from .infer_image import  transform_image
from .utils import get_recogn_model, device, mtcnn
from database.firebase import get_data, delete_person, add_person, get_person_ids_from_bucket
from .blazeFace import detect_face_and_nose



class EmbeddingManager:
    def __init__(self, bucket_name, recognition_model_name='ms1mv3_arcface', local_root="local_embeddings"):
        self.bucket_name = bucket_name
        self.recognition_model_name = recognition_model_name
        self.cloud_folder = f"{bucket_name}/Embeddings"
        self.local_dir = Path(local_root) / bucket_name
        self.local_dir.mkdir(parents=True, exist_ok=True)

        self.embeddings_file = self.local_dir / f"{recognition_model_name}_embeddings.npy"
        self.metadata_file = self.local_dir / f"{recognition_model_name}_metadata.pkl"

    def save_local(self, embeddings, image2class, index2class):
        np.save(self.embeddings_file, embeddings)
        with open(self.metadata_file, "wb") as f:
            pickle.dump((image2class, index2class), f)
        print(f"📎 Saved locally → {self.embeddings_file}, {self.metadata_file}")

    def upload_to_cloudinary(self, embeddings, image2class, index2class):
        """
        Upload embeddings and metadata to Cloudinary, invalidating cache to ensure latest load.
        """
        # Upload embeddings
        npy_buffer = BytesIO()
        np.save(npy_buffer, embeddings)
        npy_buffer.seek(0)
        cloudinary.uploader.upload(
            npy_buffer,
            folder=self.cloud_folder,
            public_id=f"{self.recognition_model_name}_embeddings",
            resource_type="raw",
            overwrite=True,
            invalidate=True
        )
        # Upload metadata
        meta_buffer = BytesIO()
        pickle.dump((image2class, index2class), meta_buffer)
        meta_buffer.seek(0)
        cloudinary.uploader.upload(
            meta_buffer,
            folder=self.cloud_folder,
            public_id=f"{self.recognition_model_name}_metadata",
            resource_type="raw",
            overwrite=True,
            invalidate=True
        )
        print("☁️ Uploaded embeddings and metadata to Cloudinary (cache invalidated)")
    def load_person_ids(self):
        person_list = get_person_ids_from_bucket(self.bucket_name)
        return person_list

    def load(self, load_local: bool = False):
        """
        Load embeddings and metadata.

        If load_local=True: load from local storage only.
        Otherwise, attempt Cloudinary first, fallback to local on failure.
        """
        embeddings_public_id = f"{self.bucket_name}/Embeddings/{self.recognition_model_name}_embeddings"
        metadata_public_id = f"{self.bucket_name}/Embeddings/{self.recognition_model_name}_metadata"

        # Nếu chọn load từ local thì bỏ qua cloudinary luôn
        if load_local:
            try:
                embeddings = np.load(self.embeddings_file)
                with open(self.metadata_file, "rb") as f:
                    image2class, index2class = pickle.load(f)
                print("✅ Loaded from local storage (by request)")
                return embeddings, image2class, index2class
            except Exception as e:
                print(f"❌ Failed to load locally: {e}")
                return None, None, None

        # Load từ Cloudinary trước, nếu thất bại thì fallback về local
        try:
            base_emb_url = cloudinary.CloudinaryImage(embeddings_public_id).build_url(resource_type="raw")
            base_meta_url = cloudinary.CloudinaryImage(metadata_public_id).build_url(resource_type="raw")
            ts = int(time.time())
            emb_url = f"{base_emb_url}?t={ts}"
            meta_url = f"{base_meta_url}?t={ts}"

            emb_response = requests.get(emb_url)
            emb_response.raise_for_status()
            embeddings = np.load(BytesIO(emb_response.content))

            meta_response = requests.get(meta_url)
            meta_response.raise_for_status()
            image2class, index2class = pickle.load(BytesIO(meta_response.content))

            print("✅ Loaded from Cloudinary (fresh)")
            return embeddings, image2class, index2class

        except Exception as e:
            print(f"⚠️ Cloudinary load failed: {e}")
            print("🔁 Trying local fallback...")
            try:
                embeddings = np.load(self.embeddings_file)
                with open(self.metadata_file, "rb") as f:
                    image2class, index2class = pickle.load(f)
                print("✅ Loaded from local storage (fallback)")
                return embeddings, image2class, index2class
            except Exception as le:
                print(f"❌ Local load failed too: {le}")
                return None, None, None

    def add_employee(self, UPLOAD_FOLDER, name, age, gender, salary, email, year):
        person_id = add_person(self.bucket_name, UPLOAD_FOLDER, name, age, gender, salary, email, year)

        print(f"➕ Adding new employee: {person_id}")

        embeddings, image2class, index2class = self.load()
        if embeddings is None:
            embeddings = np.zeros((0, 512))
            image2class = {}
            index2class = {}

        recognition_model = get_recogn_model()

        # ---------- Bước 1: Lấy ảnh ----------
        images = []
        local_image_paths = glob(os.path.join(UPLOAD_FOLDER, "*"))

        if local_image_paths:
            print(f"📂 Found {len(local_image_paths)} local images in {UPLOAD_FOLDER}")
            for img_path in local_image_paths:
                try:
                    image = Image.open(img_path).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"❌ Error loading local image {img_path}: {e}")
        else:
            print(f"🌐 No local images found, loading images from Firebase...")
            person_data = get_data(f"{self.bucket_name}/Employees/{person_id}")
            image_urls = person_data.get("images", []) if person_data else []
            if not image_urls:
                print(f"⚠️ No images found for {person_id}")
                return
            for url in image_urls:
                try:
                    response = requests.get(url)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    images.append(image)
                except Exception as e:
                    print(f"❌ Error loading image from URL {url}: {e}")

        # Nếu không có ảnh thì dừng
        if not images:
            print(f"❌ No valid images loaded for {person_id}")
            return

        # ---------- Bước 2: Xử lý ảnh ----------
        aligned = []
        image_index = len(image2class)
        
        # Tìm class_index mới bằng cách tìm khoảng trống trong index2class
        used_indices = set(index2class.keys())
        class_index = 0
        while class_index in used_indices:
            class_index += 1

        for image in images:
            try:
                x_aligned = mtcnn(image)
                if x_aligned is not None:
                    x_aligned = transform_image(x_aligned)
                else:
                    image = np.array(image)
                    face, _, prob = detect_face_and_nose(image)
                    if face is not None and prob > 0.7:
                        x1, y1, x2, y2 = map(int, face)
                        face_crop = image.crop((x1, y1, x2, y2))
                        x_aligned = transform_image(face_crop)
                    else:
                        x_aligned = transform_image(image)

                aligned.append(x_aligned)
                image2class[image_index] = class_index
                image_index += 1
            except Exception as e:
                print(f"❌ Error processing image: {e}")

        if not aligned:
            print(f"❌ No valid images processed for {person_id}")
            return

        # ---------- Bước 3: Tính embedding ----------
        batch = torch.cat(aligned, dim=0).to(device)
        with torch.no_grad():
            new_embeddings = recognition_model(batch).detach().cpu().numpy()

        embeddings = np.vstack([embeddings, new_embeddings])
        index2class[class_index] = person_id

        # ---------- Bước 4: Lưu ----------
        self.save_local(embeddings, image2class, index2class)
        self.upload_to_cloudinary(embeddings, image2class, index2class)

        print(f"✅ {person_id} added successfully with {len(new_embeddings)} embeddings")
        return embeddings, image2class, index2class


    def delete_employee(self, person_id):
        print(f"➖ Deleting employee: {person_id}")

          # Xóa luôn dữ liệu trên Firebase và Cloudinary
        delete_person(self.bucket_name, person_id)

        # Load embeddings và metadata
        embeddings, image2class, index2class = self.load()
        if embeddings is None:
            print("❌ Cannot load embeddings and metadata.")
            return False

        # Tìm class_index tương ứng với person_id
        class_indices_to_delete = [idx for idx, pid in index2class.items() if pid == person_id]
        if not class_indices_to_delete:
            print(f"⚠️ Person ID {person_id} not found in metadata.")
            return False

        class_index_to_delete = class_indices_to_delete[0]

        # Xác định image indices cần giữ lại
        image_indices_to_keep = [img_idx for img_idx, cls_idx in image2class.items() if cls_idx != class_index_to_delete]

        if len(image_indices_to_keep) == len(image2class):
            print(f"⚠️ No images found for person {person_id} to delete.")
            return False

        # Backup image2class cũ
        old_image2class = image2class.copy()

        # Cập nhật embeddings
        embeddings = embeddings[image_indices_to_keep]

        # Rebuild image2class
        new_image2class = {}
        for new_idx, old_idx in enumerate(image_indices_to_keep):
            if old_idx in old_image2class:
                new_image2class[new_idx] = old_image2class[old_idx]

        # Xóa person_id khỏi index2class
        del index2class[class_index_to_delete]

        # Save lại local và upload lên cloud
        self.save_local(embeddings, new_image2class, index2class)
        self.upload_to_cloudinary(embeddings, new_image2class, index2class)

        print(f"✅ Deleted {person_id} successfully.")
        return embeddings, new_image2class, index2class


if __name__ == '__main__':
    
    # create_data_embeddings('Neu')
    manager = EmbeddingManager('Hust')
    embeddings, image2class, index2class = manager.load(load_local = True)
    if embeddings is None:
        print("❌ Failed to load embeddings!")
    else:
        print(f"✅ Loaded {embeddings.shape} embeddings")
        print(f"Image to class mapping: {image2class}")
        print(f"Index to class mapping: {index2class}")
    # manager.delete_employee('000010')




