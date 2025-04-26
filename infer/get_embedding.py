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


from .infer_image import  transform_image
from .utils import get_recogn_model, device, mtcnn
from database.firebase import get_data
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

    def add_employee(self, person_id):
        print(f"➕ Adding new employee: {person_id}")

        embeddings, image2class, index2class = self.load()
        if embeddings is None:
            embeddings = np.zeros((0, 512))
            image2class = {}
            index2class = {}

        recognition_model = get_recogn_model()

        person_data = get_data(f"{self.bucket_name}/Employees/{person_id}")
        image_urls = person_data.get("images", []) if person_data else []
        if not image_urls:
            print(f"⚠️ No images found for {person_id}")
            return

        aligned = []
        image_index = len(image2class)
        class_index = max(index2class.keys(), default=-1) + 1

        for url in image_urls:
            try:
                response = requests.get(url)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
                x_aligned = mtcnn(image)
                if x_aligned is not None:
                    x_aligned = transform_image(x_aligned)
                else:
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
                print(f"❌ Error processing image {url}: {e}")

        if not aligned:
            print(f"❌ No valid images processed for {person_id}")
            return

        batch = torch.cat(aligned, dim=0).to(device)
        with torch.no_grad():
            new_embeddings = recognition_model(batch).detach().cpu().numpy()

        embeddings = np.vstack([embeddings, new_embeddings])
        index2class[class_index] = person_id

        self.save_local(embeddings, image2class, index2class)
        self.upload_to_cloudinary(embeddings, image2class, index2class)
        print(f"✅ {person_id} added successfully with {len(new_embeddings)} embeddings")
        return embeddings, image2class, index2class



def create_data_embeddings(bucket_name, batch_size: int = 32):
    # Tải mô hình nhận dạng
    recognition_model = get_recogn_model()
    # Lấy dữ liệu employees từ Firebase hoặc từ nơi nào đó
    employees_data = get_data(f'{bucket_name}/Employees')
    if not employees_data:
        print("No employees found in Employees bucket!")
        return None, None, None

    aligned = []
    image2class = {}
    index2class = {}
    image_index = 0
    class_index = 0

    print("🔍 Fetching data from Firebase Realtime Database (Employees bucket)")
    total_images = sum(len(employee_data.get("images", [])) for employee_data in employees_data.values())
    print(f"📂 Total images: {total_images}")
    print(f"📦 Batch size: {batch_size}")

    progress_bar = tqdm(total=total_images, desc="Processing images")
    
    for employee_id, employee_data in employees_data.items():
        image_urls = employee_data.get("images", [])
        if not image_urls:
            continue

        index2class[class_index] = employee_id

        for url in image_urls:
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    continue

                image = Image.open(BytesIO(response.content)).convert("RGB")
                x_aligned = mtcnn(image)  # Dùng MTCNN để căn chỉnh khuôn mặt
                image2class[image_index] = class_index

                if x_aligned is not None:
                    x_aligned = transform_image(x_aligned)
                    aligned.append(x_aligned)
                else:
                    face, center_point, prob = detect_face_and_nose(image)
                    if face is not None and prob > 0.7:
                        x1, y1, x2, y2 = map(int, face)
                        face = image.crop(x1, y1, x2, y2)
                        x_aligned = transform_image(face)
                    else:
                        x_aligned = transform_image(image)

                    aligned.append(x_aligned)
                
                image_index += 1

                if len(aligned) >= batch_size:
                    batch = torch.cat(aligned[:batch_size], dim=0).to(device)
                    with torch.no_grad():
                        embeddings_batch = recognition_model(batch).detach().cpu().numpy()
                    embeddings = embeddings_batch if 'embeddings' not in locals() else np.vstack((embeddings, embeddings_batch))
                    aligned = aligned[batch_size:]

            except Exception as e:
                continue
            finally:
                progress_bar.update(1)

        class_index += 1

    progress_bar.close()
    
    # Xử lý các ảnh còn lại
    if aligned:
        batch = torch.cat(aligned, dim=0).to(device)
        with torch.no_grad():
            embeddings_batch = recognition_model(batch).detach().cpu().numpy()
        embeddings = embeddings_batch if 'embeddings' not in locals() else np.vstack((embeddings, embeddings_batch))

    if 'embeddings' not in locals():
        print("❌ No valid embeddings created!")
        return None

    # Lúc này bạn có thể lưu embeddings & metadata vào Cloudinary và local
    manager = EmbeddingManager(bucket_name)
    manager.save_local(embeddings, image2class, index2class)
    manager.upload_to_cloudinary(embeddings, image2class, index2class)

    return embeddings, image2class, index2class


if __name__ == '__main__':
    
    # create_data_embeddings('Hust', 'ms1mv3_arcface', 'r100')
    manager = EmbeddingManager('Hust')
    embeddings, image2class, index2class = manager.load(load_local = True)
    if embeddings is None:
        print("❌ Failed to load embeddings!")
    else:
        print(f"✅ Loaded {embeddings.shape[0]} embeddings")
        print(f"Image to class mapping: {image2class}")
        print(f"Index to class mapping: {index2class}")




