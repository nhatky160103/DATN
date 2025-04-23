import torch
import numpy as np
import os
import pickle
from PIL import Image
from tqdm import tqdm
from io import BytesIO
import cloudinary.uploader
import io
import requests
from .infer_image import  transform_image, mtcnn
from .utils import get_recogn_model
from database.firebase import get_data
from pathlib import Path
from .blazeFace import detect_face_and_nose

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

workers = 0 if os.name == 'nt' else 4


def custom_transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_batch = transform(img) 
    img_batch = img_batch.unsqueeze(0) 
    return img_batch

class EmbeddingManager:
    def __init__(self, bucket_name, recognition_model_name, local_root="local_embeddings"):
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
        print(f"💾 Saved locally → {self.embeddings_file}, {self.metadata_file}")

    def upload_to_cloudinary(self, embeddings, image2class, index2class):
        # Upload embeddings
        npy_buffer = io.BytesIO()
        np.save(npy_buffer, embeddings)
        npy_buffer.seek(0)
        upload_result = cloudinary.uploader.upload(
            npy_buffer,
            folder=self.cloud_folder,
            public_id=f"{self.recognition_model_name}_embeddings",
            resource_type="raw",
            overwrite=True
        )
        print(f"☁️ Uploaded embeddings → {upload_result['secure_url']}")

        # Upload metadata
        meta_buffer = io.BytesIO()
        pickle.dump((image2class, index2class), meta_buffer)
        meta_buffer.seek(0)
        upload_result = cloudinary.uploader.upload(
            meta_buffer,
            folder=self.cloud_folder,
            public_id=f"{self.recognition_model_name}_metadata",
            resource_type="raw",
            overwrite=True
        )
        print(f"☁️ Uploaded metadata → {upload_result['secure_url']}")

    def load(self):
        embeddings_public_id = f"{self.bucket_name}/Embeddings/{self.recognition_model_name}_embeddings"
        metadata_public_id = f"{self.bucket_name}/Embeddings/{self.recognition_model_name}_metadata"

        try:
            # Build URL trực tiếp
            emb_url = cloudinary.CloudinaryImage(embeddings_public_id).build_url(resource_type="raw")
            meta_url = cloudinary.CloudinaryImage(metadata_public_id).build_url(resource_type="raw")

            emb_response = requests.get(emb_url)
            emb_response.raise_for_status()
            embeddings = np.load(io.BytesIO(emb_response.content))

            meta_response = requests.get(meta_url)
            meta_response.raise_for_status()
            image2class, index2class = pickle.load(io.BytesIO(meta_response.content))

            print("✅ Loaded from Cloudinary")

            return embeddings, image2class, index2class

        except Exception as e:
            print(f"⚠️ Cloudinary load failed: {e}")
            print("🔁 Trying local fallback...")

            try:
                embeddings = np.load(self.embeddings_file)
                with open(self.metadata_file, "rb") as f:
                    image2class, index2class = pickle.load(f)
                print("✅ Loaded from local storage")
                return embeddings, image2class, index2class

            except Exception as le:
                print(f"❌ Local load failed too: {le}")
                return None, None, None


def create_data_embeddings(bucket_name, recognition_model_name = 'ms1mv3_arcface', backbone_name='r50', batch_size: int = 32, device='cpu'):
    # Tải mô hình nhận dạng
    recognition_model = get_recogn_model(recognition_model_name).to(device)
    recognition_model.eval()

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
                        x_aligned = custom_transform_image(face)
                    else:
                        x_aligned = custom_transform_image(image)

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
    manager = EmbeddingManager(bucket_name, recognition_model_name)
    manager.save_local(embeddings, image2class, index2class)
    manager.upload_to_cloudinary(embeddings, image2class, index2class)

    return embeddings, image2class, index2class


if __name__ == '__main__':
    
    # create_data_embeddings('Hust', 'glint360k_cosface', 'r50')
    manager = EmbeddingManager('Hust', 'glint360k_cosface')
    embeddings, image2class, index2class = manager.load()
    if embeddings is None:
        print("❌ Failed to load embeddings!")
    else:
        print(f"✅ Loaded {embeddings.shape[0]} embeddings")
        print(f"Image to class mapping: {image2class}")
        print(f"Index to class mapping: {index2class}")




