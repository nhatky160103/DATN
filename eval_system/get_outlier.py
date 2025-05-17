from sklearn.metrics.pairwise import cosine_similarity
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


from infer.infer_image import transform_image
from infer.blazeFace import detect_face_and_nose
from infer.utils import mtcnn, device, get_recogn_model


def detect_outlier_images(model, parent_folder, threshold=0.5):
    """
    Dùng embedding để phát hiện ảnh không cùng người trong mỗi folder và hiển thị ảnh lạc loài.

    Args:
        model: Mô hình nhận diện khuôn mặt.
        parent_folder (str): Thư mục cha chứa các folder con (mỗi người).
        threshold (float): Ngưỡng khoảng cách để coi là lạc loài.

    Returns:
        None: Hiển thị ảnh nghi ngờ từ từng thư mục.
    """
    person_folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    person_folders.sort()

    for person_id in tqdm(person_folders, desc="Detecting outliers"):
        folder_path = os.path.join(parent_folder, person_id)
        image_paths = glob(os.path.join(folder_path, "*"))

        aligned = []
        valid_paths = []

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                x_aligned = mtcnn(image)

                if x_aligned is None:
                    face, _, prob = detect_face_and_nose(image)
                    if face is not None and prob > 0.7:
                        x1, y1, x2, y2 = map(int, face)
                        x_aligned = image.crop((x1, y1, x2, y2))
                    else:
                        # Nếu không phát hiện khuôn mặt, lấy ảnh gốc
                        x_aligned = image
                x_aligned = transform_image(x_aligned)
                aligned.append(x_aligned)
                valid_paths.append(img_path)

            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")

        if len(aligned) < 3:
            continue  # Không đủ ảnh để đánh giá

        batch = torch.cat(aligned, dim=0).to(device)
        with torch.no_grad():
            embs = model(batch).detach().cpu().numpy()

        # Tính similarity ma trận
        sim_matrix = cosine_similarity(embs)
        sim_scores = sim_matrix.mean(axis=1)  # độ tương đồng trung bình của mỗi ảnh với phần còn lại

        # Ảnh nào có độ tương đồng thấp bất thường thì đánh dấu
        mean_sim = np.mean(sim_scores)
        std_sim = np.std(sim_scores)

        person_outliers = []
        for i, score in enumerate(sim_scores):
            if score < mean_sim - std_sim:  # thấp hơn 1 std → nghi ngờ
                person_outliers.append(valid_paths[i])

        if person_outliers:
            print(f"{person_id}:")
            for img_path in person_outliers:
                print(f"    - {os.path.basename(img_path)}")  # Hiển thị chỉ tên file
            print()

if __name__ == "__main__":
    model = get_recogn_model()
    parent_folder = 'models/Recognition/Arcface_torch/dataset/VN-celeb'
    detect_outlier_images(model, parent_folder)
