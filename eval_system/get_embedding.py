import os
import torch
import numpy as np
import pickle
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from glob import glob
import cv2

from infer.infer_image import transform_image
from infer.blazeFace import detect_face_and_nose
from infer.utils import mtcnn, device, get_recogn_model

def create_embeddings_from_folder(model, parent_folder, save_dir="local_embeddings", model_name="arcface"):
    """
    Tạo embeddings từ thư mục ảnh và lưu kết quả tại local.

    Args:
        model (torch.nn.Module): Mô hình nhận diện (đã load sẵn).
        parent_folder (str): Thư mục cha chứa các folder con, mỗi folder ứng với 1 người.
        save_dir (str): Thư mục lưu trữ kết quả local.
        model_name (str): Tên mô hình (dùng làm prefix cho file lưu).

    Output:
        Lưu 2 file:
        - {model_name}_embeddings.npy
        - {model_name}_metadata.pkl
    """

    embeddings = []
    image2class = {}
    index2class = {}

    image_index = 0
    class_index = 0

    person_folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    person_folders.sort()  # đảm bảo thứ tự nhất quán

    for person_id in tqdm(person_folders, desc="Processing persons"):
        folder_path = os.path.join(parent_folder, person_id)
        image_paths = glob(os.path.join(folder_path, "*"))
        aligned = []

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                x_aligned = mtcnn(image)
                if x_aligned is not None:
                    x_aligned = transform_image(x_aligned)
                else:
                    blaze_input = cv2.imread(img_path)
                    face, _, prob = detect_face_and_nose(blaze_input)
                    if face is not None and prob > 0.8:
                        x1, y1, x2, y2 = map(int, face)
                        face_crop = image.crop((x1, y1, x2, y2))
                        x_aligned = transform_image(face_crop)
                    else:
                        x_aligned = transform_image(image)

                aligned.append(x_aligned)
                image2class[image_index] = class_index
                image_index += 1

            except Exception as e:
                print(f" Error loading or processing {img_path}: {e}")

        if not aligned:
            print(f" No valid images found in {folder_path}")
            continue

        batch = torch.cat(aligned, dim=0).to(device)
        with torch.no_grad():
            person_embeddings = model(batch).detach().cpu().numpy()

        embeddings.append(person_embeddings)
        index2class[class_index] = person_id
        class_index += 1

    # Stack embeddings
    embeddings = np.vstack(embeddings)

    # Tạo thư mục lưu
    save_path = Path(os.path.join(save_dir, 'Test'))
    save_path.mkdir(parents=True, exist_ok=True)
    emb_file = save_path / f"{model_name}_embeddings.npy"
    meta_file = save_path / f"{model_name}_metadata.pkl"

    # Lưu file
    np.save(emb_file, embeddings)
    with open(meta_file, "wb") as f:
        pickle.dump((image2class, index2class), f)

    print(f" Saved embeddings to {emb_file}")
    print(f" Saved metadata to {meta_file}")

def load_embeddings_and_metadata(save_dir="local_embeddings", model_name="arcface", subfolder="Test"):
    """
    Load embeddings và metadata đã lưu từ ổ đĩa.

    Args:
        save_dir (str): Thư mục chứa kết quả lưu embeddings và metadata.
        model_name (str): Tên mô hình, dùng để xác định file cần load.
        subfolder (str): Tên thư mục con (ví dụ "Test").

    Returns:
        Tuple: (embeddings: np.ndarray, image2class: dict, index2class: dict)
    """
    save_path = Path(save_dir) / subfolder
    emb_file = save_path / f"{model_name}_embeddings.npy"
    meta_file = save_path / f"{model_name}_metadata.pkl"

    # Kiểm tra tồn tại
    if not emb_file.exists() or not meta_file.exists():
        raise FileNotFoundError(f"Không tìm thấy file tại {save_path}")

    # Load dữ liệu
    embeddings = np.load(emb_file)
    with open(meta_file, "rb") as f:
        image2class, index2class = pickle.load(f)

    # print(f" Loaded embeddings từ {emb_file}")
    # print(f" Loaded metadata từ {meta_file}")

    return embeddings, image2class, index2class


if __name__ == "__main__":
    
    model = get_recogn_model()
    folder = 'models/Recognition/Arcface_torch/dataset/VN-celeb-mini'
    # create_embeddings_from_folder(model, folder)

    embeddings, image2class, index2class = load_embeddings_and_metadata(
    save_dir="local_embeddings", model_name="arcface", subfolder="Test"
    )

    print(embeddings.shape)
    print(image2class)
    print(index2class)
