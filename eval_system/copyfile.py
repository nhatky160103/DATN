import os
import shutil
from glob import glob
from tqdm import tqdm
from PIL import Image
import cv2
import torch

from infer.infer_image import transform_image
from infer.blazeFace import detect_face_and_nose
from infer.utils import mtcnn, device, get_recogn_model
from eval_system.get_embedding import load_embeddings_and_metadata
from infer.identity_person import find_closest_person

# Đường dẫn
VN_CELEB = r"models/Recognition/Arcface_torch/dataset/VN-celeb"
VN_CELEB_MINI = r"models/Recognition/Arcface_torch/dataset/VN-celeb-mini"

# Tham số nhận diện
DISTANCE_MODE = "cosine"
L2_THRESHOLD = 1.0
COSINE_THRESHOLD = 0.7  # Có thể điều chỉnh
MODEL_NAME = "arcface"
SUBFOLDER = "Test"
SAVE_DIR = "local_embeddings"


def main():
    # Load model và embeddings của VN-celeb-mini
    print("Loading model and embeddings...")
    model = get_recogn_model()
    embeddings, image2class, index2class = load_embeddings_and_metadata(
        save_dir=SAVE_DIR, model_name=MODEL_NAME, subfolder=SUBFOLDER
    )
    class2index = {v: k for k, v in index2class.items()}

    # Duyệt qua từng người trong VN-celeb
    person_folders = sorted([
        d for d in os.listdir(VN_CELEB)
        if os.path.isdir(os.path.join(VN_CELEB, d))
    ])

    for person_id in tqdm(person_folders, desc="Processing VN-celeb"):
        folder_path = os.path.join(VN_CELEB, person_id)
        image_paths = glob(os.path.join(folder_path, "*"))

        for img_path in image_paths:
            try:
                image = Image.open(img_path).convert("RGB")
                x_aligned = mtcnn(image)
                if x_aligned is None:
                    blaze_input = cv2.imread(img_path)
                    face, _, prob = detect_face_and_nose(blaze_input)
                    if face is not None and prob > 0.7:
                        x1, y1, x2, y2 = map(int, face)
                        image = image.crop((x1, y1, x2, y2))
                x_aligned = transform_image(image).to(device)
                with torch.no_grad():
                    pred_embed = model(x_aligned).detach().cpu()
                pred_class = find_closest_person(
                    pred_embed,
                    embeddings,
                    image2class,
                    DISTANCE_MODE,
                    L2_THRESHOLD,
                    COSINE_THRESHOLD
                )
                # Nếu xác định đúng (danh tính trùng với folder gốc)
                    # Tạo folder đích nếu chưa có
                    dest_folder = os.path.join(VN_CELEB_MINI, person_id)
                    os.makedirs(dest_folder, exist_ok=True)
                    dest_path = os.path.join(dest_folder, os.path.basename(img_path))
                    # Nếu file chưa tồn tại thì copy
                    if not os.path.exists(dest_path):
                        shutil.copy2(img_path, dest_path)
            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")


def rename_folders_sequentially(parent_folder):
    folders = [d for d in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, d))]
    folders_sorted = sorted(folders)
    tmp_names = []

    # Bước 1: Đổi tất cả sang tên tạm
    for folder in folders_sorted:
        old_path = os.path.join(parent_folder, folder)
        tmp_path = os.path.join(parent_folder, folder + "_tmp_rename")
        os.rename(old_path, tmp_path)
        tmp_names.append(folder + "_tmp_rename")

    # Bước 2: Đổi từ tên tạm sang tên số thứ tự
    for idx, tmp_folder in enumerate(sorted(tmp_names)):
        tmp_path = os.path.join(parent_folder, tmp_folder)
        new_path = os.path.join(parent_folder, str(idx))
        os.rename(tmp_path, new_path)


if __name__ == "__main__":
    main() 
    