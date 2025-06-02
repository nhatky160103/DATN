import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine
from ..backbones import get_model

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).float().unsqueeze(0)
    img.div_(255).sub_(0.5).div_(0.5)
    return img.to(device)

@torch.no_grad()
def get_embedding(model, img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = preprocess_image(img)
    embedding = model(img)
    return embedding.squeeze(0).cpu().numpy()

def keep_top_5_similar_images(model, folder_path):
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_paths = [os.path.join(folder_path, f) for f in image_files]

    embeddings = []
    valid_paths = []

    # Tính embedding cho tất cả ảnh

    if len(image_paths) <= 5:
        print(f"Skipping {folder_path} (only {len(image_paths)} valid images)")
        return 

    for path in image_paths:
        emb = get_embedding(model, path)
        if emb is not None:
            embeddings.append(emb)
            valid_paths.append(path)



    # Chọn ảnh trung tâm làm mean embedding
    mean_emb = np.mean(embeddings, axis=0)

    # Tính độ tương đồng với vector trung bình
    similarities = [1 - cosine(e, mean_emb) for e in embeddings]

    # Lấy 5 ảnh có độ tương đồng cao nhất
    top5_indices = np.argsort(similarities)[-5:]
    top5_paths = [valid_paths[i] for i in top5_indices]

    # Xóa các ảnh còn lại
    for path in valid_paths:
        if path not in top5_paths:
            os.remove(path)

    print(f"✔️ {folder_path}: Kept {len(top5_paths)} images, removed {len(valid_paths) - len(top5_paths)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help="Path to model weight (.pth)")
    parser.add_argument('--network', type=str, default='r50', help="Backbone name (e.g., r50)")
    parser.add_argument('--root_folder', type=str, required=True, help="Folder chứa các folder con (mỗi folder là 1 người)")

    args = parser.parse_args()

    model = get_model(args.network, fp16=False)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval().to(device)

    subfolders = [os.path.join(args.root_folder, d) for d in os.listdir(args.root_folder) if os.path.isdir(os.path.join(args.root_folder, d))]

    for subfolder in tqdm(subfolders):
        keep_top_5_similar_images(model, subfolder)
