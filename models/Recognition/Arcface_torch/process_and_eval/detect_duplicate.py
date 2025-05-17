import torch
import cv2
import numpy as np
from scipy.spatial.distance import cosine
import os
from tqdm import tqdm
from ..backbones import get_model
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image


def transform_image(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if not isinstance(img, torch.Tensor):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = transform(img)
    img = img.unsqueeze(0) 
    return img



def extract_folder_embedding(model, folder_path, device='cuda'):
    model.eval()
    model.to(device)
    embeddings = []
    for fname in os.listdir(folder_path):
        if not fname.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        img_path = os.path.join(folder_path, fname)
        try:
            img = transform_image(img_path).to(device)
            with torch.no_grad():
                emb = model(img).cpu().numpy().flatten()
            embeddings.append(emb)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return None

def find_duplicate_folders(model, root_dir, threshold=0.6, device='cuda'):
    folder_paths = [os.path.join(root_dir, name) for name in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, name))]
    
    print(f"Tìm {len(folder_paths)} thư mục...")
    folder_embeddings = {}
    for folder in tqdm(folder_paths, desc="Trích xuất đặc trưng các thư mục"):
        emb = extract_folder_embedding(model, folder, device)
        if emb is not None:
            folder_embeddings[folder] = emb

    folders = list(folder_embeddings.keys())
    duplicates = []

    print("So sánh các thư mục...")
    for i in tqdm(range(len(folders))):
        for j in range(i + 1, len(folders)):
            emb_i = torch.tensor(folder_embeddings[folders[i]], dtype=torch.float32)
            emb_j = torch.tensor(folder_embeddings[folders[j]], dtype=torch.float32)

            similarity = F.cosine_similarity(emb_i.unsqueeze(0), emb_j.unsqueeze(0)).item()
            distance = 1 - similarity

            if distance < threshold:
                duplicates.append((folders[i], folders[j], distance))

    return duplicates

if __name__ == "__main__":
    import argparse

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, required=True, help="Path to model weight (.pth)")
    parser.add_argument('--network', type=str, default='r50', help="Backbone name (e.g., r50)")
    parser.add_argument('--root_folder', type=str, required=True, help="Folder chứa các folder con (mỗi folder là 1 người)")

    args = parser.parse_args()

    model = get_model(args.network, fp16=False)
    model.load_state_dict(torch.load(args.weight, map_location=device))
    model.eval().to(device)
    duplicates = find_duplicate_folders(
        model=model,
        root_dir= args.root_folder,
        threshold=0.43,
        device=device  
    )

    for folder1, folder2, dist in duplicates:
        print(f"[MATCH] {os.path.basename(folder1)} <--> {os.path.basename(folder2)} (distance = {dist:.4f})")
