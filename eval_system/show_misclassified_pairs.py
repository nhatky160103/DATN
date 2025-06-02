import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import cv2

from infer.infer_image import transform_image
from infer.utils import mtcnn, device, get_recogn_model
from eval_system.get_embedding import load_embeddings_and_metadata

def get_image_embedding(model, img_path):
    image = Image.open(img_path).convert("RGB")
    x_aligned = mtcnn(image)
    if x_aligned is not None:
        x_aligned = transform_image(x_aligned)
    else:
        blaze_input = cv2.imread(img_path)
        from infer.blazeFace import detect_face_and_nose
        face, _, prob = detect_face_and_nose(blaze_input)
        if face is not None and prob > 0.7:
            x1, y1, x2, y2 = map(int, face)
            image = image.crop((x1, y1, x2, y2))
        x_aligned = transform_image(image)
    x_aligned = x_aligned.to(device)
    with torch.no_grad():
        emb = model(x_aligned).detach().cpu().numpy().squeeze()
    return emb

def cosine_distance(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    return 1 - np.dot(a, b)

def parse_line(line):
    # Ví dụ: models/Recognition/Arcface_torch/dataset/VN-celeb-mini/1030/4.png | True: 1030 | Predicted: -1
    path, true_str, pred_str = line.strip().split('|')
    path = path.strip()
    true_cls = true_str.replace('True:', '').strip()
    pred_cls = pred_str.replace('Predicted:', '').strip()
    return path, true_cls, pred_cls

def get_all_images_in_class(dataset_root, class_id):
    class_dir = os.path.join(dataset_root, class_id)
    if not os.path.exists(class_dir):
        return []
    return [os.path.join(class_dir, f) for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def show_pair(img1_path, img2_path, title1, title2, distance):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(Image.open(img1_path))
    axs[0].set_title(f"Infer: {title1}")
    axs[0].axis('off')
    axs[1].imshow(Image.open(img2_path))
    axs[1].set_title(f"Predicted class: {title2}\nDist: {distance:.4f}")
    axs[1].axis('off')
    plt.tight_layout()
    plt.show()

def main():
    misclassified_file = "eval_system/evaluation_results_cosine_0.7/misclassified_samples.txt"
    dataset_root = "models/Recognition/Arcface_torch/dataset/VN-celeb-mini"
    print("Loading model...")
    model = get_recogn_model()
    print("Model loaded.")
    with open(misclassified_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    for line in tqdm(lines):
        img_path, true_cls, pred_cls = parse_line(line)
        if pred_cls == '-1' or pred_cls == true_cls:
            continue  # Bỏ qua các trường hợp không nhận diện được hoặc nhận đúng

        # Lấy embedding của ảnh infer
        try:
            emb_infer = get_image_embedding(model, img_path)
        except Exception as e:
            print(f"Không lấy được embedding cho {img_path}: {e}")
            continue

        # Lấy tất cả ảnh trong class bị nhận nhầm
        pred_class_images = get_all_images_in_class(dataset_root, pred_cls)
        if not pred_class_images:
            print(f"Không tìm thấy ảnh trong class {pred_cls}")
            continue

        min_dist = float('inf')
        min_img = None
        for img in pred_class_images:
            try:
                emb = get_image_embedding(model, img)
                dist = cosine_distance(emb_infer, emb)
                if dist < min_dist:
                    min_dist = dist
                    min_img = img
            except Exception as e:
                print(f"Không lấy được embedding cho {img}: {e}")
                continue
        if min_img:
            show_pair(img_path, min_img, true_cls, pred_cls, min_dist)

if __name__ == "__main__":
    main() 