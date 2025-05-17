import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from PIL import Image
import cv2
from glob import glob
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)

from infer.infer_image import transform_image
from infer.blazeFace import detect_face_and_nose
from infer.utils import mtcnn, device, get_recogn_model
from .get_embedding import load_embeddings_and_metadata
from infer.identity_person import find_closest_person


def compute_tar_far(scores, labels, threshold):
    labels = np.array(labels)
    scores = np.array(scores)
    preds = (scores >= threshold).astype(int)

    true_accepts = np.logical_and(preds == 1, labels == 1).sum()
    false_accepts = np.logical_and(preds == 1, labels == 0).sum()
    total_accepts = (labels == 1).sum()
    total_rejects = (labels == 0).sum()

    tar = true_accepts / (total_accepts + 1e-8)
    far = false_accepts / (total_rejects + 1e-8)
    return tar, far


def compute_eer(fpr, tpr):
    # EER xảy ra khi FAR = FRR => FRR = 1 - TPR
    eer_threshold = np.nanargmin(np.abs(fpr - (1 - tpr)))
    eer = fpr[eer_threshold]
    return eer


def evaluate_recognition_system(
    model,
    parent_folder,
    embeddings,
    image2class,
    index2class,
    distance_mode="cosine",
    l2_threshold=1.0,
    cosine_threshold=0.4,
    show_plot=True,
    save_dir="evaluation_results"
):
    os.makedirs(save_dir, exist_ok=True)

    y_true, y_pred, scores, labels = [], [], [], []
    misclassified_samples = []
    class2index = {v: k for k, v in index2class.items()}

    person_folders = sorted([
        d for d in os.listdir(parent_folder)
        if os.path.isdir(os.path.join(parent_folder, d))
    ])

    for person_id in tqdm(person_folders, desc="Evaluating"):
        folder_path = os.path.join(parent_folder, person_id)
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
                    distance_mode,
                    l2_threshold,
                    cosine_threshold
                )

                true_class = class2index[person_id]
                y_true.append(true_class)
                y_pred.append(pred_class)

                is_correct = (pred_class == true_class)
                labels.append(int(is_correct))

              
                if distance_mode == 'l2':
                    distances = np.linalg.norm(embeddings - pred_embed.numpy(), axis=1)
                    scores.append(-np.min(distances))  # closer is better
                else:
                    sims = np.dot(embeddings, pred_embed.numpy().T).squeeze()
                    scores.append(np.max(sims))  # higher is better


                if not is_correct:
                    pred_class_name = index2class[pred_class] if pred_class != -1 else -1
                    misclassified_samples.append(
                        f"{img_path} | True: {index2class[true_class]} | Predicted: {pred_class_name}"
                    )

            except Exception as e:
                print(f"❌ Error processing {img_path}: {e}")

    # Convert
    y_true, y_pred, scores, labels = map(np.array, (y_true, y_pred, scores, labels))
    valid = y_pred != -1


    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true[valid], y_pred[valid], average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true[valid], y_pred[valid])

    # ROC & AUC
    fpr, tpr, thresholds = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)
    eer = compute_eer(fpr, tpr)

    # TAR/FAR tại vài threshold
    tar_far_table = {}
    for th in [0.3, 0.5, 0.7, 0.8, 0.9]:
        tar, far = compute_tar_far(scores, labels, threshold=th)
        tar_far_table[th] = {"TAR": tar, "FAR": far}

    # Save errors
    with open(os.path.join(save_dir, "misclassified_samples.txt"), "w", encoding="utf-8") as f:
        for line in misclassified_samples:
            f.write(line + "\n")
    print(f"📄 Misclassified samples saved to {save_dir}/misclassified_samples.txt")

    # Confusion matrix
    if show_plot:
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
        plt.show()

        # ROC Curve
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "roc_curve.png"))
        plt.show()

    return {
        "accuracy": (y_true == y_pred).sum() / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "eer": eer,
        "tar_far": tar_far_table,
        "num_total": len(y_true),
        "num_matched": y_true.sum()
    }


if __name__ == "__main__":
    model = get_recogn_model()
    embeddings, image2class, index2class = load_embeddings_and_metadata(
        save_dir="local_embeddings", model_name="arcface", subfolder="Test"
    )
    metrics = evaluate_recognition_system(
        model,
        parent_folder="models/Recognition/Arcface_torch/dataset/VN-celeb-mini",
        embeddings=embeddings,
        image2class=image2class,
        index2class=index2class,
        distance_mode="cosine",
        cosine_threshold=0.8,
        show_plot=True
    )
    for k, v in metrics.items():
        print(f"{k}: {v}")
