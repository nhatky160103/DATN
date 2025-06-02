import os
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import multiprocessing
from infer.utils import get_recogn_model
from .get_embedding import load_embeddings_and_metadata
from eval_system.metric import evaluate_recognition_system

def evaluate_for_threshold(args):
    cosine_threshold, embeddings_path, parent_folder, device_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    device = f"cuda:{device_id}" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    save_dir = f"evaluation_results_cosine_{cosine_threshold}"
    # Load model và embeddings trong mỗi process
    model = get_recogn_model()
    embeddings, image2class, index2class = load_embeddings_and_metadata(
        save_dir=embeddings_path, model_name="arcface", subfolder="Test"
    )
    metrics = evaluate_recognition_system(
        model,
        parent_folder=parent_folder,
        embeddings=embeddings,
        image2class=image2class,
        index2class=index2class,
        distance_mode="cosine",
        cosine_threshold=cosine_threshold,
        show_plot=True,
        save_dir=save_dir,
        device=device
    )
    # Lưu metrics vào file txt
    metrics_txt_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_txt_path, "w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")
    print(f"📄 Metrics saved to {metrics_txt_path}")
    return {
        "fpr": metrics["fpr"],
        "tpr": metrics["tpr"],
        "roc_auc": metrics["roc_auc"],
        "threshold": cosine_threshold
    }

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    # Đường dẫn dữ liệu
    parent_folder = "models/Recognition/Arcface_torch/dataset/VN-celeb-mini"
    embeddings_path = "local_embeddings"
    thresholds = [0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
    num_gpus = 2
    # Chuẩn bị args cho mỗi process, chia đều cho 2 GPU
    args_list = [
        (th, embeddings_path, parent_folder, i % num_gpus)
        for i, th in enumerate(thresholds)
    ]
    roc_curves = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(evaluate_for_threshold, args_list))
    roc_curves.extend(results)

    # Vẽ tổng hợp các đường ROC
    plt.figure(figsize=(8, 8))
    for roc in roc_curves:
        plt.plot(roc["fpr"], roc["tpr"], label=f"Threshold={roc['threshold']}, AUC={roc['roc_auc']:.2f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("All ROC Curves for Different Cosine Thresholds (Parallel)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join('eval_system', "all_roc_curves_parallel.png"))
    plt.show() 