import os
import re
import matplotlib.pyplot as plt

# Thư mục chứa các kết quả
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = BASE_DIR

# Regex để lấy threshold từ tên thư mục
THRESHOLD_PATTERN = re.compile(r"evaluation_results_cosine_(\d+\.?\d*)")

# Các chỉ số cần vẽ (bỏ roc_auc, eer)
METRICS = [
    ("accuracy", "TPR"),
    "precision", "recall", "f1_score", "FAR", "FRR"
]

# Tìm tất cả các thư mục evaluation_results_cosine_*
subdirs = [d for d in os.listdir(RESULTS_DIR)
           if os.path.isdir(os.path.join(RESULTS_DIR, d)) and THRESHOLD_PATTERN.match(d)]

# Đọc metrics từ từng thư mục
all_metrics = []
for subdir in subdirs:
    m = THRESHOLD_PATTERN.match(subdir)
    if not m:
        continue
    threshold = float(m.group(1))
    metrics_path = os.path.join(RESULTS_DIR, subdir, "metrics.txt")
    if not os.path.exists(metrics_path):
        continue
    metrics = {"threshold": threshold}
    with open(metrics_path, "r", encoding="utf-8") as f:
        for line in f:
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            k = k.strip()
            v = v.strip()
            if k in ["accuracy", "TPR", "precision", "recall", "f1_score", "FAR", "FRR"]:
                try:
                    metrics[k] = float(v)
                except ValueError:
                    pass
    all_metrics.append(metrics)

# Sắp xếp theo threshold tăng dần
all_metrics.sort(key=lambda x: x["threshold"])

# Vẽ biểu đồ
fig, axes = plt.subplots(3, 2, figsize=(14, 12))
font_title = 18
font_label = 15
font_tick = 13
axes = axes.flatten()

for i, metric in enumerate(METRICS):
    ax = axes[i]
    if isinstance(metric, tuple):
        x = [m["threshold"] for m in all_metrics if metric[0] in m and metric[1] in m]
        y1 = [m[metric[0]] for m in all_metrics if metric[0] in m and metric[1] in m]
        y2 = [m[metric[1]] for m in all_metrics if metric[0] in m and metric[1] in m]
        ax.plot(x, y1, marker="o", label="Accuracy")
        ax.plot(x, y2, marker="s", label="TAR")
        ax.set_title("Accuracy & TAR", fontsize=font_title)
        ax.legend(fontsize=font_label)
        ax.set_ylabel("Value", fontsize=font_label)
    else:
        x = [m["threshold"] for m in all_metrics if metric in m]
        y = [m[metric] for m in all_metrics if metric in m]
        ax.plot(x, y, marker="o")
        ax.set_title(metric, fontsize=font_title)
        ax.set_ylabel(metric, fontsize=font_label)
    # ax.set_xlabel("Cosine threshold", fontsize=font_label)
    ax.grid(True)
    ax.tick_params(axis='both', labelsize=font_tick)

# Ẩn các ô thừa nếu có
for j in range(len(METRICS), len(axes)):
    axes[j].axis('off')

plt.tight_layout(h_pad=2.0)
plt.savefig(os.path.join(RESULTS_DIR, "metrics_vs_threshold.png"))
plt.show() 