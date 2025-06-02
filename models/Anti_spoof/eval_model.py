import time
import psutil
import numpy as np

def count_parameters(model):
    """Đếm số tham số có thể train của một nn.Module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_fasnet(fasnet_model, img: np.ndarray, facial_area: tuple, repeat: int = 1000):
    """
    Tính các thông số cho mô hình Fasnet:
      - total_params: tổng số tham số của cả hai backbone
      - avg_infer_ms: thời gian inference trung bình (ms)
      - ram_usage_mb: bộ nhớ RAM đang sử dụng (MB)

    Args:
        fasnet_model: instance của class Fasnet (đã load weight, .eval()).
        img (np.ndarray): ảnh gốc (H x W x C).
        facial_area (tuple): (x, y, w, h) bounding box khuôn mặt.
        repeat (int): số lần lặp để đo thời gian (mặc định 100).
    """
    # 1. Đếm tham số
    total_params = (
        count_parameters(fasnet_model.first_model) +
        count_parameters(fasnet_model.second_model)
    )

    start = time.time()
    for _ in range(repeat):
        _ = fasnet_model.analyze(img, facial_area)
    elapsed = time.time() - start
    avg_infer_ms = elapsed / repeat * 1000.0

    # 3. Đo RAM sử dụng hiện tại
    process = psutil.Process()
    ram_usage_mb = process.memory_info().rss / 1024**2

    return {
        'total_params': total_params,
        'avg_infer_ms': avg_infer_ms,
        'ram_usage_mb': ram_usage_mb
    }

# Ví dụ sử dụng:
if __name__ == "__main__":
    import cv2
    from .FasNet import Fasnet   # sửa lại path import của bạn

    img = cv2.imread('data/img1.jpg')
    box = (0, 0, img.shape[1], img.shape[0])

    model = Fasnet()
    stats = profile_fasnet(model, img, box, repeat=100)
    print(f"Số tham số: {stats['total_params']:,}")
    print(f"Avg inference time: {stats['avg_infer_ms']:.2f} ms")
    print(f"RAM usage: {stats['ram_usage_mb']:.2f} MB")
