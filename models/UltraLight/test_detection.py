"""
Test Ultra-Light Face Detector trên folder ảnh.
Hiển thị thời gian inference từng ảnh và thống kê tổng.

Cách dùng:
    python -m models.Detection.test_detection --folder /path/to/images
    python -m models.Detection.test_detection --folder /path/to/images --save
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

# Hỗ trợ cả chạy trực tiếp lẫn chạy qua -m
if __package__:
    from .pipeline import UltraLightDetector
else:
    sys.path.insert(0, os.path.dirname(__file__))
    from pipeline import UltraLightDetector

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: str):
    paths = []
    for fname in sorted(os.listdir(folder)):
        if os.path.splitext(fname)[1].lower() in SUPPORTED_EXTS:
            paths.append(os.path.join(folder, fname))
    return paths


def run_test(folder: str, model_path: str, conf: float, save: bool):
    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model: {model_path}")
    detector = UltraLightDetector(model_path=model_path, conf_threshold=conf)
    print("✓ Model loaded.\n")

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_images(folder)
    if not images:
        print(f"Không tìm thấy ảnh nào trong: {folder}")
        sys.exit(1)
    print(f"Tìm thấy {len(images)} ảnh trong '{folder}'\n")

    # ── Tạo output folder nếu --save ─────────────────────────────────────────
    save_dir = None
    if save:
        save_dir = os.path.join(folder, "detection_output")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Ảnh kết quả sẽ được lưu vào: {save_dir}\n")

    # ── Header bảng ───────────────────────────────────────────────────────────
    col_w = [5, 40, 12, 8]
    header = (
        f"{'#':<{col_w[0]}}"
        f"{'Tên file':<{col_w[1]}}"
        f"{'Time (ms)':>{col_w[2]}}"
        f"{'Faces':>{col_w[3]}}"
    )
    print(header)
    print("-" * sum(col_w))

    # ── Inference từng ảnh ────────────────────────────────────────────────────
    latencies = []
    total_faces = 0

    for idx, img_path in enumerate(images, 1):
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"[!] Không đọc được: {img_path}")
            continue

        t0 = time.perf_counter()
        boxes, scores = detector.detect(frame)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        latencies.append(elapsed_ms)
        total_faces += len(boxes)

        fname = os.path.basename(img_path)
        fname_display = fname
        if len(fname_display) > col_w[1] - 2:
            fname_display = "…" + fname_display[-(col_w[1] - 3):]

        print(
            f"{idx:<{col_w[0]}}"
            f"{fname_display:<{col_w[1]}}"
            f"{elapsed_ms:>{col_w[2]}.2f}"
            f"{len(boxes):>{col_w[3]}}"
        )

        # ── Lưu ảnh kết quả nếu --save ───────────────────────────────────────
        if save and save_dir:
            vis = detector.detect_and_draw(frame)
            cv2.imwrite(os.path.join(save_dir, fname), vis)

    # ── Thống kê tổng ─────────────────────────────────────────────────────────
    if not latencies:
        return

    print("-" * sum(col_w))
    print(f"\nThống kê:")
    print(f"  Số ảnh đã test  : {len(latencies)}")
    print(f"  Tổng faces      : {total_faces}")
    print(f"  Avg / ảnh       : {np.mean(latencies):.2f} ms")
    print(f"  Min             : {np.min(latencies):.2f} ms")
    print(f"  Max             : {np.max(latencies):.2f} ms")
    print(f"  Std             : {np.std(latencies):.2f} ms")
    print(f"  Avg FPS         : {1000/np.mean(latencies):.1f}")

    if save:
        print(f"\n  Ảnh đã lưu vào : {save_dir}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Ultra-Light Face Detector")
    parser.add_argument("--folder", required=True,  help="Folder chứa ảnh test")
    parser.add_argument("--model",  default=os.path.join(os.path.dirname(__file__), "version-slim-320.onnx"))
    parser.add_argument("--conf",   type=float, default=0.6, help="Confidence threshold")
    parser.add_argument("--save",   action="store_true", help="Lưu ảnh kết quả (có bounding box) ra folder detection_output/")
    args = parser.parse_args()

    run_test(args.folder, args.model, args.conf, args.save)
