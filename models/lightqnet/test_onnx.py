"""
Test LightQNet ONNX model on a folder of images.
Input images must be cropped and aligned face images (96x96).

Usage:
    python -m models.lightqnet.test_onnx --folder /path/to/faces
    python -m models.lightqnet.test_onnx --folder /path/to/faces --model lightqnet-dm100.onnx
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np

if __package__:
    pass
else:
    sys.path.insert(0, os.path.dirname(__file__))

CURRENT_DIR    = os.path.dirname(os.path.abspath(__file__))
IMG_SIZE       = 96
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path: str) -> np.ndarray | None:
    """Read image like the original LightQNet pipeline."""
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def preprocess(img_bgr: np.ndarray, grayscale: bool) -> np.ndarray:
    """Match original preprocessing: resize -> optional grayscale -> normalize -> NHWC."""
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[:, :, np.newaxis]
    img = (img.astype(np.float32) - 128.0) / 128.0
    return img[np.newaxis]


def collect_images(folder: str):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])


def run_test(folder: str, onnx_path: str):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[ERROR] onnxruntime not installed. Run: pip install onnxruntime")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading: {onnx_path}")
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inp = session.get_inputs()[0]
    print(f"  Input  : {inp.name}  shape={inp.shape}")
    print(f"  Output : {session.get_outputs()[0].name}\n")

    # Determine grayscale or RGB from model input shape
    grayscale = (inp.shape[-1] == 1)

    # ── Collect images ────────────────────────────────────────────────────────
    images = collect_images(folder)
    if not images:
        print(f"No images found in: {folder}")
        sys.exit(1)
    print(f"Found {len(images)} images\n")

    # ── Table header ──────────────────────────────────────────────────────────
    col_w = [5, 40, 12, 10]
    print(
        f"{'#':<{col_w[0]}}"
        f"{'Filename':<{col_w[1]}}"
        f"{'Time (ms)':>{col_w[2]}}"
        f"{'Score':>{col_w[3]}}"
    )
    print("-" * sum(col_w))

    # ── Inference ─────────────────────────────────────────────────────────────
    latencies = []
    scores    = []

    for idx, img_path in enumerate(images, 1):
        frame = read_image(img_path)
        if frame is None:
            print(f"[!] Cannot read: {img_path}")
            continue

        inp_tensor = preprocess(frame, grayscale)

        t0 = time.perf_counter()
        result = session.run(None, {inp.name: inp_tensor})[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        score = float(result[0][0])
        latencies.append(elapsed_ms)
        scores.append(score)

        fname = os.path.basename(img_path)
        if len(fname) > col_w[1] - 2:
            fname = "…" + fname[-(col_w[1] - 3):]

        print(
            f"{idx:<{col_w[0]}}"
            f"{fname:<{col_w[1]}}"
            f"{elapsed_ms:>{col_w[2]}.2f}"
            f"{score:>{col_w[3]}.4f}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    if not latencies:
        return

    print("-" * sum(col_w))
    print(f"\nSummary ({len(latencies)} images):")
    print(f"  Avg time   : {np.mean(latencies):.2f} ms")
    print(f"  Min / Max  : {np.min(latencies):.2f} / {np.max(latencies):.2f} ms")
    print(f"  Avg FPS    : {1000/np.mean(latencies):.1f}")
    print(f"  Score avg  : {np.mean(scores):.4f}")
    print(f"  Score min  : {np.min(scores):.4f}  (lowest quality)")
    print(f"  Score max  : {np.max(scores):.4f}  (highest quality)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test LightQNet ONNX")
    parser.add_argument("--folder", required=True, help="Folder containing cropped face images")
    parser.add_argument("--model",  default="lightqnet-dm050.onnx", help=".onnx filename")
    args = parser.parse_args()

    onnx_path = os.path.join(CURRENT_DIR, "weights",  args.model)
    if not os.path.exists(onnx_path):
        print(f"[ERROR] File not found: {onnx_path}")
        print("  → Run convert first: python -m models.lightqnet.convert_to_onnx")
        sys.exit(1)

    run_test(args.folder, onnx_path)
