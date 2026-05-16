"""
Benchmark Anti-Spoof: ONNX vs PyTorch

Usage:
    python -m models.Anti_spoof.test_antispoof --folder /path/to/images
    python -m models.Anti_spoof.test_antispoof --folder /path/to/images --debug
"""

import os
import sys
import argparse
import time

import cv2
import numpy as np
from requests import patch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(CURRENT_DIR, "weights")
ASSETS_DIR  = os.path.join(CURRENT_DIR, "assets")

ONNX_V2   = os.path.join(WEIGHTS_DIR, "2.7_80x80_MiniFASNetV2.onnx")
ONNX_V1SE = os.path.join(WEIGHTS_DIR, "4_0_0_80x80_MiniFASNetV1SE.onnx")

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def collect_images(folder: str):
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS
    ])


def xyxy_to_xywh(box):
    """Convert [x1, y1, x2, y2] → [x, y, w, h]"""
    x1, y1, x2, y2 = box
    return [int(x1), int(y1), int(x2 - x1), int(y2 - y1)]


# ── PyTorch backend — reuse FasNet.py directly ───────────────────────────────

class AntispoofPyTorch:
    def __init__(self):
        from models.Fasnet.FasNet import Fasnet
        self.model = Fasnet()

    def predict(self, img: np.ndarray, bbox):
        is_real, score = self.model.analyze(img, bbox)
        return is_real, score


# ── ONNX backend ──────────────────────────────────────────────────────────────

class AntispoofONNX:
    def __init__(self):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        providers = ["CPUExecutionProvider"]

        self.sess_v2   = ort.InferenceSession(ONNX_V2,   sess_options=opts, providers=providers)
        self.sess_v1se = ort.InferenceSession(ONNX_V1SE, sess_options=opts, providers=providers)
        self.inp_v2    = self.sess_v2.get_inputs()[0].name
        self.inp_v1se  = self.sess_v1se.get_inputs()[0].name

    def predict(self, img: np.ndarray, bbox, debug=False):
        x, y, w, h = bbox

        patch_v2   = self.crop(img, (x, y, w, h), 2.7, 80, 80)
        patch_v1se = self.crop(img, (x, y, w, h), 4.0, 80, 80)

        def to_numpy(patch):
            return patch.astype(np.float32).transpose(2, 0, 1)[None, ...]

        inp_v2   = to_numpy(patch_v2)
        inp_v1se = to_numpy(patch_v1se)

        logits_v2   = self.sess_v2.run(None,   {self.inp_v2:   inp_v2})[0]
        logits_v1se = self.sess_v1se.run(None, {self.inp_v1se: inp_v1se})[0]

        prob_v2   = softmax(logits_v2)
        prob_v1se = softmax(logits_v1se)

        if debug:
            print(f"    [ONNX V2  ] logits={logits_v2[0]}  probs={prob_v2[0]}")
            print(f"    [ONNX V1SE] logits={logits_v1se[0]}  probs={prob_v1se[0]}")

        prediction = prob_v2 + prob_v1se
        label      = int(np.argmax(prediction))
        is_real    = label == 1
        score      = float(prediction[0, label] / 2)
        return is_real, score


# ── Test runner ───────────────────────────────────────────────────────────────

def run_test(folder: str, n_repeat: int, debug: bool):
    from models.UltraLight.pipeline import UltraLightDetector

    images = collect_images(folder)
    if not images:
        print(f"No images found in: {folder}")
        sys.exit(1)

    print("Loading models...")
    detector    = UltraLightDetector()
    onnx_model  = AntispoofONNX()
    torch_model = AntispoofPyTorch()
    print("✓ All models loaded.\n")
    print(f"Found {len(images)} image(s) in '{folder}'\n")

    col = [5, 28, 6, 14, 8, 14, 8, 10, 10]
    header = (
        f"{'#':<{col[0]}}"
        f"{'Filename':<{col[1]}}"
        f"{'Faces':>{col[2]}}"
        f"{'ONNX (ms)':>{col[3]}}"
        f"{'Real?':>{col[4]}}"
        f"{'Torch (ms)':>{col[5]}}"
        f"{'Real?':>{col[6]}}"
        f"{'Score diff':>{col[7]}}"
        f"{'Match':>{col[8]}}"
    )
    sep = "-" * sum(col)
    print(header)
    print(sep)

    onnx_times, torch_times = [], []
    mismatches = 0
    no_face    = 0

    for idx, img_path in enumerate(images, 1):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[!] Cannot read: {img_path}")
            continue

        # ── Detect faces ──────────────────────────────────────────────────────
        boxes, det_scores = detector.detect(img)

        fname = os.path.basename(img_path)
        if len(fname) > col[1] - 2:
            fname = "…" + fname[-(col[1]-3):]

        if len(boxes) == 0:
            no_face += 1
            print(f"{idx:<{col[0]}}{fname:<{col[1]}}{'0':>{col[2]}}  [no face detected]")
            continue

        # Dùng face có confidence cao nhất
        best_idx = int(np.argmax(det_scores))
        bbox = xyxy_to_xywh(boxes[best_idx])   # [x, y, w, h]

        if debug:
            print(f"\n  [{idx}] {os.path.basename(img_path)}  bbox={bbox}")

        # ── ONNX ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        for i in range(n_repeat):
            onnx_real, onnx_score = onnx_model.predict(img, bbox, debug=debug and i == 0)
        onnx_ms = (time.perf_counter() - t0) * 1000 / n_repeat

        # ── PyTorch ──────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        for i in range(n_repeat):
            torch_real, torch_score = torch_model.predict(img, bbox)
        torch_ms = (time.perf_counter() - t0) * 1000 / n_repeat

        onnx_times.append(onnx_ms)
        torch_times.append(torch_ms)

        score_diff = abs(onnx_score - torch_score)
        match      = "✅" if onnx_real == torch_real else "❌"
        if onnx_real != torch_real:
            mismatches += 1

        print(
            f"{idx:<{col[0]}}"
            f"{fname:<{col[1]}}"
            f"{len(boxes):>{col[2]}}"
            f"{onnx_ms:>{col[3]}.2f}"
            f"{'Real' if onnx_real else 'Fake':>{col[4]}}"
            f"{torch_ms:>{col[5]}.2f}"
            f"{'Real' if torch_real else 'Fake':>{col[6]}}"
            f"{score_diff:>{col[7]}.4f}"
            f"{match:>{col[8]}}"
        )

    n = len(onnx_times)
    if not n:
        return

    print(sep)
    print(f"\nSummary")
    print(f"  Images tested    : {n}  (skipped {no_face} with no face detected)")
    print(f"  Prediction match : {n - mismatches}/{n}  {'✅ all match' if mismatches == 0 else f'❌ {mismatches} mismatch(es)'}")
    print()
    print(f"  {'':30s} {'ONNX':>12}   {'PyTorch':>12}")
    print(f"  {'Avg latency (ms)':<30s} {np.mean(onnx_times):>12.2f}   {np.mean(torch_times):>12.2f}")
    print(f"  {'Min latency (ms)':<30s} {np.min(onnx_times):>12.2f}   {np.min(torch_times):>12.2f}")
    print(f"  {'Max latency (ms)':<30s} {np.max(onnx_times):>12.2f}   {np.max(torch_times):>12.2f}")
    print(f"  {'Avg FPS':<30s} {1000/np.mean(onnx_times):>12.1f}   {1000/np.mean(torch_times):>12.1f}")
    speedup = np.mean(torch_times) / np.mean(onnx_times)
    tag = f"ONNX is {speedup:.2f}x faster" if speedup >= 1 else f"PyTorch is {1/speedup:.2f}x faster"
    print(f"\n  {tag}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", default=ASSETS_DIR)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--debug",  action="store_true")
    args = parser.parse_args()

    run_test(args.folder, args.repeat, args.debug)
