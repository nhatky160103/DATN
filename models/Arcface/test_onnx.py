from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from models.Arcface.backbones import get_model


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def preprocess_arcface(image_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    image = np.transpose(image, (2, 0, 1))
    return image[None, ...].astype(np.float32, copy=False)


def collect_images(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_file():
        return [input_path]
    return sorted(
        file
        for file in input_path.rglob("*")
        if file.is_file() and file.suffix.lower() in SUPPORTED_EXTS
    )


def load_torch_model(weights: str | Path, network: str) -> torch.nn.Module:
    model = get_model(network, dropout=0.0, fp16=False, num_features=512)
    state = torch.load(str(weights), map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def load_onnx_session(model_path: str | Path, providers: list[str]):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is required. Install it with: pip install onnxruntime") from exc

    return ort.InferenceSession(str(model_path), providers=providers)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float32)
    b = b.reshape(-1).astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def run_one(
    image_path: Path,
    torch_model: torch.nn.Module,
    onnx_session: Any,
    input_name: str,
) -> dict[str, Any]:
    image = read_image(image_path)
    input_array = preprocess_arcface(image)

    with torch.no_grad():
        torch_input = torch.from_numpy(input_array)
        started = time.perf_counter()
        torch_output = torch_model(torch_input).detach().cpu().numpy()
        torch_ms = (time.perf_counter() - started) * 1000.0

    started = time.perf_counter()
    onnx_output = onnx_session.run(None, {input_name: input_array})[0]
    onnx_ms = (time.perf_counter() - started) * 1000.0

    diff = np.abs(torch_output - onnx_output)
    return {
        "image": str(image_path),
        "torch_ms": round(torch_ms, 3),
        "onnx_ms": round(onnx_ms, 3),
        "max_abs_diff": round(float(np.max(diff)), 8),
        "mean_abs_diff": round(float(np.mean(diff)), 8),
        "cosine_similarity": round(cosine_similarity(torch_output, onnx_output), 8),
        "embedding_dim": int(onnx_output.reshape(-1).shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ArcFace ONNX inference and compare with PyTorch.")
    parser.add_argument("--image", help="Image file or folder of aligned face images.")
    parser.add_argument("--folder", help="Folder of aligned face images.")
    parser.add_argument("--weights", default="models/Arcface/weights/backbone.pth")
    parser.add_argument("--onnx", default="models/Arcface/weights/model_r18.onnx")
    parser.add_argument("--network", default="r18")
    parser.add_argument("--provider", action="append", default=None)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    args = parser.parse_args()

    input_path = args.folder or args.image
    if not input_path:
        raise SystemExit("Use --image /path/to/image or --folder /path/to/folder")

    images = collect_images(input_path)
    if not images:
        raise SystemExit(f"No supported images found: {input_path}")

    torch_model = load_torch_model(args.weights, args.network)
    providers = args.provider or ["CPUExecutionProvider"]
    onnx_session = load_onnx_session(args.onnx, providers)
    input_name = onnx_session.get_inputs()[0].name

    results = [run_one(path, torch_model, onnx_session, input_name) for path in images]
    for result in results:
        result["ok"] = result["max_abs_diff"] <= args.tolerance
        print(json.dumps(result, ensure_ascii=False))

    max_diff = max(result["max_abs_diff"] for result in results)
    mean_torch_ms = float(np.mean([result["torch_ms"] for result in results]))
    mean_onnx_ms = float(np.mean([result["onnx_ms"] for result in results]))
    summary = {
        "images": len(results),
        "ok": max_diff <= args.tolerance,
        "max_abs_diff": max_diff,
        "avg_torch_ms": round(mean_torch_ms, 3),
        "avg_onnx_ms": round(mean_onnx_ms, 3),
        "tolerance": args.tolerance,
    }
    print(json.dumps(summary, ensure_ascii=False))

    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
