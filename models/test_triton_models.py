from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _status(name: str, ok: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result = {"name": name, "ok": ok, "detail": detail}
    if extra:
        result.update(extra)
    print(json.dumps(result, ensure_ascii=False))
    return result


def _softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values, axis=1, keepdims=True)
    exp = np.exp(values)
    return exp / np.sum(exp, axis=1, keepdims=True)


def read_image(path: str | Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def preprocess_ultralight(image_bgr: np.ndarray) -> np.ndarray:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (320, 240)).astype(np.float32)
    image = (image - np.array([127.0, 127.0, 127.0], dtype=np.float32)) / 128.0
    return np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def preprocess_arcface(face_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(face_bgr, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    return np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def preprocess_lightqnet(face_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(face_bgr, (96, 96)).astype(np.float32)
    return ((image - 128.0) / 128.0)[None, ...].astype(np.float32, copy=False)


def preprocess_fasnet(face_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(face_bgr, (80, 80)).astype(np.float32)
    return np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


class TritonTester:
    def __init__(self, url: str, timeout_sec: float, protocol: str) -> None:
        if protocol == "grpc":
            import tritonclient.grpc as client_module

            self.client = client_module.InferenceServerClient(url=url)
        else:
            import tritonclient.http as client_module

            self.client = client_module.InferenceServerClient(url=url, connection_timeout=timeout_sec)

        self.client_module = client_module
        self.timeout_sec = timeout_sec
        self.protocol = protocol

    def infer(
        self,
        model_name: str,
        input_name: str,
        input_array: np.ndarray,
        output_names: list[str],
    ) -> tuple[list[np.ndarray], float]:
        infer_input = self.client_module.InferInput(input_name, input_array.shape, "FP32")
        infer_input.set_data_from_numpy(input_array.astype(np.float32, copy=False))
        requested_outputs = [self.client_module.InferRequestedOutput(name) for name in output_names]

        started = time.perf_counter()
        result = self.client.infer(
            model_name,
            [infer_input],
            outputs=requested_outputs,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return [result.as_numpy(name) for name in output_names], elapsed_ms


def crop_best_face(frame_bgr: np.ndarray, scores: np.ndarray, boxes: np.ndarray, threshold: float) -> tuple[np.ndarray, int]:
    height, width = frame_bgr.shape[:2]
    face_scores = scores[0, :, 1]
    best_idx = int(np.argmax(face_scores))
    best_score = float(face_scores[best_idx])
    if best_score < threshold:
        return frame_bgr, 0

    box = boxes[0, best_idx]
    x1 = int(np.clip(box[0] * width, 0, width - 1))
    y1 = int(np.clip(box[1] * height, 0, height - 1))
    x2 = int(np.clip(box[2] * width, 0, width - 1))
    y2 = int(np.clip(box[3] * height, 0, height - 1))
    if x2 <= x1 or y2 <= y1:
        return frame_bgr, 0
    return frame_bgr[y1:y2, x1:x2], 1


def test_health(tester: TritonTester) -> dict[str, Any]:
    ok = bool(tester.client.is_server_ready())
    return _status("triton_health", ok, "server ready" if ok else "server not ready")


def test_ultralight(tester: TritonTester, image: np.ndarray, conf: float) -> tuple[dict[str, Any], np.ndarray]:
    input_array = preprocess_ultralight(image)
    try:
        outputs, elapsed_ms = tester.infer("ultralight", "input", input_array, ["scores", "boxes"])
        scores, boxes = outputs
        detected = int(np.sum(scores[0, :, 1] > conf))
        face, best_count = crop_best_face(image, scores, boxes, conf)
        result = _status(
            "ultralight",
            scores.shape == (1, 4420, 2) and boxes.shape == (1, 4420, 4),
            "detector inference ok",
            {
                "input_shape": list(input_array.shape),
                "scores_shape": list(scores.shape),
                "boxes_shape": list(boxes.shape),
                "raw_detections_above_threshold": detected,
                "best_face_crop": bool(best_count),
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
        return result, face
    except Exception as exc:
        return _status("ultralight", False, str(exc)), image


def test_arcface(tester: TritonTester, face: np.ndarray) -> dict[str, Any]:
    input_array = preprocess_arcface(face)
    try:
        outputs, elapsed_ms = tester.infer("arcface", "input", input_array, ["embedding"])
        embedding = outputs[0]
        return _status(
            "arcface",
            embedding.shape == (1, 512),
            "embedding inference ok",
            {
                "input_shape": list(input_array.shape),
                "output_shape": list(embedding.shape),
                "embedding_norm": round(float(np.linalg.norm(embedding.reshape(-1))), 6),
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
    except Exception as exc:
        return _status("arcface", False, str(exc))


def test_lightqnet(tester: TritonTester, face: np.ndarray) -> dict[str, Any]:
    input_array = preprocess_lightqnet(face)
    try:
        outputs, elapsed_ms = tester.infer("lightqnet", "input:0", input_array, ["confidence_st:0"])
        score = outputs[0]
        return _status(
            "lightqnet",
            score.shape == (1, 1),
            "quality inference ok",
            {
                "input_shape": list(input_array.shape),
                "output_shape": list(score.shape),
                "score": round(float(score.reshape(-1)[0]), 6),
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
    except Exception as exc:
        return _status("lightqnet", False, str(exc))


def test_fasnet(tester: TritonTester, face: np.ndarray, model_name: str) -> dict[str, Any]:
    input_array = preprocess_fasnet(face)
    try:
        outputs, elapsed_ms = tester.infer(model_name, "input", input_array, ["logits"])
        logits = outputs[0]
        probs = _softmax(logits)
        label = int(np.argmax(probs, axis=1)[0])
        return _status(
            model_name,
            logits.shape == (1, 3),
            "anti-spoof inference ok",
            {
                "input_shape": list(input_array.shape),
                "output_shape": list(logits.shape),
                "predicted_label": label,
                "probabilities": probs.reshape(-1).round(6).tolist(),
                "elapsed_ms": round(elapsed_ms, 3),
            },
        )
    except Exception as exc:
        return _status(model_name, False, str(exc))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test all Triton models in this repository.")
    parser.add_argument("--url", default="localhost:8000", help="Triton HTTP endpoint.")
    parser.add_argument("--protocol", choices=["http", "grpc"], default="http")
    parser.add_argument("--image", default="models/UltraLight/acess/test.jpg", help="Image used for detector and crops.")
    parser.add_argument("--det-threshold", type=float, default=0.5)
    parser.add_argument("--timeout-sec", type=float, default=5.0)
    args = parser.parse_args()

    tester = TritonTester(args.url, args.timeout_sec, args.protocol)
    image = read_image(args.image)

    results: list[dict[str, Any]] = [test_health(tester)]
    ultralight_result, face = test_ultralight(tester, image, args.det_threshold)
    results.append(ultralight_result)
    results.append(test_arcface(tester, face))
    results.append(test_lightqnet(tester, face))
    results.append(test_fasnet(tester, face, "fasnet_v1se"))
    results.append(test_fasnet(tester, face, "fasnet_v2"))

    if any(not item["ok"] for item in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
