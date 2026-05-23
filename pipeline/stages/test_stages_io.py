from __future__ import annotations

import argparse
import base64
import os
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.schemas import FaceDetection, TrackedFace
from pipeline.stages.bytetrack_adapter import FaceTrackingStage, preprocess_bytetrack
from pipeline.stages.detection import FaceDetectionStage, preprocess_ultralight
from pipeline.stages.embedding import FaceEmbeddingStage, preprocess_arcface
from pipeline.stages.liveness import FaceLivenessStage, preprocess_fasnet
from pipeline.stages.quality import FaceQualityStage, preprocess_lightqnet
from pipeline.stages.vector_search import IdentitySearchStage, SearchMatch
from pipeline.triton_client import TritonInferenceClient


def _read_image(path: str) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def _assert_shape(name: str, actual: tuple[int, ...], expected: tuple[int, ...]) -> None:
    assert actual == expected, f"{name}: expected shape {expected}, got {actual}"


def _face_from_detection(detections: list[FaceDetection]) -> tuple[np.ndarray, list[int]]:
    if not detections:
        raise ValueError("No face detected; use an image with a detectable face.")
    detection = detections[0]
    face = cv2.imdecode(np.frombuffer(base64.b64decode(detection.crop_jpeg_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
    if face is None:
        raise ValueError("Cannot decode detection crop_jpeg_b64")
    return face, detection.bbox


def test_detection(triton: TritonInferenceClient, frame: np.ndarray) -> list[FaceDetection]:
    image = preprocess_ultralight(frame, input_width=640, input_height=480)
    _assert_shape("ultralight input", image.shape, (1, 3, 480, 640))
    assert image.dtype == np.float32

    detections = FaceDetectionStage(triton, threshold=0.7, input_width=640, input_height=480).predict(frame)
    assert isinstance(detections, list)
    for item in detections:
        assert isinstance(item, FaceDetection)
        assert len(item.bbox) == 4
        assert item.score >= 0.7
        assert item.crop_bytes()
    print(f"OK detection: {len(detections)} face(s)")
    return detections


def test_quality(triton: TritonInferenceClient, face: np.ndarray) -> None:
    image = preprocess_lightqnet(face)
    _assert_shape("lightqnet input", image.shape, (1, 96, 96, 3))
    assert image.dtype == np.float32

    accepted, score = FaceQualityStage(0.4, triton).accept(face)
    assert isinstance(accepted, bool)
    assert isinstance(score, float)
    print(f"OK quality: accepted={accepted} score={score:.6f}")


def test_embedding(triton: TritonInferenceClient, face: np.ndarray) -> np.ndarray:
    image = preprocess_arcface(face)
    _assert_shape("arcface input", image.shape, (1, 3, 112, 112))
    assert image.dtype == np.float32

    embedding = FaceEmbeddingStage(triton).predict(face)
    _assert_shape("arcface output", embedding.shape, (512,))
    assert embedding.dtype == np.float32
    print(f"OK embedding: shape={embedding.shape} norm={np.linalg.norm(embedding):.6f}")
    return embedding


def test_liveness(triton: TritonInferenceClient, frame: np.ndarray, bbox: list[int]) -> None:
    image = preprocess_fasnet(frame, bbox, scale=2.7)
    _assert_shape("fasnet input", image.shape, (1, 3, 80, 80))
    assert image.dtype == np.float32

    accepted, score = FaceLivenessStage(True, 0.9, triton).predict(frame, bbox)
    assert isinstance(accepted, bool)
    assert isinstance(score, float)
    print(f"OK liveness: accepted={accepted} score={score:.6f}")


def test_tracking(detections: list[FaceDetection]) -> None:
    det_array = preprocess_bytetrack(detections)
    assert det_array.shape == (len(detections), 5)
    assert det_array.dtype == np.float32

    tracks = FaceTrackingStage().predict(detections)
    assert isinstance(tracks, list)
    for item in tracks:
        assert isinstance(item, TrackedFace)
        assert len(item.bbox) == 4
    print(f"OK tracking: {len(tracks)} track(s)")


def test_vector_search(embedding: np.ndarray) -> None:
    embeddings = np.stack([embedding, np.roll(embedding, 1)], axis=0).astype(np.float32)
    stage = IdentitySearchStage(embeddings, ["emp-1", "emp-2"], match_threshold=0.2)
    match = stage.search(embedding)
    assert isinstance(match, SearchMatch)
    assert match.employee_id == "emp-1"
    assert match.accepted is True
    print(f"OK vector_search: employee_id={match.employee_id} score={match.score:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test pipeline stages against a real Triton server.")
    parser.add_argument("--url", default=os.getenv("TRITON_URL", "localhost:8000"))
    parser.add_argument("--image", required=True, help="Image used to smoke-test all stages.")
    args = parser.parse_args()

    triton = TritonInferenceClient(args.url, timeout_sec=10.0, enabled=True)
    try:
        ready = bool(triton.client.is_server_ready())
    except Exception as exc:
        raise SystemExit(f"Triton is not reachable at {args.url}: {exc}") from exc
    if not ready:
        raise SystemExit(f"Triton is not ready: {args.url}")
    print(f"OK triton: {args.url}")

    frame = _read_image(args.image)
    detections = test_detection(triton, frame)
    face, bbox = _face_from_detection(detections)
    test_quality(triton, face)
    embedding = test_embedding(triton, face)
    test_liveness(triton, frame, bbox)
    test_tracking(detections)
    test_vector_search(embedding)


if __name__ == "__main__":
    main()
