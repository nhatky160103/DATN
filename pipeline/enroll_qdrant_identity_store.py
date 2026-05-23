from __future__ import annotations

import argparse
import hashlib
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from .config import load_pipeline_config
from .qdrant_identity_store import QdrantIdentitySearchStage, QdrantIdentityStoreConfig
from .stages.detection import FaceDetectionStage
from .stages.embedding import FaceEmbeddingStage
from .stages.quality import FaceQualityStage
from .triton_client import TritonInferenceClient


SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _collect_images(employee_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(employee_dir.iterdir())
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS
    ]


def _point_id(bucket_name: str, employee_id: str, image_path: Path, image_bytes: bytes) -> str:
    digest = hashlib.sha256(image_bytes).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{bucket_name}:{employee_id}:{image_path.as_posix()}:{digest}"))


def _decode_crop(crop_bytes: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(crop_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)


def enroll_dataset(config_path: str, dataset_root: str, min_quality: float | None = None) -> None:
    cfg = load_pipeline_config(config_path)
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(root)

    triton = TritonInferenceClient(cfg.triton.url, cfg.triton.timeout_sec, cfg.triton.enabled)
    detector = FaceDetectionStage(
        triton,
        model_name=cfg.triton.detector_model,
        threshold=cfg.bbox_threshold,
        iou_threshold=cfg.detection.iou_threshold,
        input_width=cfg.detection.input_width,
        input_height=cfg.detection.input_height,
        crop_margin=cfg.detection.crop_margin,
        crop_margin_x=cfg.detection.crop_margin_x,
        crop_margin_y=cfg.detection.crop_margin_y,
    )
    quality = FaceQualityStage(min_quality if min_quality is not None else cfg.qscore_threshold, triton, cfg.triton.quality_model)
    embedder = FaceEmbeddingStage(triton, cfg.triton.arcface_model)
    qdrant = QdrantIdentitySearchStage(
        QdrantIdentityStoreConfig(
            url=cfg.qdrant.url,
            api_key=cfg.qdrant.api_key,
            collection=cfg.qdrant.collection,
            bucket_name=cfg.bucket_name,
            match_threshold=cfg.match_threshold,
        )
    )

    employee_dirs = [path for path in sorted(root.iterdir()) if path.is_dir()]
    if not employee_dirs:
        raise ValueError(f"No employee folders found in {root}")

    points = []
    skipped = 0
    for employee_dir in employee_dirs:
        employee_id = employee_dir.name
        enrolled = 0
        for image_path in _collect_images(employee_dir):
            raw = image_path.read_bytes()
            image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                print(f"Skipped unreadable image: {image_path}")
                skipped += 1
                continue

            detections = detector.predict(image)
            if not detections:
                print(f"Skipped no-face image: {image_path}")
                skipped += 1
                continue

            detection = max(detections, key=lambda item: item.score)
            face = _decode_crop(detection.crop_bytes())
            if face is None:
                print(f"Skipped invalid crop: {image_path}")
                skipped += 1
                continue

            accepted, quality_score = quality.accept(face)
            if not accepted:
                print(f"Skipped low-quality image: {image_path} quality={quality_score:.6f}")
                skipped += 1
                continue

            embedding = embedder.predict(face).astype(np.float32).reshape(-1)
            if embedding.shape != (512,):
                raise ValueError(f"Unexpected embedding shape for {image_path}: {embedding.shape}")

            point_id = _point_id(cfg.bucket_name, employee_id, image_path, raw)
            points.append(
                {
                    "id": point_id,
                    "vector": embedding.tolist(),
                    "payload": {
                        "bucket_name": cfg.bucket_name,
                        "employee_id": employee_id,
                        "image_path": image_path.as_posix(),
                        "det_score": float(detection.score),
                        "quality_score": float(quality_score),
                        "bbox": detection.bbox,
                        "model_name": cfg.triton.arcface_model,
                        "model_version": "arcface",
                        "active": True,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                    },
                }
            )
            enrolled += 1
        print(f"Prepared {employee_id}: {enrolled} embedding(s)")

    if points:
        from qdrant_client import models

        qdrant.client.upsert(
            collection_name=cfg.qdrant.collection,
            points=[
                models.PointStruct(
                    id=item["id"],
                    vector=item["vector"],
                    payload=item["payload"],
                )
                for item in points
            ],
            wait=True,
        )
    print(f"Upserted {len(points)} embedding(s) to Qdrant collection={cfg.qdrant.collection}; skipped={skipped}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Enroll employee face embeddings into Qdrant.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset-root", required=True, help="Root folder with subfolders named by employee_id")
    parser.add_argument("--min-quality", type=float, default=None)
    args = parser.parse_args()
    enroll_dataset(args.config, args.dataset_root, args.min_quality)


if __name__ == "__main__":
    main()
