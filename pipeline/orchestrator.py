from __future__ import annotations

import base64
import os
import time
from collections import Counter, defaultdict, deque
from typing import Any

import cv2
import numpy as np

from .config import PipelineConfig
from .qdrant_identity_store import build_qdrant_identity_search
from .schemas import FaceDetection, RecognitionResult, TrackedFace
from .stages.detection import FaceDetectionStage
from .stages.embedding import FaceEmbeddingStage
from .stages.liveness import FaceLivenessStage
from .stages.quality import FaceQualityStage
from .stages.bytetrack_adapter import FaceTrackingStage
from .triton_client import TritonInferenceClient


class RecognitionOrchestrator:
    """Coordinates preprocessing, Triton model calls, tracking, vector search, and decision logic."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.triton = TritonInferenceClient(cfg.triton.url, cfg.triton.timeout_sec, cfg.triton.enabled)
        self.detector = FaceDetectionStage(
            self.triton,
            model_name=cfg.triton.detector_model,
            threshold=cfg.bbox_threshold,
            iou_threshold=cfg.detection.iou_threshold,
            input_width=cfg.detection.input_width,
            input_height=cfg.detection.input_height,
            crop_margin=cfg.detection.crop_margin,
            crop_margin_x=cfg.detection.crop_margin_x,
            crop_margin_y=cfg.detection.crop_margin_y,
        )
        self.trackers: dict[str, FaceTrackingStage] = defaultdict(self._new_tracker)
        self.quality = FaceQualityStage(cfg.qscore_threshold, self.triton, cfg.triton.quality_model)
        self.liveness = FaceLivenessStage(
            cfg.anti_spoof_enabled,
            cfg.anti_spoof_threshold,
            self.triton,
            cfg.triton.fasnet_v1_model,
            cfg.triton.fasnet_v2_model,
        )
        self.embedding = FaceEmbeddingStage(self.triton, cfg.triton.arcface_model)
        self.identity_index = self._load_identity_index()
        self.track_buffers: dict[str, dict[int, deque[dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=cfg.max_track_buffer))
        )

    def _new_tracker(self) -> FaceTrackingStage:
        return FaceTrackingStage(
            track_thresh=self.cfg.tracking.track_thresh,
            match_thresh=self.cfg.tracking.match_thresh,
            track_buffer=self.cfg.tracking.track_buffer,
            frame_rate=self.cfg.tracking.frame_rate,
        )

    def _load_identity_index(self):
        return build_qdrant_identity_search(self.cfg)

    def recognize(
        self,
        frame_bgr: np.ndarray,
        camera_id: str = "api",
        frame_id: str | None = None,
        use_tracking: bool = True,
        use_voting: bool | None = None,
    ) -> dict[str, Any]:
        frame_id = frame_id or str(time.time_ns())
        use_voting = self.cfg.use_voting if use_voting is None else use_voting
        started = time.perf_counter()
        detections = self.detector.predict(frame_bgr)
        faces = self._track_faces(camera_id, detections) if use_tracking else self._untracked_faces(detections)

        results = []
        for tracked in faces:
            result = self._recognize_face(camera_id, frame_id, frame_bgr, tracked, use_voting)
            if result is not None:
                results.append(result)

        return {
            "status": "ok",
            "bucket_name": self.cfg.bucket_name,
            "camera_id": camera_id,
            "frame_id": frame_id,
            "detections": len(detections),
            "faces": results,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        }

    def recognition_results(self, response: dict[str, Any]) -> list[RecognitionResult]:
        output = []
        for face in response.get("faces", []):
            output.append(
                RecognitionResult(
                    bucket_name=response["bucket_name"],
                    employee_id=face.get("employee_id") or "UNKNOWN",
                    track_id=face.get("track_id"),
                    score=face.get("identity_score"),
                    status=face["status"],
                    metadata={
                        "camera_id": response["camera_id"],
                        "frame_id": response["frame_id"],
                        "bbox": face.get("bbox"),
                        "det_score": face.get("det_score"),
                        "quality_score": face.get("quality_score"),
                        "liveness_score": face.get("liveness_score"),
                        "vote_employee_id": face.get("vote_employee_id"),
                        "vote_ratio": face.get("vote_ratio"),
                        "unknown_ratio": face.get("unknown_ratio"),
                        "valid_frames": face.get("valid_frames"),
                        "required_images": face.get("required_images"),
                    },
                )
            )
        return output

    def _track_faces(self, camera_id: str, detections: list[FaceDetection]) -> list[TrackedFace]:
        return self.trackers[camera_id].predict(detections)

    def _untracked_faces(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        return [
            TrackedFace(
                track_id=index + 1,
                bbox=detection.bbox,
                score=detection.score,
                crop_jpeg_b64=detection.crop_jpeg_b64,
            )
            for index, detection in enumerate(detections)
        ]

    def _recognize_face(
        self,
        camera_id: str,
        frame_id: str,
        frame_bgr: np.ndarray,
        tracked: TrackedFace,
        use_voting: bool,
    ) -> dict[str, Any] | None:
        x1, y1, x2, y2 = tracked.bbox
        frame_area = frame_bgr.shape[0] * frame_bgr.shape[1]
        face_area = max(0, x2 - x1) * max(0, y2 - y1)
        base = {
            "track_id": tracked.track_id,
            "bbox": tracked.bbox,
            "det_score": round(float(tracked.score), 6),
            "frame_id": frame_id,
        }

        if face_area < self.cfg.min_face_area * frame_area:
            return {**base, "status": "face_too_small"}

        face = cv2.imdecode(np.frombuffer(base64.b64decode(tracked.crop_jpeg_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
        if face is None:
            return {**base, "status": "invalid_crop"}

        quality_ok, quality_score = self.quality.accept(face)
        base["quality_score"] = round(float(quality_score), 6)
        if os.getenv("SAVE_QUALITY_DEBUG"):
            os.makedirs("debug_quality", exist_ok=True)
            safe_frame_id = frame_id.replace("/", "_")
            cv2.imwrite(
                f"debug_quality/{camera_id}_{safe_frame_id}_track-{tracked.track_id}_q-{quality_score:.6f}.jpg",
                face,
            )
        if not quality_ok:
            return {**base, "status": "quality_rejected"}

        liveness_ok, liveness_score = self.liveness.predict(frame_bgr, tracked.bbox)
        base["liveness_score"] = round(float(liveness_score), 6)
        if not liveness_ok:
            return {**base, "status": "spoof_rejected", "employee_id": "UNKNOWN", "identity_score": None}

        embedding = self.embedding.predict(face)
        match = self.identity_index.search(embedding)
        employee_id = match.employee_id if match and match.accepted else "UNKNOWN"
        identity_score = float(match.score) if match else None
        base["identity_score"] = None if identity_score is None else round(identity_score, 6)

        if use_voting:
            return self._aggregate_track(
                camera_id,
                tracked.track_id,
                base,
                employee_id,
                identity_score,
                embedding,
                quality_score,
                liveness_score,
            )

        return {
            **base,
            "employee_id": employee_id,
            "status": "recognized" if employee_id != "UNKNOWN" else "unknown",
        }

    def _aggregate_track(
        self,
        camera_id: str,
        track_id: int,
        base: dict[str, Any],
        employee_id: str,
        identity_score: float | None,
        embedding: np.ndarray,
        quality_score: float,
        liveness_score: float,
    ) -> dict[str, Any]:
        buffer = self.track_buffers[camera_id][track_id]
        buffer.append(
            {
                "employee_id": employee_id,
                "score": identity_score,
                "det_score": float(base["det_score"]),
                "quality_score": float(quality_score),
                "liveness_score": float(liveness_score),
            }
        )

        valid_frames = len(buffer)
        if valid_frames < self.cfg.required_images:
            return {
                **base,
                "employee_id": employee_id,
                "status": "pending",
                "valid_frames": valid_frames,
                "required_images": self.cfg.required_images,
            }

        predictions = [item["employee_id"] for item in buffer]
        known_predictions = [value for value in predictions if value != "Theo"]
        unknown_ratio = predictions.count("UNKNOWN") / valid_frames
        if known_predictions:
            vote_employee_id, vote_count = Counter(known_predictions).most_common(1)[0]
            vote_ratio = vote_count / valid_frames
        else:
            vote_employee_id = "UNKNOWN"
            vote_ratio = 0.0

        vote_scores = [
            float(item["score"])
            for item in buffer
            if item["employee_id"] == vote_employee_id and item["score"] is not None
        ]
        vote_score = float(np.mean(vote_scores)) if vote_scores else None

        result_base = {
            **base,
            "employee_id": vote_employee_id,
            "identity_score": None if vote_score is None else round(vote_score, 6),
            "vote_employee_id": vote_employee_id,
            "vote_ratio": round(float(vote_ratio), 6),
            "unknown_ratio": round(float(unknown_ratio), 6),
            "valid_frames": valid_frames,
            "required_images": self.cfg.required_images,
            "max_track_buffer": self.cfg.max_track_buffer,
            "mean_quality_score": round(float(np.mean([item["quality_score"] for item in buffer])), 6),
            "mean_liveness_score": round(float(np.mean([item["liveness_score"] for item in buffer])), 6),
        }

        if not known_predictions and unknown_ratio >= self.cfg.validation_threshold:
            buffer.clear()
            return {**result_base, "employee_id": "UNKNOWN", "status": "unknown"}

        if vote_employee_id != "UNKNOWN" and vote_ratio >= self.cfg.validation_threshold:
            buffer.clear()
            return {**result_base, "status": "recognized"}

        if valid_frames >= self.cfg.max_track_buffer:
            buffer.clear()
            return {**result_base, "employee_id": "UNKNOWN", "status": "unknown"}

        return {
            **result_base,
            "employee_id": vote_employee_id,
            "status": "pending",
        }

    def _aggregate_embeddings(self, buffer: deque[dict[str, Any]]) -> np.ndarray:
        embeddings = np.stack(
            [self._normalize_embedding(item["embedding"]) for item in buffer],
            axis=0,
        ).astype(np.float32)
        weights = np.asarray(
            [
                max(1e-6, item["det_score"] * item["quality_score"] * item["liveness_score"])
                for item in buffer
            ],
            dtype=np.float32,
        )
        mean_embedding = np.average(embeddings, axis=0, weights=weights).astype(np.float32)
        return self._normalize_embedding(mean_embedding)

    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return (vector / norm).astype(np.float32)
