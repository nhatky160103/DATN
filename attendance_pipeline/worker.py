from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict, deque

import cv2
import numpy as np

from infer.get_embedding import EmbeddingManager

from .config import PipelineConfig, load_pipeline_config
from .redis_queue import RedisListQueue
from .response import ResponseWriter
from .schemas import DetectionsMessage, FaceDetection, FrameMessage, RecognitionResult, from_json
from .stages.detection import MTCNNFaceDetector
from .stages.embedding import FaceEmbeddingStage
from .stages.liveness import LivenessStage
from .stages.quality import FaceQualityStage
from .stages.tracking import ByteTrackFaceTracker
from .triton_client import TritonInferenceClient
from .vector_search import FaissIdentityIndex


class AttendancePipelineWorker:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.frame_queue = RedisListQueue(cfg.redis.url, cfg.redis.frame_queue, cfg.redis.max_queue_size)
        self.detection_queue = RedisListQueue(cfg.redis.url, cfg.redis.detection_queue, cfg.redis.max_queue_size)
        self.response_writer = ResponseWriter(cfg.redis.url, cfg.redis.result_queue, cfg.redis.max_queue_size)
        self.triton = TritonInferenceClient(cfg.triton.url, cfg.triton.timeout_sec, cfg.triton.enabled)
        self.detector = MTCNNFaceDetector(
            threshold=cfg.bbox_threshold,
            max_side=cfg.detection.mtcnn_max_side,
            min_face_size=cfg.detection.mtcnn_min_face_size,
            factor=cfg.detection.mtcnn_factor,
            keep_all=cfg.detection.mtcnn_keep_all,
            torch_num_threads=cfg.detection.torch_num_threads,
        )
        self.tracker = ByteTrackFaceTracker()
        self.quality = FaceQualityStage(cfg.qscore_threshold, self.triton, cfg.triton.quality_model)
        self.liveness = LivenessStage(
            cfg.anti_spoof_enabled,
            cfg.anti_spoof_threshold,
            self.triton,
            cfg.triton.fasnet_v1_model,
            cfg.triton.fasnet_v2_model,
        )
        self.embedding = FaceEmbeddingStage(self.triton, cfg.triton.arcface_model)
        self.identity_index = self._load_identity_index()
        self.track_predictions: dict[int, deque[str]] = defaultdict(lambda: deque(maxlen=cfg.required_images))
        self.track_scores: dict[int, deque[float]] = defaultdict(lambda: deque(maxlen=cfg.required_images))

    def _load_identity_index(self) -> FaissIdentityIndex:
        manager = EmbeddingManager(self.cfg.bucket_name)
        embeddings, employee_ids = manager.load()
        if embeddings is None or employee_ids is None:
            return FaissIdentityIndex.empty(
                self.cfg.distance_mode,
                self.cfg.cosine_threshold,
                self.cfg.l2_threshold,
            )
        return FaissIdentityIndex(
            embeddings,
            employee_ids,
            self.cfg.distance_mode,
            self.cfg.cosine_threshold,
            self.cfg.l2_threshold,
        )

    def run_forever(self) -> None:
        while True:
            payload = self._pop_payload()
            if payload is None:
                continue
            try:
                if self.cfg.detection.provider == "external_cpp":
                    self.process_detections(from_json(payload, DetectionsMessage))
                else:
                    self.process_frame(from_json(payload, FrameMessage))
            except Exception as exc:
                print(f"Pipeline worker failed to process frame: {exc}")

    def _pop_payload(self) -> str | None:
        if self.cfg.detection.provider == "external_cpp":
            return self.detection_queue.pop(timeout_sec=1)
        return self.frame_queue.pop(timeout_sec=1)

    def process_frame(self, message: FrameMessage) -> None:
        frame_bytes = np.frombuffer(message.image_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return

        detections = self.detector.detect(frame)
        self._process_detected_faces(message.frame_id, frame, detections)

    def process_detections(self, message: DetectionsMessage) -> None:
        frame_bytes = np.frombuffer(message.image_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
        if frame is None:
            return
        self._process_detected_faces(message.frame_id, frame, message.face_detections())

    def _process_detected_faces(self, frame_id: str, frame: np.ndarray, detections: list[FaceDetection]) -> None:
        tracked_faces = self.tracker.update(detections)
        frame_area = frame.shape[0] * frame.shape[1]

        for tracked in tracked_faces:
            x1, y1, x2, y2 = tracked.bbox
            if ((x2 - x1) * (y2 - y1)) < self.cfg.min_face_area * frame_area:
                continue

            face = cv2.imdecode(np.frombuffer(base64.b64decode(tracked.crop_jpeg_b64), dtype=np.uint8), cv2.IMREAD_COLOR)
            if face is None:
                continue

            quality_ok, quality_score = self.quality.accept(face)
            if not quality_ok:
                continue

            liveness_ok, liveness_score = self.liveness.accept(frame, tracked.bbox)
            if not liveness_ok:
                self.response_writer.write(
                    RecognitionResult(
                        bucket_name=self.cfg.bucket_name,
                        employee_id="UNKNOWN",
                        track_id=tracked.track_id,
                        score=liveness_score,
                        status="spoof_rejected",
                        metadata={"frame_id": frame_id, "quality_score": quality_score},
                    )
                )
                continue

            embedding = self.embedding.embed(face)
            match = self.identity_index.search(embedding)
            employee_id = match.employee_id if match and match.accepted else "UNKNOWN"
            score = match.score if match else None

            predictions = self.track_predictions[tracked.track_id]
            scores = self.track_scores[tracked.track_id]
            predictions.append(employee_id)
            if score is not None:
                scores.append(score)

            if len(predictions) < self.cfg.required_images:
                continue

            candidate, count = Counter(predictions).most_common(1)[0]
            if count >= self.cfg.required_images * self.cfg.validation_threshold:
                mean_score = float(np.mean(scores)) if scores else None
                self.response_writer.write(
                    RecognitionResult(
                        bucket_name=self.cfg.bucket_name,
                        employee_id=candidate,
                        track_id=tracked.track_id,
                        score=mean_score,
                        status="recognized" if candidate != "UNKNOWN" else "unknown",
                        metadata={
                            "frame_id": frame_id,
                            "quality_score": quality_score,
                            "liveness_score": liveness_score,
                        },
                    )
                )
                predictions.clear()
                scores.clear()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    AttendancePipelineWorker(load_pipeline_config(args.config)).run_forever()


if __name__ == "__main__":
    main()
