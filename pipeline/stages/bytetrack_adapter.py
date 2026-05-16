from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from models.ByteTrack.yolox.tracker.byte_tracker import BYTETracker
from pipeline.schemas import FaceDetection, TrackedFace


def preprocess_bytetrack(detections: list[FaceDetection]) -> np.ndarray:
    if not detections:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray(
        [detection.bbox + [float(detection.score)] for detection in detections],
        dtype=np.float32,
    )


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FaceTrackingStage:
    """Adapter from pipeline FaceDetection objects to official ByteTrack."""

    def __init__(
        self,
        track_thresh: float = 0.6,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        frame_rate: int = 30,
        mot20: bool = False,
    ) -> None:
        args = SimpleNamespace(
            track_thresh=track_thresh,
            match_thresh=match_thresh,
            track_buffer=track_buffer,
            mot20=mot20,
        )
        self.tracker = BYTETracker(args, frame_rate=frame_rate)

    def predict(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        det_array = preprocess_bytetrack(detections)
        # img_info and img_size are equal so official ByteTrack keeps bbox scale unchanged.
        tracks = self.tracker.update(det_array, img_info=(1, 1), img_size=(1, 1))
        det_boxes = [np.asarray(det.bbox, dtype=np.float32) for det in detections]
        output: list[TrackedFace] = []

        for track in tracks:
            bbox = np.asarray(track.tlbr, dtype=np.float32)
            crop_jpeg_b64 = ""
            if det_boxes:
                best_index = int(np.argmax([_iou(bbox, det_box) for det_box in det_boxes]))
                crop_jpeg_b64 = detections[best_index].crop_jpeg_b64
            output.append(
                TrackedFace(
                    track_id=int(track.track_id),
                    bbox=[int(round(value)) for value in bbox.tolist()],
                    score=float(track.score),
                    crop_jpeg_b64=crop_jpeg_b64,
                )
            )

        return output

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        return self.predict(detections)
