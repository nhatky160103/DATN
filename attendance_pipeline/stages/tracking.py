from __future__ import annotations

from dataclasses import dataclass

from attendance_pipeline.schemas import FaceDetection, TrackedFace


def _iou(a: list[int], b: list[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom else 0.0


@dataclass
class _TrackState:
    track_id: int
    bbox: list[int]
    missed: int = 0


class ByteTrackFaceTracker:
    """CPU fallback tracker with ByteTrack-compatible responsibility.

    Install a real ByteTrack implementation and replace this class if strict
    ByteTrack behavior is required. The pipeline contract stays the same:
    detections in, stable face track IDs out.
    """

    def __init__(self, iou_threshold: float = 0.3, max_missed: int = 10):
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[FaceDetection]) -> list[TrackedFace]:
        assigned_tracks: set[int] = set()
        results: list[TrackedFace] = []

        for detection in detections:
            best_id = None
            best_iou = 0.0
            for track_id, state in self._tracks.items():
                if track_id in assigned_tracks:
                    continue
                score = _iou(detection.bbox, state.bbox)
                if score > best_iou:
                    best_iou = score
                    best_id = track_id

            if best_id is None or best_iou < self.iou_threshold:
                best_id = self._next_id
                self._next_id += 1

            assigned_tracks.add(best_id)
            self._tracks[best_id] = _TrackState(best_id, detection.bbox, missed=0)
            results.append(
                TrackedFace(
                    track_id=best_id,
                    bbox=detection.bbox,
                    score=detection.score,
                    crop_jpeg_b64=detection.crop_jpeg_b64,
                )
            )

        for track_id in list(self._tracks):
            if track_id not in assigned_tracks:
                self._tracks[track_id].missed += 1
                if self._tracks[track_id].missed > self.max_missed:
                    del self._tracks[track_id]

        return results

