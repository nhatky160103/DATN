from __future__ import annotations

import base64
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FrameMessage:
    camera_id: str
    frame_id: str
    timestamp: float
    image_jpeg_b64: str

    @classmethod
    def from_jpeg(cls, camera_id: str, frame_id: str, image_jpeg: bytes) -> "FrameMessage":
        return cls(
            camera_id=camera_id,
            frame_id=frame_id,
            timestamp=time.time(),
            image_jpeg_b64=base64.b64encode(image_jpeg).decode("ascii"),
        )

    def image_bytes(self) -> bytes:
        return base64.b64decode(self.image_jpeg_b64)


@dataclass
class FaceDetection:
    bbox: list[int]
    score: float
    crop_jpeg_b64: str

    def crop_bytes(self) -> bytes:
        return base64.b64decode(self.crop_jpeg_b64)


@dataclass
class DetectionsMessage:
    camera_id: str
    frame_id: str
    timestamp: float
    image_jpeg_b64: str
    detections: list[dict[str, Any]]

    def image_bytes(self) -> bytes:
        return base64.b64decode(self.image_jpeg_b64)

    def face_detections(self) -> list[FaceDetection]:
        return [FaceDetection(**item) for item in self.detections]


@dataclass
class TrackedFace:
    track_id: int
    bbox: list[int]
    score: float
    crop_jpeg_b64: str


@dataclass
class RecognitionResult:
    bucket_name: str
    employee_id: str
    track_id: int | None
    score: float | None
    status: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


def to_json(data: Any) -> str:
    if hasattr(data, "__dataclass_fields__"):
        data = asdict(data)
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def from_json(payload: bytes | str, cls: type[Any]) -> Any:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    return cls(**json.loads(payload))
