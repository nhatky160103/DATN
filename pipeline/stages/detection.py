from __future__ import annotations

import base64

import cv2
import numpy as np

from pipeline.schemas import FaceDetection
from pipeline.triton_client import TritonInferenceClient, TritonModelInput


def preprocess_ultralight(frame_bgr: np.ndarray, input_width: int = 320, input_height: int = 240) -> np.ndarray:
    image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height)).astype(np.float32)
    image = (image - np.array([127.0, 127.0, 127.0], dtype=np.float32)) / 128.0
    return np.transpose(image, (2, 0, 1))[None, ...].astype(np.float32, copy=False)


def expand_box(
    box: list[int],
    width: int,
    height: int,
    margin_x: float,
    margin_y: float | None = None,
) -> list[int]:
    if margin_y is None:
        margin_y = margin_x
    x1, y1, x2, y2 = box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)
    pad_x = box_width * margin_x
    pad_y = box_height * margin_y
    return [
        int(max(0, round(x1 - pad_x))),
        int(max(0, round(y1 - pad_y))),
        int(min(width - 1, round(x2 + pad_x))),
        int(min(height - 1, round(y2 + pad_y))),
    ]


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int64)

    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.argsort(scores)[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[i] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= threshold]

    return np.asarray(keep, dtype=np.int64)


class FaceDetectionStage:
    def __init__(
        self,
        triton: TritonInferenceClient,
        model_name: str = "ultralight",
        threshold: float = 0.7,
        iou_threshold: float = 0.4,
        input_width: int = 320,
        input_height: int = 240,
        crop_margin: float = 0.25,
        crop_margin_x: float | None = None,
        crop_margin_y: float | None = None,
    ):
        self.triton = triton
        self.model_name = model_name
        self.threshold = threshold
        self.iou_threshold = iou_threshold
        self.input_width = input_width
        self.input_height = input_height
        self.crop_margin_x = crop_margin if crop_margin_x is None else crop_margin_x
        self.crop_margin_y = crop_margin if crop_margin_y is None else crop_margin_y

    def _preprocess(self, frame_bgr: np.ndarray) -> np.ndarray:
        return preprocess_ultralight(frame_bgr, self.input_width, self.input_height)

    def predict(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        height, width = frame_bgr.shape[:2]
        scores, boxes = self.triton.infer(
            self.model_name,
            [TritonModelInput("input", self._preprocess(frame_bgr))],
            ["scores", "boxes"],
        )

        face_scores = scores[0, :, 1].astype(np.float32)
        normalized_boxes = boxes[0].astype(np.float32)
        mask = face_scores >= self.threshold
        if not np.any(mask):
            return []

        face_scores = face_scores[mask]
        normalized_boxes = normalized_boxes[mask]
        pixel_boxes = np.stack(
            [
                normalized_boxes[:, 0] * width,
                normalized_boxes[:, 1] * height,
                normalized_boxes[:, 2] * width,
                normalized_boxes[:, 3] * height,
            ],
            axis=1,
        )
        pixel_boxes[:, [0, 2]] = np.clip(pixel_boxes[:, [0, 2]], 0, width - 1)
        pixel_boxes[:, [1, 3]] = np.clip(pixel_boxes[:, [1, 3]], 0, height - 1)

        keep = _nms(pixel_boxes, face_scores, self.iou_threshold)
        detections: list[FaceDetection] = []
        for box, score in zip(pixel_boxes[keep], face_scores[keep]):
            x1, y1, x2, y2 = [int(round(value)) for value in box]
            if x2 <= x1 or y2 <= y1:
                continue
            face_crop = frame_bgr[y1:y2, x1:x2]
            ok, encoded_face = cv2.imencode(".jpg", face_crop)
            if not ok:
                continue
            crop_x1, crop_y1, crop_x2, crop_y2 = expand_box(
                [x1, y1, x2, y2],
                width,
                height,
                self.crop_margin_x,
                self.crop_margin_y,
            )
            quality_crop = frame_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
            ok, encoded_quality = cv2.imencode(".jpg", quality_crop)
            if not ok:
                continue
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    score=float(score),
                    crop_jpeg_b64=base64.b64encode(encoded_face.tobytes()).decode("ascii"),
                    crop_bbox=[crop_x1, crop_y1, crop_x2, crop_y2],
                    quality_crop_jpeg_b64=base64.b64encode(encoded_quality.tobytes()).decode("ascii"),
                    quality_bbox=[crop_x1, crop_y1, crop_x2, crop_y2],
                )
            )
        return detections

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        return self.predict(frame_bgr)
