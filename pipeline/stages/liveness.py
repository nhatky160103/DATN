from __future__ import annotations

import cv2
import numpy as np

from pipeline.triton_client import TritonInferenceClient, TritonModelInput


def _get_new_box(src_w: int, src_h: int, bbox_xywh: tuple[int, int, int, int], scale: float) -> tuple[int, int, int, int]:
    x, y, box_w, box_h = bbox_xywh
    scale = min((src_h - 1) / box_h, min((src_w - 1) / box_w, scale))
    new_width = box_w * scale
    new_height = box_h * scale
    center_x, center_y = box_w / 2 + x, box_h / 2 + y
    left_top_x = center_x - new_width / 2
    left_top_y = center_y - new_height / 2
    right_bottom_x = center_x + new_width / 2
    right_bottom_y = center_y + new_height / 2
    if left_top_x < 0:
        right_bottom_x -= left_top_x
        left_top_x = 0
    if left_top_y < 0:
        right_bottom_y -= left_top_y
        left_top_y = 0
    if right_bottom_x > src_w - 1:
        left_top_x -= right_bottom_x - src_w + 1
        right_bottom_x = src_w - 1
    if right_bottom_y > src_h - 1:
        left_top_y -= right_bottom_y - src_h + 1
        right_bottom_y = src_h - 1
    return int(left_top_x), int(left_top_y), int(right_bottom_x), int(right_bottom_y)


def _crop_scaled(frame_bgr: np.ndarray, bbox_xywh: tuple[int, int, int, int], scale: float) -> np.ndarray:
    src_h, src_w, _ = np.shape(frame_bgr)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox_xywh, scale)
    face = frame_bgr[left_top_y : right_bottom_y + 1, left_top_x : right_bottom_x + 1]
    return cv2.resize(face, (80, 80))


def preprocess_fasnet(frame_bgr: np.ndarray, bbox_xyxy: list[int], scale: float) -> np.ndarray:
    src_h, src_w, _ = np.shape(frame_bgr)
    x1, y1, x2, y2 = bbox_xyxy
    x1 = int(np.clip(x1, 0, src_w - 1))
    y1 = int(np.clip(y1, 0, src_h - 1))
    x2 = int(np.clip(x2, 0, src_w - 1))
    y2 = int(np.clip(y2, 0, src_h - 1))
    box_w = x2 - x1
    box_h = y2 - y1
    if box_w <= 0 or box_h <= 0:
        raise ValueError(f"Invalid liveness bbox: {bbox_xyxy}")

    image = _crop_scaled(frame_bgr, (x1, y1, box_w, box_h), scale).astype(np.float32)
    return np.transpose(image, (2, 0, 1))[None, ...]


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


class FaceLivenessStage:
    def __init__(
        self,
        enabled: bool,
        threshold: float,
        triton: TritonInferenceClient,
        fasnet_v1_model: str = "fasnet_v1se",
        fasnet_v2_model: str = "fasnet_v2",
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.triton = triton
        self.fasnet_v1_model = fasnet_v1_model
        self.fasnet_v2_model = fasnet_v2_model

    def predict(self, frame_bgr: np.ndarray, bbox: list[int]) -> tuple[bool, float]:
        if not self.enabled:
            return True, 1.0

        try:
            v1_input = preprocess_fasnet(frame_bgr, bbox, scale=4.0)
            v2_input = preprocess_fasnet(frame_bgr, bbox, scale=2.7)
        except ValueError:
            return False, 0.0

        v1 = self.triton.infer(self.fasnet_v1_model, [TritonModelInput("input", v1_input)], ["logits"])[0]
        v2 = self.triton.infer(self.fasnet_v2_model, [TritonModelInput("input", v2_input)], ["logits"])[0]
        v1 = _softmax(v1)
        v2 = _softmax(v2)
        prediction = (v1 + v2) / 2.0
        label = int(np.argmax(prediction))
        score = float(prediction.reshape(-1)[label])
        return label == 1 and score >= self.threshold, score

    def accept(self, frame_bgr: np.ndarray, bbox: list[int]) -> tuple[bool, float]:
        return self.predict(frame_bgr, bbox)
