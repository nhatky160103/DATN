from __future__ import annotations

import cv2
import numpy as np

from attendance_pipeline.triton_client import TritonInferenceClient, TritonModelInput


class LivenessStage:
    def __init__(
        self,
        enabled: bool,
        threshold: float,
        triton: TritonInferenceClient | None = None,
        fasnet_v1_model: str = "fasnet_v1se",
        fasnet_v2_model: str = "fasnet_v2",
    ):
        self.enabled = enabled
        self.threshold = threshold
        self.triton = triton
        self.fasnet_v1_model = fasnet_v1_model
        self.fasnet_v2_model = fasnet_v2_model
        self._legacy_model = None

    @property
    def legacy_model(self):
        if self._legacy_model is None:
            from models.Anti_spoof.FasNet import Fasnet

            self._legacy_model = Fasnet()
        return self._legacy_model

    def accept(self, frame_bgr: np.ndarray, bbox: list[int]) -> tuple[bool, float]:
        if not self.enabled:
            return True, 1.0

        if self.triton is not None and self.triton.enabled:
            try:
                x1, y1, x2, y2 = bbox
                face = frame_bgr[y1:y2, x1:x2]
                image = cv2.resize(face, (80, 80)).astype(np.float32)
                image = np.transpose(image, (2, 0, 1))[None, ...]
                v1 = self.triton.infer(self.fasnet_v1_model, [TritonModelInput("input", image)], ["logits"])[0]
                v2 = self.triton.infer(self.fasnet_v2_model, [TritonModelInput("input", image)], ["logits"])[0]
                v1 = np.exp(v1) / np.sum(np.exp(v1), axis=1, keepdims=True)
                v2 = np.exp(v2) / np.sum(np.exp(v2), axis=1, keepdims=True)
                prediction = (v1 + v2) / 2.0
                label = int(np.argmax(prediction))
                score = float(prediction.reshape(-1)[label])
                return label == 1 and score >= self.threshold, score
            except Exception as exc:
                print(f"Triton FASNet failed, falling back to legacy model: {exc}")

        is_real, score = self.legacy_model.analyze(frame_bgr, bbox)
        return bool(is_real and score >= self.threshold), float(score)
