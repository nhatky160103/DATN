from __future__ import annotations

import cv2
import numpy as np

from attendance_pipeline.triton_client import TritonInferenceClient, TritonModelInput


class FaceQualityStage:
    def __init__(
        self,
        threshold: float,
        triton: TritonInferenceClient | None = None,
        model_name: str = "lightqnet",
    ):
        self.threshold = threshold
        self.triton = triton
        self.model_name = model_name
        self._legacy_model = None

    @property
    def legacy_model(self):
        if self._legacy_model is None:
            from models.lightqnet.tf_face_quality_model import TfFaceQaulityModel

            self._legacy_model = TfFaceQaulityModel()
        return self._legacy_model

    def score(self, face_bgr: np.ndarray) -> float:
        if self.triton is not None and self.triton.enabled:
            try:
                image = cv2.resize(face_bgr, (96, 96)).astype(np.float32)
                image = ((image - 128.0) / 128.0)[None, ...]
                output = self.triton.infer(
                    self.model_name,
                    [TritonModelInput("input", image)],
                    ["confidence_st"],
                )[0]
                return float(output.reshape(-1)[0])
            except Exception as exc:
                print(f"Triton LightQNet failed, falling back to legacy model: {exc}")
        return float(self.legacy_model.inference(face_bgr))

    def accept(self, face_bgr: np.ndarray) -> tuple[bool, float]:
        value = self.score(face_bgr)
        return value >= self.threshold, value
