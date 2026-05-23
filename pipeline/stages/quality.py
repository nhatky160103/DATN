from __future__ import annotations

import cv2
import numpy as np

from pipeline.triton_client import TritonInferenceClient, TritonModelInput


def preprocess_lightqnet(face_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(face_bgr, (96, 96)).astype(np.float32)
    return ((image - 128.0) / 128.0)[None, ...]


class FaceQualityStage:
    def __init__(
        self,
        threshold: float,
        triton: TritonInferenceClient,
        model_name: str = "lightqnet",
    ):
        self.threshold = threshold
        self.triton = triton
        self.model_name = model_name

    def predict(self, face_bgr: np.ndarray) -> float:
        output = self.triton.infer(
            self.model_name,
            [TritonModelInput("input:0", preprocess_lightqnet(face_bgr))],
            ["confidence_st:0"],
        )[0]
        return float(output.reshape(-1)[0])

    def score(self, face_bgr: np.ndarray) -> float:
        return self.predict(face_bgr)

    def accept(self, face_bgr: np.ndarray) -> tuple[bool, float]:
        value = self.predict(face_bgr)
        return value >= self.threshold, value
