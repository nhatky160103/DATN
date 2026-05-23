from __future__ import annotations

import cv2
import numpy as np

from pipeline.triton_client import TritonInferenceClient, TritonModelInput


def preprocess_arcface(face_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(face_bgr, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    return np.transpose(image, (2, 0, 1))[None, ...]


class FaceEmbeddingStage:
    def __init__(self, triton: TritonInferenceClient, model_name: str = "arcface"):
        self.triton = triton
        self.model_name = model_name

    def predict(self, face_bgr: np.ndarray) -> np.ndarray:
        output = self.triton.infer(
            self.model_name,
            [TritonModelInput("input", preprocess_arcface(face_bgr))],
            ["embedding"],
        )[0]
        return output.reshape(-1).astype(np.float32)

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        return self.predict(face_bgr)
