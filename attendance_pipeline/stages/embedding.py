from __future__ import annotations

import cv2
import numpy as np

from attendance_pipeline.triton_client import TritonInferenceClient, TritonModelInput


def preprocess_arcface(face_bgr: np.ndarray) -> np.ndarray:
    image = cv2.resize(face_bgr, (112, 112))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = (image / 255.0 - 0.5) / 0.5
    return np.transpose(image, (2, 0, 1))[None, ...]


class FaceEmbeddingStage:
    def __init__(self, triton: TritonInferenceClient | None = None, model_name: str = "arcface"):
        self.triton = triton
        self.model_name = model_name
        self._legacy_model = None

    @property
    def legacy_model(self):
        if self._legacy_model is None:
            from infer.utils import get_recogn_model

            self._legacy_model = get_recogn_model()
        return self._legacy_model

    def embed(self, face_bgr: np.ndarray) -> np.ndarray:
        if self.triton is not None and self.triton.enabled:
            try:
                output = self.triton.infer(
                    self.model_name,
                    [TritonModelInput("input", preprocess_arcface(face_bgr))],
                    ["embedding"],
                )[0]
                return output.reshape(-1).astype(np.float32)
            except Exception as exc:
                print(f"Triton ArcFace failed, falling back to legacy PyTorch model: {exc}")

        import torch
        from PIL import Image
        from infer.infer_image import transform_image
        from infer.utils import device

        image = Image.fromarray(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB))
        tensor = transform_image(image).to(device)
        with torch.no_grad():
            embedding = self.legacy_model(tensor).detach().cpu().numpy()
        return embedding.reshape(-1).astype(np.float32)

