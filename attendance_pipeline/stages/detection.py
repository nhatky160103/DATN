from __future__ import annotations

import base64

import cv2
import numpy as np
from PIL import Image

from attendance_pipeline.schemas import FaceDetection


class MTCNNFaceDetector:
    def __init__(
        self,
        threshold: float = 0.7,
        max_side: int = 640,
        min_face_size: int = 40,
        factor: float = 0.8,
        keep_all: bool = True,
        torch_num_threads: int = 2,
    ):
        self.threshold = threshold
        self.max_side = max_side
        self.min_face_size = min_face_size
        self.factor = factor
        self.keep_all = keep_all
        self.torch_num_threads = torch_num_threads
        self._mtcnn = None

    @property
    def mtcnn(self):
        if self._mtcnn is None:
            import torch
            from infer.utils import device
            from models.Detection.mtcnn import MTCNN

            if self.torch_num_threads > 0:
                torch.set_num_threads(self.torch_num_threads)

            self._mtcnn = MTCNN(
                image_size=112,
                margin=0,
                min_face_size=self.min_face_size,
                thresholds=[0.6, 0.7, 0.7],
                factor=self.factor,
                post_process=False,
                select_largest=not self.keep_all,
                keep_all=self.keep_all,
                device=device,
            )
        return self._mtcnn

    def detect(self, frame_bgr: np.ndarray) -> list[FaceDetection]:
        height, width = frame_bgr.shape[:2]
        scale = 1.0
        detect_frame = frame_bgr
        longest_side = max(height, width)
        if self.max_side > 0 and longest_side > self.max_side:
            scale = self.max_side / float(longest_side)
            detect_frame = cv2.resize(
                frame_bgr,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_AREA,
            )

        image_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        boxes, probs = self.mtcnn.detect(pil_image, landmarks=False)

        if boxes is None or probs is None:
            return []

        detections: list[FaceDetection] = []
        for box, prob in zip(boxes, probs):
            if prob is None or float(prob) < self.threshold:
                continue
            x1, y1, x2, y2 = [int(v / scale) for v in box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(width - 1, x2), min(height - 1, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            crop = frame_bgr[y1:y2, x1:x2]
            ok, encoded = cv2.imencode(".jpg", crop)
            if not ok:
                continue
            detections.append(
                FaceDetection(
                    bbox=[x1, y1, x2, y2],
                    score=float(prob),
                    crop_jpeg_b64=base64.b64encode(encoded.tobytes()).decode("ascii"),
                )
            )
        return detections
