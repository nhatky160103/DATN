"""
Ultra-Light-Fast Face Detector - ONNX Inference Pipeline
Model: version-slim-320
Source: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
"""

import os
import cv2
import numpy as np
import onnxruntime as ort

# ── Cấu hình mặc định ────────────────────────────────────────────────────────
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(__file__), "weights", "version-slim-320.onnx")
INPUT_SIZE = (320, 240)        # (width, height) — khớp với model slim-320
CONF_THRESHOLD = 0.5           # Ngưỡng confidence để giữ detection
IOU_THRESHOLD  = 0.4           # Ngưỡng IoU cho NMS
# ─────────────────────────────────────────────────────────────────────────────


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> np.ndarray:
    """NMS trên box dạng [x1, y1, x2, y2], đồng bộ với runtime pipeline."""
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


class UltraLightDetector:
    """
    Wrapper ONNX cho Ultra-Light-Fast Face Detector (version-slim-320).

    Cách dùng:
        detector = UltraLightDetector()
        boxes, scores = detector.detect(frame)   # frame: BGR numpy array
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = CONF_THRESHOLD,
        iou_threshold: float  = IOU_THRESHOLD,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Không tìm thấy file model: {model_path}\n"
                "Hãy download 'version-slim-320.onnx' và đặt vào folder này."
            )

        # Chạy trên CPU
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name  = self.session.get_inputs()[0].name
        self.conf_threshold = conf_threshold
        self.iou_threshold  = iou_threshold

        # Warm-up để tránh latency ở lần inference đầu tiên
        dummy = np.zeros((1, 3, INPUT_SIZE[1], INPUT_SIZE[0]), dtype=np.float32)
        self.session.run(None, {self.input_name: dummy})

    # ── Tiền xử lý ───────────────────────────────────────────────────────────
    def _preprocess(self, image_bgr: np.ndarray) -> np.ndarray:
        """BGR HxWxC  →  float32 1xCxHxW, chuẩn hóa mean/std ImageNet."""
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, INPUT_SIZE)
        img = img.astype(np.float32)

        # Chuẩn hoá theo ImageNet mean / std
        mean = np.array([127.0, 127.0, 127.0], dtype=np.float32)
        std  = 128.0
        img  = (img - mean) / std

        img = img.transpose(2, 0, 1)    # HWC → CHW
        img = np.expand_dims(img, 0)    # CHW → 1CHW
        return img

    # ── Hậu xử lý ────────────────────────────────────────────────────────────
    def _postprocess(
        self,
        confidences: np.ndarray,
        boxes: np.ndarray,
        orig_h: int,
        orig_w: int,
    ):
        """
        Lọc theo confidence và chạy NMS.

        Returns:
            boxes_out  : np.ndarray shape (N, 4)  — [x1, y1, x2, y2] pixel coords
            scores_out : np.ndarray shape (N,)
        """
        # confidences shape: (1, num_anchors, 2)  — col 1 là prob "face"
        scores = confidences[0, :, 1]

        # boxes shape: (1, num_anchors, 4)  — normalized [x1, y1, x2, y2]
        bboxes = boxes[0]

        # Lọc theo ngưỡng confidence
        mask   = scores >= self.conf_threshold
        scores = scores[mask]
        bboxes = bboxes[mask]

        if len(scores) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        # Chuyển normalized [x1,y1,x2,y2] → pixel [x1,y1,x2,y2]
        x1 = (bboxes[:, 0] * orig_w).clip(0, orig_w - 1)
        y1 = (bboxes[:, 1] * orig_h).clip(0, orig_h - 1)
        x2 = (bboxes[:, 2] * orig_w).clip(0, orig_w - 1)
        y2 = (bboxes[:, 3] * orig_h).clip(0, orig_h - 1)
        pixel_boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

        indices = _nms(pixel_boxes, scores, self.iou_threshold)

        if len(indices) == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32)

        return np.rint(pixel_boxes[indices]).astype(np.int32), scores[indices]

    # ── API chính ─────────────────────────────────────────────────────────────
    def detect(self, image_bgr: np.ndarray):
        """
        Phát hiện khuôn mặt trong ảnh BGR.

        Args:
            image_bgr: numpy array BGR, shape (H, W, 3)

        Returns:
            boxes  : np.ndarray (N, 4)  — [[x1,y1,x2,y2], ...]  đơn vị pixel
            scores : np.ndarray (N,)    — confidence score mỗi box
        """
        orig_h, orig_w = image_bgr.shape[:2]
        inp = self._preprocess(image_bgr)
        confidences, boxes = self.session.run(None, {self.input_name: inp})
        return self._postprocess(confidences, boxes, orig_h, orig_w)

    def detect_and_draw(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Phát hiện và vẽ bounding box lên ảnh (dùng để debug/visualize).

        Returns:
            Ảnh BGR đã vẽ box.
        """
        boxes, scores = self.detect(image_bgr)
        result = image_bgr.copy()
        for (x1, y1, x2, y2), score in zip(boxes, scores):
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                result,
                f"{score:.2f}",
                (x1, max(y1 - 6, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        return result


# ── Demo nhanh khi chạy trực tiếp ────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from time import time

    parser = argparse.ArgumentParser(description="Ultra-Light Face Detector Demo")
    parser.add_argument("--source", type=str, default="0",
                        help="Đường dẫn ảnh/video, hoặc '0' để dùng webcam")
    parser.add_argument("--model",  type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--conf",   type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    detector = UltraLightDetector(model_path=args.model, conf_threshold=args.conf)
    print("✓ Model loaded.")

    # Nguồn: ảnh tĩnh
    if args.source.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
        frame = cv2.imread(args.source)
        if frame is None:
            raise FileNotFoundError(f"Không đọc được ảnh: {args.source}")
        t0 = time()
        boxes, scores = detector.detect(frame)
        print(f"Detected {len(boxes)} face(s) in {(time()-t0)*1000:.1f} ms")
        for i, (box, sc) in enumerate(zip(boxes, scores)):
            print(f"  [{i}] box={box.tolist()}  score={sc:.3f}")

    # Nguồn: webcam hoặc video
    else:
        src = 0 if args.source == "0" else args.source
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"Không mở được nguồn: {args.source}")
        print("Nhấn Ctrl+C để thoát.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            t0 = time()
            boxes, scores = detector.detect(frame)
            fps = 1.0 / (time() - t0)
            print(f"\rFPS: {fps:.1f}  Faces: {len(boxes)}", end="", flush=True)
        cap.release()
