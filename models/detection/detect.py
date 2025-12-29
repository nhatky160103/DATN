import cv2
import numpy as np
import onnxruntime as ort
from .box_utils_numpy import hard_nms


onnx_path="models/weights/version-RFB-320.onnx"
session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def detect_face_and_nose(image: np.ndarray,
                         input_size=(320, 240),
                         threshold=0.7):
    """
    Detect a single face in an image using ONNX model.
    
    Args:
        image (np.ndarray): input BGR image
        onnx_path (str): path to ONNX model
        input_size (tuple): model input size (w, h)
        threshold (float): confidence threshold

    Returns:
        (bbox, nose, prob) or None
        bbox = (x1, y1, x2, y2)
        nose = (nose_x, nose_y) center of bbox
        prob = float
    """

    def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.3, top_k=-1):
        boxes = boxes[0]
        confidences = confidences[0]
        picked_box_probs = []
        for class_index in range(1, confidences.shape[1]):
            probs = confidences[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.shape[0] == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
            box_probs = hard_nms(box_probs, iou_threshold=iou_threshold, top_k=top_k)
            picked_box_probs.append(box_probs)

        if not picked_box_probs:
            return np.array([]), np.array([])

        picked_box_probs = np.concatenate(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height

        return picked_box_probs[:, :4].astype(np.int32), picked_box_probs[:, 4]

    # Load model
    
    input_name = session.get_inputs()[0].name

    orig_h, orig_w = image.shape[:2]

    # Preprocess
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, input_size)
    img_mean = np.array([127, 127, 127])
    img = (img - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # Inference
    confidences, boxes = session.run(None, {input_name: img})

    # Postprocess
    boxes, probs = predict(orig_w, orig_h, confidences, boxes, threshold)

    if boxes.shape[0] == 0:
        return (0,0,0,0), (0,0), 0.0

    # chọn box có prob cao nhất
    idx = np.argmax(probs)
    x1, y1, x2, y2 = boxes[idx]
    prob = float(probs[idx])
    nose_x = int((x1 + x2) / 2)
    nose_y = int((y1 + y2) / 2)

    return (x1, y1, x2, y2), (nose_x, nose_y), prob


if __name__ == "__main__":
    img = cv2.imread("models/detection/test.jpg")

    for i in range(100):
        result = detect_face_and_nose(img)

        if result:
            bbox, nose, prob = result
            print("BBox:", bbox)
            print("Nose:", nose)
            print("Prob:", prob)
            print("_____"* 10)
        else:
            print("No face detected")
