import cv2
import numpy as np
import onnxruntime as ort
from typing import Union


first_model_onnx_file = 'models/weights/MiniFASNetV1SE.onnx'
second_model_onnx_file = 'models/weights/MiniFASNetV2.onnx'


class FasnetOnnx:
    def __init__(self):
        # Load 2 models vào session
        self.first_sess = ort.InferenceSession(first_model_onnx_file, providers=['CPUExecutionProvider'])
        self.second_sess = ort.InferenceSession(second_model_onnx_file, providers=['CPUExecutionProvider'])

        # Lấy tên input/output
        self.first_input_name = self.first_sess.get_inputs()[0].name
        self.first_output_name = self.first_sess.get_outputs()[0].name

        self.second_input_name = self.second_sess.get_inputs()[0].name
        self.second_output_name = self.second_sess.get_outputs()[0].name

        print(f"First model input: {self.first_input_name}, output: {self.first_output_name}")
        print(f"Second model input: {self.second_input_name}, output: {self.second_output_name}")

    def analyze(self, img: np.ndarray, facial_area: Union[list, tuple]):
        """
        Analyze a given image spoofed or not using ONNX models
        Args:
            img (np.ndarray): pre loaded image
            facial_area (list or tuple): facial rectangle area coordinates with x, y, w, h respectively
        Returns:
            result (tuple): (is_real, score)
        """
        x, y, w, h = facial_area
        first_img = crop(img, (x, y, w, h), 2.7, 80, 80)
        second_img = crop(img, (x, y, w, h), 4, 80, 80)

        # Chuẩn hóa như trong PyTorch: [H, W, C] -> [1, 3, 80, 80], float32
        first_input = np.transpose(first_img.astype(np.float32) , (2, 0, 1))[None, ...]
        second_input = np.transpose(second_img.astype(np.float32) , (2, 0, 1))[None, ...]

        # Inference ONNX
        first_result = self.first_sess.run([self.first_output_name], {self.first_input_name: first_input})[0]
        second_result = self.second_sess.run([self.second_output_name], {self.second_input_name: second_input})[0]

        # Softmax (nếu model chưa có softmax trong graph)
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
            return e_x / np.sum(e_x, axis=1, keepdims=True)

        first_result = softmax(first_result)
        second_result = softmax(second_result)

        # Cộng kết quả
        prediction = np.zeros((1, 3))
        prediction += first_result
        prediction += second_result

        label = np.argmax(prediction)
        is_real = True if label == 1 else False
        score = prediction[0][label] / 2

        return is_real, score


# -----------------------------
# Giữ nguyên crop() và helper từ code cũ
def _get_new_box(src_w, src_h, bbox, scale):
    x = bbox[0]
    y = bbox[1]
    box_w = bbox[2]
    box_h = bbox[3]
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


def crop(org_img, bbox, scale, out_w, out_h):
    src_h, src_w, _ = np.shape(org_img)
    left_top_x, left_top_y, right_bottom_x, right_bottom_y = _get_new_box(src_w, src_h, bbox, scale)
    img = org_img[left_top_y: right_bottom_y + 1, left_top_x: right_bottom_x + 1]
    dst_img = cv2.resize(img, (out_w, out_h))
    return dst_img

