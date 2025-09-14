"""Main implementation class of PFE with ONNX Runtime
"""
import argparse
from abc import ABC
import numpy as np
import cv2
from time import time
import onnxruntime as ort


ONNX_MODEL_PATH = 'models/weights/lightqnet-dm025.onnx'
ONNX_FEAT_IMG_W = 96
ONNX_FEAT_IMG_H = 96


class OnnxFaceQualityModel(ABC):
    def __init__(self, model_data_path=ONNX_MODEL_PATH, image_w=ONNX_FEAT_IMG_W, image_h=ONNX_FEAT_IMG_H):
        self.image_w = image_w
        self.image_h = image_h
        self.model_data_path = model_data_path

        start_time = time()
        # Load ONNX model
        self.session = ort.InferenceSession(self.model_data_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name   # thường là "input:0"
        self.output_name = self.session.get_outputs()[0].name # thường là "confidence_st:0"
        end_time = time()
        print(f"ONNX model loaded in {(end_time - start_time) * 1000:.2f} ms")
        print("Input name:", self.input_name)
        print("Output name:", self.output_name)

    def inference(self, img, to_rgb=True, verbose=0):
        """Run inference with ONNX Runtime."""
        img = cv2.resize(img, (self.image_w, self.image_h))
        if to_rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = (img - 128.) / 128.
        images = np.expand_dims(img, axis=0).astype(np.float32)

        start_time = time()
        qscore = self.session.run([self.output_name], {self.input_name: images})[0]
        end_time = time()

        if verbose > 0:
            print('Inference {:.2f} ms'.format((end_time - start_time) * 1000))

        return qscore[0][0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',
                        help='image path',
                        default='models/lightqnet/test.jpg')
    parser.add_argument('--batch_size',
                        type=int, default=1,
                        help='Batch size.')
    parser.add_argument('--num_batches',
                        type=int, default=1,
                        help='Number of batches to run.')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Không load được ảnh từ {args.image}")

    face_quality_predictor = OnnxFaceQualityModel()

    # Test 10 lần inference
    for _ in range(10):
        score = face_quality_predictor.inference(img)
        print("qscore:", score)

    # Benchmark
    t1 = time()
    iters = 10
    for i in range(iters):
        img2 = img + i * np.random.random(img.shape) * 10
        img2 = img2.astype(np.uint8)
        score = face_quality_predictor.inference(img2)
        print(score)
    print('average %.2f ms' % ((time() - t1) * 1000. / iters))
    print('----------------')
