# import cv2
# import numpy as np
# import onnxruntime as ort

# onnx_path = "models/weights/backbone.onnx"

# # 1. Load ONNX model
# sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
# input_name = sess.get_inputs()[0].name
# output_name = sess.get_outputs()[0].name

# print(f"Input name: {input_name}, Output name: {output_name}")

# # 2. Load và tiền xử lý ảnh
# img = cv2.imread("models/test.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (112, 112))
# img = img.astype(np.float32)

# # PyTorch style: HWC -> CHW -> NCHW
# img = np.transpose(img, (2, 0, 1))  # [H,W,C] -> [C,H,W]
# img = np.expand_dims(img, axis=0)   # [1,C,H,W]

# # 3. Inference
# embedding = sess.run([output_name], {input_name: img})[0]  # shape [1,512]

# # 4. Kiểm tra kết quả
# print("Embedding shape:", embedding.shape)
# print(embedding)
# print("Vector norm:", np.linalg.norm(embedding, axis=1)[0])
