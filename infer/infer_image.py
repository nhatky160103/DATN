import numpy as np
import cv2

def transform_image(img):
    """
    Tiền xử lý 1 ảnh cho ONNX ArcFace.
    Input: img (np.ndarray, BGR hoặc RGB).
    Output: numpy [1,3,112,112], float32.
    """
    if img is None:
        raise ValueError("Ảnh đầu vào None")

    # Resize -> 112x112
    img = cv2.resize(img, (112, 112))

    # Nếu ảnh là BGR (OpenCV) thì convert sang RGB
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # float32 + scale [0,255] -> [0,1]
    img = img.astype(np.float32) / 255.0

    # Normalize theo ArcFace: (x - 0.5) / 0.5
    img = (img - 0.5) / 0.5

    # HWC -> CHW -> NCHW
    img = np.transpose(img, (2, 0, 1))  # [H,W,C] -> [C,H,W]
    img = np.expand_dims(img, axis=0)   # [1,C,H,W]

    return img.astype(np.float32)


def transform_batch_image(imgs):
    """
    Tiền xử lý batch ảnh -> numpy [B,3,112,112].
    imgs: list[np.ndarray] hoặc np.ndarray duy nhất.
    """
    if isinstance(imgs, (list, tuple)):
        batch = [transform_image(im)[0] for im in imgs]  # lấy [C,H,W]
        return np.stack(batch, axis=0).astype(np.float32)
    else:
        return transform_image(imgs)  # [1,3,112,112]


def getEmbedding(rec_sess=None, images=None,
                 transform=transform_batch_image, keep_all=False):
    """
    Inference ONNX model để lấy embeddings.
    rec_sess: onnxruntime.InferenceSession
    images: list[np.ndarray] hoặc np.ndarray (ảnh gốc BGR/RGB)
    """
    # 1. Tiền xử lý ảnh
    images = transform(images)  # numpy [B,3,112,112]

    # 2. Lấy tên input/output từ session
    input_name = rec_sess.get_inputs()[0].name
    output_name = rec_sess.get_outputs()[0].name

    # 3. Inference ONNX
    embeddings = rec_sess.run([output_name], {input_name: images})[0]  # numpy [B,512]

    return embeddings
