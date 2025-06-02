import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from infer.utils import get_recogn_model, device  # Thay bằng module chứa get_model bạn đã đưa
import mxnet as mx
import numpy as np
from mxnet import ndarray as nd


# 1. Load .bin file (chuẩn với MXNet)
def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')

    data_list = []
    for flip in [0, 1]:
        data = torch.empty((len(issame_list) * 2, 3, image_size[0], image_size[1]))
        data_list.append(data)

    for idx in range(len(issame_list) * 2):
        _bin = bins[idx]
        img = mx.image.imdecode(_bin)
        if img.shape[1] != image_size[0]:
            img = mx.image.resize_short(img, image_size[0])
        img = nd.transpose(img, axes=(2, 0, 1))
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

# 2. Chuẩn hóa ảnh (MXNet -> PyTorch)
def preprocess_batch(imgs):
    imgs = imgs.float() / 255.0
    imgs = (imgs - 0.5) / 0.5
    return imgs

# 3. Lấy embedding
def get_embeddings(model, data):
    model.eval()
    embeddings = []
    data = preprocess_batch(data).to(device)
    with torch.no_grad():
        for i in range(0, data.size(0), 64):
            batch = data[i:i+64]
            emb = model(batch)
            emb = torch.nn.functional.normalize(emb, dim=1)
            embeddings.append(emb.cpu())
    return torch.cat(embeddings, dim=0).numpy()

# 4. Tính góc giữa các vector embedding
def compute_angles(embeddings, issame):
    pos_angles = []
    neg_angles = []
    for i in range(0, len(embeddings), 2):
        emb1 = embeddings[i]
        emb2 = embeddings[i + 1]
        cos_sim = np.dot(emb1, emb2)
        angle = np.arccos(np.clip(cos_sim, -1.0, 1.0)) * 180 / np.pi
        if issame[i // 2]:
            pos_angles.append(angle)
        else:
            neg_angles.append(angle)
    return pos_angles, neg_angles

# 5. Vẽ histogram
def plot_angle_histogram(pos_angles, neg_angles):
    plt.hist(pos_angles, bins=100, alpha=0.6, color='red', label='Positive')
    plt.hist(neg_angles, bins=100, alpha=0.6, color='blue', label='Negative')
    plt.xlabel('Angle between Embeddings (degrees)')
    plt.ylabel('Pair Numbers')
    plt.legend()
    plt.title('Angle Distribution of Positive and Negative Pairs')
    plt.grid(True)
    plt.show()

# 6. Main
if __name__ == "__main__":
    image_size = (112, 112)

    # Load model
    model = get_recogn_model()

    bin_path = 'models/Recognition/Arcface_torch/dataset/lfw.bin'
    data_list, issame_list = load_bin(bin_path, image_size)
    data = data_list[0]  # Dùng ảnh gốc, không lật

    # Extract embeddings
    embeddings = get_embeddings(model, data)

    # Compute angles
    pos_angles, neg_angles = compute_angles(embeddings, issame_list)

    # Plot
    plot_angle_histogram(pos_angles, neg_angles)
