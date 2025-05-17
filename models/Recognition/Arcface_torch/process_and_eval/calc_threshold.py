import pickle
import torch
import mxnet as mx
from mxnet import ndarray as nd
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from models.Recognition.Arcface_torch.backbones import get_model
from tqdm import tqdm
import torch.nn.functional as F


device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


net = get_model("r100", fp16=False)
net.load_state_dict(torch.load("models/Recognition/Arcface_torch/weights/glint360k_cosface_r100_fp16_0.1/backbone.pth", map_location=device))
net.to(device)
net.eval()


def load_bin(path, image_size):
    try:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f)  # py2
    except UnicodeDecodeError as e:
        with open(path, 'rb') as f:
            bins, issame_list = pickle.load(f, encoding='bytes')  # py3
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
        img = img.astype(np.float32) / 255.0 
        for flip in [0, 1]:
            if flip == 1:
                img = mx.ndarray.flip(data=img, axis=2)
            data_list[flip][idx][:] = torch.from_numpy(img.asnumpy())
        if idx % 1000 == 0:
            print('loading bin', idx)
    print(data_list[0].shape)
    return data_list, issame_list

data_list, issame_list = load_bin('models/Recognition/Arcface_torch/dataset/VN-celeb.bin', (112, 112))


def get_embeddings(data_list, model, batch_size=512):
    num_samples = data_list[0].shape[0] 
    embeddings_flip0 = []
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, num_samples, batch_size), desc="Processing flip=0"):
            batch = data_list[0][i:i + batch_size].to(device)
            emb = model(batch)
            embeddings_flip0.append(emb)
    
    embeddings_flip0 = torch.cat(embeddings_flip0, dim=0)  
    return embeddings_flip0

embeddings = get_embeddings(data_list, net, batch_size=256)
print("Embeddings shape:", embeddings.shape)  

def compute_distances(embeddings, issame_list):
    l2_distances = []
    cosine_distances = []
    labels = []

    for i in range(len(issame_list)): 
        emb1 = embeddings[2 * i]
        emb2 = embeddings[2 * i + 1]
        l2_dist = torch.norm(emb1 - emb2, p=2).item()
        l2_distances.append(l2_dist)
        cosine_sim = F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        cosine_dist = 1 - cosine_sim
        cosine_distances.append(cosine_dist)
        labels.append(issame_list[i])

    l2_distances = np.array(l2_distances)
    cosine_distances = np.array(cosine_distances)
    labels = np.array(labels)
    return l2_distances, cosine_distances, labels

if __name__ == "__main__":
    l2_distances, cosine_distances, labels = compute_distances(embeddings, issame_list)

    print("L2 distances shape:", l2_distances.shape) 
    print("Cosine distances shape:", cosine_distances.shape) 
    print("Labels shape:", labels.shape) 


    l2_same_class_distances = l2_distances[labels == True]
    l2_diff_class_distances = l2_distances[labels == False]
    print("Mean distance L2 (same class):", l2_same_class_distances.mean())
    print("Mean distance L2 (diff class):", l2_diff_class_distances.mean())
    l2_threshold = (l2_same_class_distances.mean() + l2_diff_class_distances.mean()) / 2
    print("Simple L2 threshold:", l2_threshold)


    cosine_same_class_distances = cosine_distances[labels == True]
    cosine_diff_class_distances = cosine_distances[labels == False]
    print("Mean distance Cosine (same class):", cosine_same_class_distances.mean())
    print("Mean distance Cosine (diff class):", cosine_diff_class_distances.mean())
    cosine_threshold = (cosine_same_class_distances.mean() + cosine_diff_class_distances.mean()) / 2
    print("Simple Cosine threshold:", cosine_threshold)
