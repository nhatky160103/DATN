import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from .utils import device

def transform_image(img, keep_all=False):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    if not isinstance(img, torch.Tensor):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        img = transform(img)
    img = img.unsqueeze(0) 
    return img


def transform_batch_image(imgs, keep_all=False):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transformed = []
    if isinstance(imgs, (list, tuple)):
        for img in imgs:
            if not isinstance(img, torch.Tensor):
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                img = transform(img)
            transformed.append(img)
        return torch.stack(transformed)

def getEmbedding(rec_model=None, images=None, transform=transform_batch_image, keep_all=False):
    # Transform images to tensor batch
    images = transform(images, keep_all)
    images = images.to(device)

    # Get embeddings for batch
    with torch.no_grad():
        embeddings = rec_model(images)
    return embeddings
