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

def getEmbedding(rec_model= None, image = None, transform = transform_image, keep_all=False):

    image = transform(image, keep_all)
    image = image.to(device)

    with torch.no_grad():
        embedding = rec_model(image)
    return embedding



    
