import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_recogn_model
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import cv2
from models.Detection.mtcnn import MTCNN


# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


mtcnn = MTCNN(
    image_size=112, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    select_largest=True,
    selection_method='largest',
    device=device,
    keep_all=False,
)
mtcnn_keep_all = MTCNN(
    image_size=112, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    select_largest=True,
    selection_method='largest',
    device=device,
    keep_all=True,
)

def getFace(dec_model=None, image=None, keep_all=False):
    if dec_model is None or image is None:
        raise ValueError("Cần truyền cả dec_model và image")

    faces, probs, landmarks = dec_model.detect(image, landmarks=True)

    if faces is None: 
        resized_image = cv2.resize(image, (112, 112))
        return resized_image, None, 0, None  

    if keep_all:
        return dec_model(image), faces, probs, landmarks 

    face, prob, landmark = faces[0], probs[0], landmarks[0]
    return dec_model(image), face, prob, landmark
        


def transform_image(img, keep_all=False):
    if not isinstance(img, torch.Tensor):
        img = transforms.ToTensor()(img)
    if not keep_all:
        img = img.unsqueeze(0) 
    return img



def getEmbedding(rec_model= None, image = None, transform = transform_image, keep_all=False, device: str = 'cpu'):

    image = transform(image, keep_all)
    image = image.to(device)

    with torch.no_grad():
        embedding = rec_model(image)
    return embedding



    
