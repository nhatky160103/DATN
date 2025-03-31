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

    if len(faces) == 0: 
        return image, None, 0, None  

    if keep_all:
        return dec_model(image), faces, probs, landmarks 

    face, prob, landmark = faces[0], probs[0], landmarks[0]
    return dec_model(image), face, prob, landmark
        

def getEmbedding(rec_model, transform, image, keep_all=False, device: str = 'cpu'):

    image = transform(image, keep_all)
    image = image.to(device)

    with torch.no_grad():
        embedding = rec_model(image)
    return embedding


def transform_image(img, keep_all=False):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_batch = transform(img) 
    if not keep_all:
        img_batch = img_batch.unsqueeze(0) 
    return img_batch



if __name__ == "__main__":
    
    keep_all_mode = False
    if keep_all_mode:
        mtcnn = mtcnn_keep_all
    else:
        mtcnn = mtcnn
    arcface_model = get_recogn_model()
    image = Image.open('data/Testset/thaotam/019.jpg').convert('RGB')

    input_image, face, prob, landmark = getFace(mtcnn, image, keep_all=keep_all_mode)
 
    print("input_image:")
    print(input_image.shape)
    print("face:")
    print(face)
    print("prob:")
    print(prob)
    print("landmark:")
    print(landmark)

    embedding =  getEmbedding(arcface_model,  transform_image, input_image, keep_all=keep_all_mode, device=device)                                          
    print(embedding.shape)

    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    x1, y1, x2, y2 = map(int, face)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.putText(image, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y) in landmark:
        cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
