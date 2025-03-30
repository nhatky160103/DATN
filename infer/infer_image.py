import torch
import matplotlib.pyplot as plt
import numpy as np
from .utils import get_recogn_model
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F
import cv2
from models.Detection.mtcnn import MTCNN


# define device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GetFace:
    def __init__(self, keep_all: bool = False, selection_method: str = 'largest', device: str = 'cpu'):
        '''
        Initializes the InferModel class with a detection model and optional transformations.

        Parameters:
        ----------
        transforms : torchvision.transforms.Compose, optional
            A composition of transformations to be applied to the input image.
        detect_model : torch.nn.Module, optional
            The face detection model to be used for detecting faces in images.
        '''
        self.detect_model = MTCNN(
                                image_size=160, margin=0, min_face_size=20,
                                thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
                                select_largest= True,
                                selection_method= selection_method,
                                device=device,
                                keep_all= keep_all,
                            )

        self.device = device
        self.keep_all = keep_all

    def __call__(self, image):
        '''
        Detects faces and returns cropped images, bounding boxes, probabilities, and landmarks.
        '''
        if self.detect_model is None:
            raise ValueError("Detection model not provided.")

        if isinstance(image, list):
            all_faces, all_boxes, all_probs, all_landmarks = [], [], [], []
            for img in image:
                faces, boxes, probs, landmarks = self.process_single_image(img)
                all_faces.append(faces)
                all_boxes.append(boxes)
                all_probs.append(probs)
                all_landmarks.append(landmarks)

            return all_faces, all_boxes, all_probs, all_landmarks
        else:
            return self.process_single_image(image)

    def process_single_image(self, image):
        face= None
        input_image = image
        prob = 0
        lanmark = None
        if self.detect_model is not None:
            boxes, probs, landmarks = self.detect_model.detect(image, landmarks=True)
            crop_image = self.detect_model(image)
            if self.keep_all: 
                return  crop_image, boxes, probs, landmarks
            return crop_image, boxes[0], probs[0], landmarks[0]
          
        else:
            raise ValueError("Detection model not provided.")

class GetEmbedding():
    '''
    Initializes the InferModel class with a detection model and optional transformations.
    Parameters:
    ----------
    transforms : torchvision.transforms.Compose, optional
        A composition of transformations to be applied to the input image.
    detect_model : torch.nn.Module, optional
        The face detection model to be used for detecting faces in images.
    '''
    def __init__(self, model, transform, device):
        self.model = model
        self.device = device
        self.model.eval()
        self.model.to(self.device)
        self.transform = transform
    def __call__(self, images):
        '''
        Detects faces and returns cropped images, bounding boxes, probabilities, and landmarks.
        '''
        if self.model is None:
            raise ValueError("Model not provided.")
        if self.transform is None:
            raise ValueError("Transform not provided.")
        images = self.transform(images)
        images = images.to(self.device)
        with torch.no_grad():
            embeddings = self.model(images)
        return embeddings
    

def transform_image(img):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    if isinstance(img, list): 
        img_tensors = [transform(im) for im in img]
        img_batch = torch.stack(img_tensors)  
    else:  
        img_batch = transform(img) 
        img_batch = img_batch.unsqueeze(0) 

    return img_batch

if __name__ == "__main__":
    
    det_model = GetFace(keep_all = True)
    arcface_model = get_recogn_model()
    rec_model = GetEmbedding(model= arcface_model, transform= transform_image, device=device)
    image1 = Image.open('data/img1.jpg').convert('RGB')
    image2 = Image.open('data/img2.jpg').convert('RGB')
    image = [image1, image2]
    input_image, face, prob, landmark = det_model(image)


    print(input_image[0].shape)
    print(input_image[1].shape)
    print(face)
    print(prob)
    print(landmark)

    # embedding = rec_model(input_image) 
    # print(embedding.shape)

    # image = np.array(image)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # x1, y1, x2, y2 = map(int, face)
    # cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # cv2.putText(image, f"Face {prob:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # for (x, y) in landmark:
    #     cv2.circle(image, (int(x), int(y)), 4, (0, 0, 255), -1)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
