from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from .infer_image import  transform_image
from .utils import get_recogn_model
import pickle
from torchvision import transforms
from PIL import Image
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from models.Detection.mtcnn import MTCNN
import matplotlib.pyplot as plt


model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)
def yolo_transform_image(img, keep_all=False):
    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    img_batch = transform(img) 
    img_batch = img_batch.unsqueeze(0) 
    return img_batch


# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

workers = 0 if os.name == 'nt' else 4

mtcnn =  MTCNN(
            image_size=112, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            select_largest= True,
            selection_method= 'largest',
            device=device,
            keep_all= False,
        )


def create_data_embeddings(data_gallary_path, recognition_model_name, backbone_name, save_path, batch_size: int=16):
      
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recognition_model = get_recogn_model(recognition_model_name)

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(data_gallary_path)
    dataset.index2class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []  # List of aligned images in batches
    image2class = {}  # Mapping from image index to class label

    for i, (x, y) in enumerate(loader):
        x_aligned = mtcnn(x)
        image2class[i] = y
        
        if x_aligned is not None:
            x_aligned = transform_image(x_aligned)
            aligned.append(x_aligned)

        else:
            results = yolo(x)
            if results[0].boxes.xyxy.shape[0] != 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, boxes[0])
                face = x.crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = yolo_transform_image(face)
                plt.imshow(x_aligned.permute(1, 2, 0).cpu().numpy())
                plt.show()
                aligned.append(x_aligned)
            else:
                face = x.resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = yolo_transform_image(face)
                aligned.append(x_aligned)

        # Process batch when it reaches the batch_size
        if len(aligned) >= batch_size:
            batch = torch.cat(aligned[:batch_size], dim=0).to(device)
            embeddings_batch = recognition_model(batch).detach().cpu().numpy()
            if 'embeddings' not in locals():
                embeddings = embeddings_batch
            else:
                embeddings = np.vstack((embeddings, embeddings_batch))

            # Remove processed items
            aligned = aligned[batch_size:]

    # Process remaining items in the aligned list
    if aligned:
        batch = torch.cat(aligned, dim=0).to(device)
        embeddings_batch = recognition_model(batch).detach().cpu().numpy()
        if 'embeddings' not in locals():
            embeddings = embeddings_batch
        else:
            embeddings = np.vstack((embeddings, embeddings_batch))

    # Save embeddings
    embedding_file_path = os.path.join(save_path, f"{recognition_model_name}_embeddings.npy")
    np.save(embedding_file_path, embeddings)

    image2class_file_path = os.path.join(save_path, f"{recognition_model_name}_image2class.pkl")
    with open(image2class_file_path, 'wb') as f:
        pickle.dump(image2class, f)

    index2class_file_path = os.path.join(save_path, f"{recognition_model_name}_index2class.pkl")
    with open(index2class_file_path, 'wb') as f:
        pickle.dump(dataset.index2class, f)

    print(f"Embeddings saved to {embedding_file_path}")
    print(f"image2class saved to {image2class_file_path}")
    print(f"index2class saved to {index2class_file_path}")

    return embeddings, image2class, dataset.index2class


def load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path):
    '''
        Loads precomputed embeddings, image-to-class mapping, and index-to-class mapping from saved files.

        Process:
        -------
        1. Loads the embeddings from a `.npy` file.
        2. Deserializes the `image2class` dictionary from a `.pkl` file.
        3. Deserializes the `index2class` dictionary from a `.pkl` file.

        Returns:
        -------
        embeddings : numpy.ndarray
            A 2D array containing the precomputed embeddings for the dataset.
            
        image2class : dict
            A dictionary mapping image indices to their corresponding class labels.
            
        index2class : dict
            A dictionary mapping index values to class names.

    '''
    embeddings = np.load(embedding_file_path)
    with open(image2class_file_path, 'rb') as f:
        image2class = pickle.load(f)

    with open(index2class_file_path, 'rb') as f:
        index2class = pickle.load(f)

    return embeddings, image2class, index2class

if __name__ == '__main__':
    
    data_gallary_path = 'data/Testset'
    embedding_save_path = 'data/data_source'
    # embeddings, image2class, index2class = create_data_embeddings(data_gallary_path, 'glint360k_cosface', 'r100', embedding_save_path )
  
    embedding_file_path= 'data/data_source/glint360k_cosface_embeddings.npy'
    image2class_file_path = 'data/data_source/glint360k_cosface_image2class.pkl'
    index2class_file_path = 'data/data_source/glint360k_cosface_index2class.pkl'

    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)

    print(embeddings.shape)
    print(image2class)
    print(index2class)

 
