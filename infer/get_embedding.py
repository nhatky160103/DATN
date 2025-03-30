from torchvision import datasets
import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from .getface import mtcnn_inceptionresnetV1
from .infer_image import get_model, inceptionresnetV1_transform
import pickle
from torchvision import transforms
from .getface import yolo
from PIL import Image

# set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

workers = 0 if os.name == 'nt' else 4


# def create_data_embeddings(data_gallary_path, recognition_model_name, save_path):
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     recognition_model = get_model(recognition_model_name)

#     def collate_fn(x):
#         return x[0]

#     dataset = datasets.ImageFolder(data_gallary_path)
#     dataset.index2class = {i: c for c, i in dataset.class_to_idx.items()}
#     loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

#     aligned = []  # List of images in the gallery
#     image2class = {}   # List of names corresponding to images


#     for i, (x, y) in enumerate(loader):
#         x_aligned = mtcnn_inceptionresnetV1(x)
#         image2class[i]= y

#         if x_aligned is not None:
#             x_aligned = inceptionresnetV1_transform(x_aligned)
#             aligned.append(x_aligned)
#         else:
#             results = yolo(x)
#             print()
#             if results[0].boxes.xyxy.shape[0] != 0:
#                 boxes = results[0].boxes.xyxy.cpu().numpy() 
#                 x1, y1, x2, y2 = map(int, boxes[0]) 
#                 face = x.crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.LANCZOS)
#                 x_aligned = inceptionresnetV1_transform(face)
#                 aligned.append(x_aligned)
#             else:
#                 face = x.resize((160, 160), Image.Resampling.LANCZOS)
#                 x_aligned = inceptionresnetV1_transform(face)
#                 aligned.append(x_aligned)

#     if aligned:
#         aligned = torch.cat(aligned, dim=0).to(device)
#         embeddings = recognition_model(aligned).detach().cpu().numpy() 
      
#         # Save embedding 
#         embedding_file_path = os.path.join(save_path, f"{recognition_model_name}_embeddings.npy")
#         np.save(embedding_file_path, embeddings)

#         image2class_file_path = os.path.join(save_path, f"{recognition_model_name}_image2class.pkl")
#         with open(image2class_file_path, 'wb') as f:
#             pickle.dump(image2class, f)

#         index2class_file_path = os.path.join(save_path, f"{recognition_model_name}_index2class.pkl")
#         with open(index2class_file_path, 'wb') as f:
#             pickle.dump(dataset.index2class, f)

#         print(f"Embeddings saved to {embedding_file_path}")
#         print(f"image2class saved to {image2class_file_path}")
#         print(f"index2class saved to {index2class_file_path}")
        
#         return embeddings, image2class, dataset.index2class
#     else:
#         print("No aligned images found.")



def create_data_embeddings(data_gallary_path, recognition_model_name, save_path, batch_size=64):
    '''
    Generates embeddings for a dataset of images using a specified recognition model and saves the results.

    Parameters:
    ----------
    data_gallary_path : str
        Path to the directory containing the image dataset. The dataset should be organized in subdirectories 
        where each subdirectory represents a class.
        
    recognition_model_name : str
        The name of the recognition model to be used for generating embeddings. This will be used to load 
        the appropriate model.
        
    save_path : str
        Path to the directory where the generated embeddings
        will be saved.
        
    batch_size : int, optional
        The number of images to process in a single batch. Default is 64.

    Returns:
    -------
    embeddings : numpy.ndarray
        A 2D array where each row is the embedding of an image in the dataset.
        
    image2class : dict
        A dictionary mapping image indices to their corresponding class labels.
        
    dataset.index2class : dict
        A dictionary mapping index values to class names in the dataset.

    '''
    
     
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    recognition_model = get_model(recognition_model_name)

    def collate_fn(x):
        return x[0]

    dataset = datasets.ImageFolder(data_gallary_path)
    dataset.index2class = {i: c for c, i in dataset.class_to_idx.items()}
    loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)

    aligned = []  # List of aligned images in batches
    image2class = {}  # Mapping from image index to class label

    for i, (x, y) in enumerate(loader):
        x_aligned = mtcnn_inceptionresnetV1(x)
        image2class[i] = y

        if x_aligned is not None:
            x_aligned = inceptionresnetV1_transform(x_aligned)
            aligned.append(x_aligned)
        else:
            results = yolo(x)
            if results[0].boxes.xyxy.shape[0] != 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                x1, y1, x2, y2 = map(int, boxes[0])
                face = x.crop((x1, y1, x2, y2)).resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
                aligned.append(x_aligned)
            else:
                face = x.resize((160, 160), Image.Resampling.LANCZOS)
                x_aligned = inceptionresnetV1_transform(face)
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
    
    data_gallary_path = 'data/dataset'
    embedding_save_path = 'data/data_source'
    # embeddings, image2class, index2class = create_data_embeddings(data_gallary_path, 'inceptionresnetV1', embedding_save_path )
 

    embedding_file_path= 'data/data_source/db1/inceptionresnetV1_embeddings.npy'
    image2class_file_path = 'data/data_source/db1/inceptionresnetV1_image2class.pkl'
    index2class_file_path = 'data/data_source/db1/inceptionresnetV1_index2class.pkl'

    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)

    print(embeddings.shape)
    print(image2class)
    print(index2class)

 
