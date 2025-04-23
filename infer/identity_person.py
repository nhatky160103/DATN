
import torch
import os
import cv2
from PIL import Image
from torch.nn.modules.distance import PairwiseDistance
import torch.nn.functional as F
from collections import Counter, defaultdict
import numpy as np
import yaml


from models.Anti_spoof.FasNet import Fasnet
from .utils import get_recogn_model
from .infer_image import getFace, mtcnn, mtcnn_keep_all, getEmbedding, transform_image


# use config file
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)['identity_person']

# define distance calculator, device and anti-spoof model
l2_distance = PairwiseDistance(p=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
antispoof_model = Fasnet()


def find_closest_person(
        pred_embed, 
        embeddings, 
        image2class, 
        distance_mode=config['distance_mode'], 
        l2_threshold=config['l2_threshold'], 
        cosine_threshold=config['cosine_threshold']):
    '''
     Finds the closest matching class (person) for a given embedding by comparing it to precomputed embeddings.

    Parameters:
    ----------
    pred_embed : torch.Tensor
        The embedding of the query image.
    embeddings : numpy.ndarray
        Precomputed embeddings of the dataset.
    image2class : dict
        Maps image indices to class labels.
    distance_mode : str, optional
        Metric for comparison ('l2' for Euclidean, 'cosine' for cosine similarity).
    l2_threshold : float, optional
        Max L2 distance for a valid match.
    cosine_threshold : float, optional
        Max cosine distance (1 - similarity) for a valid match.

    Returns:
    -------
    best_class : int
        Class label of the closest match, or -1 if no match meets the threshold.

    '''
 
    embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)

    if distance_mode == 'l2':
        distances = torch.norm(embeddings_tensor - pred_embed, dim=1).detach().cpu().numpy()
    else:
     
        similarities = F.cosine_similarity(pred_embed, embeddings_tensor)
        distances = (1 - similarities).detach().cpu().numpy()

    image2class_np = np.array([image2class[i] for i in range(len(embeddings))])
    
    num_classes = max(image2class.values()) + 1
    
    total_distances = np.zeros(num_classes, dtype=np.float32)
    np.add.at(total_distances, image2class_np, distances)

    counts = np.zeros(num_classes, dtype=np.int32)
    np.add.at(counts, image2class_np, 1)

    avg_distances = np.divide(total_distances, counts, out=np.full_like(total_distances, np.inf), where=counts > 0)
    print(avg_distances)
    if distance_mode == 'l2': # l2
        best_class = np.argmin(avg_distances) 
        if avg_distances[best_class] < l2_threshold:
            return best_class
    else:  # Cosine
        best_class = np.argmin(avg_distances)
        if avg_distances[best_class] < cosine_threshold:
            return best_class
    
    return -1
 

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    embedding_file_path= 'data/data_source/glint360k_cosface_embeddings.npy'
    image2class_file_path = 'data/data_source/glint360k_cosface_image2class.pkl'
    index2class_file_path = 'data/data_source/glint360k_cosface_index2class.pkl'

    embeddings, image2class, index2class = load_embeddings_and_names(embedding_file_path, image2class_file_path, index2class_file_path)
    
    keep_all_mode = False
    arcface_model = get_recogn_model()

  
    image_folder = "data/Testset/thaotam"
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path).convert('RGB')
        align_image, faces, probs, lanmark  = getFace(mtcnn, image)
        pred_embed = getEmbedding(rec_model = arcface_model, transform = transform_image , image = align_image, keep_all = keep_all_mode)

        result = find_closest_person(pred_embed, embeddings, image2class, index2class)
        print(result)






