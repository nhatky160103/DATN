
import torch.nn.functional as F
import torch
import numpy as np


def find_closest_person(
        pred_embed, 
        embeddings, 
        image2class, 
        distance_mode, 
        l2_threshold, 
        cosine_threshold):
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
    # print(avg_distances)
    
    best_class = np.argmin(avg_distances)
    print(avg_distances[best_class])
    
    if distance_mode == 'l2': # l2
        if avg_distances[best_class] < l2_threshold:
            return best_class
    else:  # Cosine
        
        if avg_distances[best_class] < cosine_threshold:
            return best_class
    
    return -1
 






