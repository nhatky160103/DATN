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

    Parameters
    ----------
    pred_embed : numpy.ndarray
        The embedding of the query image. Shape (512,)
    embeddings : numpy.ndarray
        Precomputed embeddings of the dataset. Shape (N, 512)
    image2class : dict
        Maps image indices to class labels.
    distance_mode : str, optional
        Metric for comparison ('l2' for Euclidean, 'cosine' for cosine similarity).
    l2_threshold : float, optional
        Max L2 distance for a valid match.
    cosine_threshold : float, optional
        Max cosine distance (1 - similarity) for a valid match.

    Returns
    -------
    best_class : int
        Class label of the closest match, or -1 if no match meets the threshold.
    '''

    pred_embed = pred_embed.reshape(1, -1)  # (1, 512)

    if distance_mode == "l2":
        # Euclidean distance
        diffs = embeddings - pred_embed  # (N, 512)
        distances = np.linalg.norm(diffs, axis=1)  # (N,)
    else:
        # Cosine similarity -> distance = 1 - similarity
        pred_norm = np.linalg.norm(pred_embed, axis=1, keepdims=True)  # (1,1)
        emb_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)  # (N,1)

        similarities = (embeddings @ pred_embed.T).squeeze() / (emb_norms.squeeze() * pred_norm.squeeze() + 1e-10)
        distances = 1 - similarities  # nhỏ hơn = giống hơn

    # map embeddings -> class
    image2class_np = np.array([image2class[i] for i in range(len(embeddings))])
    num_classes = max(image2class.values()) + 1

    # tính tổng khoảng cách cho từng class
    total_distances = np.zeros(num_classes, dtype=np.float32)
    np.add.at(total_distances, image2class_np, distances)

    # đếm số ảnh trong mỗi class
    counts = np.zeros(num_classes, dtype=np.int32)
    np.add.at(counts, image2class_np, 1)

    # tính khoảng cách trung bình theo class
    avg_distances = np.divide(
        total_distances, counts,
        out=np.full_like(total_distances, np.inf),
        where=counts > 0
    )

    best_class = np.argmin(avg_distances)
    print("Best class avg distance:", avg_distances[best_class])

    if distance_mode == "l2":
        if avg_distances[best_class] < l2_threshold:
            return best_class
    else:  # cosine
        if avg_distances[best_class] < cosine_threshold:
            return best_class

    return -1
