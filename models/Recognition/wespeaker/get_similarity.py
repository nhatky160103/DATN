

import time
start = time.time()
from .speaker import load_model_local
model = load_model_local()
end = time.time()
def voice_similarrity(path1, path2):
    similarity = model.compute_similarity(path1, path2)
    return similarity

if __name__ == "__main__":
    path1 = "models/Recognition/wespeaker/data/id10003/_JpHD6VnJ3I/00001.wav"
    path2 = "models/Recognition/wespeaker/data/id10003/_JpHD6VnJ3I/00002.wav"   
    embedding = model.extract_embedding(path1)
    print(embedding.shape)
    print('init model time:', end- start)