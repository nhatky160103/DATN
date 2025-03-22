import argparse

import cv2
import numpy as np
import torch
import numpy as np
from scipy.spatial.distance import cosine


from backbones import get_model

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

def preprocess_image(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))  # Chuyển đổi HWC -> CHW
    img = torch.from_numpy(img).float().unsqueeze(0)  # Thêm batch dimension
    img.div_(255).sub_(0.5).div_(0.5)  # Chuẩn hóa về khoảng [-1, 1]
    return img


@torch.no_grad()
def inference(weight, name, img1, img2):
    if img1 is None:
        img1 = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img1 = cv2.imread(img1)
        img1 = cv2.resize(img1, (112, 112))
    if img2 is None:
        img2 = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img2 = cv2.imread(img2)
        img2 = cv2.resize(img2, (112, 112))
    
    img1 = preprocess_image(img1)
    img2 = preprocess_image(img2)

    batch = torch.cat([img1, img2], dim=0) 
    
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight,  map_location=device))
    net.eval()
    
    feat = net(batch).numpy()
    distance = cosine(feat[0], feat[1])
    print("Cosine Distance:", distance)
    return distance



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img1', type=str, default=None)
    parser.add_argument('--img2', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img1, args.img2)