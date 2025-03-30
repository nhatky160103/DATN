import argparse

import cv2
import numpy as np
import torch

from models.Recognition.Arcface_torch.backbones import get_model

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight,  map_location=device))
    net.eval()
    feat = net(img).numpy()
    print(feat)



if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    # parser.add_argument('--network', type=str, default='r50', help='backbone network')
    # parser.add_argument('--weight', type=str, default='')
    # parser.add_argument('--img', type=str, default=None)
    # args = parser.parse_args()
    # inference(args.weight, args.network, args.img)

    net = get_model("r100", fp16=False)
    net.load_state_dict(torch.load("models/Recognition/Arcface_torch/weights/glint360k_cosface_r100_fp16_0.1/backbone.pth",  map_location=device))
    net.eval()
    img  = torch.rand(16, 3, 112, 112)
    img.to(device)
    feat = net(img).detach().numpy()
    print(feat.shape)


