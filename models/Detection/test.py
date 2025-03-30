import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from .mtcnn import MTCNN
import torch

# set device
device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define mtcnn model
mtcnn_model = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    select_largest= True,
    selection_method= 'largest',
    device=device,
    keep_all=True,
)



if __name__ == '__main__':
    
    from PIL import Image
    image_path = 'models/Detection/assets/img7.jpg'
    image = Image.open(image_path).convert('RGB')
    crop_image = mtcnn_model(image)
    boxes, probs = mtcnn_model.detect(image)
    print(boxes)
    print(crop_image.shape)


    crop_images = crop_image.permute(0, 2, 3, 1).cpu().numpy()  # Đưa kênh màu về đúng thứ tự

    num_faces = crop_images.shape[0]

    fig, axes = plt.subplots(1, num_faces, figsize=(num_faces * 3, 3))

    if num_faces == 1:
        axes = [axes]

    for i, img in enumerate(crop_images):
        axes[i].imshow(img)
        axes[i].set_title(f"Khuôn mặt {i+1}")
        axes[i].axis("off")  # Ẩn trục tọa độ

    plt.show()




