import torch
import time
import psutil
from PIL import Image
from torchvision import transforms

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def profile_mtcnn(mtcnn_model, image_pil, device='cpu', repeat=100):
    mtcnn_model.eval()

    # Tính tổng số tham số của cả 3 mạng con: PNet, RNet, ONet
    total_params = (
        count_parameters(mtcnn_model.pnet) +
        count_parameters(mtcnn_model.rnet) +
        count_parameters(mtcnn_model.onet)
    )

    # Đo thời gian inference trung bình
    with torch.no_grad():
        start_time = time.time()
        for _ in range(repeat):
            _ = mtcnn_model(image_pil)
        infer_time = time.time() - start_time

    # Đo RAM sử dụng
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_usage = mem_info.rss / (1024 ** 2)  # MB

    print(f"Số tham số: {total_params:,}")
    print(f"Thời gian infer trung bình: {(infer_time / repeat) * 1000:.2f} ms")
    print(f"Bộ nhớ RAM sử dụng: {memory_usage:.2f} MB")


if __name__ == "__main__":
    from .mtcnn import MTCNN  # Đảm bảo đường dẫn đúng nếu import nội bộ

    mtcnn = MTCNN(keep_all=False, device='cpu')
    image_path = 'data/img1.jpg'
    image = Image.open(image_path).convert('RGB')

    profile_mtcnn(mtcnn, image, device='cpu', repeat=50)
