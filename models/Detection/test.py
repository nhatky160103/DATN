import time
start_time = time.time()  # Đặt ngay trước tất cả các import
# import torch
# from torch.nn.functional import interpolate
# from torchvision.transforms import functional as F
# from torchvision.ops.boxes import batched_nms
# from PIL import Image
# import numpy as np
# import os
# import math

# --- Tính thời gian ---

from .face_detect import detect_face, extract_face
end_time = time.time()
elapsed = end_time - start_time
print(f"⏱️ Thời gian import & khởi tạo: {elapsed:.4f} giây")

if __name__ == "__main__":
    print('✅ Chương trình bắt đầu!')
