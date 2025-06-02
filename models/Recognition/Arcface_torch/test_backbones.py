import torch
import time
import numpy as np
from backbones.iresnet_plus import (
    iresnet18_plus, 
    iresnet34_plus, 
    iresnet50_plus, 
    iresnet100_plus, 
    iresnet200_plus
)
import cv2
from tqdm import tqdm

def count_parameters(model):
    """Đếm số lượng tham số của model"""
    return sum(p.numel() for p in model.parameters())

def measure_inference_time(model, input_tensor, num_runs=100):
    """Đo thời gian inference"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(input_tensor)
            end = time.time()
            times.append(end - start)
    
    return np.mean(times), np.std(times)

def test_backbone(backbone_name, model, input_tensor, num_runs=100):
    """Test một backbone cụ thể"""
    print(f"\nTesting {backbone_name}:")
    
    # Đếm số tham số
    num_params = count_parameters(model)
    print(f"Number of parameters: {num_params:,}")
    
    # Đo thời gian inference
    mean_time, std_time = measure_inference_time(model, input_tensor, num_runs)
    print(f"Inference time: {mean_time*1000:.2f}ms ± {std_time*1000:.2f}ms")
    
    # Đo memory usage
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(input_tensor)
    max_memory = torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB
    print(f"Peak memory usage: {max_memory:.2f}MB")
    
    return {
        'name': backbone_name,
        'parameters': num_params,
        'inference_time': mean_time,
        'memory_usage': max_memory
    }

def main():
    # Chuẩn bị input
    img = cv2.imread("data/img1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
    input_tensor = torch.from_numpy(img).float().unsqueeze(0)  # [1,C,H,W]
    
    # Danh sách các backbone cần test
    backbones = {
        'iresnet18_plus': iresnet18_plus(),
        'iresnet34_plus': iresnet34_plus(),
        'iresnet50_plus': iresnet50_plus(),
        'iresnet100_plus': iresnet100_plus(),
        'iresnet200_plus': iresnet200_plus()
    }
    
    # Lưu kết quả
    results = []
    
    # Test từng backbone
    for name, model in backbones.items():
        result = test_backbone(name, model, input_tensor)
        results.append(result)
    
    # In bảng so sánh
    print("\nComparison Table:")
    print("-" * 80)
    print(f"{'Backbone':<15} {'Parameters':<15} {'Inference Time (ms)':<20} {'Memory (MB)':<15}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<15} {result['parameters']:<15,} {result['inference_time']*1000:<20.2f} {result['memory_usage']:<15.2f}")

if __name__ == "__main__":
    main() 