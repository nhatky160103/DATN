import torch
import time
from ptflops import get_model_complexity_info
import numpy as np
import psutil

def analyze_model(model, input_size=(3, 112, 112), device='cpu', n_runs=100, dummy_img=None):
    model.eval().to(device)

    # Dummy input
    if dummy_img is None:
        dummy_img = torch.randn(1, *input_size).to(device)

    # Tính số tham số & FLOPs
    flops, params = get_model_complexity_info(model, input_size,
                                              as_strings=False, print_per_layer_stat=False)

    # Ước lượng bộ nhớ của model (params) - 4 bytes per param (float32)
    model_mem_MB = params * 4 / (1024 ** 2)

    # Ước lượng bộ nhớ trung gian (chạy RAM tracking)
    def get_ram_usage_MB():
        return psutil.Process().memory_info().rss / 1024 ** 2

    mem_before = get_ram_usage_MB()

    # Tính thời gian inference trung bình
    with torch.no_grad():
        torch.cuda.empty_cache()
        start = time.time()
        for _ in range(n_runs):
            _ = model(dummy_img)
        end = time.time()
    infer_time = (end - start) / n_runs

    mem_after = get_ram_usage_MB()
    extra_mem_MB = mem_after - mem_before

    print("🧠 Thống kê model:")
    print(f"- Parameters      : {params:,} ({params * 4 / 1024**2:.2f} MB)")
    print(f"- FLOPs           : {flops / 1e9:.2f} GFLOPs")
    print(f"- Bộ nhớ model    : {model_mem_MB:.2f} MB")
    print(f"- Bộ nhớ tăng thêm: {extra_mem_MB:.2f} MB (gồm tensor, buffer,...)")
    print(f"- Inference time  : {infer_time*1000:.2f} ms/image ({n_runs} lần chạy)")
    print("- Device          :", device)

if __name__ == "__main__":
    from .iresnet import  iresnet18, iresnet34, iresnet50, iresnet100
    from .iresnet_lite import iresnet100_lite

    model_r18 = iresnet18()
    model_r34 = iresnet34()
    model_r50 = iresnet50()
    model_r100 = iresnet100()
    analyze_model(model_r18, input_size=(3, 112, 112), device='cpu', n_runs=100)
    analyze_model(model_r34, input_size=(3, 112, 112), device='cpu', n_runs=100)
    analyze_model(model_r50, input_size=(3, 112, 112), device='cpu', n_runs=100)
    analyze_model(model_r100, input_size=(3, 112, 112), device='cpu', n_runs=100)
 