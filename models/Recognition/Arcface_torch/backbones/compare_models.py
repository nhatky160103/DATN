import torch
import time
from iresnet_lightweight import iresnet_lightweight
from iresnet_ultra_lightweight import iresnet_ultra_lightweight

def count_parameters(model):
    """Đếm số tham số của model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def benchmark_model(model, device, num_runs=50):
    """Benchmark model performance"""
    model.eval()
    
    # Prepare dummy input
    dummy_input = torch.randn(1, 3, 112, 112).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    with torch.no_grad():
        start_time = time.time()
        for _ in range(num_runs):
            output = model(dummy_input)
        end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs
    return avg_time, output.shape

def main():
    print("🔍 Model Comparison: Lightweight vs Ultra Lightweight")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print()
    
    # Test Lightweight model
    print("📊 Lightweight Model (Original):")
    model_lightweight = iresnet_lightweight()
    model_lightweight = model_lightweight.to(device)
    
    total_params, trainable_params = count_parameters(model_lightweight)
    avg_time, output_shape = benchmark_model(model_lightweight, device)
    
    print(f"  • Total parameters: {total_params:,}")
    print(f"  • Trainable parameters: {trainable_params:,}")
    print(f"  • Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"  • Average inference time: {avg_time:.6f} seconds")
    print(f"  • Output shape: {output_shape}")
    print(f"  • Target achieved: {'✅' if total_params <= 5000000 else '❌'} (≤ 5M params)")
    print()
    
    # Test Ultra Lightweight model
    print("📊 Ultra Lightweight Model (Optimized):")
    model_ultra = iresnet_ultra_lightweight()
    model_ultra = model_ultra.to(device)
    
    total_params_ultra, trainable_params_ultra = count_parameters(model_ultra)
    avg_time_ultra, output_shape_ultra = benchmark_model(model_ultra, device)
    
    print(f"  • Total parameters: {total_params_ultra:,}")
    print(f"  • Trainable parameters: {trainable_params_ultra:,}")
    print(f"  • Model size: {total_params_ultra * 4 / 1024 / 1024:.2f} MB")
    print(f"  • Average inference time: {avg_time_ultra:.6f} seconds")
    print(f"  • Output shape: {output_shape_ultra}")
    print(f"  • Target achieved: {'✅' if total_params_ultra <= 5000000 else '❌'} (≤ 5M params)")
    print()
    
    # Comparison
    print("📈 Comparison:")
    param_reduction = ((total_params - total_params_ultra) / total_params) * 100
    speed_improvement = ((avg_time - avg_time_ultra) / avg_time) * 100 if avg_time > avg_time_ultra else 0
    
    print(f"  • Parameter reduction: {param_reduction:.1f}%")
    print(f"  • Speed improvement: {speed_improvement:.1f}%")
    print(f"  • Memory reduction: {((total_params - total_params_ultra) * 4 / 1024 / 1024):.2f} MB")
    print()
    
    # Recommendations
    print("💡 Recommendations:")
    if total_params_ultra <= 5000000:
        print("  ✅ Ultra Lightweight model meets the 5M parameter target!")
        print("  ✅ Recommended for deployment on resource-constrained devices")
    else:
        print("  ⚠️  Ultra Lightweight model still exceeds 5M parameters")
        print("  💡 Consider further optimizations:")
        print("     - Reduce output features from 512 to 256")
        print("     - Use depthwise separable convolutions")
        print("     - Implement model pruning")
    
    print()
    print("🎯 Summary:")
    print(f"  • Lightweight: {total_params:,} params ({total_params * 4 / 1024 / 1024:.2f} MB)")
    print(f"  • Ultra Lightweight: {total_params_ultra:,} params ({total_params_ultra * 4 / 1024 / 1024:.2f} MB)")
    print(f"  • Target: ≤ 5,000,000 params (≤ 19.07 MB)")

if __name__ == "__main__":
    main() 