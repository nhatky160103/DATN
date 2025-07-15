import torch
import torch.nn as nn
from iresnet_lightweight import iresnet_lightweight

def analyze_model_parameters(model):
    """Phân tích chi tiết số tham số của từng thành phần trong model"""
    
    print("🔍 PHÂN TÍCH THAM SỐ MODEL")
    print("=" * 60)
    
    total_params = 0
    component_params = {}
    
    # Phân tích từng module
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Chỉ lấy leaf modules
            params = sum(p.numel() for p in module.parameters())
            if params > 0:
                component_params[name] = params
                total_params += params
    
    # Sắp xếp theo số tham số giảm dần
    sorted_components = sorted(component_params.items(), key=lambda x: x[1], reverse=True)
    
    print(f"📊 TỔNG SỐ THAM SỐ: {total_params:,}")
    print()
    
    print("🏆 TOP 10 THÀNH PHẦN CHIẾM NHIỀU THAM SỐ NHẤT:")
    print("-" * 60)
    
    for i, (name, params) in enumerate(sorted_components[:10], 1):
        percentage = (params / total_params) * 100
        print(f"{i:2d}. {name:<40} {params:>10,} ({percentage:>5.1f}%)")
    
    print()
    
    # Phân tích theo loại layer
    conv_params = 0
    bn_params = 0
    fc_params = 0
    other_params = 0
    
    for name, params in component_params.items():
        if 'conv' in name.lower():
            conv_params += params
        elif 'bn' in name.lower() or 'batch' in name.lower():
            bn_params += params
        elif 'fc' in name.lower() or 'linear' in name.lower():
            fc_params += params
        else:
            other_params += params
    
    print("📈 PHÂN TÍCH THEO LOẠI LAYER:")
    print("-" * 40)
    print(f"Convolutional layers: {conv_params:>10,} ({(conv_params/total_params)*100:>5.1f}%)")
    print(f"BatchNorm layers:     {bn_params:>10,} ({(bn_params/total_params)*100:>5.1f}%)")
    print(f"Fully Connected:      {fc_params:>10,} ({(fc_params/total_params)*100:>5.1f}%)")
    print(f"Others:               {other_params:>10,} ({(other_params/total_params)*100:>5.1f}%)")
    print()
    
    # Phân tích theo layer depth
    layer_params = {}
    for name, params in component_params.items():
        if 'layer' in name:
            layer_num = name.split('.')[0]  # layer1, layer2, etc.
            if layer_num not in layer_params:
                layer_params[layer_num] = 0
            layer_params[layer_num] += params
    
    print("🏗️  PHÂN TÍCH THEO LAYER DEPTH:")
    print("-" * 40)
    for layer_name in sorted(layer_params.keys()):
        params = layer_params[layer_name]
        percentage = (params / total_params) * 100
        print(f"{layer_name:<10} {params:>10,} ({percentage:>5.1f}%)")
    print()
    
    # Tìm thành phần chiếm nhiều nhất
    max_component = max(component_params.items(), key=lambda x: x[1])
    print(f"🎯 THÀNH PHẦN CHIẾM NHIỀU NHẤT:")
    print(f"   {max_component[0]}")
    print(f"   {max_component[1]:,} parameters ({(max_component[1]/total_params)*100:.1f}%)")
    print()
    
    # Gợi ý tối ưu
    print("💡 GỢI Ý TỐI ƯU:")
    if fc_params > conv_params * 0.5:
        print("   ⚠️  FC layer chiếm quá nhiều tham số!")
        print("   💡 Giảm output features từ 512 xuống 256 hoặc 128")
    
    if max_component[1] > total_params * 0.3:
        print(f"   ⚠️  {max_component[0]} chiếm {(max_component[1]/total_params)*100:.1f}% tham số!")
        print("   💡 Cân nhắc giảm kích thước layer này")
    
    # Tính toán model size
    model_size_mb = total_params * 4 / 1024 / 1024  # 4 bytes per float32
    print(f"📦 MODEL SIZE: {model_size_mb:.2f} MB (float32)")
    print(f"🎯 TARGET: ≤ 5M params = ≤ 19.07 MB")
    print(f"✅ Status: {'Đạt' if total_params <= 5000000 else 'Chưa đạt'} mục tiêu")

def detailed_fc_analysis(model):
    """Phân tích chi tiết FC layer"""
    print("\n🔍 PHÂN TÍCH CHI TIẾT FC LAYER:")
    print("-" * 40)
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            input_features = module.in_features
            output_features = module.out_features
            params = sum(p.numel() for p in module.parameters())
            
            print(f"Layer: {name}")
            print(f"  Input features:  {input_features:,}")
            print(f"  Output features: {output_features:,}")
            print(f"  Parameters:      {params:,}")
            print(f"  Size:            {params * 4 / 1024 / 1024:.2f} MB")
            print()

def main():
    # Tạo model
    model = iresnet_lightweight()
    model.eval()
    
    # Phân tích
    analyze_model_parameters(model)
    detailed_fc_analysis(model)
    
    # Thống kê tổng quan
    print("📊 THỐNG KÊ TỔNG QUAN:")
    print("-" * 40)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable:       {total_params - trainable_params:,}")
    print(f"Model size (float32): {total_params * 4 / 1024 / 1024:.2f} MB")
    print(f"Model size (float16): {total_params * 2 / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    main() 