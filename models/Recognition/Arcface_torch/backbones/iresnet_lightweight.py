import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet_lightweight']
using_ckpt = False


# Depthwise Separable Convolution
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride,
                                   padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


# Basic Block using Depthwise Separable Convolution
class IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = DepthwiseSeparableConv(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = DepthwiseSeparableConv(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        self.downsample = downsample
        self.stride = stride

    def forward_impl(self, x):
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


# ResNet Model
class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, block, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16
        self.inplanes = 32
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)

        self.layer1 = self._make_layer(block, 32, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 192, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn2 = nn.BatchNorm2d(192 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(192 * block.expansion * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x.float() if self.fp16 else x)
        x = self.features(x)
        return x


def _iresnet_lite(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("No pretrained weights available.")
    return model


def iresnet_light(pretrained=False, progress=True, **kwargs):
    return _iresnet_lite('iresnet_light', IBasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def count_parameters(model):
    """Tính số lượng parameters của model"""
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    """Tính số lượng trainable parameters của model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model):
    """Tính dung lượng model tính bằng MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb


def count_flops(model, input_shape=(1, 3, 112, 112)):
    """Tính số FLOPs của model"""
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(flops_count)
            m_key = f"{class_name}_{module_idx}"
            flops_count[m_key] = 0
            
            if isinstance(module, nn.Conv2d):
                output_dims = list(output.size())
                batch_size = output_dims[0]
                output_height = output_dims[2]
                output_width = output_dims[3]
                
                kernel_dims = list(module.kernel_size)
                in_channels = module.in_channels
                out_channels = module.out_channels
                groups = module.groups
                
                # Tính FLOPs cho Conv2d
                kernel_ops = kernel_dims[0] * kernel_dims[1] * in_channels // groups
                bias_ops = 1 if module.bias is not None else 0
                ops_per_element = kernel_ops + bias_ops
                output_elements = output_height * output_width
                flops_count[m_key] = batch_size * out_channels * output_elements * ops_per_element
                
            elif isinstance(module, nn.Linear):
                output_dims = list(output.size())
                batch_size = output_dims[0]
                output_elements = output_dims[1]
                input_elements = module.in_features
                bias_ops = 1 if module.bias is not None else 0
                flops_count[m_key] = batch_size * output_elements * (input_elements + bias_ops)
                
            elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                output_dims = list(output.size())
                batch_size = output_dims[0]
                output_elements = output_dims[1] if len(output_dims) > 1 else output_dims[0]
                flops_count[m_key] = batch_size * output_elements * 2  # 2 operations per element
                
            elif isinstance(module, nn.PReLU):
                output_dims = list(output.size())
                batch_size = output_dims[0]
                output_elements = output_dims[1] if len(output_dims) > 1 else output_dims[0]
                flops_count[m_key] = batch_size * output_elements  # 1 operation per element
                
            elif isinstance(module, nn.Dropout):
                # Dropout không có FLOPs đáng kể
                flops_count[m_key] = 0
                
        hooks.append(module.register_forward_hook(hook))
    
    flops_count = {}
    hooks = []
    
    # Đăng ký hook cho tất cả modules
    model.apply(register_hook)
    
    # Forward pass với dummy input
    dummy_input = torch.randn(input_shape)
    model(dummy_input)
    
    # Xóa hooks
    for hook in hooks:
        hook.remove()
    
    # Tính tổng FLOPs
    total_flops = sum(flops_count.values())
    return total_flops


# Test script
if __name__ == "__main__":
    import cv2
    import time

    model = iresnet_light()
    model.eval()

    # Chuẩn bị ảnh
    img = cv2.imread("data/img1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (112, 112))
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).float().unsqueeze(0)  # [1,C,H,W]

    # Tính các metrics
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    model_size_mb = get_model_size_mb(model)
    total_flops = count_flops(model)
    gflops = total_flops / 1e9

    # Inference test
    with torch.no_grad():
        start = time.time()
        for _ in range(50):
            y = model(img)
        end = time.time()

    infer_time = (end - start) / 50

    print("=" * 50)
    print("MODEL METRICS")
    print("=" * 50)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Model size: {model_size_mb:.2f} MB')
    print(f'Total FLOPs: {total_flops:,}')
    print(f'GFLOPs: {gflops:.2f}')
    print("=" * 50)
    print("INFERENCE TEST")
    print("=" * 50)
    print(f'Average inference time: {infer_time:.6f} seconds')
    print(f'FPS: {1/infer_time:.2f}')
    print('Output shape:', y.shape)
    print("Vector norm:", torch.norm(y, p=2, dim=1).item())
    print("=" * 50)
