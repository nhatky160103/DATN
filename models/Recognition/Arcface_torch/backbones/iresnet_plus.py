import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
import math

__all__ = ['iresnet18_plus', 'iresnet34_plus', 'iresnet50_plus', 'iresnet100_plus', 'iresnet200_plus']
using_ckpt = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class IBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        
        # Add dilated convolution support
        self.dilation = dilation
        self.stride = stride
        
        # First conv layer with dilation
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05)
        self.conv1 = conv3x3(inplanes, planes, stride=stride, dilation=dilation)
        
        # Second conv layer with dilation
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, dilation=dilation)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05)
        
        # Downsample layer
        self.downsample = downsample
        
        # Initialize weights using Kaiming initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward_impl(self, x):
        identity = x
        
        # First conv block
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn3(out)
        
        # Add skip connection
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Add residual connection
        out += identity
        return out

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)


class IResNet(nn.Module):
    def __init__(self,
                 block, layers, dropout=0, num_features=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False, image_size=112):
        super(IResNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
        self.extra_gflops = 0.0
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        
        # Calculate feature map size after all stride operations
        self.fc_scale = (image_size // 16)**2  # 4 stride=2 operations: 2^4 = 16
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        
        self.groups = groups
        self.base_width = width_per_group
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        self.prelu = nn.PReLU(self.inplanes)
        
        # Create layers with skip connections
        self.layer1 = self._make_layer(block, 96, layers[0], stride=2)
        self.layer2 = self._make_layer(block, 160, layers[1], stride=2,
                                     dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 320, layers[2], stride=2,
                                     dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                     dilate=replace_stride_with_dilation[2])
        
        # Final layers
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        
        # Calculate input size for fc layer
        fc_input_size = 512 * block.expansion * self.fc_scale
        self.fc = nn.Linear(fc_input_size, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-05)
        
        # Initialize weights
        self._initialize_weights()
        
        # Set feature weights to 1 and freeze
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                          self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                              base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        
        # Forward through layers with skip connections
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



def _iresnet_plus(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model


def iresnet18_plus(pretrained=False, progress=True, **kwargs):
    return _iresnet_plus('iresnet18_plus', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34_plus(pretrained=False, progress=True, **kwargs):
    return _iresnet_plus('iresnet34_plus', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50_plus(pretrained=False, progress=True, **kwargs):
    return _iresnet_plus('iresnet50_plus', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100_plus(pretrained=False, progress=True, **kwargs):
    return _iresnet_plus('iresnet100_plus', IBasicBlock, [4, 8, 16, 3], pretrained,
                    progress, **kwargs)


def iresnet200_plus(pretrained=False, progress=True, **kwargs):
    return _iresnet_plus('iresnet200_plus', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)



if __name__ == "__main__":
    import cv2
    import time
    model = iresnet100_plus(image_size=96)  # Specify image size explicitly
    model.eval()

    img = cv2.imread("data/img1.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (96, 96))
    img = img.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
    img = torch.from_numpy(img).float().unsqueeze(0)  # [1,C,H,W]

    # Test forward pass
    with torch.no_grad():
        start = time.time()
        for _ in range(10):
            y = model(img)
        end = time.time()
    print(y.shape)
    infer_time = (end - start) / 10

    # Results
    print(f'Average inference time: {infer_time:.6f} seconds')
    print('Output shape:', y.shape)  # [1, 512]
    print("Vector norm:", torch.norm(y, p=2, dim=1).item())
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")

