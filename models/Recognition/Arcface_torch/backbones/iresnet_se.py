import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet50_se']

using_ckpt = False

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride,
                     padding=dilation, groups=groups,
                     bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 1, stride, bias=False)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.avg_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w

class IBasicBlockSE(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, reduction=16):
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlockSE only supports groups=1 and base_width=64')
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.se = SEBlock(planes, reduction)
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
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity

    def forward(self, x):
        if self.training and using_ckpt:
            return checkpoint(self.forward_impl, x)
        else:
            return self.forward_impl(x)

class IResNetSE(nn.Module):
    fc_scale = 7 * 7

    def __init__(self, layers, dropout=0, num_features=512,
                 zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None,
                 fp16=False):
        super().__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        self.groups = groups
        self.base_width = width_per_group

        # conv1 giống IResNet
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        # 4 layer
        self.layer1 = self._make_layer(IBasicBlockSE,  64, layers[0], stride=2)
        self.layer2 = self._make_layer(IBasicBlockSE, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(IBasicBlockSE, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(IBasicBlockSE, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])

        self.bn2     = nn.BatchNorm2d(512 * IBasicBlockSE.expansion, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc      = nn.Linear(512 * IBasicBlockSE.expansion * self.fc_scale,
                                 num_features, bias=False)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        # init BN-final
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # init conv & fc như IResNet
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlockSE):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers = [block(self.inplanes, planes, stride, downsample,
                        self.groups, self.base_width, previous_dilation)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups,
                                base_width=self.base_width,
                                dilation=self.dilation))
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

def _iresnet_se(arch, block, layers, pretrained, progress, **kwargs):
    model = IResNetSE(block, layers, **kwargs)
    if pretrained:
        raise ValueError()
    return model

def iresnet50_se(pretrained=False, progress=True, **kwargs):
    """
    IResNet‑50 with SE modules.
    Nếu pretrained=True sẽ ném ValueError (chưa có weight pretrained).
    """
    if pretrained:
        raise ValueError("Pretrained weights not available for IResNet‑SE")
    # layers = [3, 4, 14, 3] tương ứng cấu hình của ResNet‑50
    return IResNetSE([3, 4, 14, 3], **kwargs)


if __name__ == "__main__":
    import cv2
    model = iresnet50_se()
    print("Params:", sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6, "M")
    x = cv2.imread("data/img1.jpg")
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = cv2.resize(x, (112, 112))
    x = x.transpose(2, 0, 1)  # [H,W,C] -> [C,H,W]
    x = torch.from_numpy(x).float()  # [C,H,W]
    x = x.unsqueeze(0)  # [C,H,W] -> [1,C,H,W]
    model.eval()
    with torch.no_grad():
        y = model(x)
        print("Shape:", y.shape)         # [1,512]
        print("Norm:", torch.norm(y,2,1)) # ~1e8–1e9

