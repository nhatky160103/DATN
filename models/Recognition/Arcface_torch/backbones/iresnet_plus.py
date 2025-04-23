import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

__all__ = ['iresnet18_plus', 'iresnet34_plus', 'iresnet50_plus', 'iresnet100_plus']

# DropPath implementation (Stochastic Depth)
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor

# Improved Residual Block (ConvNeXt-inspired)
class ImprovedBlock(nn.Module):
    expansion = 1
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # normalization before MLP
        self.norm = nn.GroupNorm(1, dim)
        self.pw_conv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pw_conv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        # x: [B, C, H, W]
        shortcut = x
        x = self.dw_conv(x)
        # apply normalization in channel-first
        x = self.norm(x)
        # prepare for point-wise MLP: BCHW -> BHWC
        x = x.permute(0, 2, 3, 1)
        x = self.pw_conv1(x)
        x = self.act(x)
        x = self.pw_conv2(x)
        # apply layer scale and back to BCHW
        x = (self.gamma * x).permute(0, 3, 1, 2)
        x = self.drop_path(x)
        return shortcut + x

# Improved IResNet Backbone
class ImprovedIResNet(nn.Module):
    def __init__(self, layers, num_features=512, drop_path_rate=0.1):
        super().__init__()
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        self.inplanes = 64
        self.total_blocks = sum(layers)
        self.cur_block = 0

        # Build stages
        self.layer1 = self._make_stage(64,  layers[0], stride=1, drop_path_rate=drop_path_rate)
        self.layer2 = self._make_stage(128, layers[1], stride=2, drop_path_rate=drop_path_rate)
        self.layer3 = self._make_stage(256, layers[2], stride=2, drop_path_rate=drop_path_rate)
        self.layer4 = self._make_stage(512, layers[3], stride=2, drop_path_rate=drop_path_rate)

        # Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * ImprovedBlock.expansion, num_features)
        self.bn = nn.BatchNorm1d(num_features)
        nn.init.constant_(self.bn.weight, 1.0)
        self.bn.weight.requires_grad = False

    def _make_stage(self, planes, num_blocks, stride, drop_path_rate):
        downsample = None
        if stride != 1 or self.inplanes != planes * ImprovedBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * ImprovedBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * ImprovedBlock.expansion)
            )
        blocks = []
        for i in range(num_blocks):
            dpr = drop_path_rate * (self.cur_block / (self.total_blocks - 1))
            if i == 0:
                if downsample is not None:
                    blocks.append(downsample)
                blocks.append(ImprovedBlock(planes * ImprovedBlock.expansion, drop_path=dpr))
            else:
                blocks.append(ImprovedBlock(planes * ImprovedBlock.expansion, drop_path=dpr))
            self.cur_block += 1
            self.inplanes = planes * ImprovedBlock.expansion
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        x = self.bn(x)
        return x

# Factory functions

def iresnet18_plus(num_features=512, drop_path_rate=0.1):
    return ImprovedIResNet([2, 2, 2, 2], num_features, drop_path_rate)

def iresnet34_plus(num_features=512, drop_path_rate=0.1):
    return ImprovedIResNet([3, 4, 6, 3], num_features, drop_path_rate)

def iresnet50_plus(num_features=512, drop_path_rate=0.1):
    return ImprovedIResNet([3, 4, 14, 3], num_features, drop_path_rate)

def iresnet100_plus(num_features=512, drop_path_rate=0.1):
    return ImprovedIResNet([3, 13, 30, 3], num_features, drop_path_rate)

if __name__ == "__main__":
    model = iresnet50_plus(num_features=512, drop_path_rate=0.1)
    x = torch.randn(1, 3, 112, 112)
    model.eval()
    with torch.no_grad():
        output = model(x)
    print("Output shape:", output.shape)  # Should be [1, 512]
    norm = output.norm(p=2, dim=1)
    print("L2 norm of output:", norm.item())