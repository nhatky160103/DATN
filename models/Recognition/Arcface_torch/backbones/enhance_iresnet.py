import torch
from torch import nn

class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(self.pool(x))
        return x * w

class EnhancedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = ConvBNAct(in_channels, out_channels, 3, stride, 1)
        self.conv2 = ConvBNAct(out_channels, out_channels, 3, 1, 1)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = ConvBNAct(in_channels, out_channels, 1, stride)
        self.norm = nn.BatchNorm2d(out_channels)  # ✅ sửa từ LayerNorm -> BatchNorm2d

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.se(out)
        out = out + identity
        out = self.norm(out)  # không cần reshape
        return out


class EnhancedIResNet(nn.Module):
    def __init__(self, layers=[2, 2, 2, 2], num_features=512, dropout=0.4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.stage1 = self._make_layer(64, 64, layers[0], stride=1)
        self.stage2 = self._make_layer(64, 128, layers[1], stride=2)
        self.stage3 = self._make_layer(128, 256, layers[2], stride=2)
        self.stage4 = self._make_layer(256, 512, layers[3], stride=2)
        self.bn = nn.BatchNorm2d(512)
        self.pool = nn.AdaptiveAvgPool2d((7, 7))  # Đảm bảo đầu vào FC luôn là 512*7*7
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7, num_features)
        self.feat_bn = nn.BatchNorm1d(num_features, eps=1e-05)
        nn.init.constant_(self.feat_bn.weight, 1.0)
        self.feat_bn.weight.requires_grad = False

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [EnhancedBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(EnhancedBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.bn(x)
        x = self.pool(x)  # Đảm bảo [B, 512, 7, 7]
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.feat_bn(x)
        return x

if __name__ == "__main__":
    model = EnhancedIResNet()
    x = torch.randn(1, 3, 112, 112)
    model.eval()
    with torch.no_grad():
        output = model(x)
        print("Output shape:", output.shape)
        print("L2 norm:", torch.norm(output))
