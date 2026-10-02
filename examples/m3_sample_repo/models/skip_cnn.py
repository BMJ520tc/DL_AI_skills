"""小跳跃 CNN（M3 验收开发期快速链路，CPU 秒级）。

覆盖要素：SkipBlock 残差加（out + identity，op 节点双输入）、stem Sequential 容器、
叶子层（Conv/BN/ReLU/AdaptiveAvgPool/Linear）、torch.flatten op。
"""
import torch
from torch import nn


class SkipBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity
        return self.relu1(out)


class SkipCNN(nn.Module):
    def __init__(self, num_classes=10, base_channels=16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.block1 = SkipBlock(base_channels)
        self.block2 = SkipBlock(base_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)
