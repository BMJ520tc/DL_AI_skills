"""M3 验收样例模型包。"""
from .resnet import BasicBlock, ResNet, resnet18
from .skip_cnn import SkipBlock, SkipCNN

__all__ = ["BasicBlock", "ResNet", "resnet18", "SkipBlock", "SkipCNN"]


def build_model(name: str, num_classes: int = 10):
    if name == "resnet18":
        return resnet18(num_classes=num_classes)
    if name == "skipcnn":
        return SkipCNN(num_classes=num_classes)
    raise ValueError(f"unknown model: {name}")
