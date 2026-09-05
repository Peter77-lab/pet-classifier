"""模型构建：ResNet-18（ImageNet 预训练）+ 37 类分类头。"""

import torch.nn as nn
from torchvision.models import resnet18

from config import NUM_CLASSES


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """构建 ResNet-18，并将最后的全连接层替换为 ``num_classes`` 分类头。

    Args:
        num_classes: 分类类别数，默认 37（Oxford-IIIT Pet 品种数）。
        pretrained: 是否加载 ImageNet 预训练权重。

    Returns:
        构建好的模型。
    """
    weights = "DEFAULT" if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model
