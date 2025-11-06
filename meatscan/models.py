# meatscan/models.py
import torch
from torch import nn
from torchvision import models

def create_model(name: str, num_classes: int, pretrained: bool=True, dropout: float=0.0):
    name = name.lower()
    if name == "mobilenetv2":
        m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        in_ch = m.classifier[1].in_features
        m.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_ch, num_classes)
        )
        return m
    # 확장 여지: efficientnet_b0 등
    raise ValueError(f"Unknown model: {name}")
