from __future__ import annotations
import timm
import torch.nn as nn

def create_model(name: str, num_classes: int, pretrained: bool=True, dropout: float=0.0) -> nn.Module:
    name = name.lower()
    if name == "resnet50":
        model = timm.create_model("resnet50.a1_in1k", pretrained=pretrained, num_classes=num_classes)
    elif name == "efficientnet_b3":
        model = timm.create_model("efficientnet_b3.ra2_in1k", pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model
