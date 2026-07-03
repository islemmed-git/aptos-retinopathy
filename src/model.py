"""
Construction du modele CNN par transfer learning.

On part d'un reseau pre-entraine sur ImageNet et on remplace sa
derniere couche par une couche a 5 sorties (les 5 stades).
"""
import torch.nn as nn
from torchvision import models

from . import config


def build_model(name=None, num_classes=None, pretrained=True):
    name = name or config.MODEL_NAME
    num_classes = num_classes or config.NUM_CLASSES

    if name == "resnet18":
        net = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet34":
        net = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "resnet50":
        net = models.resnet50(weights="IMAGENET1K_V2" if pretrained else None)
        net.fc = nn.Linear(net.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        net = models.efficientnet_b0(
            weights="IMAGENET1K_V1" if pretrained else None)
        net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Modele inconnu : {name}")

    return net
