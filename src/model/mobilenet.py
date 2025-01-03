import torch
import torch.nn as nn
from torchvision import models


def build_mobilenet_v3(num_classes=7):
    # Load the pretrained MobileNetV3 model
    model = models.mobilenet_v3_large(pretrained=True)
    in_features = model.classifier[0].in_features
    # Modify the classifier to fit your number of classes
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(1024, num_classes)
    )
    
    return model
