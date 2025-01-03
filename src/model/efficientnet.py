import torch
import torch.nn as nn
from torchvision import models

def build_efficientnet_b0(num_classes=7):
    # Load the pretrained EfficientNet-B0 model
    model = models.efficientnet_b0(pretrained=True)

    # Get the input features of the original classifier
    in_features = model.classifier[1].in_features  # Layer 1 is the last linear layer

    # Replace the classifier with a new one tailored for `num_classes`
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    return model

def build_efficientnet_b4(num_classes=7):
    # Load pretrained EfficientNet-B4
    model = models.efficientnet_b4(pretrained=True)
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace the classifier for your custom number of classes
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.4),  # Higher dropout for larger models
        nn.Linear(in_features, num_classes)
    )
    return model


if __name__ == "__main__":
    print(build_efficientnet_b4(num_classes=7))
