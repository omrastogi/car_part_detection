import torch
import torch.nn as nn
from torchvision import models

def build_convnext(num_classes=7, model_type='convnext_tiny', pretrained=True):
    """
    Builds and returns a ConvNeXt model for multiclass classification.

    Args:
        num_classes (int): Number of output classes for classification.
        model_type (str): Type of ConvNeXt model ('convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large').
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        model (torch.nn.Module): ConvNeXt model modified for the given number of classes.
    """
    # Load the pretrained ConvNeXt model
    if model_type == 'convnext_tiny':
        model = models.convnext_tiny(pretrained=pretrained)
    elif model_type == 'convnext_small':
        model = models.convnext_small(pretrained=pretrained)
    elif model_type == 'convnext_base':
        model = models.convnext_base(pretrained=pretrained)
    elif model_type == 'convnext_large':
        model = models.convnext_large(pretrained=pretrained)
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Choose from 'convnext_tiny', 'convnext_small', 'convnext_base', 'convnext_large'.")

    # Modify the classifier for the specified number of classes
    in_features = model.classifier[2].in_features  # Get the input features of the last layer
    model.classifier[2] = nn.Linear(in_features, num_classes)  # Replace the final classification layer

    return model

# Example usage
if __name__ == "__main__":
    num_classes = 7
    model_type = 'convnext_tiny'  # Change to 'convnext_tiny' or others as needed
    model = build_convnext(num_classes=num_classes, model_type=model_type, pretrained=True)
    
    # Check the model
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Example input to test the output
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch of 1, 3-channel image, 224x224 resolution
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")  # Should be [1, num_classes]
