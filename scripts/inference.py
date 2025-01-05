import json
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.model.convnext import build_convnext  # Replace with your model import
from src.dataset.dataset import CarDataset  # Replace with your dataset import

# Class names
CLASS_NAMES = ["Front", "Rear", "Front-Right", "Front-Left", "Rear-Right", "Rear-Left", "None"]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
model = build_convnext(num_classes=len(CLASS_NAMES))
model.load_state_dict(torch.load("best_model.pth", map_location=device))  # Replace "best_model.pth" with your saved model
model = model.to(device)
model.eval()

# Image transformation (same as training preprocessing)
transform = transforms.Compose([
    transforms.Resize((380, 380)),  # Ensure it matches your training resize
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def infer_single_image(image_path):
    """
    Perform inference on a single image.

    Args:
        image_path (str): Path to the input image.

    Returns:
        str: Predicted class name.
        float: Confidence score for the prediction.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert("RGB")  # Ensure RGB
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Forward pass
    with torch.inference_mode():
        output = model(image_tensor)
        probabilities = F.softmax(output, dim=1)  # Convert logits to probabilities
        predicted_idx = probabilities.argmax(dim=1).item()
        confidence = probabilities.max().item()

    return CLASS_NAMES[predicted_idx], confidence

def batch_infer(image_paths):
    """
    Perform inference on a batch of images.

    Args:
        image_paths (list): List of paths to input images.

    Returns:
        list of tuples: Each tuple contains the predicted class name and confidence score for an image.
    """
    predictions = []
    for image_path in image_paths:
        try:
            prediction = infer_single_image(image_path)
            predictions.append((image_path, prediction[0], prediction[1]))  # (Path, Class, Confidence)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            predictions.append((image_path, "Error", 0.0))
    return predictions

if __name__ == "__main__":
    # Example image paths
    image_paths = [
        "data/test_image1.jpg",
        "data/test_image2.jpg",
        "data/test_image3.jpg"  # Add more image paths
    ]

    # Run inference
    results = batch_infer(image_paths)

    # Print results
    for image_path, class_name, confidence in results:
        print(f"Image: {image_path} | Predicted: {class_name} | Confidence: {confidence:.4f}")
