import argparse
import autoroot
import autorootcwd
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from src.model.convnext import build_convnext  # Replace with your model import
from src.dataset.dataset import CarDataset  # Replace with your dataset import

# Class names
CLASS_NAMES = ["Front", "Rear", "Front-Right", "Front-Left", "Rear-Right", "Rear-Left", "None"]

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test dataset
def load_json_lines(file_path):
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def evaluate_model(model, test_loader, criterion):
    """
    Evaluates the model on the test set.

    Args:
        model: Trained model.
        test_loader: DataLoader for the test dataset.
        criterion: Loss function.

    Returns:
        dict: Evaluation metrics including loss, accuracy, and F1 scores.
    """
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.inference_mode():
        for batch in test_loader:
            inputs = batch["image"].to(device)
            labels = batch["final_label"].to(device)  # Assuming labels are one-hot encoded

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Accumulate loss
            test_loss += loss.item()

            # Convert outputs to predictions
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.argmax(dim=1).cpu().numpy())

    # Metrics calculation
    avg_loss = test_loss / len(test_loader)
    accuracy = accuracy_score(all_labels, all_preds)
    class_f1_scores = f1_score(all_labels, all_preds, average=None)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(all_labels, all_preds, target_names=CLASS_NAMES)

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Class-wise F1 Scores: {class_f1_scores}")
    print("")
    print("Classification Report:\n", report)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "class_f1_scores": dict(zip(CLASS_NAMES, class_f1_scores))
    }

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Evaluate a trained model on the test set.")

    # Add arguments
    parser.add_argument("--test_data_path", type=str, required=True, help="Path to the test data JSON file.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the saved model checkpoint.")
    parser.add_argument("--base_dir", type=str, default="data", help="Base directory for test images.")
    parser.add_argument("--resize_to", type=int, nargs=2, default=(380, 380), help="Resize dimensions for input images.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for the DataLoader.")

    # Parse arguments
    args = parser.parse_args()

    # Load the trained model
    model = build_convnext(num_classes=len(CLASS_NAMES))
    state_dict = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])  # Replace with your saved model path
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_data = load_json_lines(args.test_data_path)
    test_dataset = CarDataset(data=test_data, base_dir=args.base_dir, 
                              resize_to=args.resize_to, augment=False)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, criterion)

    # Print final metrics
    print("Overall Metrics:")
    print("Accuracy", metrics["accuracy"])
    print("F1 Score", metrics["weighted_f1"])
