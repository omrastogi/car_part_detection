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
    print("Classification Report:\n", report)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "weighted_f1": weighted_f1,
        "class_f1_scores": dict(zip(CLASS_NAMES, class_f1_scores))
    }

if __name__ == "__main__":
    # Paths and parameters
    test_data_path = "data/test_data.json"
    base_dir = "data"
    resize_to = (380, 380)
    batch_size = 32
    
    # Load the trained model
    model = build_convnext(num_classes=len(CLASS_NAMES))
    state_dict = torch.load("checkpoints/checkpoint_iter_5000.pth", map_location=device)
    model.load_state_dict(state_dict["model_state_dict"])  # Replace with your saved model path
    model = model.to(device)
    model.eval()

    # Load test dataset
    test_data = load_json_lines(test_data_path)
    test_dataset = CarDataset(data=test_data, base_dir=base_dir, 
                              resize_to=resize_to, augment=False)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Define loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    # Evaluate the model
    metrics = evaluate_model(model, test_loader, criterion)

    # Print final metrics
    print("Overall Metrics:")
    print("Acccuracy", metrics["accuracy"])
    print("f1 score", metrics["weighted_f1"])
    
    # for key, value in metrics.items():
    #     if isinstance(value, dict):
    #         print(f"{key}:")
    #         for sub_key, sub_value in value.items():
    #             print(f"  {sub_key}: {sub_value:.4f}")
    #     else:
    #         print(f"{key}: {value:.4f}")
