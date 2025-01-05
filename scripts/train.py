import os
import json
import torch
import numpy as np
import wandb  # W&B
import itertools  # For itertools.cycle
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.dataset.dataset import CarDataset
from src.model.mobilenet import build_mobilenet_v3
from src.model.efficientnet import build_efficientnet_b4
from src.model.convnext import build_convnext

ACCUM_STEPS = 2

def summarize_layer_status(model):
    print("Layer-wise Status:")
    for name, param in model.named_parameters():
        status = "Trainable" if param.requires_grad else "Frozen"
        print(f"{name}: {status}")

def unfreeze_layers(model, num_layers_to_unfreeze):
    """
    Unfreezes the top `num_layers_to_unfreeze` layers in the model's feature extractor.
    """
    total_layers = len(list(model.features.children()))
    unfreeze_start = total_layers - num_layers_to_unfreeze

    for i, child in enumerate(model.features.children()):
        if i >= unfreeze_start:
            for param in child.parameters():
                param.requires_grad = True

def smooth_labels(labels, smoothing=0.1):
    """
    Apply label smoothing to one-hot encoded labels.
    
    Args:
        labels (Tensor): One-hot encoded target labels, shape (batch_size, num_classes).
        smoothing (float): Smoothing factor.
    
    Returns:
        Tensor: Smoothed labels.
    """
    num_classes = labels.size(1)
    smooth = smoothing / num_classes
    labels = labels * (1 - smoothing) + smooth
    return labels

def save_checkpoint(model, optimizer, epoch, iteration, metrics, checkpoint_dir="checkpoints"):
    """
    Saves a checkpoint file containing model/optimizer states.
    Dynamically creates a JSON config that extracts details from `model` and `optimizer`.
    If a detail can't be extracted, it is omitted.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Prepare file paths
    ckpt_filename = f"checkpoint_iter_{iteration+1}.pth"
    json_filename = f"config_iter_{iteration+1}.json"
    checkpoint_path = os.path.join(checkpoint_dir, ckpt_filename)
    config_path = os.path.join(checkpoint_dir, json_filename)
    
    # Dynamically build config
    config = {
        "epoch": epoch,
        "iteration": iteration + 1,  # human-friendly iteration
        "metrics": metrics 
    }
    
    # Try to extract model name
    try:
        config["model_name"] = model.__class__.__name__
    except AttributeError:
        pass  # If not found, just skip
    
    # Try to extract number of layers (assuming your model has a `model.features`)
    try:
        config["num_layers"] = len(list(model.features.children()))
    except AttributeError:
        pass
    
    # Try to extract learning rate & weight decay from the optimizer
    # (Assuming the first param_group is representative)
    try:
        config["learning_rate"] = optimizer.param_groups[0]["lr"]
        config["weight_decay"] = optimizer.param_groups[0]["weight_decay"]
    except (IndexError, KeyError):
        pass
    
    # Save checkpoint: model state, optimizer state, etc.
    torch.save({
        'epoch': epoch,
        'iteration': iteration,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)
    
    # Save config details to a JSON file
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print(f"[Checkpoint Saved] {checkpoint_path}")
    print(f"[Config Saved]     {config_path}")

def train(model, train_loader, val_loader, criterion, optimizer, lr_scheduler, args, device):
    """Training function with argparse support."""
    # W&B: Watch the model to log gradients and parameters
    wandb.watch(model, criterion, log="all", log_freq=args.log_interval)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    iteration = 0
    epoch = 0
    num_layers_unfrozen = 0
    unfreeze_interval = args.unfreeze_interval
    total_layers = len(list(model.features.children()))
    validate(model, val_loader, criterion, wandb, iteration, device)

    for iteration in tqdm(range(args.num_iterations), desc="Training Progress"):
        # Reinitialize the iterator at the start of each epoch
        if iteration % len(train_loader) == 0:
            epoch += 1
            train_loader_iter = iter(train_loader)  # Create a new iterator for the DataLoader

        # Progressive unfreezing based on iteration count
        if (iteration + 1) % unfreeze_interval == 0 and num_layers_unfrozen < total_layers:
            num_layers_unfrozen += 1
            unfreeze_layers(model, num_layers_unfrozen)

            # Update optimizer to include newly unfrozen layers
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=args.learning_rate, weight_decay=1e-4
            )
            print(f"Unfroze {num_layers_unfrozen}/{total_layers} layers at iteration {iteration}.")

        # Fetch the next batch
        batch = next(train_loader_iter)

        # Extract "image" and "final_label"
        inputs = batch["image"].to(device)
        labels = batch["final_label"].to(device)

        # Forward pass
        outputs = model(inputs)
        smoothed_labels = smooth_labels(labels, smoothing=0.1)
        loss = criterion(outputs, smoothed_labels)

        loss.backward()
        # Only step optimizer after grad_accumulation iterations
        if (iteration + 1) % args.grad_accumulation == 0:
            optimizer.step()
            optimizer.zero_grad()

        # Backward pass
        lr_scheduler.step()
        current_lr = lr_scheduler.get_last_lr()[0]

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(dim=1)).sum().item()

        # Logging
        if (iteration + 1) % args.log_interval == 0:
            avg_loss = total_loss / args.log_interval
            accuracy = correct / total

            # W&B: Log metrics
            wandb.log({
                "train/loss": avg_loss,
                "train/accuracy": accuracy,
                "epoch": epoch,
                "lr": current_lr,
                "layers_unfrozen": num_layers_unfrozen
            }, step=iteration + 1)

            # Reset for next logging interval
            total_loss = 0.0
            correct = 0
            total = 0

        # Validation
        if (iteration + 1) % args.val_interval == 0:
            validate(model, val_loader, criterion, wandb, iteration, device)
            print(f"Iteration {iteration + 1}/{args.num_iterations}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

        # Checkpoint saving
        if (iteration + 1) % args.checkpoint_interval == 0:
            avg_loss, accuracy, f1 = validate(model, val_loader, criterion, wandb, iteration, device)
            metrics = dict(
                loss=avg_loss,
                accuracy=accuracy,
                f1_score=f1
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                iteration=iteration,  # `save_checkpoint` adds +1 in config
                metrics=metrics,
                checkpoint_dir="checkpoints",
            )

    print("Training complete.")



@torch.inference_mode()  # or @torch.no_grad()
def validate(model, val_loader, criterion, wandb, iteration, device):
    """
    Validation function to evaluate the model on the validation set.
    Logs F1 scores for each class as well as overall metrics.
    """
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    class_names = ["Front", "Rear", "Front-Right", "Front-Left", "Rear-Right", "Rear-Left", "None"]


    # Iterate through validation data
    for batch in val_loader:
        # Move inputs and labels to the appropriate device
        inputs = batch["image"].to(device)
        labels = batch["final_label"].to(device)  # Assumes labels are one-hot encoded

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        val_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels.argmax(dim=1)).sum().item()
        all_preds.extend(predicted.cpu().numpy())  # Add predictions
        all_labels.extend(labels.argmax(dim=1).cpu().numpy())  # Add true labels

    # Compute metrics
    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")  # Weighted F1 score for the dataset

    # Class-wise F1 scores
    class_f1_scores = f1_score(all_labels, all_preds, average=None)  # F1 for each class

    # Logging to WandB
    wandb.log({
        "validation/loss": avg_loss,
        "validation/accuracy": accuracy,
        "validation/f1-score": f1,
        **{f"F1/{class_name}": class_f1_scores[i] for i, class_name in enumerate(class_names)}
    }, step=iteration + 1)

    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, Weighted F1: {f1:.4f}")
    print("Class-wise F1 Scores:")
    for class_name, f1_score_val in zip(class_names, class_f1_scores):
        print(f"{class_name}: {f1_score_val:.4f}")

    model.train()  # Switch back to training mode
    return avg_loss, accuracy, f1

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a car classification model.")

    # Add arguments
    parser.add_argument("--base_data_dir", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data JSON file.")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to the validation data JSON file.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the optimizer.")
    parser.add_argument("--num_iterations", type=int, default=10000, help="Number of training iterations.")
    parser.add_argument("--log_interval", type=int, default=5, help="Interval for logging training metrics.")
    parser.add_argument("--val_interval", type=int, default=50, help="Interval for validation during training.")
    parser.add_argument("--checkpoint_interval", type=int, default=1000, help="Interval for saving model checkpoints.")
    parser.add_argument("--grad_accumulation", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--resize_to", type=int, nargs=2, default=(380, 380), help="Resize dimensions for input images.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for training (cuda or cpu).")
    parser.add_argument("--unfreeze_interval", type=int, default=400, help="Interval for progressively unfreezing model layers.")

    # Parse arguments
    args = parser.parse_args()

    # W&B initialization
    wandb.init(project="my_car_classification_project", config=vars(args))

    # Load train and validation data
    def load_json_lines(file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    train_data = load_json_lines(args.train_data_path)
    val_data = load_json_lines(args.val_data_path)

    # Create datasets
    train_dataset = CarDataset(data=train_data, base_dir=args.base_data_dir, 
                               resize_to=args.resize_to, augment=True)
    val_dataset = CarDataset(data=val_data, base_dir=args.base_data_dir, 
                             resize_to=args.resize_to, augment=False)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Model and optimizer setup
    model = build_convnext(num_classes=7)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    lr_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.num_iterations // 2, eta_min=1e-6
    )

    # Start training
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        args=args,
        device=device
    )
