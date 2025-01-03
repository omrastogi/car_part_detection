import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import itertools  # Make sure to import itertools for itertools.cycle
# W&B
import wandb
from tqdm import tqdm

from src.dataset.dataset import CarDataset  # Replace with your dataset class
from src.model.mobilenet import build_mobilenet_v3  # Replace with your model function
from src.model.efficientnet import build_efficientnet_b4
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import CosineAnnealingLR

ACCUM_STEPS = 2

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

def train(model, train_loader, val_loader, criterion, optimizer, 
          num_iterations, log_interval, val_interval, lr_scheduler, device):
    
    # W&B: Watch the model to log gradients and parameters
    wandb.watch(model, criterion, log="all", log_freq=log_interval)
    
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    iteration = 0
    epoch = 0
    num_layers_unfrozen = 0
    unfreeze_interval = 10
    total_layers = len(list(model.features.children()))
    validate(model, val_loader, criterion, wandb, iteration, device)
    
    for iteration in tqdm(range(num_iterations), desc="Training Progress"):
        # Reinitialize the iterator at the start of each epoch
        if iteration % len(train_loader) == 0:
            epoch += 1
            train_loader_iter = iter(train_loader)  # Create a new iterator for the DataLoader

        # Progressive unfreezing based on iteration count
        if iteration % unfreeze_interval == 0 and num_layers_unfrozen < total_layers:
            num_layers_unfrozen += 1
            unfreeze_layers(model, num_layers_unfrozen)

            # Update optimizer to include newly unfrozen layers
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4, weight_decay=1e-4
            )
            print(f"Unfroze {num_layers_unfrozen}/{total_layers} layers at iteration {iteration}.")

        # Fetch the next batch
        batch = next(train_loader_iter)

        # Extract "image" and "final_label"
        inputs = batch["image"].to(device)
        labels = batch["final_label"].to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        # Only step optimizer after ACCUM_STEPS iterations
        if (iteration + 1) % ACCUM_STEPS == 0:
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
        if (iteration + 1) % log_interval == 0:
            avg_loss = total_loss / log_interval
            accuracy = correct / total
            print(f"Iteration {iteration + 1}/{num_iterations}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

            # W&B: Log metrics
            wandb.log({
                "train/loss": avg_loss,
                "train/accuracy": accuracy,
                "iteration": iteration + 1,
                "epoch":epoch,
                "lr":current_lr
                
            })

            # Reset for next logging interval
            total_loss = 0.0
            correct = 0
            total = 0

        # Validation
        if (iteration + 1) % val_interval == 0:
            validate(model, val_loader, criterion, wandb, iteration, device)
            
        iteration += 1

    print("Training complete.")

# Validation function
@torch.inference_mode()  # or @torch.no_grad()
def validate(model, val_loader, criterion, wandb, iteration, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

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


    avg_loss = val_loss / len(val_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_preds, average="weighted")  # Use "weighted" for multiclass

    print(f"Validation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
    # W&B: Log validation metrics
    wandb.log({
        "val/loss": avg_loss,
        "val/accuracy": accuracy,
        "val/f1-score":f1,
        "iteration": iteration + 1
    })


    model.train()  # Switch back to training mode
    return avg_loss, accuracy, f1


if __name__ == "__main__":
    # W&B: Initialize the project
    wandb.init(project="my_car_classification_project")  
    # Optionally, wandb.config can store hyperparams:
    wandb.config = {
        "batch_size": 8,
        "learning_rate": 0.0001,
        "num_iterations": 5000,
        "log_interval": 5,
        "val_interval": 50
    }

    # Paths and parameters
    base_dir = "E:/Om/Other Projects/ClearQuote_Assignment/data"
    resize_to = (380, 380)
    train_data_path = "training_data.json"
    val_data_path = "val_data.json"
    batch_size = wandb.config["batch_size"]
    learning_rate = wandb.config["learning_rate"]
    num_iterations = wandb.config["num_iterations"]
    log_interval = wandb.config["log_interval"]
    val_interval = wandb.config["val_interval"]

    # Load train and validation data from JSON files
    def load_json_lines(file_path):
        with open(file_path, 'r') as file:
            return [json.loads(line) for line in file]

    train_data = load_json_lines(train_data_path)
    val_data = load_json_lines(val_data_path)

    # Create datasets
    train_dataset = CarDataset(data=train_data, base_dir=base_dir, 
                               resize_to=resize_to, augment=True)
    val_dataset = CarDataset(data=val_data, base_dir=base_dir, 
                             resize_to=resize_to, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, num_workers=4)

    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    # Move the model to GPU if available
    model = build_efficientnet_b4(num_classes=7)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    lr_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=num_iterations,  # # of steps until reaching eta_min
        eta_min=5e-5
    )

    # Start training with iterations
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_iterations=num_iterations,
        log_interval=log_interval,
        val_interval=val_interval,
        lr_scheduler = lr_scheduler,
        device=device
    )
