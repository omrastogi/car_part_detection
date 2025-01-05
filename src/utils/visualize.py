import torchvision.transforms.functional as F
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import torch

def visualize_and_save_augmentations(dataset, num_samples=5, save_dir="visualized_samples", augment=True):
    """
    Visualizes and saves augmented images from the dataset.

    Args:
        dataset (CustomCarDataset): The dataset instance.
        num_samples (int): Number of samples to visualize.
        save_dir (str): Directory to save the visualized images.
        augment (bool): Whether to visualize with augmentations.
    """
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(num_samples):
        sample = dataset[i]
        image = sample['image']
        final_label = sample['final_label']

        # If augmentations are enabled, apply them
        if augment and dataset.augmentations:
            image = dataset.augmentations(image)

        # Convert tensor to PIL image and save it
        save_path = os.path.join(save_dir, f"sample_{i}_{final_label}.png")
        save_image(image, save_path)

        print(f"Saved augmented image for sample {i} at: {save_path}")


