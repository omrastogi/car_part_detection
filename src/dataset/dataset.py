import os
import json
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import cv2

class CarDataset(Dataset):
    def __init__(self, data, base_dir, resize_to=(224, 224), augment=False):
        """
        Initializes the dataset.

        Args:
            data (list): List of dictionaries with image paths, annotations, etc.
            base_dir (str): Base directory containing the images.
            resize_to (tuple): Dimensions to resize the image to (height, width).
            augment (bool): Whether to apply augmentations.
        """
        self.data = data
        self.base_dir = base_dir
        self.resize_to = resize_to
        self.augment = augment
        self.class_labels = ["Front", "Rear", "Front-Right", "Front-Left", "Rear-Right", "Rear-Left", "None"]
        self.label_to_idx = {label: idx for idx, label in enumerate(self.class_labels)}
        self.non_flippable_labels = ["Rear-Right", "Rear-Left", "Front-Right", "Front-Left"]

        # Preprocessing transformations
        self.preprocess = T.Compose([
            T.ToTensor(),
            T.Resize(self.resize_to)
        ])
        
        self.augmentations = T.Compose([
            T.RandomAffine(degrees=45, scale=(0.7, 1.3)),  # Random rotation and scaling
            T.RandomResizedCrop(size=self.resize_to, scale=(0.8, 1.2)),  # Random crop and scale
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color adjustments
        ]) if self.augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Fetches a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary with the image tensor and its corresponding labels.
        """
        item = self.data[idx]
        image_path = os.path.join(self.base_dir, item['path'], item['image'])
        image = cv2.imread(image_path)
        
        if image is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply preprocessing
        image = self.preprocess(image)
        # Prepare annotations and labels
        final_label = item['final_label']

        # Apply augmentations if enabled
        if self.augment:
            if final_label in self.non_flippable_labels:
                image = T.RandomHorizontalFlip(p=0.5)(image)  # Apply flipping only if label is flippable
            image = self.augmentations(image)

        # Convert final label to one-hot encoding
        final_label_one_hot = torch.zeros(len(self.class_labels))
        if final_label in self.label_to_idx:
            final_label_one_hot[self.label_to_idx[final_label]] = 1.0            
        
        return {
            "image": image,
            "final_label": final_label_one_hot
        }



# Example Usage
if __name__ == "__main__":
    import autoroot 
    import autorootcwd
    from src.utils.visualize import visualize_and_save_augmentations
    # Load the JSON file
    json_file = "E:/Om/Other Projects/ClearQuote_Assignment/notebooks/training_data.json"  # Replace with your actual file path
    with open(json_file, 'r') as f:
        data = [json.loads(line) for line in f.readlines()]  # Assuming one JSON object per line

    base_dir = "E:/Om/Other Projects/ClearQuote_Assignment/data"  # Base directory
    dataset = CarDataset(data=data, base_dir=base_dir, resize_to=(224, 224), augment=True)
    visualize_and_save_augmentations(dataset, num_samples=10, save_dir="visualized_samples")

    # Fetch a single sample
    import random
    for _ in range(4):
        idx = random.randint(0, len(dataset) - 1)
        sample = dataset[idx]
        print(f"Sample {idx}:")
        print("Image shape:", sample['image'].shape)
        print("Final Label:", sample['final_label'])
