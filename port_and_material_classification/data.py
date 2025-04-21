import os
import torch
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from omegaconf import DictConfig, OmegaConf
import logging
import wandb
from typing import Tuple

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageClassificationDataset(data.Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the image classification dataset.
        
        Args:
            root_dir (str): Directory containing class subfolders
            transform: Optional transforms to apply to images
        """
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) 
                              if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        
        # Load all image paths and labels
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Get all images in this class folder
            for fname in os.listdir(class_dir):
                if self._is_image_file(fname):
                    path = os.path.join(class_dir, fname)
                    self.samples.append((path, class_idx))
        
        logger.info(f"Loaded {len(self.samples)} images from {root_dir} with {len(self.classes)} classes")
    
    def _is_image_file(self, filename):
        """Check if a file is an image based on extension."""
        img_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        return any(filename.lower().endswith(ext) for ext in img_extensions)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Return a sample from the dataset."""
        path, label = self.samples[idx]
        
        try:
            # Load image
            with open(path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                
            # Apply transforms if available
            if self.transform:
                img = self.transform(img)
                
            return {
                "image": img,
                "label": label,
                "class_name": self.classes[label],
                "path": path
            }
            
        except Exception as e:
            logger.warning(f"Error loading image {path}: {str(e)}. Loading a different sample.")
            # Return a different sample on error
            return self.__getitem__(np.random.randint(0, len(self)))

def get_transforms(img_size=224):
    """Get standard image transforms for training and validation."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        # transforms.RandomResizedCrop(img_size),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(90),
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        # transforms.RandomGrayscale(p=0.05),
        # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
    ])
    
    return train_transform, val_transform

def load_classification_datasets(img_size=224, train_data_budget=1.0) -> Tuple[data.Dataset, data.Dataset]:
    """
    Load train and test datasets for port and material classification.
    
    Args:
        img_size (int): Input image size (default: 224)
        train_data_budget (float): Fraction of training data to use (0-1)
    
    Returns:
        Tuple of training and test datasets
    """
    # Get transforms
    train_transform, val_transform = get_transforms(img_size)
    
    # Load training dataset
    train_dataset = ImageClassificationDataset(
        root_dir="data/objects",
        transform=train_transform
    )
    
    # Load test dataset
    test_dataset = ImageClassificationDataset(
        root_dir="data/objects_20140",
        transform=val_transform
    )
    
    # Apply train data budget if needed
    if train_data_budget < 1.0:
        train_size = int(len(train_dataset) * train_data_budget)
        indices = torch.randperm(len(train_dataset))[:train_size]
        train_dataset = data.Subset(train_dataset, indices)
        logger.info(f"Applied train_data_budget={train_data_budget}, using {train_size} training samples")
    
    # Log dataset stats
    wandb.log({
        "train_dataset_size": len(train_dataset),
        "test_dataset_size": len(test_dataset),
        "num_classes": len(train_dataset.classes) if hasattr(train_dataset, 'classes') else None,
        "class_names": train_dataset.classes if hasattr(train_dataset, 'classes') else None
    })
    
    return train_dataset, test_dataset
