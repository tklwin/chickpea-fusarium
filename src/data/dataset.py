"""
Dataset class for Fusarium-22 chickpea disease dataset.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Callable, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from .transforms import get_train_transforms, get_val_transforms
from .split import load_splits, compute_class_weights, get_sample_weights


class FusariumDataset(Dataset):
    """
    PyTorch Dataset for Fusarium-22 chickpea disease images.
    
    Args:
        dataframe: DataFrame with 'image_path' and 'label' columns
        transform: Albumentations transform to apply
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform: Optional[Callable] = None,
    ):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        row = self.df.iloc[idx]
        
        # Load image
        image_path = row['image_path']
        image = cv2.imread(image_path)
        
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        label = int(row['label'])
        
        return image, label
    
    def get_labels(self) -> List[int]:
        """Get all labels (useful for computing class weights)."""
        return self.df['label'].tolist()


def get_dataloaders(
    splits_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 2,
    pin_memory: bool = True,
    use_weighted_sampler: bool = False,
    augmentation_config: Optional[dict] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, val, and test sets.
    
    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv
        batch_size: Batch size for training
        image_size: Target image size
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for faster GPU transfer
        use_weighted_sampler: Use WeightedRandomSampler for class imbalance
        augmentation_config: Optional dict with augmentation parameters
    
    Returns:
        Dict with 'train', 'val', 'test' DataLoaders
    """
    # Load splits
    splits = load_splits(splits_dir)
    
    # Get transforms
    aug_kwargs = augmentation_config or {}
    train_transform = get_train_transforms(image_size=image_size, **aug_kwargs)
    val_transform = get_val_transforms(image_size=image_size)
    
    # Create datasets
    train_dataset = FusariumDataset(splits['train'], transform=train_transform)
    val_dataset = FusariumDataset(splits['val'], transform=val_transform)
    test_dataset = FusariumDataset(splits['test'], transform=val_transform)
    
    # Prepare sampler for class imbalance (optional)
    train_sampler = None
    shuffle = True
    
    if use_weighted_sampler:
        # Compute class weights
        class_weights = compute_class_weights(splits['train'], num_classes=3)
        sample_weights = get_sample_weights(splits['train'], class_weights)
        
        train_sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        shuffle = False  # Sampler handles shuffling
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    print(f"\nDataLoaders created:")
    print(f"  Train: {len(train_dataset)} images, {len(train_loader)} batches")
    print(f"  Val: {len(val_dataset)} images, {len(val_loader)} batches")
    print(f"  Test: {len(test_dataset)} images, {len(test_loader)} batches")
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
    }
