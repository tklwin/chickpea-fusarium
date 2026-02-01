"""
Data splitting utilities for reproducible train/val/test splits.
Handles stratified splitting with class imbalance awareness.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def create_splits(
    data_dir: str,
    output_dir: str,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    class_mapping: Optional[Dict[str, int]] = None,
    seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """
    Create stratified train/val/test splits and save to CSV files.
    
    Args:
        data_dir: Path to dataset_raw folder containing class subfolders
        output_dir: Where to save split CSV files
        train_ratio: Proportion for training (default: 0.70)
        val_ratio: Proportion for validation (default: 0.15)
        test_ratio: Proportion for testing (default: 0.15)
        class_mapping: Dict mapping folder names to class labels
        seed: Random seed for reproducibility
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Default class mapping (5 -> 3 classes)
    if class_mapping is None:
        class_mapping = {
            "1(HR)": 0,   # Resistant
            "3(R)": 0,    # Resistant
            "5(MR)": 1,   # Moderate
            "7(S)": 2,    # Susceptible
            "9(HS)": 2,   # Susceptible
        }
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all image paths and labels
    data = []
    for folder_name, label in class_mapping.items():
        folder_path = data_dir / folder_name
        if not folder_path.exists():
            print(f"Warning: Folder not found: {folder_path}")
            continue
            
        for img_file in folder_path.iterdir():
            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                data.append({
                    'image_path': str(img_file),
                    'original_class': folder_name,
                    'label': label
                })
    
    df = pd.DataFrame(data)
    print(f"Total images found: {len(df)}")
    print(f"\nOriginal class distribution:")
    print(df['original_class'].value_counts().sort_index())
    print(f"\nMerged class distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Stratified split: first split train from (val+test)
    val_test_ratio = val_ratio + test_ratio
    
    train_df, val_test_df = train_test_split(
        df,
        test_size=val_test_ratio,
        stratify=df['label'],
        random_state=seed
    )
    
    # Split val from test
    test_ratio_adjusted = test_ratio / val_test_ratio
    val_df, test_df = train_test_split(
        val_test_df,
        test_size=test_ratio_adjusted,
        stratify=val_test_df['label'],
        random_state=seed
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # Save to CSV
    train_df.to_csv(output_dir / 'train.csv', index=False)
    val_df.to_csv(output_dir / 'val.csv', index=False)
    test_df.to_csv(output_dir / 'test.csv', index=False)
    
    # Print split statistics
    print(f"\n{'='*50}")
    print("Split Statistics:")
    print(f"{'='*50}")
    
    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        print(f"\n{name}: {len(split_df)} images ({len(split_df)/len(df)*100:.1f}%)")
        print(f"  Class distribution:")
        for label in sorted(split_df['label'].unique()):
            count = (split_df['label'] == label).sum()
            print(f"    Class {label}: {count} ({count/len(split_df)*100:.1f}%)")
    
    # Save split info for reproducibility
    info = {
        'seed': seed,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'total_images': len(df),
        'train_size': len(train_df),
        'val_size': len(val_df),
        'test_size': len(test_df),
    }
    
    info_df = pd.DataFrame([info])
    info_df.to_csv(output_dir / 'split_info.csv', index=False)
    
    print(f"\n✓ Splits saved to: {output_dir}")
    
    return {'train': train_df, 'val': val_df, 'test': test_df}


def load_splits(splits_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Load existing splits from CSV files.
    
    Args:
        splits_dir: Directory containing train.csv, val.csv, test.csv
    
    Returns:
        Dict with 'train', 'val', 'test' DataFrames
    """
    splits_dir = Path(splits_dir)
    
    splits = {}
    for name in ['train', 'val', 'test']:
        csv_path = splits_dir / f'{name}.csv'
        if not csv_path.exists():
            raise FileNotFoundError(f"Split file not found: {csv_path}")
        splits[name] = pd.read_csv(csv_path)
    
    return splits


def compute_class_weights(
    train_df: pd.DataFrame,
    num_classes: int = 3,
    method: str = "inverse_freq"
) -> List[float]:
    """
    Compute class weights for handling imbalanced data.
    
    Args:
        train_df: Training DataFrame with 'label' column
        num_classes: Number of classes
        method: Weighting method
            - "inverse_freq": 1 / class_count (normalized)
            - "inverse_sqrt": 1 / sqrt(class_count) (smoother)
            - "effective_samples": Based on effective number of samples
    
    Returns:
        List of class weights
    """
    import numpy as np
    
    class_counts = train_df['label'].value_counts().sort_index().values
    
    if method == "inverse_freq":
        weights = 1.0 / class_counts
        weights = weights / weights.sum() * num_classes  # normalize
        
    elif method == "inverse_sqrt":
        weights = 1.0 / np.sqrt(class_counts)
        weights = weights / weights.sum() * num_classes
        
    elif method == "effective_samples":
        # From "Class-Balanced Loss Based on Effective Number of Samples"
        beta = 0.9999
        effective_num = 1.0 - np.power(beta, class_counts)
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.sum() * num_classes
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    print(f"\nClass weights ({method}):")
    for i, w in enumerate(weights):
        print(f"  Class {i}: {w:.4f} (count: {class_counts[i]})")
    
    return weights.tolist()


def get_sample_weights(train_df: pd.DataFrame, class_weights: List[float]) -> List[float]:
    """
    Get per-sample weights for WeightedRandomSampler.
    
    Args:
        train_df: Training DataFrame with 'label' column
        class_weights: List of class weights
    
    Returns:
        List of sample weights (same length as train_df)
    """
    sample_weights = [class_weights[label] for label in train_df['label'].values]
    return sample_weights
