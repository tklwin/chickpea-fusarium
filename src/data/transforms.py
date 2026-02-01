"""
Data augmentation transforms using Albumentations.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(
    image_size: int = 224,
    # Geometric
    horizontal_flip_p: float = 0.5,
    vertical_flip_p: float = 0.5,
    rotate_limit: int = 30,
    rotate_p: float = 0.5,
    # Color
    brightness_limit: float = 0.2,
    contrast_limit: float = 0.2,
    color_jitter_p: float = 0.5,
    # Crop
    random_resized_crop: bool = True,
    crop_scale: tuple = (0.8, 1.0),
    # Regularization
    coarse_dropout_p: float = 0.3,
    max_holes: int = 8,
    max_height: int = 28,
    max_width: int = 28,
) -> A.Compose:
    """
    Get training augmentation transforms.
    
    Designed to:
    1. Increase dataset diversity
    2. Improve generalization
    3. Help with class imbalance (more variations for minority classes)
    
    Returns:
        Albumentations Compose object
    """
    transforms_list = []
    
    # Resize or RandomResizedCrop
    if random_resized_crop:
        transforms_list.append(
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=crop_scale,
                ratio=(0.9, 1.1),
                p=1.0
            )
        )
    else:
        transforms_list.append(
            A.Resize(image_size, image_size)
        )
    
    # Geometric transforms
    transforms_list.extend([
        A.HorizontalFlip(p=horizontal_flip_p),
        A.VerticalFlip(p=vertical_flip_p),
        A.Rotate(limit=rotate_limit, p=rotate_p, border_mode=0),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=0,
            p=0.3,
            border_mode=0
        ),
    ])
    
    # Color transforms
    transforms_list.extend([
        A.ColorJitter(
            brightness=brightness_limit,
            contrast=contrast_limit,
            saturation=0.2,
            hue=0.1,
            p=color_jitter_p
        ),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.MotionBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.GaussNoise(std_range=(0.02, 0.1), p=0.2),
    ])
    
    # Regularization (Cutout-like) - Updated for albumentations 2.0+
    if coarse_dropout_p > 0:
        transforms_list.append(
            A.CoarseDropout(
                num_holes_range=(1, max_holes),
                hole_height_range=(8, max_height),
                hole_width_range=(8, max_width),
                fill=0,  # Fill with black (0) - numeric value required
                p=coarse_dropout_p
            )
        )
    
    # Normalize (ImageNet stats) and convert to tensor
    transforms_list.extend([
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    return A.Compose(transforms_list)


def get_val_transforms(image_size: int = 224) -> A.Compose:
    """
    Get validation/test transforms (no augmentation).
    
    Returns:
        Albumentations Compose object
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])


def get_tta_transforms(image_size: int = 224, n_augments: int = 5) -> list:
    """
    Get Test-Time Augmentation (TTA) transforms.
    
    For inference: apply multiple augmentations and average predictions.
    
    Args:
        image_size: Target image size
        n_augments: Number of augmented versions to generate
    
    Returns:
        List of Albumentations Compose objects
    """
    tta_list = [
        # Original (no augmentation)
        get_val_transforms(image_size),
        
        # Horizontal flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Vertical flip
        A.Compose([
            A.Resize(image_size, image_size),
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Small rotation
        A.Compose([
            A.Resize(image_size, image_size),
            A.Rotate(limit=(10, 10), p=1.0, border_mode=0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Center crop
        A.Compose([
            A.Resize(int(image_size * 1.1), int(image_size * 1.1)),
            A.CenterCrop(image_size, image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]
    
    return tta_list[:n_augments]
