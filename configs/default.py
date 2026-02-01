"""
Default configuration for Chickpea Fusarium experiments.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    # Kaggle dataset path
    data_dir: str = "/kaggle/input/fusarium-wilt-disease-in-chickpea-dataset/FUSARIUM-22/dataset_raw"
    splits_dir: str = "./splits"
    
    # Image settings
    image_size: int = 224
    
    # Split ratios
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Class mapping (5 -> 3 classes)
    class_mapping: dict = field(default_factory=lambda: {
        "1(HR)": 0,   # Resistant
        "3(R)": 0,    # Resistant
        "5(MR)": 1,   # Moderate
        "7(S)": 2,    # Susceptible
        "9(HS)": 2,   # Susceptible
    })
    
    class_names: List[str] = field(default_factory=lambda: [
        "Resistant",
        "Moderate", 
        "Susceptible"
    ])
    
    num_classes: int = 3
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 2
    pin_memory: bool = True


@dataclass
class AugmentationConfig:
    # Training augmentation strength
    enable_augmentation: bool = True
    
    # Geometric transforms
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate_limit: int = 30
    rotate_p: float = 0.5
    
    # Color transforms
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    color_jitter_p: float = 0.5
    
    # Advanced augmentations
    random_resized_crop: bool = True
    crop_scale: tuple = (0.8, 1.0)
    
    # Regularization augmentations
    coarse_dropout_p: float = 0.3
    max_holes: int = 8
    max_height: int = 28  # 224 / 8
    max_width: int = 28


@dataclass
class ModelConfig:
    # Model selection
    model_name: str = "squeezenet1_1"  # squeezenet1_1, squeezenet1_1_cbam, mobilenetv2, efficientnet_b0, shufflenetv2
    pretrained: bool = True
    
    # CBAM settings (when using squeezenet_cbam)
    cbam_reduction: int = 16
    cbam_kernel_size: int = 7
    
    # Dropout
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    # Basic training
    epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"  # adam, adamw, sgd
    momentum: float = 0.9  # for SGD
    
    # Scheduler
    scheduler: str = "cosine"  # cosine, step, plateau
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Class imbalance handling
    use_class_weights: bool = True      # Weighted CrossEntropyLoss
    use_weighted_sampler: bool = False  # WeightedRandomSampler (alternative to class weights)
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    
    # Checkpointing
    save_best_only: bool = True
    checkpoint_dir: str = "./checkpoints"
    
    # Reproducibility
    seed: int = 42


@dataclass
class WandbConfig:
    enabled: bool = True
    entity: str = "tklwin_msds"
    project: str = "chickpea"
    
    # Run naming
    run_name: Optional[str] = None  # Auto-generated if None
    tags: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def get_config(**overrides) -> Config:
    """
    Get default config with optional overrides.
    
    Usage:
        config = get_config()
        config = get_config(model={"model_name": "squeezenet1_1_cbam"})
    """
    config = Config()
    
    # Apply overrides
    for section, params in overrides.items():
        if hasattr(config, section):
            section_config = getattr(config, section)
            for key, value in params.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
    
    return config


# Experiment presets
def get_baseline_config() -> Config:
    """SqueezeNet baseline configuration."""
    return get_config(
        model={"model_name": "squeezenet1_1"},
        wandb={"tags": ["baseline", "squeezenet"]}
    )


def get_cbam_config() -> Config:
    """SqueezeNet + CBAM configuration."""
    return get_config(
        model={"model_name": "squeezenet1_1_cbam"},
        wandb={"tags": ["cbam", "squeezenet", "attention"]}
    )


def get_mobilenet_config() -> Config:
    """MobileNetV2 comparison configuration."""
    return get_config(
        model={"model_name": "mobilenetv2"},
        wandb={"tags": ["comparison", "mobilenet"]}
    )
