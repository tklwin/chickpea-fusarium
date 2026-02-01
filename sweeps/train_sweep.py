"""
W&B Sweep Training Script
==========================
This script is called by W&B sweep agent with different hyperparameter configs.

Usage:
    1. Create sweep: wandb sweep sweeps/sweep_config.yaml
    2. Run agent:    wandb agent <sweep_id>

For Kaggle: See sweeps/kaggle_sweep_agent.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import wandb
from configs.default import get_config
from src.training import Trainer


def train():
    """Training function called by W&B sweep agent."""
    
    # Initialize W&B run (sweep agent passes config automatically)
    wandb.init()
    
    # Get hyperparameters from sweep
    sweep_config = wandb.config
    
    print(f"\n{'='*60}")
    print("SWEEP RUN CONFIGURATION:")
    print(f"{'='*60}")
    for key, value in sweep_config.items():
        print(f"  {key}: {value}")
    print(f"{'='*60}\n")
    
    # Build config from sweep parameters
    config = get_config(
        data={
            "data_dir": os.environ.get("DATA_DIR", "/kaggle/input/fusarium-wilt-disease-in-chickpea-dataset/FUSARIUM-22/dataset_raw"),
            "splits_dir": os.environ.get("SPLITS_DIR", "/kaggle/working/splits"),
            "batch_size": sweep_config.get("batch_size", 32),
        },
        model={
            "model_name": sweep_config.get("model_name", "squeezenet1_1"),
            "pretrained": True,
            "dropout": sweep_config.get("dropout", 0.5),
        },
        training={
            "epochs": sweep_config.get("epochs", 50),
            "learning_rate": sweep_config.get("learning_rate", 1e-3),
            "weight_decay": sweep_config.get("weight_decay", 1e-4),
            "optimizer": sweep_config.get("optimizer", "adamw"),
            "scheduler": sweep_config.get("scheduler", "cosine"),
            "use_class_weights": sweep_config.get("use_class_weights", True),
            "use_weighted_sampler": False,
            "seed": 42,
            "checkpoint_dir": os.environ.get("CHECKPOINT_DIR", "./checkpoints"),
        },
        wandb={
            "enabled": True,  # Already initialized by sweep
            "entity": os.environ.get("WANDB_ENTITY", "tklwin_msds"),
            "project": os.environ.get("WANDB_PROJECT", "chickpea"),
        }
    )
    
    # Override wandb since it's already initialized by sweep
    config.wandb.enabled = False  # Trainer won't re-init
    
    # Create trainer and train
    trainer = Trainer(config)
    trainer.fit()
    
    # Log final metrics to sweep
    wandb.log({
        "final_val_acc": trainer.best_val_acc,
        "final_val_f1": trainer.best_val_f1,
    })
    
    print(f"\n{'='*60}")
    print(f"SWEEP RUN COMPLETE!")
    print(f"Best Val Accuracy: {trainer.best_val_acc * 100:.2f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    train()
