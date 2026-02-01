"""
Kaggle Training Script (Background Run / Save Version)
=======================================================
For Kaggle "Save Version" background execution.

Instructions:
1. Create a new Kaggle notebook
2. Add the Fusarium-22 dataset as input
3. Enable GPU (P100 or T4)
4. Add W&B API key to Kaggle Secrets (key name: WANDB_API_KEY)
5. Copy this entire file into a single code cell
6. ⭐ ONLY CHANGE THE 2 LINES IN "EXPERIMENT CONFIG" SECTION BELOW ⭐
7. Click "Save Version" > "Save & Run All (Commit)"
"""

# ╔═══════════════════════════════════════════════════════════╗
# ║  ⭐ EXPERIMENT CONFIG - ONLY CHANGE THESE 2 LINES! ⭐     ║
# ╚═══════════════════════════════════════════════════════════╝

EXPERIMENT_NAME = "exp1_squeezenet_baseline"   # ← Change for each run
MODEL_NAME = "squeezenet1_1"                   # ← Options below

# MODEL_NAME options:
#   "squeezenet1_1"       - Baseline SqueezeNet (724K params)
#   "squeezenet1_1_cbam"  - SqueezeNet + CBAM attention
#   "mobilenetv2"         - MobileNetV2 comparison
#   "efficientnet_b0"     - EfficientNet-B0 comparison
#   "shufflenetv2"        - ShuffleNetV2 comparison

# ============================================================
# SETUP: Clone repository and install dependencies
# ============================================================

import subprocess
import sys
import os
import shutil

REPO_PATH = '/kaggle/working/chickpea-fusarium'

# Always get fresh code for background runs
if os.path.exists(REPO_PATH):
    shutil.rmtree(REPO_PATH)
    
subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/tklwin/chickpea-fusarium.git",
    REPO_PATH
], check=True)
print(f"✓ Repository cloned to {REPO_PATH}")

# Add to path (MUST be done before any src imports)
sys.path.insert(0, REPO_PATH)
print(f"✓ Added {REPO_PATH} to Python path")

# Install additional dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations", "wandb"], check=True)
print("✓ Dependencies installed")

# ============================================================
# FIXED CONFIG (don't change unless needed)
# ============================================================

CONFIG = {
    # Data (fixed paths for Kaggle)
    "data_dir": "/kaggle/input/fusarium-wilt-disease-in-chickpea-dataset/FUSARIUM-22/dataset_raw",
    "splits_dir": "/kaggle/working/splits",
    
    # Model (uses MODEL_NAME from above)
    "model_name": MODEL_NAME,
    "pretrained": True,
    "dropout": 0.5,
    
    # Training hyperparameters (same for fair comparison)
    "batch_size": 32,
    "epochs": 50,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adamw",
    "scheduler": "cosine",
    
    # Class imbalance handling
    "use_class_weights": True,
    "use_weighted_sampler": False,
    
    # W&B tracking
    "wandb_enabled": True,
    "wandb_entity": "tklwin_msds",
    "wandb_project": "chickpea",
    
    # Reproducibility
    "seed": 42,
}

# ============================================================
# STEP 1: Create Data Splits
# ============================================================

# Import AFTER path is set
from src.data.split import create_splits

splits = create_splits(
    data_dir=CONFIG["data_dir"],
    output_dir=CONFIG["splits_dir"],
    seed=CONFIG["seed"]
)

print("\n✓ Data splits created!")

# ============================================================
# STEP 2: Setup W&B (login with API key)
# ============================================================

import wandb
import os

# Option 1: Set API key directly (less secure, but works)
# os.environ["WANDB_API_KEY"] = "your-api-key-here"

# Option 2: Use Kaggle Secrets (recommended)
# In Kaggle, go to Add-ons > Secrets and add WANDB_API_KEY
from kaggle_secrets import UserSecretsClient
try:
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
    os.environ["WANDB_API_KEY"] = wandb_api_key
    print("✓ W&B API key loaded from Kaggle Secrets")
except:
    print("⚠ Could not load W&B API key. Set it manually or disable W&B logging.")
    CONFIG["wandb_enabled"] = False

# ============================================================
# STEP 3: Build Configuration and Train
# ============================================================

from configs.default import get_config
from src.training import Trainer

# Build config from our settings
config = get_config(
    data={
        "data_dir": CONFIG["data_dir"],
        "splits_dir": CONFIG["splits_dir"],
        "batch_size": CONFIG["batch_size"],
    },
    model={
        "model_name": CONFIG["model_name"],
        "pretrained": CONFIG["pretrained"],
        "dropout": CONFIG["dropout"],
    },
    training={
        "epochs": CONFIG["epochs"],
        "learning_rate": CONFIG["learning_rate"],
        "weight_decay": CONFIG["weight_decay"],
        "optimizer": CONFIG["optimizer"],
        "scheduler": CONFIG["scheduler"],
        "use_class_weights": CONFIG["use_class_weights"],
        "use_weighted_sampler": CONFIG["use_weighted_sampler"],
        "seed": CONFIG["seed"],
        "checkpoint_dir": "/kaggle/working/checkpoints",
    },
    wandb={
        "enabled": CONFIG["wandb_enabled"],
        "entity": CONFIG["wandb_entity"],
        "project": CONFIG["wandb_project"],
        "run_name": EXPERIMENT_NAME,
        "tags": [CONFIG["model_name"], "experiment"],
    }
)

# Create trainer and train!
trainer = Trainer(config)
trainer.fit()

# ============================================================
# STEP 4: Save Final Model for Mobile Deployment
# ============================================================

from src.utils import save_model_for_mobile

# Load best model
trainer._load_checkpoint('best.pth')

# Export to TorchScript
save_model_for_mobile(
    trainer.model,
    "/kaggle/working/model_mobile.pt",
    format="torchscript"
)

# Export to ONNX
save_model_for_mobile(
    trainer.model,
    "/kaggle/working/model_mobile.onnx",
    format="onnx"
)

print("\n✓ Models exported for mobile deployment!")
print("Download from /kaggle/working/")

# ============================================================
# STEP 5: Generate Evaluation Report
# ============================================================

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for background runs
import matplotlib.pyplot as plt
from src.training.metrics import plot_confusion_matrix, plot_training_curves

# Plot training curves
fig = plot_training_curves(
    trainer.history['train_loss'],
    trainer.history['val_loss'],
    trainer.history['train_acc'],
    trainer.history['val_acc'],
    save_path="/kaggle/working/training_curves.png"
)
plt.close(fig)
print("✓ Training curves saved to /kaggle/working/training_curves.png")

# Plot confusion matrix (need predictions)
import torch
import numpy as np

trainer.model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in trainer.dataloaders['test']:
        images = images.to(trainer.device)
        outputs = trainer.model(images)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

fig = plot_confusion_matrix(
    np.array(all_labels),
    np.array(all_preds),
    save_path="/kaggle/working/confusion_matrix.png"
)
plt.close(fig)
print("✓ Confusion matrix saved to /kaggle/working/confusion_matrix.png")

print("\n" + "=" * 60)
print("EXPERIMENT COMPLETE!")
print("=" * 60)
print(f"Results logged to W&B: {CONFIG['wandb_entity']}/{CONFIG['wandb_project']}")
print(f"Best validation accuracy: {trainer.best_val_acc * 100:.2f}%")
print("=" * 60)
