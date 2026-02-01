"""
Kaggle Sweep Agent Script
=========================
For running W&B sweep experiments on Kaggle.

Instructions:
1. First, create the sweep locally or on any machine:
   $ wandb sweep sweeps/sweep_config.yaml
   
   This outputs a SWEEP_ID like: tklwin_msds/chickpea/abc123xyz

2. Copy this entire file to a Kaggle notebook

3. Set the SWEEP_ID below

4. Run on Kaggle with GPU enabled

5. Each "Save Version" will run one sweep trial
"""

# ╔═══════════════════════════════════════════════════════════╗
# ║  CONFIGURATION - SET YOUR SWEEP ID HERE                   ║
# ╚═══════════════════════════════════════════════════════════╝

SWEEP_ID = "tklwin_msds/chickpea/YOUR_SWEEP_ID"  # ← Replace with your sweep ID
NUM_RUNS = 1  # Number of runs per Kaggle session (1 is safest)

# ============================================================
# SETUP
# ============================================================

import subprocess
import sys
import os
import shutil

# Clone repository
REPO_PATH = '/kaggle/working/chickpea-fusarium'
if os.path.exists(REPO_PATH):
    shutil.rmtree(REPO_PATH)

subprocess.run([
    "git", "clone", "--depth", "1",
    "https://github.com/tklwin/chickpea-fusarium.git",
    REPO_PATH
], check=True)
print(f"✓ Repository cloned")

sys.path.insert(0, REPO_PATH)

# Install dependencies
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "albumentations", "wandb"], check=True)
print("✓ Dependencies installed")

# ============================================================
# W&B LOGIN
# ============================================================

import wandb
from kaggle_secrets import UserSecretsClient

try:
    user_secrets = UserSecretsClient()
    wandb_api_key = user_secrets.get_secret("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)
    print("✓ W&B logged in")
except Exception as e:
    print(f"⚠ W&B login failed: {e}")

# ============================================================
# CREATE DATA SPLITS (once)
# ============================================================

os.environ["DATA_DIR"] = "/kaggle/input/fusarium-wilt-disease-in-chickpea-dataset/FUSARIUM-22/dataset_raw"
os.environ["SPLITS_DIR"] = "/kaggle/working/splits"
os.environ["CHECKPOINT_DIR"] = "/kaggle/working/checkpoints"
os.environ["WANDB_ENTITY"] = "tklwin_msds"
os.environ["WANDB_PROJECT"] = "chickpea"

from src.data.split import create_splits

if not os.path.exists(os.environ["SPLITS_DIR"]):
    splits = create_splits(
        data_dir=os.environ["DATA_DIR"],
        output_dir=os.environ["SPLITS_DIR"],
        seed=42
    )
    print("✓ Data splits created")
else:
    print("✓ Data splits already exist")

# ============================================================
# RUN SWEEP AGENT
# ============================================================

from sweeps.train_sweep import train

print(f"\n{'='*60}")
print(f"Starting W&B Sweep Agent")
print(f"Sweep ID: {SWEEP_ID}")
print(f"Number of runs: {NUM_RUNS}")
print(f"{'='*60}\n")

# Run sweep agent
wandb.agent(
    SWEEP_ID,
    function=train,
    count=NUM_RUNS  # How many runs this agent should do
)

print(f"\n{'='*60}")
print("SWEEP AGENT COMPLETE!")
print(f"View results at: https://wandb.ai/{SWEEP_ID}")
print(f"{'='*60}")
