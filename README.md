# Chickpea Fusarium Wilt Disease Detection

Master's Thesis Project: Lightweight CNN with Attention for Mobile Deployment

## Project Structure

```
chickpea-fusarium/
├── configs/
│   └── default.yaml          # Training configurations
├── src/
│   ├── data/
│   │   ├── dataset.py        # Dataset class
│   │   ├── transforms.py     # Albumentations augmentations
│   │   └── split.py          # Train/val/test splitting
│   ├── models/
│   │   ├── squeezenet.py     # SqueezeNet baseline
│   │   ├── cbam.py           # CBAM attention module
│   │   └── factory.py        # Model factory
│   ├── training/
│   │   ├── trainer.py        # Training loop with W&B
│   │   └── metrics.py        # Evaluation metrics
│   └── utils/
│       └── helpers.py        # Utility functions
├── splits/                   # Generated train/val/test CSVs
├── notebooks/
│   └── kaggle_train.py       # Kaggle notebook template
└── requirements.txt
```

## Setup

### Local Development (MacBook)

```bash
pip install -r requirements.txt
```

### Kaggle Usage

```python
!git clone --depth 1 https://github.com/YOUR_USERNAME/chickpea-fusarium.git
import sys
sys.path.append('/kaggle/working/chickpea-fusarium')
```

## Quick Start

### 1. Create Data Splits (run once)

```python
from src.data.split import create_splits

create_splits(
    data_dir="/kaggle/input/fusarium-wilt-disease-in-chickpea-dataset/FUSARIUM-22/dataset_raw",
    output_dir="./splits",
    seed=42
)
```

### 2. Train Model

```python
from src.training.trainer import Trainer
from configs.default import get_config

config = get_config()
trainer = Trainer(config)
trainer.fit()
```

## Class Mapping (5 → 3 classes)

| Original    | Merged Class | Label |
| ----------- | ------------ | ----- |
| 1(HR), 3(R) | Resistant    | 0     |
| 5(MR)       | Moderate     | 1     |
| 7(S), 9(HS) | Susceptible  | 2     |

## Experiments

- Baseline: SqueezeNet 1.1 (ImageNet pretrained)
- Proposed: SqueezeNet + CBAM
- Comparisons: MobileNetV2, EfficientNet-B0, ShuffleNetV2

## W&B Project

- Entity: tklwin_msds
- Project: chickpea
