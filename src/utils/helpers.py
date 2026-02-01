"""
Utility functions.
"""

import os
import random
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For CUDA determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Returns:
        Dict with total, trainable, non-trainable counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'size_mb': total * 4 / (1024 ** 2),  # Assuming float32
    }


def save_model_for_mobile(
    model: nn.Module,
    save_path: str,
    input_shape: tuple = (1, 3, 224, 224),
    format: str = "torchscript"
):
    """
    Export model for mobile deployment.
    
    Args:
        model: Trained PyTorch model
        save_path: Path to save exported model
        input_shape: Input tensor shape
        format: Export format ('torchscript' or 'onnx')
    """
    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(input_shape).to(device)
    
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "torchscript":
        # TorchScript for PyTorch Mobile
        traced_model = torch.jit.trace(model, dummy_input)
        traced_model.save(str(save_path))
        print(f"TorchScript model saved to: {save_path}")
        
        # Get file size
        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        print(f"Model size: {size_mb:.2f} MB")
        
    elif format == "onnx":
        # ONNX for cross-platform deployment
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        print(f"ONNX model saved to: {save_path}")
        
        size_mb = os.path.getsize(save_path) / (1024 ** 2)
        print(f"Model size: {size_mb:.2f} MB")
        
    else:
        raise ValueError(f"Unknown format: {format}")


def get_model_summary(model: nn.Module, input_shape: tuple = (1, 3, 224, 224)) -> str:
    """
    Get model summary string.
    
    Args:
        model: PyTorch model
        input_shape: Input tensor shape
    
    Returns:
        Summary string
    """
    from io import StringIO
    
    summary_str = StringIO()
    
    summary_str.write("=" * 60 + "\n")
    summary_str.write("Model Summary\n")
    summary_str.write("=" * 60 + "\n")
    
    # Parameter count
    params = count_parameters(model)
    summary_str.write(f"Total parameters: {params['total']:,}\n")
    summary_str.write(f"Trainable parameters: {params['trainable']:,}\n")
    summary_str.write(f"Model size: {params['size_mb']:.2f} MB\n")
    summary_str.write("=" * 60 + "\n")
    
    return summary_str.getvalue()


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
