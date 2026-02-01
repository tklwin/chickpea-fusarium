"""
Model factory for creating different architectures.

Supports:
- SqueezeNet 1.1 (baseline)
- SqueezeNet 1.1 + CBAM (proposed)
- MobileNetV2 (comparison)
- EfficientNet-B0 (comparison)
- ShuffleNetV2 (comparison)
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from .squeezenet import SqueezeNet, SqueezeNetCBAM, count_parameters


# Registry of available models
MODEL_REGISTRY = {
    "squeezenet1_1": "SqueezeNet 1.1 (baseline)",
    "squeezenet1_1_cbam": "SqueezeNet 1.1 + CBAM (proposed)",
    "mobilenetv2": "MobileNetV2",
    "mobilenetv3_small": "MobileNetV3-Small",
    "efficientnet_b0": "EfficientNet-B0",
    "shufflenetv2": "ShuffleNetV2 x1.0",
}


def get_model(
    model_name: str,
    num_classes: int = 3,
    pretrained: bool = True,
    dropout: float = 0.5,
    cbam_reduction: int = 16,
    cbam_kernel_size: int = 7,
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_name: Name of the model architecture
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout probability
        cbam_reduction: CBAM reduction ratio (for SqueezeNet+CBAM)
        cbam_kernel_size: CBAM spatial kernel size
    
    Returns:
        PyTorch model
    """
    model_name = model_name.lower()
    
    if model_name == "squeezenet1_1":
        model = SqueezeNet(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        
    elif model_name == "squeezenet1_1_cbam":
        model = SqueezeNetCBAM(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout,
            cbam_reduction=cbam_reduction,
            cbam_kernel_size=cbam_kernel_size
        )
        
    elif model_name == "mobilenetv2":
        weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.last_channel, num_classes)
        )
        
    elif model_name == "mobilenetv3_small":
        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier[0].in_features, 256),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes)
        )
        
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.classifier[1].in_features, num_classes)
        )
        
    elif model_name == "shufflenetv2" or model_name == "shufflenetv2_x1_0":
        weights = models.ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.shufflenet_v2_x1_0(weights=weights)
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(model.fc.in_features, num_classes)
        )
        
    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )
    
    return model


def list_available_models() -> dict:
    """List all available models with descriptions."""
    return MODEL_REGISTRY.copy()


def compare_models(num_classes: int = 3) -> None:
    """
    Print comparison table of all available models.
    
    Shows:
    - Parameter count
    - Model size (MB)
    - Inference speed (relative)
    """
    print("\n" + "=" * 70)
    print("Model Comparison (Lightweight CNNs for Mobile Deployment)")
    print("=" * 70)
    print(f"{'Model':<25} {'Params (M)':<12} {'Size (MB)':<12} {'Notes'}")
    print("-" * 70)
    
    for model_name in MODEL_REGISTRY.keys():
        try:
            model = get_model(model_name, num_classes=num_classes, pretrained=False)
            stats = count_parameters(model)
            
            params_m = stats['total'] / 1e6
            size_mb = stats['total_mb']
            
            print(f"{model_name:<25} {params_m:<12.2f} {size_mb:<12.2f}")
            
            del model
        except Exception as e:
            print(f"{model_name:<25} Error: {e}")
    
    print("=" * 70)


if __name__ == "__main__":
    compare_models()
