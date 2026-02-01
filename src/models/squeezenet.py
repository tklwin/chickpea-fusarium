"""
SqueezeNet models for Fusarium detection.

- SqueezeNet: Baseline SqueezeNet 1.1 with custom classifier
- SqueezeNetCBAM: SqueezeNet 1.1 with CBAM attention modules

Reference: "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters"
https://arxiv.org/abs/1602.07360
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional

from .cbam import CBAM


class SqueezeNet(nn.Module):
    """
    Baseline SqueezeNet 1.1 for chickpea disease classification.
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout probability before final classifier
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Load pretrained SqueezeNet 1.1
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        squeezenet = models.squeezenet1_1(weights=weights)
        
        # Extract features (everything except classifier)
        self.features = squeezenet.features
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.num_classes = num_classes
        
        # Initialize new layers
        self._init_classifier()
        
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), -1)  # Flatten
        return x
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classifier (for visualization/analysis)."""
        return self.features(x)


class SqueezeNetCBAM(nn.Module):
    """
    SqueezeNet 1.1 with CBAM attention modules.
    
    CBAM is inserted after key feature extraction stages to help the model
    focus on disease-relevant regions.
    
    Architecture:
    - features[0:5]: Early features + CBAM (128 channels)
    - features[5:8]: Mid features + CBAM (256 channels)
    - features[8:]:  Late features + CBAM (512 channels)
    
    Args:
        num_classes: Number of output classes
        pretrained: Whether to use ImageNet pretrained weights
        dropout: Dropout probability before final classifier
        cbam_reduction: Reduction ratio for CBAM channel attention
        cbam_kernel_size: Kernel size for CBAM spatial attention
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.5,
        cbam_reduction: int = 16,
        cbam_kernel_size: int = 7
    ):
        super().__init__()
        
        # Load pretrained SqueezeNet 1.1
        weights = models.SqueezeNet1_1_Weights.IMAGENET1K_V1 if pretrained else None
        squeezenet = models.squeezenet1_1(weights=weights)
        
        # Split features into stages
        features = list(squeezenet.features.children())
        
        # Stage 1: Conv -> MaxPool -> Fire2 -> Fire3 (output: 128 channels)
        self.stage1 = nn.Sequential(*features[:5])
        self.cbam1 = CBAM(128, reduction=cbam_reduction, kernel_size=cbam_kernel_size)
        
        # Stage 2: MaxPool -> Fire4 -> Fire5 (output: 256 channels)
        self.stage2 = nn.Sequential(*features[5:8])
        self.cbam2 = CBAM(256, reduction=cbam_reduction, kernel_size=cbam_kernel_size)
        
        # Stage 3: MaxPool -> Fire6 -> Fire7 -> Fire8 -> Fire9 (output: 512 channels)
        self.stage3 = nn.Sequential(*features[8:])
        self.cbam3 = CBAM(512, reduction=cbam_reduction, kernel_size=cbam_kernel_size)
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Conv2d(512, num_classes, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.num_classes = num_classes
        
        # Initialize new layers
        self._init_classifier()
        
    def _init_classifier(self):
        """Initialize classifier weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Stage 1 + CBAM
        x = self.stage1(x)
        x = self.cbam1(x)
        
        # Stage 2 + CBAM
        x = self.stage2(x)
        x = self.cbam2(x)
        
        # Stage 3 + CBAM
        x = self.stage3(x)
        x = self.cbam3(x)
        
        # Classifier
        x = self.classifier(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        return x
    
    def get_features(self, x: torch.Tensor, return_attention: bool = False):
        """
        Extract features at each stage (for visualization/analysis).
        
        Args:
            x: Input tensor
            return_attention: If True, also return attention maps
        
        Returns:
            Dict of features (and optionally attention maps)
        """
        features = {}
        
        # Stage 1
        f1 = self.stage1(x)
        f1_att = self.cbam1(f1)
        features['stage1'] = f1_att
        
        # Stage 2
        f2 = self.stage2(f1_att)
        f2_att = self.cbam2(f2)
        features['stage2'] = f2_att
        
        # Stage 3
        f3 = self.stage3(f2_att)
        f3_att = self.cbam3(f3)
        features['stage3'] = f3_att
        
        return features


def count_parameters(model: nn.Module) -> dict:
    """
    Count model parameters.
    
    Returns:
        Dict with total, trainable, and non-trainable parameter counts
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'non_trainable': total - trainable,
        'total_mb': total * 4 / (1024 ** 2),  # Assuming float32
    }


if __name__ == "__main__":
    # Quick test
    print("Testing SqueezeNet models...")
    
    # Test baseline
    model = SqueezeNet(num_classes=3, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"SqueezeNet output shape: {out.shape}")
    print(f"SqueezeNet params: {count_parameters(model)}")
    
    # Test CBAM variant
    model_cbam = SqueezeNetCBAM(num_classes=3, pretrained=False)
    out_cbam = model_cbam(x)
    print(f"SqueezeNet+CBAM output shape: {out_cbam.shape}")
    print(f"SqueezeNet+CBAM params: {count_parameters(model_cbam)}")
