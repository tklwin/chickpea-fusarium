"""
CBAM: Convolutional Block Attention Module

Paper: "CBAM: Convolutional Block Attention Module" (ECCV 2018)
https://arxiv.org/abs/1807.06521

CBAM applies attention sequentially:
1. Channel Attention: "what" to focus on
2. Spatial Attention: "where" to focus
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module.
    
    Uses both max-pooling and average-pooling to capture different aspects,
    then combines them through a shared MLP.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for the MLP
    """
    
    def __init__(self, in_channels: int, reduction: int = 16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        reduced_channels = max(in_channels // reduction, 8)  # minimum 8 channels
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1, bias=False),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Channel attention weights of shape (B, C, 1, 1)
        """
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        
        attention = torch.sigmoid(avg_out + max_out)
        return attention


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    Uses channel-wise max and average pooling followed by convolution
    to generate spatial attention map.
    
    Args:
        kernel_size: Convolution kernel size (default: 7)
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_channels=2,  # avg + max pooled
            out_channels=1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Spatial attention weights of shape (B, 1, H, W)
        """
        # Channel-wise pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        
        # Concatenate and convolve
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = torch.sigmoid(self.conv(concat))
        
        return attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module.
    
    Applies channel attention followed by spatial attention.
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention convolution
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Attention-refined tensor of shape (B, C, H, W)
        """
        # Channel attention
        x = x * self.channel_attention(x)
        
        # Spatial attention
        x = x * self.spatial_attention(x)
        
        return x


class CBAMBlock(nn.Module):
    """
    CBAM with residual connection.
    
    output = x + CBAM(x)
    
    Args:
        in_channels: Number of input channels
        reduction: Channel reduction ratio
        kernel_size: Kernel size for spatial attention
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ):
        super().__init__()
        self.cbam = CBAM(in_channels, reduction, kernel_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.cbam(x)
