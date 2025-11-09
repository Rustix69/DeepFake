"""
CNN Feature Encoder
Uses EfficientNet-B0 for spatial feature extraction from rPPG signals
"""

import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class CNNEncoder(nn.Module):
    """
    CNN encoder for extracting spatial features from rPPG signals
    
    Uses pretrained EfficientNet-B0 as backbone
    Optimized for both accuracy and efficiency
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        feature_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False
    ):
        """
        Initialize CNN Encoder
        
        Args:
            input_channels: Number of input channels (default: 3 for RGB)
            feature_dim: Dimension of output features
            pretrained: Use ImageNet pretrained weights
            freeze_backbone: Freeze backbone weights for fine-tuning
        """
        super(CNNEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # EfficientNet-B0 backbone
        if pretrained:
            weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
            self.backbone = models.efficientnet_b0(weights=weights)
        else:
            self.backbone = models.efficientnet_b0(weights=None)
        
        # Modify first conv layer if input_channels != 3
        if input_channels != 3:
            original_conv = self.backbone.features[0][0]
            self.backbone.features[0][0] = nn.Conv2d(
                input_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
            )
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Get backbone output dimension
        backbone_out_dim = self.backbone.classifier[1].in_features
        
        # Replace classifier with feature projection
        self.backbone.classifier = nn.Identity()
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(backbone_out_dim, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
               or (batch_size, num_rois, channels, height, width)
        
        Returns:
            Features of shape (batch_size, feature_dim) or 
            (batch_size, num_rois, feature_dim)
        """
        # Handle both single and multi-ROI inputs
        if x.dim() == 5:  # (B, ROI, C, H, W)
            batch_size, num_rois = x.shape[:2]
            # Reshape to (B*ROI, C, H, W)
            x = x.view(-1, *x.shape[2:])
            
            # Extract features
            features = self.backbone(x)
            features = self.projection(features)
            
            # Reshape back to (B, ROI, feature_dim)
            features = features.view(batch_size, num_rois, -1)
        else:  # (B, C, H, W)
            features = self.backbone(x)
            features = self.projection(features)
        
        return features


class TemporalCNN(nn.Module):
    """
    Temporal CNN for processing time-series rPPG signals
    Uses 1D convolutions for temporal feature extraction
    """
    
    def __init__(
        self,
        input_length: int = 150,
        input_channels: int = 3,
        feature_dim: int = 256
    ):
        """
        Initialize Temporal CNN
        
        Args:
            input_length: Length of input sequence (number of frames)
            input_channels: Number of input channels (3 for RGB)
            feature_dim: Dimension of output features
        """
        super(TemporalCNN, self).__init__()
        
        self.input_length = input_length
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        
        # Temporal convolution layers
        self.conv1 = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        
        # Calculate output length after pooling
        out_length = input_length // (2 ** 3)  # 3 pooling layers
        
        # Global average pooling + projection
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.projection = nn.Sequential(
            nn.Linear(256, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_rois, input_channels, input_length)
        
        Returns:
            Features of shape (batch_size, num_rois, feature_dim)
        """
        batch_size, num_rois = x.shape[:2]
        
        # Reshape to (B*ROI, C, T)
        x = x.view(-1, self.input_channels, self.input_length)
        
        # Temporal convolutions
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        # Global average pooling
        x = self.gap(x).squeeze(-1)  # (B*ROI, 256)
        
        # Projection
        features = self.projection(x)
        
        # Reshape back to (B, ROI, feature_dim)
        features = features.view(batch_size, num_rois, -1)
        
        return features


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CNN ENCODER MODULE - TESTING")
    print("="*70 + "\n")
    
    # Test CNNEncoder
    print("Testing CNNEncoder (EfficientNet-B0)...")
    encoder = CNNEncoder(input_channels=3, feature_dim=256, pretrained=False)
    
    # Single image
    x_single = torch.randn(4, 3, 224, 224)
    out_single = encoder(x_single)
    print(f"  Single image input: {x_single.shape} -> {out_single.shape}")
    
    # Multi-ROI images
    x_multi = torch.randn(4, 7, 3, 64, 64)  # 7 ROIs
    out_multi = encoder(x_multi)
    print(f"  Multi-ROI input: {x_multi.shape} -> {out_multi.shape}")
    
    # Test TemporalCNN
    print("\nTesting TemporalCNN...")
    temporal = TemporalCNN(input_length=150, input_channels=3, feature_dim=256)
    
    x_temporal = torch.randn(4, 7, 3, 150)  # 4 videos, 7 ROIs, RGB, 150 frames
    out_temporal = temporal(x_temporal)
    print(f"  Temporal input: {x_temporal.shape} -> {out_temporal.shape}")
    
    # Count parameters
    cnn_params = sum(p.numel() for p in encoder.parameters())
    temporal_params = sum(p.numel() for p in temporal.parameters())
    
    print(f"\nParameter counts:")
    print(f"  CNNEncoder: {cnn_params:,} parameters")
    print(f"  TemporalCNN: {temporal_params:,} parameters")
    
    print("\n" + "="*70)
    print("✅ CNN Encoder modules loaded successfully!")
    print("="*70 + "\n")

