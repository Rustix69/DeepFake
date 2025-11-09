"""
Spatio-Temporal Fusion Module
Combines spatial, temporal, and consistency features for final classification
"""

import torch
import torch.nn as nn
from typing import Dict, Optional


class SpatioTemporalFusion(nn.Module):
    """
    Fuses multiple feature streams for deepfake detection
    
    Combines:
    1. Spatial features (from CNN/Transformer)
    2. Temporal features (from Temporal CNN)
    3. Consistency features (from ConsistencyAnalyzer)
    4. Hand-crafted features (optional)
    """
    
    def __init__(
        self,
        spatial_dim: int = 256,
        temporal_dim: int = 256,
        consistency_dim: int = 21,  # 7*6/2 = 21 pairs
        handcrafted_dim: int = 49,  # From feature extraction
        hidden_dim: int = 512,
        output_dim: int = 2,  # Binary classification
        use_handcrafted: bool = True,
        dropout: float = 0.3
    ):
        """
        Initialize Spatio-Temporal Fusion
        
        Args:
            spatial_dim: Dimension of spatial features
            temporal_dim: Dimension of temporal features
            consistency_dim: Dimension of consistency features
            handcrafted_dim: Dimension of hand-crafted features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (2 for binary classification)
            use_handcrafted: Whether to use hand-crafted features
            dropout: Dropout probability
        """
        super(SpatioTemporalFusion, self).__init__()
        
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.consistency_dim = consistency_dim
        self.handcrafted_dim = handcrafted_dim if use_handcrafted else 0
        self.use_handcrafted = use_handcrafted
        
        # Calculate total input dimension
        total_dim = spatial_dim + temporal_dim + consistency_dim
        if use_handcrafted:
            total_dim += handcrafted_dim
        
        # Attention-based fusion
        self.spatial_attn = nn.Sequential(
            nn.Linear(spatial_dim, spatial_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(spatial_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.temporal_attn = nn.Sequential(
            nn.Linear(temporal_dim, temporal_dim // 4),
            nn.ReLU(inplace=True),
            nn.Linear(temporal_dim // 4, 1),
            nn.Sigmoid()
        )
        
        self.consistency_attn = nn.Sequential(
            nn.Linear(consistency_dim, consistency_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(consistency_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(
        self,
        spatial_features: torch.Tensor,
        temporal_features: torch.Tensor,
        consistency_features: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            spatial_features: (batch_size, spatial_dim)
            temporal_features: (batch_size, temporal_dim)
            consistency_features: (batch_size, consistency_dim)
            handcrafted_features: (batch_size, handcrafted_dim) [optional]
        
        Returns:
            Dictionary with:
                - logits: (batch_size, output_dim)
                - attention_weights: Dictionary of attention weights
        """
        # Compute attention weights
        spatial_weight = self.spatial_attn(spatial_features)
        temporal_weight = self.temporal_attn(temporal_features)
        consistency_weight = self.consistency_attn(consistency_features)
        
        # Apply attention
        spatial_weighted = spatial_features * spatial_weight
        temporal_weighted = temporal_features * temporal_weight
        consistency_weighted = consistency_features * consistency_weight
        
        # Concatenate features
        if self.use_handcrafted and handcrafted_features is not None:
            fused = torch.cat([
                spatial_weighted,
                temporal_weighted,
                consistency_weighted,
                handcrafted_features
            ], dim=-1)
        else:
            fused = torch.cat([
                spatial_weighted,
                temporal_weighted,
                consistency_weighted
            ], dim=-1)
        
        # Final classification
        logits = self.fusion(fused)
        
        return {
            'logits': logits,
            'attention_weights': {
                'spatial': spatial_weight,
                'temporal': temporal_weight,
                'consistency': consistency_weight
            }
        }


class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion that learns to weight different feature streams
    More flexible than fixed concatenation
    """
    
    def __init__(
        self,
        feature_dims: Dict[str, int],
        hidden_dim: int = 512,
        output_dim: int = 2,
        dropout: float = 0.3
    ):
        """
        Initialize Adaptive Fusion
        
        Args:
            feature_dims: Dictionary mapping feature names to dimensions
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(AdaptiveFusion, self).__init__()
        
        self.feature_dims = feature_dims
        self.feature_names = list(feature_dims.keys())
        
        # Per-feature projections
        self.projections = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            for name, dim in feature_dims.items()
        })
        
        # Adaptive weighting
        total_dim = len(feature_dims) * hidden_dim
        self.weight_net = nn.Sequential(
            nn.Linear(total_dim, len(feature_dims)),
            nn.Softmax(dim=-1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: Dictionary mapping feature names to tensors
        
        Returns:
            Dictionary with logits and weights
        """
        # Project all features
        projected = {}
        for name in self.feature_names:
            projected[name] = self.projections[name](features[name])
        
        # Concatenate for weighting
        all_projected = torch.stack([projected[name] for name in self.feature_names], dim=1)
        B, N, D = all_projected.shape
        
        # Compute adaptive weights
        all_flat = all_projected.view(B, -1)
        weights = self.weight_net(all_flat)  # (B, N)
        
        # Apply weights
        weighted = (all_projected * weights.unsqueeze(-1)).sum(dim=1)
        
        # Classification
        logits = self.classifier(weighted)
        
        # Create weight dictionary
        weight_dict = {
            name: weights[:, i]
            for i, name in enumerate(self.feature_names)
        }
        
        return {
            'logits': logits,
            'weights': weight_dict
        }


if __name__ == "__main__":
    print("\n" + "="*70)
    print("SPATIO-TEMPORAL FUSION MODULE - TESTING")
    print("="*70 + "\n")
    
    # Test SpatioTemporalFusion
    print("Testing SpatioTemporalFusion...")
    fusion = SpatioTemporalFusion(
        spatial_dim=256,
        temporal_dim=256,
        consistency_dim=21,
        handcrafted_dim=49,
        hidden_dim=512,
        use_handcrafted=True
    )
    
    spatial = torch.randn(4, 256)
    temporal = torch.randn(4, 256)
    consistency = torch.randn(4, 21)
    handcrafted = torch.randn(4, 49)
    
    output = fusion(spatial, temporal, consistency, handcrafted)
    print(f"  Spatial: {spatial.shape}")
    print(f"  Temporal: {temporal.shape}")
    print(f"  Consistency: {consistency.shape}")
    print(f"  Handcrafted: {handcrafted.shape}")
    print(f"  Output logits: {output['logits'].shape}")
    print(f"  Attention weights keys: {list(output['attention_weights'].keys())}")
    
    # Test AdaptiveFusion
    print("\nTesting AdaptiveFusion...")
    feature_dims = {
        'spatial': 256,
        'temporal': 256,
        'consistency': 21,
        'handcrafted': 49
    }
    adaptive_fusion = AdaptiveFusion(feature_dims, hidden_dim=512)
    
    features = {
        'spatial': spatial,
        'temporal': temporal,
        'consistency': consistency,
        'handcrafted': handcrafted
    }
    
    output_adaptive = adaptive_fusion(features)
    print(f"  Output logits: {output_adaptive['logits'].shape}")
    print(f"  Learned weights: {list(output_adaptive['weights'].keys())}")
    
    # Count parameters
    fusion_params = sum(p.numel() for p in fusion.parameters())
    adaptive_params = sum(p.numel() for p in adaptive_fusion.parameters())
    
    print(f"\nParameter counts:")
    print(f"  SpatioTemporalFusion: {fusion_params:,} parameters")
    print(f"  AdaptiveFusion: {adaptive_params:,} parameters")
    
    print("\n" + "="*70)
    print("✅ Fusion modules loaded successfully!")
    print("="*70 + "\n")

