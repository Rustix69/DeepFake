"""
DeepfakeDetector - Main Model
Hybrid CNN-Transformer architecture for deepfake detection
using localized physiological signal inconsistency
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

try:
    from .cnn_encoder import CNNEncoder, TemporalCNN
    from .transformer_module import CrossRegionTransformer, ConsistencyAnalyzer
    from .fusion_module import SpatioTemporalFusion
except ImportError:
    from cnn_encoder import CNNEncoder, TemporalCNN
    from transformer_module import CrossRegionTransformer, ConsistencyAnalyzer
    from fusion_module import SpatioTemporalFusion


class DeepfakeDetector(nn.Module):
    """
    Complete deepfake detection model
    
    Architecture:
    1. Temporal CNN: Extract features from rPPG signals (RGB over time)
    2. Cross-Region Transformer: Analyze consistency across facial regions
    3. Consistency Analyzer: Compute pairwise region consistency
    4. Spatio-Temporal Fusion: Combine all features for classification
    
    Designed for maximum accuracy without compromises
    """
    
    def __init__(
        self,
        # Temporal CNN params
        sequence_length: int = 150,
        input_channels: int = 3,
        num_regions: int = 7,
        
        # Feature dimensions
        temporal_feature_dim: int = 256,
        transformer_dim: int = 256,
        
        # Transformer params
        num_transformer_layers: int = 4,
        num_heads: int = 8,
        
        # Fusion params
        fusion_hidden_dim: int = 512,
        handcrafted_dim: int = 49,
        use_handcrafted: bool = True,
        
        # Regularization
        dropout: float = 0.3,
        
        # Output
        num_classes: int = 2
    ):
        """
        Initialize DeepfakeDetector
        
        Args:
            sequence_length: Number of frames in input sequence
            input_channels: Number of input channels (3 for RGB)
            num_regions: Number of facial regions (7 ROIs)
            temporal_feature_dim: Dimension of temporal features
            transformer_dim: Dimension of transformer features
            num_transformer_layers: Number of transformer layers
            num_heads: Number of attention heads
            fusion_hidden_dim: Hidden dimension for fusion module
            handcrafted_dim: Dimension of hand-crafted features
            use_handcrafted: Whether to use hand-crafted features
            dropout: Dropout probability
            num_classes: Number of output classes (2 for binary)
        """
        super(DeepfakeDetector, self).__init__()
        
        self.sequence_length = sequence_length
        self.input_channels = input_channels
        self.num_regions = num_regions
        self.use_handcrafted = use_handcrafted
        
        # 1. Temporal CNN for rPPG signal processing
        self.temporal_cnn = TemporalCNN(
            input_length=sequence_length,
            input_channels=input_channels,
            feature_dim=temporal_feature_dim
        )
        
        # 2. Cross-Region Transformer
        self.transformer = CrossRegionTransformer(
            feature_dim=transformer_dim,
            num_regions=num_regions,
            num_layers=num_transformer_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_cls_token=True
        )
        
        # Feature projection if dimensions differ
        if temporal_feature_dim != transformer_dim:
            self.feature_proj = nn.Linear(temporal_feature_dim, transformer_dim)
        else:
            self.feature_proj = nn.Identity()
        
        # 3. Consistency Analyzer
        self.consistency_analyzer = ConsistencyAnalyzer(
            feature_dim=transformer_dim,
            num_regions=num_regions
        )
        
        # 4. Spatio-Temporal Fusion
        consistency_dim = num_regions * (num_regions - 1) // 2  # Number of pairs
        self.fusion = SpatioTemporalFusion(
            spatial_dim=transformer_dim,
            temporal_dim=temporal_feature_dim,
            consistency_dim=consistency_dim,
            handcrafted_dim=handcrafted_dim if use_handcrafted else 0,
            hidden_dim=fusion_hidden_dim,
            output_dim=num_classes,
            use_handcrafted=use_handcrafted,
            dropout=dropout
        )
    
    def forward(
        self,
        rppg_signals: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            rppg_signals: rPPG signals of shape (B, ROI, C, T)
                         B = batch size
                         ROI = number of regions (7)
                         C = channels (3 for RGB)
                         T = sequence length (150 frames)
            handcrafted_features: Optional hand-crafted features (B, feature_dim)
            return_attention: Whether to return attention maps
        
        Returns:
            Dictionary containing:
                - logits: Classification logits (B, num_classes)
                - probabilities: Class probabilities (B, num_classes)
                - attention_weights: Attention weights (if requested)
                - fusion_weights: Fusion module attention weights
        """
        batch_size = rppg_signals.shape[0]
        
        # 1. Extract temporal features from each ROI
        temporal_features = self.temporal_cnn(rppg_signals)  # (B, ROI, temporal_dim)
        
        # Average temporal features across regions for global representation
        temporal_global = temporal_features.mean(dim=1)  # (B, temporal_dim)
        
        # 2. Project to transformer dimension
        transformer_input = self.feature_proj(temporal_features)  # (B, ROI, transformer_dim)
        
        # 3. Apply Cross-Region Transformer
        if return_attention:
            spatial_features, attention_maps = self.transformer(
                transformer_input,
                return_attention=True
            )
        else:
            spatial_features = self.transformer(transformer_input)
            attention_maps = None
        # spatial_features: (B, transformer_dim) [CLS token output]
        
        # 4. Compute consistency features
        consistency_features = self.consistency_analyzer(transformer_input)
        # consistency_features: (B, num_pairs)
        
        # 5. Fuse all features
        fusion_output = self.fusion(
            spatial_features=spatial_features,
            temporal_features=temporal_global,
            consistency_features=consistency_features,
            handcrafted_features=handcrafted_features if self.use_handcrafted else None
        )
        
        logits = fusion_output['logits']
        probabilities = torch.softmax(logits, dim=-1)
        
        # Prepare output
        output = {
            'logits': logits,
            'probabilities': probabilities,
            'predictions': torch.argmax(probabilities, dim=-1),
            'confidence': torch.max(probabilities, dim=-1)[0],
            'fusion_weights': fusion_output['attention_weights']
        }
        
        if return_attention:
            output['attention_maps'] = attention_maps
        
        return output
    
    def predict(
        self,
        rppg_signals: torch.Tensor,
        handcrafted_features: Optional[torch.Tensor] = None,
        threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict whether videos are real or fake
        
        Args:
            rppg_signals: Input rPPG signals
            handcrafted_features: Optional hand-crafted features
            threshold: Classification threshold (default: 0.5)
        
        Returns:
            predictions: Binary predictions (0=real, 1=fake)
            confidences: Prediction confidences
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(rppg_signals, handcrafted_features)
            fake_prob = output['probabilities'][:, 1]  # Probability of fake
            predictions = (fake_prob > threshold).long()
            confidences = torch.where(
                predictions == 1,
                fake_prob,
                1 - fake_prob
            )
        
        return predictions, confidences


if __name__ == "__main__":
    print("\n" + "="*70)
    print("DEEPFAKE DETECTOR - COMPLETE MODEL TESTING")
    print("="*70 + "\n")
    
    # Initialize model
    print("Initializing DeepfakeDetector...")
    model = DeepfakeDetector(
        sequence_length=150,
        input_channels=3,
        num_regions=7,
        temporal_feature_dim=256,
        transformer_dim=256,
        num_transformer_layers=4,
        num_heads=8,
        fusion_hidden_dim=512,
        handcrafted_dim=49,
        use_handcrafted=True,
        dropout=0.3,
        num_classes=2
    )
    
    print("✅ Model initialized successfully!\n")
    
    # Test forward pass
    print("Testing forward pass...")
    batch_size = 4
    rppg_signals = torch.randn(batch_size, 7, 3, 150)
    handcrafted = torch.randn(batch_size, 49)
    
    output = model(rppg_signals, handcrafted, return_attention=True)
    
    print(f"  Input rPPG signals: {rppg_signals.shape}")
    print(f"  Input handcrafted: {handcrafted.shape}")
    print(f"\n  Output shapes:")
    print(f"    Logits: {output['logits'].shape}")
    print(f"    Probabilities: {output['probabilities'].shape}")
    print(f"    Predictions: {output['predictions'].shape}")
    print(f"    Confidence: {output['confidence'].shape}")
    print(f"    Attention maps: {output['attention_maps'].shape}")
    print(f"    Fusion weights: {list(output['fusion_weights'].keys())}")
    
    # Test prediction
    print("\nTesting prediction method...")
    preds, confs = model.predict(rppg_signals, handcrafted)
    print(f"  Predictions: {preds}")
    print(f"  Confidences: {confs}")
    
    # Count total parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n📊 Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size: ~{total_params * 4 / (1024**2):.2f} MB (float32)")
    
    # Component breakdown
    print(f"\n🔍 Component breakdown:")
    temporal_params = sum(p.numel() for p in model.temporal_cnn.parameters())
    transformer_params = sum(p.numel() for p in model.transformer.parameters())
    consistency_params = sum(p.numel() for p in model.consistency_analyzer.parameters())
    fusion_params = sum(p.numel() for p in model.fusion.parameters())
    
    print(f"  Temporal CNN: {temporal_params:,} ({temporal_params/total_params*100:.1f}%)")
    print(f"  Transformer: {transformer_params:,} ({transformer_params/total_params*100:.1f}%)")
    print(f"  Consistency: {consistency_params:,} ({consistency_params/total_params*100:.1f}%)")
    print(f"  Fusion: {fusion_params:,} ({fusion_params/total_params*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ DeepfakeDetector is ready for training!")
    print("="*70 + "\n")

