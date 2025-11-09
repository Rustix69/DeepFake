"""
Cross-Region Transformer Module
Uses self-attention to analyze consistency across facial regions
KEY for deepfake detection based on physiological signal inconsistency
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    Allows model to jointly attend to information from different representation subspaces
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        """
        Initialize multi-head attention
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadSelfAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch_size, num_tokens, embed_dim)
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor of same shape as input
            Optionally attention weights if return_attention=True
        """
        B, N, C = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, N, D)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        
        # Output projection
        x = self.proj(x)
        x = self.proj_drop(x)
        
        if return_attention:
            return x, attn
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head attention and feed-forward network
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        """
        Initialize transformer block
        
        Args:
            embed_dim: Dimension of embeddings
            num_heads: Number of attention heads
            mlp_ratio: Ratio of MLP hidden dim to embedding dim
            dropout: Dropout probability
        """
        super(TransformerBlock, self).__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass with residual connections
        
        Args:
            x: Input tensor
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor
        """
        # Self-attention with residual
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + attn_out
        else:
            x = x + self.attn(self.norm1(x))
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        if return_attention:
            return x, attn_weights
        return x


class CrossRegionTransformer(nn.Module):
    """
    Cross-Region Transformer for analyzing physiological signal consistency
    
    Uses self-attention to model relationships between different facial regions
    Key innovation: Detects inconsistencies in fake videos
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_regions: int = 7,
        num_layers: int = 4,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        use_cls_token: bool = True
    ):
        """
        Initialize Cross-Region Transformer
        
        Args:
            feature_dim: Dimension of input features
            num_regions: Number of facial regions (ROIs)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout probability
            use_cls_token: Whether to use a CLS token for classification
        """
        super(CrossRegionTransformer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_regions = num_regions
        self.num_layers = num_layers
        self.use_cls_token = use_cls_token
        
        # CLS token (learnable)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Position embeddings for each region
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_regions + (1 if use_cls_token else 0), feature_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        self.pos_drop = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(feature_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(feature_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features of shape (batch_size, num_regions, feature_dim)
            return_attention: Whether to return attention weights from last layer
        
        Returns:
            If use_cls_token: (batch_size, feature_dim)
            Otherwise: (batch_size, num_regions, feature_dim)
        """
        B = x.shape[0]
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Apply transformer blocks
        attention_weights = None
        for i, block in enumerate(self.blocks):
            if return_attention and i == len(self.blocks) - 1:
                x, attention_weights = block(x, return_attention=True)
            else:
                x = block(x)
        
        # Layer norm
        x = self.norm(x)
        
        # Return CLS token or all tokens
        if self.use_cls_token:
            output = x[:, 0]  # (B, feature_dim)
        else:
            output = x  # (B, num_regions, feature_dim)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention maps from the last layer
        Useful for visualization and interpretation
        
        Args:
            x: Input features (batch_size, num_regions, feature_dim)
        
        Returns:
            Attention weights (batch_size, num_heads, num_tokens, num_tokens)
        """
        _, attention_weights = self.forward(x, return_attention=True)
        return attention_weights


class ConsistencyAnalyzer(nn.Module):
    """
    Analyzes cross-region consistency using transformer features
    Computes consistency metrics that help detect deepfakes
    """
    
    def __init__(
        self,
        feature_dim: int = 256,
        num_regions: int = 7
    ):
        """
        Initialize Consistency Analyzer
        
        Args:
            feature_dim: Dimension of features
            num_regions: Number of regions
        """
        super(ConsistencyAnalyzer, self).__init__()
        
        self.feature_dim = feature_dim
        self.num_regions = num_regions
        
        # Pairwise comparison network
        self.comparison_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise consistency scores
        
        Args:
            x: Region features (batch_size, num_regions, feature_dim)
        
        Returns:
            Consistency scores (batch_size, num_pairs)
        """
        B = x.shape[0]
        consistency_scores = []
        
        # Compute pairwise comparisons
        for i in range(self.num_regions):
            for j in range(i + 1, self.num_regions):
                # Concatenate pairs
                pair = torch.cat([x[:, i], x[:, j]], dim=-1)
                # Compute consistency score
                score = self.comparison_net(pair)
                consistency_scores.append(score)
        
        # Stack all scores
        consistency_scores = torch.cat(consistency_scores, dim=-1)
        
        return consistency_scores


if __name__ == "__main__":
    print("\n" + "="*70)
    print("CROSS-REGION TRANSFORMER MODULE - TESTING")
    print("="*70 + "\n")
    
    # Test CrossRegionTransformer
    print("Testing CrossRegionTransformer...")
    transformer = CrossRegionTransformer(
        feature_dim=256,
        num_regions=7,
        num_layers=4,
        num_heads=8,
        use_cls_token=True
    )
    
    x = torch.randn(4, 7, 256)  # 4 videos, 7 ROIs, 256 features
    out = transformer(x)
    print(f"  Input: {x.shape} -> Output: {out.shape}")
    
    # Test attention maps
    _, attn = transformer(x, return_attention=True)
    print(f"  Attention shape: {attn.shape}")
    
    # Test without CLS token
    print("\nTesting without CLS token...")
    transformer_no_cls = CrossRegionTransformer(
        feature_dim=256,
        num_regions=7,
        use_cls_token=False
    )
    out_no_cls = transformer_no_cls(x)
    print(f"  Input: {x.shape} -> Output: {out_no_cls.shape}")
    
    # Test ConsistencyAnalyzer
    print("\nTesting ConsistencyAnalyzer...")
    analyzer = ConsistencyAnalyzer(feature_dim=256, num_regions=7)
    consistency = analyzer(x)
    num_pairs = 7 * 6 // 2  # n*(n-1)/2
    print(f"  Input: {x.shape} -> Consistency scores: {consistency.shape}")
    print(f"  Number of pairwise comparisons: {num_pairs}")
    
    # Count parameters
    transformer_params = sum(p.numel() for p in transformer.parameters())
    analyzer_params = sum(p.numel() for p in analyzer.parameters())
    
    print(f"\nParameter counts:")
    print(f"  CrossRegionTransformer: {transformer_params:,} parameters")
    print(f"  ConsistencyAnalyzer: {analyzer_params:,} parameters")
    
    print("\n" + "="*70)
    print("✅ Transformer modules loaded successfully!")
    print("="*70 + "\n")

