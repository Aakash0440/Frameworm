
"""
ViT-GAN: GAN with Vision Transformer discriminator.

Reference: "TransGAN: Two Pure Transformers Can Make One Strong GAN"
           Jiang et al., 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Converts (B, C, H, W) → (B, N, D) where N = (H/P)*(W/P)
    
    Args:
        image_size: Input image size
        patch_size: Size of each patch
        in_channels: Input channels
        embed_dim: Embedding dimension
    """
    
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, \
            f"Image size {image_size} must be divisible by patch size {patch_size}"
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Linear projection of flattened patches
        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Learnable positional embeddings
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim)
        )
        
        # Initialize
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch projection: (B, C, H, W) → (B, D, H/P, W/P)
        x = self.projection(x)
        
        # Flatten spatial: (B, D, H/P, W/P) → (B, N, D)
        x = x.flatten(2).transpose(1, 2)
        
        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embedding
        
        return x


class TransformerEncoderBlock(nn.Module):
    """
    Standard Transformer encoder block with multi-head self-attention.
    
    Pre-LN variant (more stable training than post-LN).
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        attn_dropout: float = 0.0
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attn_dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x


class ViTDiscriminator(nn.Module):
    """
    Vision Transformer discriminator for GAN training.
    
    Uses patch-based ViT architecture to classify real vs fake images.
    Captures long-range spatial dependencies better than CNN discriminators.
    
    Args:
        image_size: Input image size (square)
        patch_size: Patch size (image_size must be divisible)
        in_channels: Input image channels
        embed_dim: Transformer embedding dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        
    Example:
        >>> disc = ViTDiscriminator(image_size=64, patch_size=8)
        >>> x = torch.randn(4, 3, 64, 64)
        >>> logits = disc(x)  # (4, 1)
    """
    
    def __init__(
        self,
        image_size: int = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        embed_dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.patch_embed = PatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        
        self.transformer_blocks = nn.Sequential(*[
            TransformerEncoderBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Binary classification head (real vs fake)
        self.head = nn.Linear(embed_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)
        
        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Use CLS token for classification
        cls_output = x[:, 0]
        
        return self.head(cls_output)  # (B, 1)

from core.registry import register_model

@register_model('vitgan')
class ViTGAN(nn.Module):
    """
    GAN with Vision Transformer discriminator and CNN generator.
    
    Architecture:
    - Generator: Deep CNN (same as DCGAN but larger)
    - Discriminator: Vision Transformer (ViT)
    
    The ViT discriminator is superior to CNN discriminators for
    capturing global coherence and long-range dependencies.
    
    Args:
        config: Model configuration
        
    Config keys:
        model.latent_dim: Generator input noise dimension (default: 128)
        model.image_size: Output image size (default: 64)
        model.image_channels: Output channels (default: 3)
        model.gen_hidden: Generator hidden channels (default: 256)
        model.vit_embed_dim: ViT embedding dim (default: 384)
        model.vit_depth: Number of ViT layers (default: 6)
        model.vit_heads: ViT attention heads (default: 6)
        model.patch_size: ViT patch size (default: 8)
        
    Example:
        >>> model = ViTGAN(config)
        >>> real = torch.randn(4, 3, 64, 64)
        >>> losses = model.compute_loss(real)
        >>> # losses: {'loss': scalar, 'g_loss': g, 'd_loss': d}
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.latent_dim = getattr(config.model, 'latent_dim', 128)
        self.image_size = getattr(config.model, 'image_size', 64)
        self.image_channels = getattr(config.model, 'image_channels', 3)
        gen_hidden = getattr(config.model, 'gen_hidden', 256)
        
        vit_embed_dim = getattr(config.model, 'vit_embed_dim', 384)
        vit_depth = getattr(config.model, 'vit_depth', 6)
        vit_heads = getattr(config.model, 'vit_heads', 6)
        patch_size = getattr(config.model, 'patch_size', 8)
        
        # Generator (CNN)
        init_size = self.image_size // 16
        self.generator = nn.Sequential(
            nn.Linear(self.latent_dim, gen_hidden * 8 * init_size * init_size),
            nn.Unflatten(1, (gen_hidden * 8, init_size, init_size)),
            
            nn.ConvTranspose2d(gen_hidden * 8, gen_hidden * 4, 4, stride=2, padding=1),
            nn.BatchNorm2d(gen_hidden * 4),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(gen_hidden * 4, gen_hidden * 2, 4, stride=2, padding=1),
            nn.BatchNorm2d(gen_hidden * 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(gen_hidden * 2, gen_hidden, 4, stride=2, padding=1),
            nn.BatchNorm2d(gen_hidden),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(gen_hidden, self.image_channels, 4, stride=2, padding=1),
            nn.Tanh()
        )
        
        # Discriminator (ViT)
        self.discriminator = ViTDiscriminator(
            image_size=self.image_size,
            patch_size=patch_size,
            in_channels=self.image_channels,
            embed_dim=vit_embed_dim,
            depth=vit_depth,
            num_heads=vit_heads
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.compute_loss(x)

    def generate(self, batch_size: int, device: str = None) -> torch.Tensor:
        """Generate a batch of images"""
        if device is None:
            device = next(self.generator.parameters()).device
        z = torch.randn(batch_size, self.latent_dim, device=device)
        return self.generator(z)
    
    def compute_loss(
        self,
        real: torch.Tensor,
        y=None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute GAN losses.
        
        Uses non-saturating GAN loss (standard GAN).
        
        Args:
            real: Real images (B, C, H, W)
            
        Returns:
            Dict with 'loss', 'g_loss', 'd_loss'
        """
        if isinstance(real, (tuple, list)):
            real = real[0]
        
        B = real.shape[0]
        device = real.device
        
        # Generate fake images
        z = torch.randn(B, self.latent_dim, device=device)
        fake = self.generator(z)
        
        # ─── Discriminator Loss ───────────────────────────────────
        real_logits = self.discriminator(real)
        fake_logits = self.discriminator(fake.detach())
        
        # Hinge loss for discriminator (more stable than BCE)
        d_loss_real = F.relu(1.0 - real_logits).mean()
        d_loss_fake = F.relu(1.0 + fake_logits).mean()
        d_loss = d_loss_real + d_loss_fake
        
        # ─── Generator Loss ───────────────────────────────────────
        fake_logits_for_g = self.discriminator(fake)
        g_loss = -fake_logits_for_g.mean()
        
        # Combined loss (alternating update handled by Trainer)
        loss = d_loss + g_loss
        
        return {
            'loss': loss,
            'g_loss': g_loss,
            'd_loss': d_loss,
            'd_loss_real': d_loss_real,
            'd_loss_fake': d_loss_fake
        }