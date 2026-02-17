"""
VQ-VAE-2: Hierarchical Vector Quantized Variational Autoencoder.

Reference: "Generating Diverse High-Fidelity Images with VQ-VAE-2"
           Razavi et al., 2019 (https://arxiv.org/abs/1906.00446)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer for VQ-VAE.
    
    Maps continuous encoder outputs to discrete codebook vectors.
    Uses straight-through estimator for gradients.
    
    Args:
        num_embeddings: Codebook size (K)
        embedding_dim: Dimension of each embedding (D)
        commitment_cost: Weight for commitment loss (beta)
        
    Example:
        >>> vq = VectorQuantizer(num_embeddings=512, embedding_dim=64)
        >>> z = torch.randn(8, 64, 16, 16)  # Encoder output
        >>> z_q, loss, indices = vq(z)
        >>> # z_q is quantized, same shape as z
        >>> # loss is VQ + commitment loss
        >>> # indices are codebook indices shape (8, 16, 16)
    """
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        commitment_cost: float = 0.25
    ):
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook embeddings
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        # Initialize uniform
        self.embedding.weight.data.uniform_(
            -1.0 / num_embeddings,
             1.0 / num_embeddings
        )
    
    def forward(
        self,
        z: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize encoder output.
        
        Args:
            z: Encoder output (B, D, H, W)
            
        Returns:
            Tuple of:
            - z_q: Quantized output (B, D, H, W)
            - loss: VQ + commitment loss scalar
            - encoding_indices: Codebook indices (B, H*W)
        """
        # Reshape z to (B*H*W, D) for distance computation
        B, D, H, W = z.shape
        z_flat = z.permute(0, 2, 3, 1).contiguous().view(-1, D)  # (B*H*W, D)
        
        # Compute L2 distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z*e
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)         # ||z||^2
            + self.embedding.weight.pow(2).sum(dim=1)       # ||e||^2
            - 2 * z_flat @ self.embedding.weight.t()        # -2*z*e
        )  # Shape: (B*H*W, K)
        
        # Find nearest codebook entry for each spatial location
        encoding_indices = distances.argmin(dim=1)  # (B*H*W,)
        
        # Quantize: replace z with nearest codebook entry
        z_q = self.embedding(encoding_indices)  # (B*H*W, D)
        z_q = z_q.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        
        # VQ loss: move codebook entries toward encoder output (no grad to encoder)
        vq_loss = F.mse_loss(z_q.detach(), z)
        
        # Commitment loss: encourage encoder to commit to codebook entries
        commitment_loss = F.mse_loss(z_q, z.detach())
        
        # Total VQ loss
        loss = vq_loss + self.commitment_cost * commitment_loss
        
        # Straight-through estimator: copy gradients from z_q to z
        z_q = z + (z_q - z).detach()
        
        return z_q, loss, encoding_indices.view(B, H * W)
    
    def get_codebook_usage(self, indices: torch.Tensor) -> float:
        """Compute fraction of codebook entries used (codebook utilization)"""
        unique = indices.unique().numel()
        return unique / self.num_embeddings


class ResidualBlock(nn.Module):
    """Residual block used in VQ-VAE encoder/decoder"""
    
    def __init__(self, channels: int, hidden_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, hidden_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, channels, 1, bias=False)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VQEncoder(nn.Module):
    """VQ-VAE-2 encoder for one hierarchical level"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        embedding_dim: int,
        num_residual_blocks: int = 2,
        downsample: int = 2
    ):
        super().__init__()
        
        # Downsampling
        layers = [nn.Conv2d(in_channels, hidden_channels, 4, stride=downsample, padding=1)]
        
        # If more than 2x downsampling
        stride = downsample // 2
        if stride > 1:
            layers.insert(0, nn.Conv2d(in_channels, in_channels, 4, stride=stride, padding=1))
            layers.insert(1, nn.ReLU(inplace=True))
            layers[2] = nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1)
        
        layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1))
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_channels, hidden_channels // 2))
        
        layers.append(nn.ReLU(inplace=True))
        
        # Project to embedding dim
        layers.append(nn.Conv2d(hidden_channels, embedding_dim, 1))
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class VQDecoder(nn.Module):
    """VQ-VAE-2 decoder for one hierarchical level"""
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_residual_blocks: int = 2,
        upsample: int = 2
    ):
        super().__init__()
        
        layers = [nn.Conv2d(in_channels, hidden_channels, 3, padding=1)]
        
        # Residual blocks
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_channels, hidden_channels // 2))
        
        layers.append(nn.ReLU(inplace=True))
        
        # Upsampling
        layers.append(nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=upsample, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        if upsample > 2:
            layers.append(nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Conv2d(hidden_channels, out_channels, 3, padding=1))
        
        self.decoder = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class VQVAE2(nn.Module):
    """
    VQ-VAE-2: Hierarchical Vector Quantized VAE.
    
    Uses two levels of quantization:
    - Bottom level: fine-grained local features (high resolution)
    - Top level: coarse global structure (low resolution)
    
    Args:
        config: Model configuration
        
    Config keys:
        model.in_channels: Input image channels (default: 3)
        model.hidden_channels: Feature map channels (default: 128)
        model.embedding_dim: Codebook vector dimension (default: 64)
        model.num_embeddings: Codebook size per level (default: 512)
        model.commitment_cost: VQ commitment loss weight (default: 0.25)
        
    Example:
        >>> config = Config.from_dict({
        ...     'model': {
        ...         'in_channels': 3,
        ...         'hidden_channels': 128,
        ...         'embedding_dim': 64,
        ...         'num_embeddings': 512
        ...     }
        ... })
        >>> model = VQVAE2(config)
        >>> x = torch.randn(4, 3, 256, 256)
        >>> output = model(x)
        >>> # output['recon']: reconstruction
        >>> # output['loss']: total loss
        >>> # output['vq_loss']: quantization loss
    """
    
    def __init__(self, config):
        super().__init__()
        
        in_ch = getattr(config.model, 'in_channels', 3)
        hidden_ch = getattr(config.model, 'hidden_channels', 128)
        emb_dim = getattr(config.model, 'embedding_dim', 64)
        num_emb = getattr(config.model, 'num_embeddings', 512)
        commitment = getattr(config.model, 'commitment_cost', 0.25)
        
        # Bottom encoder (4x downsampling → 64x64 for 256x256 input)
        self.enc_bottom = VQEncoder(in_ch, hidden_ch, emb_dim, downsample=4)
        
        # Top encoder (additional 4x downsampling → 16x16)
        self.enc_top = VQEncoder(emb_dim, hidden_ch, emb_dim, downsample=4)
        
        # Vector quantizers
        self.vq_top = VectorQuantizer(num_emb, emb_dim, commitment)
        self.vq_bottom = VectorQuantizer(num_emb, emb_dim, commitment)
        
        # Upsample top to bottom level for decoder input
        self.top_to_bottom = nn.Sequential(
            nn.ConvTranspose2d(emb_dim, emb_dim, 4, stride=4, padding=0),
            nn.ReLU(inplace=True)
        )
        
        # Bottom decoder input: bottom + upsampled top
        self.dec_bottom = VQDecoder(emb_dim * 2, hidden_ch, in_ch, upsample=4)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode image to quantized latent codes"""
        z_bottom = self.enc_bottom(x)
        z_top = self.enc_top(z_bottom)
        
        z_q_top, loss_top, idx_top = self.vq_top(z_top)
        z_q_bottom, loss_bottom, idx_bottom = self.vq_bottom(z_bottom)
        
        return z_q_top, z_q_bottom, loss_top + loss_bottom
    
    def decode(self, z_q_top: torch.Tensor, z_q_bottom: torch.Tensor) -> torch.Tensor:
        """Decode quantized codes to image"""
        # Upsample top-level codes to bottom resolution
        top_upsampled = self.top_to_bottom(z_q_top)
        
        # Concatenate along channel dimension
        dec_input = torch.cat([z_q_bottom, top_upsampled], dim=1)
        
        return torch.tanh(self.dec_bottom(dec_input))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        z_q_top, z_q_bottom, vq_loss = self.encode(x)
        recon = self.decode(z_q_top, z_q_bottom)
        
        recon_loss = F.mse_loss(recon, x)
        
        return {
            'recon': recon,
            'recon_loss': recon_loss,
            'vq_loss': vq_loss,
            'loss': recon_loss + vq_loss
        }
    
    def compute_loss(self, x: torch.Tensor, y=None) -> Dict[str, torch.Tensor]:
        """Training interface for Trainer compatibility"""
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self(x)
    
    @torch.no_grad()
    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruct images without computing gradients"""
        return self(x)['recon']