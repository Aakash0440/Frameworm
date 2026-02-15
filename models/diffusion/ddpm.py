"""
DDPM (Denoising Diffusion Probabilistic Model) implementation.

Based on: "Denoising Diffusion Probabilistic Models" (Ho et al., 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models import BaseModel
from core import register_model
from typing import Optional


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.
    
    Args:
        timesteps: Timestep indices (B,)
        embedding_dim: Embedding dimension
        
    Returns:
        Embeddings (B, embedding_dim)
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    
    if embedding_dim % 2 == 1:  # Zero pad if odd
        emb = F.pad(emb, (0, 1))
    
    return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input (B, C, H, W)
            time_emb: Time embedding (B, time_emb_dim)
            
        Returns:
            Output (B, out_channels, H, W)
        """
        h = self.conv1(F.relu(x))
        
        # Add time embedding
        time_emb = self.time_mlp(F.relu(time_emb))
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(F.relu(h))
        
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net for noise prediction"""
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        time_emb_dim: int = 128
    ):
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
        )
        
        # Encoder (downsampling)
        self.down1 = ResidualBlock(in_channels, base_channels, time_emb_dim)
        self.down2 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        self.pool = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        # Decoder (upsampling)
        self.up1 = ResidualBlock(base_channels * 8, base_channels * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)
        self.up3 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)
        
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        # Output
        self.out = nn.Conv2d(base_channels, in_channels, 1)
    
    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """
        Predict noise.
        
        Args:
            x: Noisy input (B, C, H, W)
            timesteps: Timestep indices (B,)
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        # Time embedding
        t_emb = get_timestep_embedding(timesteps, 128)
        t_emb = self.time_mlp(t_emb)
        
        # Encoder
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)
        
        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)
        
        # Decoder with skip connections
        u1 = self.up1(torch.cat([self.upsample(b), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.upsample(u1), d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.upsample(u2), d1], dim=1), t_emb)
        
        # Output
        return self.out(u3)


@register_model("ddpm", version="1.0.0", author="Frameworm Team")
class DDPM(BaseModel):
    """
    Denoising Diffusion Probabilistic Model.
    
    Based on "Denoising Diffusion Probabilistic Models" (Ho et al., 2020).
    
    Args:
        config: Configuration with model.timesteps, model.base_channels
    """
    
    def __init__(self, config):
        super().__init__(config)
        
        self.timesteps = config.model.get('timesteps', 1000)
        channels = config.model.get('channels', 3)
        base_channels = config.model.get('base_channels', 128)
        
        # U-Net for noise prediction
        self.model = UNet(
            in_channels=channels,
            base_channels=base_channels,
            time_emb_dim=128
        )
        
        # Register beta schedule
        self.register_buffer('betas', self._get_beta_schedule())
        
        # Precompute alpha values
        alphas = 1.0 - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer(
            'alphas_cumprod_prev',
            F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        
        # Precompute values for q(x_t | x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer(
            'sqrt_one_minus_alphas_cumprod',
            torch.sqrt(1.0 - self.alphas_cumprod)
        )
        
        # Precompute values for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_variance',
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        
        self.init_weights()
    
    def _get_beta_schedule(self) -> torch.Tensor:
        """
        Linear beta schedule.
        
        Returns:
            Beta values (timesteps,)
        """
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.timesteps)
    
    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward diffusion: q(x_t | x_0).
        
        Args:
            x_start: Clean images (B, C, H, W)
            t: Timesteps (B,)
            noise: Optional noise (B, C, H, W)
            
        Returns:
            Noisy images (B, C, H, W)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t]
        
        # Reshape for broadcasting
        sqrt_alpha_cumprod_t = sqrt_alpha_cumprod_t[:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alpha_cumprod_t[:, None, None, None]
        
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict noise.
        
        Args:
            x: Noisy images (B, C, H, W)
            t: Timesteps (B,)
            
        Returns:
            Predicted noise (B, C, H, W)
        """
        return self.model(x, t)
    
    def compute_loss(
        self,
        x_start: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Compute training loss.
        
        Args:
            x_start: Clean images (B, C, H, W)
            t: Optional timesteps (B,), if None will be sampled randomly
            noise: Optional noise
            
        Returns:
            Dictionary with loss
        """
        batch_size = x_start.shape[0]
        device = x_start.device
        
        # Sample timesteps
        if t is None:
            t = torch.randint(0, self.timesteps, (batch_size,), device=device)
        
        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Add noise
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.forward(x_noisy, t)
        
        # MSE loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return {'loss': loss}
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        t_index: int
    ) -> torch.Tensor:
        """
        Reverse diffusion: sample x_{t-1} given x_t.
        
        Args:
            x: Noisy image at timestep t (B, C, H, W)
            t: Current timestep (scalar)
            t_index: Index of t in [0, timesteps-1]
            
        Returns:
            Less noisy image (B, C, H, W)
        """
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(1.0 - betas_t)
        
        # Predict noise
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        predicted_noise = self.forward(x, t_tensor)
        
        # Mean of p(x_{t-1} | x_t)
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t_index == 0:
            return model_mean
        else:
            # Add noise
            posterior_variance_t = self.posterior_variance[t_index]
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise
    
    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        image_size: int = 64,
        channels: int = 3
    ) -> torch.Tensor:
        """
        Generate samples.
        
        Args:
            batch_size: Number of samples
            image_size: Image size
            channels: Number of channels
            
        Returns:
            Generated images (B, C, H, W)
        """
        device = self.get_device()
        
        # Start from pure noise
        img = torch.randn(batch_size, channels, image_size, image_size, device=device)
        
        # Iteratively denoise
        for i in reversed(range(self.timesteps)):
            img = self.p_sample(img, i, i)
        
        return img
    
    @torch.no_grad()
    def interpolate(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        t: int,
        lambda_: float = 0.5
    ) -> torch.Tensor:
        """
        Interpolate between two images in latent space.
        
        Args:
            x1: First image (1, C, H, W)
            x2: Second image (1, C, H, W)
            t: Timestep for interpolation
            lambda_: Interpolation factor (0 = x1, 1 = x2)
            
        Returns:
            Interpolated image (1, C, H, W)
        """
        # Add noise to both images
        t_tensor = torch.tensor([t], device=x1.device)
        noise1 = torch.randn_like(x1)
        noise2 = torch.randn_like(x2)
        
        x1_noisy = self.q_sample(x1, t_tensor, noise1)
        x2_noisy = self.q_sample(x2, t_tensor, noise2)
        
        # Interpolate in noisy space
        x_interp = (1 - lambda_) * x1_noisy + lambda_ * x2_noisy
        
        # Denoise
        for i in reversed(range(t + 1)):
            x_interp = self.p_sample(x_interp, i, i)
        
        return x_interp