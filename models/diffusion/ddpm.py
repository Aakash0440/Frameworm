"""
DDPM (Denoising Diffusion Probabilistic Model) implementation for Frameworm.

Compatible with Frameworm registry and unit tests.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import register_model
from models import BaseModel


def get_timestep_embedding(timesteps: torch.Tensor, embedding_dim: int) -> torch.Tensor:
    """
    Sinusoidal timestep embeddings.
    Args:
        timesteps: (B,) tensor of timestep indices
        embedding_dim: dimension of embedding
    Returns:
        (B, embedding_dim) embeddings
    """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""

    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = h + self.time_mlp(F.relu(t_emb))[:, :, None, None]
        h = F.relu(self.conv2(h))
        return h + self.skip(x)


class UNet(nn.Module):
    """U-Net for noise prediction"""

    def __init__(self, in_ch=3, base_ch=128, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.ReLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Encoder
        self.down1 = ResidualBlock(in_ch, base_ch, time_emb_dim)
        self.down2 = ResidualBlock(base_ch, base_ch * 2, time_emb_dim)
        self.down3 = ResidualBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ResidualBlock(base_ch * 4, base_ch * 4, time_emb_dim)

        # Decoder
        self.up1 = ResidualBlock(base_ch * 8, base_ch * 2, time_emb_dim)
        self.up2 = ResidualBlock(base_ch * 4, base_ch, time_emb_dim)
        self.up3 = ResidualBlock(base_ch * 2, base_ch, time_emb_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Output
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_mlp(get_timestep_embedding(t, 128))

        # Encoder
        d1 = self.down1(x, t_emb)
        d2 = self.down2(self.pool(d1), t_emb)
        d3 = self.down3(self.pool(d2), t_emb)

        # Bottleneck
        b = self.bottleneck(self.pool(d3), t_emb)

        # Decoder
        u1 = self.up1(torch.cat([self.upsample(b), d3], dim=1), t_emb)
        u2 = self.up2(torch.cat([self.upsample(u1), d2], dim=1), t_emb)
        u3 = self.up3(torch.cat([self.upsample(u2), d1], dim=1), t_emb)

        return self.out(u3)


@register_model("ddpm", version="1.0.0", author="Frameworm Team")
class DDPM(BaseModel):
    """Denoising Diffusion Probabilistic Model"""

    def __init__(self, config=None):
        super().__init__(config)
        cfg = config.model if config else {}
        self.timesteps = cfg.get("timesteps", 100)
        channels = cfg.get("channels", 3)
        base_ch = cfg.get("base_channels", 128)

        self.model = UNet(in_ch=channels, base_ch=base_ch, time_emb_dim=128)

        # Beta schedule
        beta_start = 0.0001
        beta_end = 0.02
        self.register_buffer("betas", torch.linspace(beta_start, beta_end, self.timesteps))
        alphas = 1.0 - self.betas
        self.register_buffer("alphas_cumprod", torch.cumprod(alphas, dim=0))
        self.register_buffer(
            "alphas_cumprod_prev", F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        )
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(self.alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - self.alphas_cumprod))
        self.register_buffer(
            "posterior_variance",
            self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict noise"""
        return self.model(x, t)

    def q_sample(
        self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward diffusion q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def compute_loss(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """Compute MSE loss"""
        B = x.shape[0]
        device = x.device
        t = t if t is not None else torch.randint(0, self.timesteps, (B,), device=device)
        noise = noise if noise is not None else torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        pred = self.forward(x_noisy, t)
        return {"loss": F.mse_loss(pred, noise)}

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int, t_index: int) -> torch.Tensor:
        """Reverse diffusion step"""
        betas_t = self.betas[t_index]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t_index]
        sqrt_recip_alphas_t = 1.0 / torch.sqrt(1.0 - betas_t)
        t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        pred_noise = self.forward(x, t_tensor)
        mean = sqrt_recip_alphas_t * (x - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        if t_index == 0:
            return mean
        else:
            return mean + torch.sqrt(self.posterior_variance[t_index]) * torch.randn_like(x)

    @torch.no_grad()
    def sample(self, batch_size: int = 1, image_size: int = 64, channels: int = 3) -> torch.Tensor:
        """Generate samples, clamped to [-1, 1]"""
        device = next(self.model.parameters()).device
        x = torch.randn(batch_size, channels, image_size, image_size, device=device)
        for i in reversed(range(self.timesteps)):
            x = self.p_sample(x, i, i)
        # FIX: clamp to [-1, 1] — an untrained model produces unbounded values,
        # and even a trained model can drift slightly outside this range.
        return torch.clamp(x, -1, 1)
