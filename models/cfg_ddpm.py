"""
DDPM with Classifier-Free Guidance (CFG).

Reference: "Classifier-Free Diffusion Guidance"
           Ho & Salimans, 2021 (https://arxiv.org/abs/2207.12598)

Enables conditional generation WITHOUT a separate classifier.
During training, randomly drop class conditioning (replace with null token).
During inference, interpolate between conditional and unconditional outputs:

    pred = uncond + guidance_scale * (cond - uncond)

Higher guidance_scale → images more faithful to condition, less diverse.
Typical values: 3.0–10.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    """
    Sinusoidal embedding for timestep t.
    
    Same as Transformer positional encoding but for diffusion timestep.
    Maps scalar t → vector of size dim.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        
        # Frequencies: exp(-log(10000) * i / (dim/2))
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=device) / half_dim
        )
        
        # Outer product: (B,) x (D/2,) → (B, D/2)
        args = t[:, None].float() * freqs[None, :]
        
        # Concatenate sin and cos
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        
        return embedding


class ResBlock(nn.Module):
    """
    Residual block with timestep and class conditioning.
    
    AdaGN: Adaptive Group Normalization scales/shifts activations
    based on time embedding + class embedding.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        num_groups: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.GroupNorm(num_groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        
        # Time embedding projection → AdaGN scale + shift
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels * 2)
        )
        
        self.norm2 = nn.GroupNorm(num_groups, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        
        # Residual connection
        self.residual = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        
        # Adaptive normalization from time embedding
        scale_shift = self.time_proj(t_emb)[:, :, None, None]
        scale, shift = scale_shift.chunk(2, dim=1)
        h = self.norm2(h) * (1 + scale) + shift
        
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.residual(x)


class SelfAttentionBlock(nn.Module):
    """Self-attention for U-Net bottleneck"""
    
    def __init__(self, channels: int, num_heads: int = 4, num_groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        h = h.view(B, C, H * W).transpose(1, 2)  # (B, HW, C)
        h, _ = self.attn(h, h, h)
        h = h.transpose(1, 2).view(B, C, H, W)
        return x + h


class CFGUNet(nn.Module):
    """
    U-Net for CFG-DDPM with time + class conditioning.
    
    Architecture: encoder → bottleneck (with attention) → decoder
    Each block conditioned on time embedding and optionally class label.
    """
    
    def __init__(
        self,
        image_size: int = 64,
        in_channels: int = 3,
        model_channels: int = 128,
        channel_mult: Tuple[int, ...] = (1, 2, 4),
        num_classes: int = 10,
        num_res_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_classes = num_classes
        time_emb_dim = model_channels * 4
        
        # Time embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbedding(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Class embedding (+1 for null class during CFG)
        self.class_emb = nn.Embedding(num_classes + 1, time_emb_dim)
        
        channels = [model_channels * m for m in channel_mult]
        
        # ─── Encoder ─────────────────────────────────────────────
        self.conv_in = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        
        self.encoder = nn.ModuleList()
        self.downsample = nn.ModuleList()
        
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            for _ in range(num_res_blocks):
                self.encoder.append(ResBlock(in_ch, out_ch, time_emb_dim, dropout=dropout))
                in_ch = out_ch
            if i < len(channels) - 1:
                self.downsample.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))
        
        # ─── Bottleneck ───────────────────────────────────────────
        self.mid_block1 = ResBlock(in_ch, in_ch, time_emb_dim, dropout=dropout)
        self.mid_attn = SelfAttentionBlock(in_ch)
        self.mid_block2 = ResBlock(in_ch, in_ch, time_emb_dim, dropout=dropout)
        
        # ─── Decoder ─────────────────────────────────────────────
        self.decoder = nn.ModuleList()
        self.upsample = nn.ModuleList()
        
        for i, out_ch in enumerate(reversed(channels)):
            for j in range(num_res_blocks + 1):
                skip_ch = channels[-(i+1)]
                self.decoder.append(ResBlock(in_ch + skip_ch, out_ch, time_emb_dim, dropout=dropout))
                in_ch = out_ch
            if i < len(channels) - 1:
                self.upsample.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(in_ch, in_ch, 3, padding=1)
                    )
                )
        
        # Output
        self.conv_out = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, in_channels, 3, padding=1)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        class_label: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict noise for denoising step.
        
        Args:
            x: Noisy input (B, C, H, W)
            t: Timestep (B,)
            class_label: Class labels (B,) or None
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Class conditioning
        if class_label is not None:
            t_emb = t_emb + self.class_emb(class_label)
        else:
            # Null class (unconditional)
            null_label = torch.full((x.shape[0],), self.num_classes, device=x.device)
            t_emb = t_emb + self.class_emb(null_label)
        
        # Encoder
        h = self.conv_in(x)
        skips = [h]
        
        enc_idx = 0
        for i, block in enumerate(self.encoder):
            h = block(h, t_emb)
            skips.append(h)
            if enc_idx < len(self.downsample) and (i + 1) % 2 == 0:
                h = self.downsample[enc_idx](h)
                skips.append(h)
                enc_idx += 1
        
        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)
        
        # Decoder
        dec_idx = 0
        for i, block in enumerate(self.decoder):
            skip = skips.pop()
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)
            if dec_idx < len(self.upsample) and (i + 1) % (2+1) == 0:
                h = self.upsample[dec_idx](h)
                dec_idx += 1
        
        return self.conv_out(h)


class CFGDDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model with Classifier-Free Guidance.
    
    Supports both conditional (class-guided) and unconditional generation.
    During training, class labels are randomly dropped (p_uncond) to train
    the unconditional model simultaneously.
    
    At inference, guidance interpolates for better quality:
        pred = uncond + scale * (cond - uncond)
    
    Args:
        config: Model configuration
        
    Config keys:
        model.image_size: Image size (default: 64)
        model.image_channels: Channels (default: 3)
        model.num_classes: Number of conditional classes (default: 10)
        model.model_channels: U-Net base channels (default: 128)
        model.timesteps: Diffusion timesteps (default: 1000)
        model.p_uncond: Probability of dropping class label (default: 0.15)
        
    Example:
        >>> model = CFGDDPM(config)
        >>> # Training
        >>> x = torch.randn(4, 3, 64, 64)
        >>> labels = torch.randint(0, 10, (4,))
        >>> losses = model.compute_loss(x, labels)
        >>> # Inference with CFG
        >>> samples = model.sample(num_samples=4, class_label=3, guidance_scale=5.0)
    """
    
    def __init__(self, config):
        super().__init__()
        
        self.image_size = getattr(config.model, 'image_size', 64)
        self.image_channels = getattr(config.model, 'image_channels', 3)
        self.num_classes = getattr(config.model, 'num_classes', 10)
        self.timesteps = getattr(config.model, 'timesteps', 1000)
        self.p_uncond = getattr(config.model, 'p_uncond', 0.15)
        
        model_channels = getattr(config.model, 'model_channels', 128)
        
        # U-Net
        self.unet = CFGUNet(
            image_size=self.image_size,
            in_channels=self.image_channels,
            model_channels=model_channels,
            num_classes=self.num_classes
        )
        
        # Noise schedule (cosine schedule — better than linear)
        self.register_buffer('betas', self._cosine_schedule(self.timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod',
                             torch.sqrt(1.0 - self.alphas_cumprod))
    
    def _cosine_schedule(self, timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine noise schedule (better than linear for small timestep counts).
        
        https://arxiv.org/abs/2102.09672
        """
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        alphas_cumprod = torch.cos((steps / timesteps + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 0, 0.999).float()
    
    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward diffusion: add noise at timestep t.
        
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        return sqrt_alpha * x0 + sqrt_one_minus * noise, noise
    
    def compute_loss(
        self,
        x0: torch.Tensor,
        y: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Training step: predict added noise"""
        if isinstance(x0, (tuple, list)):
            x0, y = x0[0], x0[1] if len(x0) > 1 else None
        
        B = x0.shape[0]
        device = x0.device
        
        # Random timesteps
        t = torch.randint(0, self.timesteps, (B,), device=device)
        
        # Add noise
        x_noisy, noise_target = self.q_sample(x0, t)
        
        # Randomly drop class label for CFG training
        if y is not None and self.p_uncond > 0:
            # Replace with null class index for some examples
            mask = torch.rand(B, device=device) < self.p_uncond
            y = y.clone()
            y[mask] = self.num_classes  # null class
        
        # Predict noise
        noise_pred = self.unet(x_noisy, t, class_label=y)
        
        # Simple MSE loss on noise prediction
        loss = F.mse_loss(noise_pred, noise_target)
        
        return {'loss': loss, 'mse': loss}
    
    @torch.no_grad()
    def sample(
        self,
        num_samples: int,
        class_label: Optional[int] = None,
        guidance_scale: float = 5.0,
        device: str = 'cpu'
    ) -> torch.Tensor:
        """
        Generate samples using DDPM reverse process with CFG.
        
        Args:
            num_samples: How many images to generate
            class_label: Class to condition on (None = unconditional)
            guidance_scale: CFG guidance strength (1.0 = no guidance, 7.5 = typical)
            device: Device to run on
            
        Returns:
            Generated images (B, C, H, W) in [-1, 1]
        """
        x = torch.randn(num_samples, self.image_channels,
                        self.image_size, self.image_size, device=device)
        
        # Class labels
        if class_label is not None:
            labels = torch.full((num_samples,), class_label, dtype=torch.long, device=device)
            null_labels = torch.full((num_samples,), self.num_classes, dtype=torch.long, device=device)
        
        # Reverse diffusion loop
        for t_idx in reversed(range(self.timesteps)):
            t = torch.full((num_samples,), t_idx, dtype=torch.long, device=device)
            
            if class_label is not None and guidance_scale > 1.0:
                # Run U-Net twice: once conditional, once unconditional
                noise_cond = self.unet(x, t, class_label=labels)
                noise_uncond = self.unet(x, t, class_label=null_labels)
                
                # Classifier-free guidance interpolation
                noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
            else:
                noise_pred = self.unet(x, t, class_label=labels if class_label else None)
            
            # DDPM reverse step
            alpha = self.alphas[t_idx]
            alpha_cumprod = self.alphas_cumprod[t_idx]
            beta = self.betas[t_idx]
            
            # Mean of reverse posterior
            coeff1 = 1.0 / torch.sqrt(alpha)
            coeff2 = beta / torch.sqrt(1.0 - alpha_cumprod)
            
            mean = coeff1 * (x - coeff2 * noise_pred)
            
            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta) * noise
            else:
                x = mean
        
        return x.clamp(-1, 1)