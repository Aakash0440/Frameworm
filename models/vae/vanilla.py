"""Vanilla VAE implementation - Variational Autoencoder"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import register_model
from models import BaseModel


class Encoder(nn.Module):
    """VAE Encoder - maps images to latent distribution"""

    def __init__(self, channels, latent_dim, input_size=32):
        super().__init__()
        self.input_size = input_size
        self.conv_layers = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        conv_output_size = self._get_conv_output_size()
        self.fc_mu = nn.Linear(conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(conv_output_size, latent_dim)

    def _get_conv_output_size(self):
        size = self.input_size
        for _ in range(4):
            size = (size + 2 * 1 - 4) // 2 + 1
        return 256 * size * size

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """VAE Decoder - reconstructs images from latent vectors"""

    def __init__(self, latent_dim, channels, output_size=32):
        super().__init__()
        self.output_size = output_size
        size = output_size
        for _ in range(4):
            size = (size + 1) // 2
        self.init_size = max(size, 1)
        self.fc = nn.Linear(latent_dim, 256 * self.init_size * self.init_size)
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, self.init_size, self.init_size)
        x = self.deconv_layers(x)
        return x


@register_model("vae", version="1.0.1", author="Frameworm Team")
class VAE(BaseModel):
    """Variational Autoencoder (VAE) with flexible input size."""

    def __init__(self, config, input_size=32):
        super().__init__(config)
        self.latent_dim = config.model.latent_dim
        self.channels = getattr(config.model, "channels", 3)
        self.beta = getattr(config.model, "beta", 1.0)
        self.input_size = input_size
        self.encoder = Encoder(self.channels, self.latent_dim, input_size=input_size)
        self.decoder = Decoder(self.latent_dim, self.channels, output_size=input_size)
        self.init_weights()

    def init_weights(self):
        self.apply(self._default_init)

    def _default_init(self, module):
        if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, num_samples, device=None):
        if device is None:
            device = self.get_device()
        z = torch.randn(num_samples, self.latent_dim, device=device)
        with torch.no_grad():
            return self.decode(z)

    def reconstruct(self, x):
        with torch.no_grad():
            recon, _, _ = self.forward(x)
        return recon

    def compute_loss(self, x, recon, mu, logvar):
        recon_loss = F.mse_loss(recon, x, reduction="sum") / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        total_loss = recon_loss + self.beta * kl_loss
        return {"loss": total_loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
