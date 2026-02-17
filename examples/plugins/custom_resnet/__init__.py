"""
Example plugin: Custom ResNet-based VAE.

Shows how to register a custom model architecture.
"""

import torch
import torch.nn as nn
from core import register_model


class ResNetBlock(nn.Module):
    """ResNet residual block"""
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + residual)


class CustomResNetVAE(nn.Module):
    """
    VAE with ResNet encoder/decoder.
    
    Stronger than vanilla VAE due to residual connections.
    """
    
    def __init__(self, config):
        super().__init__()
        
        latent_dim = getattr(config.model, 'latent_dim', 128)
        hidden_ch = getattr(config.model, 'hidden_channels', 128)
        in_ch = getattr(config.model, 'in_channels', 3)
        
        # Encoder with ResNet blocks
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_ch // 2),
            nn.ReLU(),
            
            ResNetBlock(hidden_ch // 2),
            
            nn.Conv2d(hidden_ch // 2, hidden_ch, 4, 2, 1),
            nn.BatchNorm2d(hidden_ch),
            nn.ReLU(),
            
            ResNetBlock(hidden_ch),
            ResNetBlock(hidden_ch),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(hidden_ch, latent_dim)
        self.fc_logvar = nn.Linear(hidden_ch, latent_dim)
        
        # Decoder
        self.fc_decode = nn.Linear(latent_dim, hidden_ch * 4 * 4)
        
        self.decoder = nn.Sequential(
            ResNetBlock(hidden_ch),
            
            nn.ConvTranspose2d(hidden_ch, hidden_ch // 2, 4, 2, 1),
            nn.BatchNorm2d(hidden_ch // 2),
            nn.ReLU(),
            
            ResNetBlock(hidden_ch // 2),
            
            nn.ConvTranspose2d(hidden_ch // 2, in_ch, 4, 2, 1),
            nn.Tanh()
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        h = self.fc_decode(z)
        h = h.view(-1, 128, 4, 4)
        return self.decoder(h)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def compute_loss(self, x, y=None):
        """FRAMEWORM Trainer compatibility"""
        if isinstance(x, (tuple, list)):
            x = x[0]
        
        recon, mu, logvar = self(x)
        
        # Reconstruction loss
        recon_loss = nn.functional.mse_loss(recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        return {
            'loss': recon_loss + kl_loss * 0.001,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


def register():
    """Plugin entry point - registers custom model"""
    register_model('resnet_vae', CustomResNetVAE)
    print("✓ Registered: resnet_vae model")