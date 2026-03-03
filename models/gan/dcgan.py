"""DCGAN implementation - Deep Convolutional GAN"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from core import register_model
from models.base import BaseModel


class Generator(nn.Module):
    """DCGAN Generator"""

    def __init__(self, latent_dim, ngf, channels):
        super().__init__()

        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # State: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # State: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # State: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # State: ngf x 32 x 32
            nn.ConvTranspose2d(ngf, channels, 4, 2, 1, bias=False),
            nn.Tanh(),
            # Output: channels x 64 x 64
        )

    def forward(self, z):
        return self.main(z)


class Discriminator(nn.Module):
    """DCGAN Discriminator"""

    def __init__(self, ndf, channels):
        super().__init__()

        self.main = nn.Sequential(
            # Input: channels x 64 x 64
            nn.Conv2d(channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # State: ndf x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # State: (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
            # Output: 1 x 1 x 1
        )

    def forward(self, img):
        return self.main(img).view(-1, 1).squeeze(1)


@register_model("dcgan", version="1.0.0", author="Frameworm Team")
class DCGAN(BaseModel):
    """
    Deep Convolutional GAN (DCGAN).

    Architecture based on:
    Radford et al., "Unsupervised Representation Learning with Deep
    Convolutional Generative Adversarial Networks", ICLR 2016.

    Args:
        config: Configuration with model.latent_dim, model.ngf, model.ndf
    """

    def __init__(self, config):
        super().__init__(config)

        latent_dim = config.model.latent_dim
        ngf = config.model.get("ngf", 64)
        ndf = config.model.get("ndf", 64)
        channels = config.model.get("channels", 3)

        self.latent_dim = latent_dim
        self.generator = Generator(latent_dim, ngf, channels)
        self.discriminator = Discriminator(ndf, channels)

        self.init_weights()

    def forward(self, z=None, batch_size=None):
        if z is None:
            if batch_size is None:
                raise ValueError("Must provide either z or batch_size")
            z = self.sample_z(batch_size)
        return self.generator(z)

    def discriminate(self, images):
        return self.discriminator(images)

    def sample_z(self, batch_size):
        device = self.get_device()
        return torch.randn(batch_size, self.latent_dim, 1, 1, device=device)

    def compute_loss(self, batch):
        """GAN loss for generator and discriminator."""
        real_images = batch
        batch_size = real_images.size(0)
        device = real_images.device

        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # Discriminator loss
        real_preds = self.discriminator(real_images)
        d_loss_real = F.binary_cross_entropy(real_preds, real_labels)

        z = torch.randn(batch_size, self.latent_dim, 1, 1, device=device)
        fake_images = self.generator(z)
        fake_preds = self.discriminator(fake_images.detach())
        d_loss_fake = F.binary_cross_entropy(fake_preds, fake_labels)

        d_loss = d_loss_real + d_loss_fake

        # Generator loss
        gen_preds = self.discriminator(fake_images)
        g_loss = F.binary_cross_entropy(gen_preds, real_labels)

        return {"loss": d_loss + g_loss, "d_loss": d_loss, "g_loss": g_loss}

    def _default_init(self, module):
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(module.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(module.weight.data, 1.0, 0.02)
            nn.init.constant_(module.bias.data, 0)

    def init_weights(self):
        self.generator.apply(self._default_init)
        self.discriminator.apply(self._default_init)