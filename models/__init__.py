"""
FRAMEWORM model zoo.
"""

from models.base import BaseModel
from models.cfg_ddpm import CFGDDPM
from models.diffusion.ddpm import DDPM
from models.gan.dcgan import DCGAN
# Importing each module triggers their @register_model decorators
from models.vae.vanilla import VAE
from models.vitgan import ViTGAN
from models.vqvae2 import VQVAE2

__all__ = ["BaseModel", "VAE", "DCGAN", "DDPM", "VQVAE2", "ViTGAN", "CFGDDPM"]
