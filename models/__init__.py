"""
FRAMEWORM model zoo.
"""
from models.base import BaseModel

# Importing each module triggers their @register_model decorators
from models.vae.vanilla import VAE
from models.gan.dcgan import DCGAN
from models.diffusion.ddpm import DDPM
from models.vqvae2 import VQVAE2
from models.vitgan import ViTGAN
from models.cfg_ddpm import CFGDDPM

__all__ = ['BaseModel', 'VAE', 'DCGAN', 'DDPM', 'VQVAE2', 'ViTGAN', 'CFGDDPM']