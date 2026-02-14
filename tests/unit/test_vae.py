# tests/unit/test_vae.py
import pytest
from types import SimpleNamespace
from models.vae.vanilla import VAE
import torch

# -----------------------------
# Dummy Config for testing
# -----------------------------
class DummyConfig:
    def __init__(self):
        # model must have latent_dim and channels
        self.model = SimpleNamespace(
            latent_dim=16,
            channels=3
        )
        # Add other configs if needed
        self.training = SimpleNamespace(
            lr=0.001,
            beta1=0.9,
            beta2=0.999
        )

# -----------------------------
# Dummy VAE patch for init_weights
# -----------------------------
class TestVAE(VAE):
    def init_weights(self):
        # Override original call for testing
        pass

# -----------------------------
# Pytest Fixture
# -----------------------------
@pytest.fixture
def vae_model():
    cfg = DummyConfig()
    model = TestVAE(cfg)
    return model

# -----------------------------
# Tests
# -----------------------------
def test_forward_pass(vae_model):
    x = torch.randn(2, 3, 32, 32)  # batch_size=2, channels=3, 32x32 images
    out = vae_model(x)
    assert out is not None
    # You can assert shapes if your VAE returns reconstruction and latent
    # assert out[0].shape == x.shape  

def test_loss_computation(vae_model):
    x = torch.randn(2, 3, 32, 32)
    recon, mu, logvar = vae_model(x)
    # Simple VAE loss
    recon_loss = torch.nn.functional.mse_loss(recon, x)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    loss = recon_loss + kl_loss
    assert loss.item() >= 0

def test_sampling(vae_model):
    z = torch.randn(2, vae_model.latent_dim)
    sample = vae_model.decode(z)
    assert sample.shape[1] == vae_model.channels  # <-- new


def test_encode_decode(vae_model):
    x = torch.randn(2, 3, 32, 32)
    mu, logvar = vae_model.encode(x)
    z = vae_model.reparameterize(mu, logvar)
    recon = vae_model.decode(z)
    assert recon.shape[1] == vae_model.channels
