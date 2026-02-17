"""Tests for DDPM model"""

import pytest
import torch
from core import Config, get_model
from models.diffusion.ddpm import DDPM, get_timestep_embedding
import tempfile
from pathlib import Path


class TestTimestepEmbedding:
    """Test timestep embedding"""

    def test_timestep_embedding_shape(self):
        """Should create correct shape"""
        t = torch.tensor([0, 10, 100, 999])
        emb = get_timestep_embedding(t, 128)

        assert emb.shape == (4, 128)

    def test_timestep_embedding_different(self):
        """Different timesteps should have different embeddings"""
        t1 = torch.tensor([0])
        t2 = torch.tensor([100])

        emb1 = get_timestep_embedding(t1, 128)
        emb2 = get_timestep_embedding(t2, 128)

        assert not torch.allclose(emb1, emb2)


class TestDDPM:
    """Test DDPM model"""

    def setup_method(self):
        """Setup test config and model"""
        config_file = Path(tempfile.mktemp(suffix=".yaml"))
        config_file.write_text("""
model:
  type: ddpm
  timesteps: 100  # Reduced for testing
  image_size: 64
  channels: 3
  base_channels: 64  # Reduced for speed
training:
  device: cpu
""")
        self.config = Config(config_file)
        self.ddpm = DDPM(self.config)
        config_file.unlink()

    def test_model_creation(self):
        """Should create DDPM successfully"""
        assert self.ddpm is not None
        assert hasattr(self.ddpm, "model")
        assert self.ddpm.timesteps == 100

    def test_forward_pass(self):
        """Should predict noise"""
        x = torch.randn(2, 3, 64, 64)
        t = torch.randint(0, 100, (2,))

        noise_pred = self.ddpm(x, t)

        assert noise_pred.shape == x.shape

    def test_q_sample(self):
        """Should add noise to images"""
        x = torch.randn(2, 3, 64, 64)
        t = torch.tensor([50, 99])

        x_noisy = self.ddpm.q_sample(x, t)

        assert x_noisy.shape == x.shape

        # Noisy image should be different from original
        assert not torch.allclose(x, x_noisy)

    def test_compute_loss(self):
        """Should compute training loss"""
        x = torch.randn(4, 3, 64, 64)

        loss_dict = self.ddpm.compute_loss(x)

        assert "loss" in loss_dict
        assert loss_dict["loss"] > 0

    def test_p_sample(self):
        """Should perform reverse diffusion step"""
        x = torch.randn(2, 3, 64, 64)

        x_prev = self.ddpm.p_sample(x, t=50, t_index=50)

        assert x_prev.shape == x.shape

    def test_sample(self):
        """Should generate samples"""
        samples = self.ddpm.sample(batch_size=1, image_size=64, channels=3)

        assert samples.shape == (1, 3, 64, 64)
        assert samples.min() >= -1  # Approximately normalized
        assert samples.max() <= 1

    def test_model_registered(self):
        """Should be registered in plugin system"""
        model_class = get_model("ddpm")
        assert model_class == DDPM

    def test_beta_schedule(self):
        """Should have valid beta schedule"""
        betas = self.ddpm.betas

        assert len(betas) == 100
        assert (betas > 0).all()
        assert (betas < 1).all()
        # Should be increasing
        assert (betas[1:] >= betas[:-1]).all()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
