"""Tests for DCGAN model"""

import pytest
import torch
from core import Config, get_model
from models.gan.dcgan import DCGAN
import tempfile
from pathlib import Path


class TestDCGAN:
    """Test DCGAN model"""

    def setup_method(self):
        """Setup test config"""
        config_file = Path(tempfile.mktemp(suffix=".yaml"))
        config_file.write_text("""
model:
  type: dcgan
  latent_dim: 100
  image_size: 64
  channels: 3
  ngf: 64
  ndf: 64
training:
  device: cpu
""")
        self.config = Config(config_file)
        self.model = DCGAN(self.config)
        config_file.unlink()

    def test_model_creation(self):
        """Should create model successfully"""
        assert self.model is not None
        assert hasattr(self.model, "generator")
        assert hasattr(self.model, "discriminator")

    def test_generator_forward(self):
        """Should generate images"""
        z = torch.randn(4, 100, 1, 1)
        images = self.model(z)

        assert images.shape == (4, 3, 64, 64)
        assert images.min() >= -1
        assert images.max() <= 1

    def test_discriminator_forward(self):
        """Should discriminate images"""
        images = torch.randn(4, 3, 64, 64)
        probs = self.model.discriminate(images)

        assert probs.shape == (4,)
        assert probs.min() >= 0
        assert probs.max() <= 1

    def test_sample_z(self):
        """Should sample latent vectors"""
        z = self.model.sample_z(8)
        assert z.shape == (8, 100, 1, 1)

    def test_generate_without_z(self):
        """Should generate images from sampling"""
        images = self.model(batch_size=4)
        assert images.shape == (4, 3, 64, 64)

    def test_model_registered(self):
        """Should be registered in registry"""
        model_class = get_model("dcgan")
        assert model_class == DCGAN

    def test_parameter_count(self):
        """Should have reasonable parameter count"""
        total_params = self.model.count_parameters()
        assert total_params > 0
        assert total_params < 10_000_000  # Less than 10M params

    def test_to_device(self):
        """Should move to device"""
        self.model.to_device("cpu")
        assert self.model.get_device() == torch.device("cpu")


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
