import pytest
import torch

from core import Config, get_model
from core.registry import register_model
from models.vqvae2 import VQVAE2 as _VQVAE2  # noqa: F401 - triggers @register_model  # isort: skip


class TestVQVAE2:
    @pytest.fixture
    def config(self):
        return Config.from_dict(
            {
                "model": {
                    "in_channels": 3,
                    "hidden_channels": 32,
                    "embedding_dim": 16,
                    "num_embeddings": 64,
                }
            }
        )

    def test_creation(self, config):
        model = get_model("vqvae2")(config)
        assert model is not None

    def test_forward(self, config):
        model = get_model("vqvae2")(config)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        assert out["recon"].shape == x.shape
        assert "loss" in out and "vq_loss" in out

    def test_reconstruction_range(self, config):
        model = get_model("vqvae2")(config)
        x = torch.randn(2, 3, 256, 256)
        out = model(x)
        # tanh output should be in [-1, 1]
        assert out["recon"].min() >= -1.0 - 1e-5
        assert out["recon"].max() <= 1.0 + 1e-5

    def test_codebook_sizes(self, config):
        from models.vqvae2 import VectorQuantizer

        vq = VectorQuantizer(num_embeddings=128, embedding_dim=32)
        assert vq.embedding.weight.shape == (128, 32)

    def test_straight_through_gradient(self, config):
        model = get_model("vqvae2")(config)
        x = torch.randn(2, 3, 256, 256, requires_grad=True)
        out = model(x)
        out["loss"].backward()
        assert x.grad is not None, "Gradients must flow through VQ layer"

    def test_trainer_compatible(self, config):
        from torch.utils.data import DataLoader, TensorDataset

        from training import Trainer

        model = get_model("vqvae2")(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer = Trainer(model, optimizer, device="cpu")

        loader = DataLoader(TensorDataset(torch.randn(20, 3, 64, 64)), batch_size=4)
        trainer.train(loader, epochs=1)
