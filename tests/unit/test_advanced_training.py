"""Tests for advanced training features"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training import Trainer
from training.advanced import EMAModel, GradientAccumulator, GradientClipper


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

    def compute_loss(self, x, y):
        return {"loss": nn.MSELoss()(self.forward(x), y)}


class TestGradientAccumulation:
    def test_accumulation_steps(self):
        acc = GradientAccumulator(4)

        # Should update every 4 steps
        for i in range(8):
            should_update = acc.should_update()
            if (i + 1) % 4 == 0:
                assert should_update
            else:
                assert not should_update

    def test_loss_scaling(self):
        acc = GradientAccumulator(4)
        loss = torch.tensor(1.0)
        scaled = acc.scale_loss(loss)
        assert scaled == 0.25


class TestGradientClipping:
    def test_clipping(self):
        model = SimpleModel()
        x = torch.randn(10, 10)
        y = torch.randn(10, 1)

        # Create large gradients
        loss = (model(x) - y).pow(2).sum() * 100
        loss.backward()

        clipper = GradientClipper(max_norm=1.0)
        norm_before = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5

        clipped_norm = clipper.clip(model.parameters())
        norm_after = sum(p.grad.norm().item() ** 2 for p in model.parameters()) ** 0.5

        assert norm_after <= 1.0 + 0.01  # Allow small numerical error


class TestEMA:
    def test_ema_update(self):
        model = SimpleModel()
        ema = EMAModel(model, decay=0.999)

        # Get initial EMA params
        initial_ema = {k: v.clone() for k, v in ema.shadow.items()}

        # Update model
        for p in model.parameters():
            p.data.add_(torch.randn_like(p) * 0.1)

        # Update EMA
        ema.update()

        # EMA should be different
        for k in initial_ema:
            assert not torch.allclose(initial_ema[k], ema.shadow[k])

    def test_apply_and_restore(self):
        model = SimpleModel()
        ema = EMAModel(model, decay=0.999)

        # Train a bit
        for _ in range(10):
            for p in model.parameters():
                p.data.add_(torch.randn_like(p) * 0.01)
            ema.update()

        # Store original
        original = [p.clone() for p in model.parameters()]

        # Apply EMA
        ema.apply_shadow()
        ema_params = [p.clone() for p in model.parameters()]

        # Restore
        ema.restore()
        restored = [p.clone() for p in model.parameters()]

        # Check
        assert not all(torch.allclose(o, e) for o, e in zip(original, ema_params))
        assert all(torch.allclose(o, r) for o, r in zip(original, restored))


class TestAdvancedTrainer:
    def test_gradient_accumulation_training(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(40, 10)
        y = torch.randn(40, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        trainer = Trainer(model, optimizer, device="cpu")
        trainer.enable_gradient_accumulation(2)

        trainer.train(loader, epochs=1)

        # Check training completed
        assert trainer.state.current_epoch == 1

    def test_gradient_clipping_training(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(40, 10)
        y = torch.randn(40, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        trainer = Trainer(model, optimizer, device="cpu")
        trainer.enable_gradient_clipping(1.0)

        trainer.train(loader, epochs=1)

        # Should have grad_norm in metrics
        assert "grad_norm" in trainer.train_tracker.batch_metrics

    def test_ema_training(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        X = torch.randn(40, 10)
        y = torch.randn(40, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        trainer = Trainer(model, optimizer, device="cpu")
        trainer.enable_ema(0.999)

        trainer.train(loader, epochs=1)

        # Check EMA was updated
        assert trainer.ema is not None

    def test_all_features_together(self):
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        X = torch.randn(40, 10)
        y = torch.randn(40, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        trainer = Trainer(model, optimizer, device="cpu")
        trainer.enable_gradient_accumulation(2)
        trainer.enable_gradient_clipping(1.0)
        trainer.enable_ema(0.999)

        trainer.train(loader, epochs=2)

        assert trainer.state.current_epoch == 2


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
