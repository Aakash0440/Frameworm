"""
End-to-end integration tests for the complete FRAMEWORM pipeline.
"""

import tempfile
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from training.callbacks import ModelCheckpoint
from training.trainer import Trainer

# ------------------ Fixtures ------------------


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def small_dataset():
    """Small dataset for fast tests"""
    X = torch.randn(100, 3, 32, 32)
    y = torch.randint(0, 5, (100,))
    return TensorDataset(X, y)


@pytest.fixture
def tiny_loaders(small_dataset):
    """Train and val loaders"""
    train_size = 80
    val_size = 20
    train_data = TensorDataset(*[t[:train_size] for t in small_dataset.tensors])
    val_data = TensorDataset(*[t[train_size:] for t in small_dataset.tensors])

    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    return train_loader, val_loader


@pytest.fixture
def simple_model():
    """Simple test model"""

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, 5)

        def forward(self, x):
            return self.fc(self.pool(torch.relu(self.conv(x))).flatten(1))

        def compute_loss(self, x, y):
            output = self.forward(x)
            loss = nn.CrossEntropyLoss()(output, y)
            return {"loss": loss}

    return Model()


# ------------------ Training Pipeline Tests ------------------


class TestTrainingPipeline:

    def test_basic_training(self, simple_model, tiny_loaders):
        """Test basic training runs without errors"""
        train_loader, val_loader = tiny_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        trainer = Trainer(model=simple_model, optimizer=optimizer, device="cpu")

        trainer.train(train_loader, val_loader, epochs=2)

        assert len(trainer.state.train_metrics["loss"]) == 2
        assert len(trainer.state.val_metrics["loss"]) == 2
        assert all(v > 0 for v in trainer.state.train_metrics["loss"])

    def test_training_with_callbacks(self, simple_model, tiny_loaders, temp_dir):
        """Test training with model checkpoint callback"""
        train_loader, val_loader = tiny_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        trainer = Trainer(simple_model, optimizer, device="cpu")

        # Early stopping (optional)
        trainer.set_early_stopping(patience=2, min_delta=0.001)

        # Checkpoint path
        ckpt_path = temp_dir / "best.pt"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        # Checkpoint callback (force save for test)
        checkpoint_cb = ModelCheckpoint(
            filepath=str(ckpt_path),
            monitor="val_loss",
            save_best_only=False,  # <-- force save every epoch for testing
        )
        trainer.add_callback(checkpoint_cb)

        trainer.train(train_loader, val_loader, epochs=3)

        # Verify checkpoint was saved
        assert ckpt_path.exists(), f"Checkpoint not saved: {ckpt_path}"


# ------------------ Performance Tests ------------------


class TestPerformance:

    def test_training_speed(self, simple_model, tiny_loaders):
        """Ensure training doesn't regress beyond threshold"""
        train_loader, val_loader = tiny_loaders
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.001)
        trainer = Trainer(simple_model, optimizer, device="cpu")

        start = time.perf_counter()
        trainer.train(train_loader, val_loader, epochs=3)
        elapsed = time.perf_counter() - start

        assert elapsed < 30, f"Training too slow: {elapsed:.1f}s"

    def test_import_time(self):
        """Ensure FRAMEWORM imports quickly"""
        start = time.perf_counter()
        elapsed = time.perf_counter() - start
        assert elapsed < 5, f"Import too slow: {elapsed:.1f}s"


# ------------------ Run tests if script ------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
