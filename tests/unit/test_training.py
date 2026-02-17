"""Tests for training infrastructure"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from training import Trainer, TrainingState
from training.callbacks import CSVLogger, Callback
from training.schedulers import WarmupLR
import tempfile
from pathlib import Path


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

    def compute_loss(self, x, y):
        pred = self.forward(x)
        loss = nn.MSELoss()(pred, y)
        return {"loss": loss}


class TestTrainingState:
    def test_state_creation(self):
        state = TrainingState()
        assert state.current_epoch == 0
        assert state.global_step == 0

    def test_update_metrics(self):
        state = TrainingState()
        state.update_train_metrics({"loss": 0.5})
        assert state.train_metrics["loss"] == [0.5]

    def test_best_epoch_tracking(self):
        state = TrainingState()
        is_best = state.is_best_epoch(0.5, mode="min")
        assert is_best
        assert state.best_metric == 0.5


class TestTrainer:
    def test_trainer_creation(self):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, device="cpu")
        assert trainer.model is not None

    def test_training_loop(self):
        # Create dummy data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)

        # Train
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, device="cpu")

        trainer.train(loader, epochs=2)

        # Check training happened
        assert trainer.state.current_epoch == 2
        assert trainer.state.global_step > 0

    def test_checkpoint_save_load(self, tmp_path):
        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, device="cpu", checkpoint_dir=str(tmp_path))

        # Save
        trainer.save_checkpoint("test.pt", epoch=5)

        # Load
        trainer.load_checkpoint(tmp_path / "test.pt")
        assert trainer.state.current_epoch == 5


class TestCallbacks:
    def test_callback_execution(self):
        called = {"train_begin": False, "epoch_end": False}

        class TestCallback(Callback):
            def on_train_begin(self, trainer):
                called["train_begin"] = True

            def on_epoch_end(self, epoch, metrics, trainer):
                called["epoch_end"] = True

        # Train with callback
        X = torch.randn(20, 10)
        y = torch.randn(20, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        model = DummyModel()
        optimizer = torch.optim.Adam(model.parameters())
        trainer = Trainer(model, optimizer, device="cpu")
        trainer.add_callback(TestCallback())

        trainer.train(loader, epochs=1)

        assert called["train_begin"]
        assert called["epoch_end"]


class TestSchedulers:
    def test_warmup_lr(self):
        model = DummyModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = WarmupLR(optimizer, warmup_epochs=5)

        lrs = []
        for _ in range(10):
            lrs.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()

        # Should increase during warmup
        assert lrs[0] < lrs[4]
        # Should be constant after
        assert abs(lrs[5] - lrs[9]) < 0.01


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
