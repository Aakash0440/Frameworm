"""
Callback system for training.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import csv

# Import your EarlyStopper correctly
from search.early_stopping import EarlyStopper
from training.schedulers import LearningRateScheduler


class Callback:
    """Base callback class. Override methods to add custom behavior."""

    def on_train_begin(self, trainer):
        pass

    def on_train_end(self, trainer):
        pass

    def on_epoch_begin(self, epoch: int, trainer):
        pass

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        pass

    def on_batch_begin(self, batch_idx: int, trainer):
        pass

    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float], trainer):
        pass


class CallbackList:
    """Container for managing multiple callbacks."""

    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def append(self, callback: Callback):
        self.callbacks.append(callback)

    def on_train_begin(self, trainer):
        for callback in self.callbacks:
            callback.on_train_begin(trainer)

    def on_train_end(self, trainer):
        for callback in self.callbacks:
            callback.on_train_end(trainer)

    def on_epoch_begin(self, epoch: int, trainer):
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, trainer)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, metrics, trainer)

    def on_batch_begin(self, batch_idx: int, trainer):
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, trainer)

    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float], trainer):
        for callback in self.callbacks:
            callback.on_batch_end(batch_idx, metrics, trainer)


class CSVLogger(Callback):
    """Log metrics to a CSV file."""

    def __init__(self, filename: str, separator: str = ",", append: bool = False):
        self.filename = Path(filename)
        self.separator = separator
        self.append = append
        self.keys = None
        self.writer = None
        self.file = None

    def on_train_begin(self, trainer):
        mode = "a" if self.append else "w"
        self.file = open(self.filename, mode, newline="")
        self.writer = csv.writer(self.file, delimiter=self.separator)

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        if self.keys is None:
            self.keys = ["epoch"] + list(metrics.keys())
            self.writer.writerow(self.keys)
        row = [epoch] + [metrics.get(k, "") for k in self.keys[1:]]
        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, trainer):
        if self.file:
            self.file.close()


class ModelCheckpoint(Callback):
    """Save model checkpoints."""

    def __init__(
        self,
        filepath: str,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = True,
    ):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_value = float("inf") if mode == "min" else float("-inf")

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        if self.monitor not in metrics:
            return
        current_value = metrics[self.monitor]
        improved = (
            (current_value < self.best_value)
            if self.mode == "min"
            else (current_value > self.best_value)
        )

        if improved or not self.save_best_only:
            self.best_value = current_value
            filepath = self.filepath.format(epoch=epoch, **metrics)
            trainer.save_checkpoint(filepath, epoch)
            if improved:
                print(f"  → Saved best model: {filepath} ({self.monitor}={current_value:.4f})")


class LearningRateMonitor(Callback):
    """Monitor and log learning rate changes."""

    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        lr = trainer.optimizer.param_groups[0]["lr"]
        print(f"  LR: {lr:.6f}")


class GradientMonitor(Callback):
    """Monitor gradient norms."""

    def __init__(self, log_every_n_batches: int = 100):
        self.log_every_n_batches = log_every_n_batches

    def on_batch_end(self, batch_idx: int, metrics: Dict[str, float], trainer):
        if batch_idx % self.log_every_n_batches != 0:
            return
        total_norm = 0.0
        for p in trainer.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm**0.5
        print(f"    Gradient norm: {total_norm:.4f}")
