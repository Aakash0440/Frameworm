"""
Training state management.

Tracks current training progress, metrics, and configuration.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainingState:
    """
    Tracks training state.

    This can be saved/loaded for resuming training.
    """

    # Training progress
    current_epoch: int = 0
    global_step: int = 0
    best_metric: float = float("inf")
    best_epoch: int = 0

    # Metrics history
    train_metrics: Dict[str, List[float]] = field(default_factory=dict)
    val_metrics: Dict[str, List[float]] = field(default_factory=dict)

    # Early stopping
    patience_counter: int = 0
    should_stop: bool = False

    # Learning rate history
    lr_history: List[float] = field(default_factory=list)

    def update_train_metrics(self, metrics: Dict[str, float]):
        """
        Update training metrics.

        Args:
            metrics: Dictionary of metric name -> value
        """
        for name, value in metrics.items():
            if name not in self.train_metrics:
                self.train_metrics[name] = []
            self.train_metrics[name].append(value)

    def update_val_metrics(self, metrics: Dict[str, float]):
        """
        Update validation metrics.

        Args:
            metrics: Dictionary of metric name -> value
        """
        for name, value in metrics.items():
            if name not in self.val_metrics:
                self.val_metrics[name] = []
            self.val_metrics[name].append(value)

    def is_best_epoch(self, metric_value: float, mode: str = "min") -> bool:
        """
        Check if current epoch is best so far.

        Args:
            metric_value: Current metric value
            mode: 'min' or 'max'

        Returns:
            True if this is the best epoch
        """
        if mode == "min":
            is_best = metric_value < self.best_metric
        else:
            is_best = metric_value > self.best_metric

        if is_best:
            self.best_metric = metric_value
            self.best_epoch = self.current_epoch
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        return is_best

    def to_dict(self) -> Dict[str, Any]:
        """Export state as dictionary"""
        return {
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "train_metrics": self.train_metrics,
            "val_metrics": self.val_metrics,
            "patience_counter": self.patience_counter,
            "should_stop": self.should_stop,
            "lr_history": self.lr_history,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingState":
        """Load state from dictionary"""
        return cls(**data)

    def save(self, path: Path):
        """Save state to file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "TrainingState":
        """Load state from file"""
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = [
            f"Epoch: {self.current_epoch}",
            f"Global Step: {self.global_step}",
            f"Best Metric: {self.best_metric:.4f} (epoch {self.best_epoch})",
        ]

        if self.train_metrics:
            latest_train = {k: v[-1] for k, v in self.train_metrics.items() if v}
            lines.append(f"Train: {latest_train}")

        if self.val_metrics:
            latest_val = {k: v[-1] for k, v in self.val_metrics.items() if v}
            lines.append(f"Val: {latest_val}")

        return " | ".join(lines)
