"""Base class for trainers"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch

from core import Config


class BaseTrainer(ABC):
    """
    Abstract base class for all trainers.

    Trainers handle the training loop logic.
    """

    def __init__(self, model, config: Config):
        self.model = model
        self.config = config
        self.current_epoch = 0
        self.global_step = 0

    @abstractmethod
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Single training step.

        Args:
            batch: Training batch
            batch_idx: Batch index

        Returns:
            Dictionary with 'loss' and optionally other metrics
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Single validation step.

        Args:
            batch: Validation batch
            batch_idx: Batch index

        Returns:
            Dictionary with metrics
        """
        pass

    def fit(self, train_loader, val_loader=None):
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
        """
        num_epochs = self.config.training.epochs

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            self.train_epoch(train_loader)

            if val_loader is not None:
                self.validate_epoch(val_loader)

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()

        for batch_idx, batch in enumerate(train_loader):
            metrics = self.training_step(batch, batch_idx)
            self.global_step += 1

            # Log metrics (implement in subclass)
            self.log_metrics(metrics, step=self.global_step)

    def validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                metrics = self.validation_step(batch, batch_idx)
                # Aggregate metrics (implement in subclass)

    def log_metrics(self, metrics: Dict[str, Any], step: int):
        """
        Log metrics.

        Override in subclass to implement actual logging.
        """
        pass
