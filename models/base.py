"""Base classes for all models"""

from abc import ABC, abstractmethod
from typing import Any, Dict
import torch
import torch.nn as nn
from core import Config
from core.exceptions import DimensionMismatchError


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.

    All models in Frameworm should inherit from this class and implement
    the required abstract methods.

    Args:
        config: Configuration object or dict
    """

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._is_built = False

    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass of the model. Must be implemented by subclasses."""
        pass

    def build(self):
        """Build the model architecture. Override for lazy building."""
        self._is_built = True

    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config)

    def summary(self):
        """Print model summary."""
        print(f"\n{self.__class__.__name__} Summary")
        print("=" * 60)
        total_params = self.count_parameters()
        print(f"Total parameters: {total_params:,}")
        print("=" * 60)
        print()

    def save(self, path: str):
        """Save model weights."""
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "config": self.get_config(),
            },
            path,
        )

    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint["model_state_dict"])

    # -----------------------
    # New utility methods
    # -----------------------

    def get_device(self) -> torch.device:
        """Return the device of the model's parameters."""
        try:
            return next(self.parameters()).device
        except StopIteration:
            # Model has no parameters yet
            return torch.device("cpu")

    def to_device(self, device: str):
        """Move the model to the specified device (CPU/GPU)."""
        self.to(device)
        return self

    def count_parameters(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """String representation"""
        num_params = self.count_parameters()
        return f"{self.__class__.__name__}(params={num_params:,})"
