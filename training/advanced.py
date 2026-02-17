"""
Advanced training features.
"""

import torch
import torch.nn as nn
from typing import Optional, Iterator
from copy import deepcopy


class GradientAccumulator:
    """
    Handles gradient accumulation.

    Args:
        accumulation_steps: Number of steps to accumulate gradients
    """

    def __init__(self, accumulation_steps: int = 1):
        self.accumulation_steps = accumulation_steps
        self.current_step = 0

    def should_update(self) -> bool:
        """Check if optimizer should step"""
        self.current_step += 1
        return self.current_step % self.accumulation_steps == 0

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Scale loss by accumulation steps"""
        return loss / self.accumulation_steps

    def reset(self):
        """Reset counter"""
        self.current_step = 0


class GradientClipper:
    """
    Handles gradient clipping.

    Args:
        max_norm: Maximum gradient norm
        norm_type: Type of norm (2 for L2)
    """

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, parameters: Iterator[nn.Parameter]) -> float:
        """
        Clip gradients and return total norm.

        Args:
            parameters: Model parameters

        Returns:
            Total gradient norm before clipping
        """
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, self.max_norm, norm_type=self.norm_type
        )
        return total_norm.item()


class EMAModel:
    """
    Exponential Moving Average of model weights.

    Maintains a moving average of model parameters which often
    leads to better generalization.

    Args:
        model: Model to track
        decay: EMA decay rate (0.999 is common)
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay

        # Create shadow parameters
        self.shadow = {}
        self.backup = {}

        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        """Update EMA parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        """Apply EMA parameters to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        """Restore original parameters"""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        """Get EMA state"""
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state_dict):
        """Load EMA state"""
        self.decay = state_dict["decay"]
        self.shadow = state_dict["shadow"]


def compute_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """
    Compute total gradient norm.

    Args:
        model: Model
        norm_type: Type of norm

    Returns:
        Total gradient norm
    """
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type

    return total_norm ** (1.0 / norm_type)
