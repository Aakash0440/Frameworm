"""
DataParallel wrapper (simple multi-GPU).
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.nn import DataParallel as DP


class DataParallelTrainer:
    """
    Simple DataParallel wrapper for multi-GPU training.

    Simpler than DDP but less efficient. Good for quick prototyping.

    Args:
        model: Model to wrap
        device_ids: List of GPU IDs (None = all available)

    Example:
        >>> model = DataParallelTrainer.wrap(model)
        >>> # Train as normal
    """

    @staticmethod
    def wrap(model: nn.Module, device_ids: Optional[List[int]] = None) -> nn.Module:
        """
        Wrap model with DataParallel.

        Args:
            model: Model to wrap
            device_ids: GPU IDs to use

        Returns:
            Wrapped model
        """
        if not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            return model

        if device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))

        if len(device_ids) == 0:
            print("No GPUs specified, using CPU")
            return model

        if len(device_ids) == 1:
            print(f"Single GPU ({device_ids[0]}), not using DataParallel")
            return model.cuda(device_ids[0])

        print(f"Using DataParallel on GPUs: {device_ids}")

        # Move to first device
        model = model.cuda(device_ids[0])

        # Wrap with DataParallel
        model = DP(model, device_ids=device_ids)

        return model


def is_data_parallel(model: nn.Module) -> bool:
    """Check if model is wrapped with DataParallel"""
    return isinstance(model, DP)


def unwrap_data_parallel(model: nn.Module) -> nn.Module:
    """Unwrap DataParallel model"""
    if is_data_parallel(model):
        return model.module
    return model
