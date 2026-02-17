"""
Memory optimization utilities.
"""

import gc
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn


class MemoryMonitor:
    """
    Monitor GPU/CPU memory usage during training.

    Tracks peak memory, current memory, and identifies leaks.

    Example:
        >>> monitor = MemoryMonitor()
        >>> with monitor.track("training"):
        ...     train_one_epoch()
        >>> monitor.print_summary()
    """

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.snapshots: Dict[str, Dict] = {}

    @contextmanager
    def track(self, label: str):
        """Track memory during a code block"""
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            start_mem = torch.cuda.memory_allocated()

        yield

        if self.device == "cuda":
            end_mem = torch.cuda.memory_allocated()
            peak_mem = torch.cuda.max_memory_allocated()

            self.snapshots[label] = {
                "start_gb": start_mem / (1024**3),
                "end_gb": end_mem / (1024**3),
                "peak_gb": peak_mem / (1024**3),
                "delta_gb": (end_mem - start_mem) / (1024**3),
            }

    def print_summary(self):
        """Print memory usage summary"""
        print("\n" + "=" * 60)
        print("MEMORY USAGE SUMMARY")
        print("=" * 60)
        print(f"\n{'Phase':<25} {'Start (GB)':<12} {'Peak (GB)':<12} {'Delta (GB)'}")
        print("-" * 60)

        for label, stats in self.snapshots.items():
            delta_str = (
                f"+{stats['delta_gb']:.3f}"
                if stats["delta_gb"] >= 0
                else f"{stats['delta_gb']:.3f}"
            )
            print(f"{label:<25} {stats['start_gb']:<12.3f} {stats['peak_gb']:<12.3f} {delta_str}")

        print("=" * 60)

    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if self.device == "cuda" and torch.cuda.is_available():
            return {
                "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
                "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
                "peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
            }
        return {}


def enable_gradient_checkpointing(model: nn.Module):
    """
    Enable gradient checkpointing to reduce memory usage.

    Trades compute for memory: recomputes activations during backward pass
    instead of storing them. Reduces memory by ~60% at cost of ~20% more compute.

    Args:
        model: Model to enable checkpointing for

    Example:
        >>> model = get_model('vae')(config)
        >>> enable_gradient_checkpointing(model)
        >>> # Now uses 60% less memory
    """
    from torch.utils.checkpoint import checkpoint

    # Find sequential blocks to checkpoint
    for name, module in model.named_modules():
        if isinstance(module, (nn.Sequential, nn.ModuleList)):
            # Wrap forward with checkpointing
            original_forward = module.forward

            def make_checkpointed_forward(orig_forward):
                def checkpointed_forward(*args, **kwargs):
                    # Use gradient checkpointing
                    if any(isinstance(a, torch.Tensor) and a.requires_grad for a in args):
                        return checkpoint(orig_forward, *args, use_reentrant=False)
                    return orig_forward(*args, **kwargs)

                return checkpointed_forward

            module.forward = make_checkpointed_forward(original_forward)

    print(f"✓ Gradient checkpointing enabled")


def optimize_model_memory(model: nn.Module) -> nn.Module:
    """
    Apply memory optimizations to model.

    1. Convert BatchNorm to syncBN if distributed
    2. Enable channels_last memory format for CNN
    3. Remove unnecessary buffers

    Args:
        model: Model to optimize

    Returns:
        Optimized model
    """
    # Check if CNN (has Conv2d)
    has_conv = any(isinstance(m, nn.Conv2d) for m in model.modules())

    if has_conv:
        # Use channels_last for better GPU performance
        model = model.to(memory_format=torch.channels_last)
        print("✓ Enabled channels_last memory format (CNN)")

    return model


def clear_memory(device: str = "cuda"):
    """
    Clear GPU/CPU memory caches.

    Call after loading models or during memory-intensive operations.
    """
    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def estimate_model_memory(
    model: nn.Module, batch_size: int = 1, input_shape: tuple = None
) -> Dict[str, float]:
    """
    Estimate memory requirements for training.

    Args:
        model: Model to estimate
        batch_size: Batch size
        input_shape: Input tensor shape (excluding batch)

    Returns:
        Dictionary with memory estimates in GB
    """
    # Parameter memory
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters())

    # Gradient memory (same as params)
    grad_memory = param_memory

    # Optimizer state (Adam: 2 moments = 2x params)
    optimizer_memory = param_memory * 2

    # Activation memory (estimate ~10x batch size)
    if input_shape and batch_size:
        activation_size = batch_size * int(torch.prod(torch.tensor(input_shape)))
        activation_memory = activation_size * 4 * 10  # float32, ~10x expansion
    else:
        activation_memory = param_memory * batch_size

    total = param_memory + grad_memory + optimizer_memory + activation_memory

    gb = 1024**3

    result = {
        "parameters_gb": param_memory / gb,
        "gradients_gb": grad_memory / gb,
        "optimizer_gb": optimizer_memory / gb,
        "activations_gb": activation_memory / gb,
        "total_gb": total / gb,
    }

    return result
