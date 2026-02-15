"""
Custom learning rate schedulers.
"""

import math
from torch.optim.lr_scheduler import _LRScheduler


class WarmupLR(_LRScheduler):
    """
    Linear warmup learning rate scheduler.
    
    Linearly increases LR from 0 to base_lr over warmup_epochs.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant after warmup
            return self.base_lrs


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with linear warmup and cosine annealing.
    
    Args:
        optimizer: Optimizer
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of epochs
        min_lr: Minimum learning rate
        last_epoch: Last epoch number
    """
    
    def __init__(
        self,
        optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = (self.last_epoch + 1) / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay.
    
    Args:
        optimizer: Optimizer
        total_epochs: Total number of epochs
        power: Polynomial power
        min_lr: Minimum learning rate
    """
    
    def __init__(
        self,
        optimizer,
        total_epochs: int,
        power: float = 2.0,
        min_lr: float = 0,
        last_epoch: int = -1
    ):
        self.total_epochs = total_epochs
        self.power = power
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        progress = min(1.0, self.last_epoch / self.total_epochs)
        decay_factor = (1 - progress) ** self.power
        
        return [
            self.min_lr + (base_lr - self.min_lr) * decay_factor
            for base_lr in self.base_lrs
        ]


def get_scheduler(
    name: str,
    optimizer,
    **kwargs
):
    """
    Get scheduler by name.
    
    Args:
        name: Scheduler name
        optimizer: Optimizer
        **kwargs: Scheduler-specific arguments
        
    Returns:
        Scheduler instance
    """
    import torch.optim.lr_scheduler as torch_schedulers
    
    schedulers = {
        # PyTorch built-in
        'step': torch_schedulers.StepLR,
        'multistep': torch_schedulers.MultiStepLR,
        'exponential': torch_schedulers.ExponentialLR,
        'cosine': torch_schedulers.CosineAnnealingLR,
        'plateau': torch_schedulers.ReduceLROnPlateau,
        'onecycle': torch_schedulers.OneCycleLR,
        
        # Custom
        'warmup': WarmupLR,
        'warmup_cosine': WarmupCosineScheduler,
        'polynomial': PolynomialLR,
    }
    
    if name not in schedulers:
        raise ValueError(f"Unknown scheduler: {name}. Available: {list(schedulers.keys())}")
    
    return schedulers[name](optimizer, **kwargs)