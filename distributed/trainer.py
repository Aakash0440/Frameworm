"""
Distributed training wrapper for Trainer.
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from typing import Optional, Dict, Any
import os

from training import Trainer
from distributed.utils import (
    is_distributed,
    get_rank,
    get_local_rank,
    is_master,
    setup_distributed,
    cleanup_distributed,
    barrier,
    all_reduce_dict
)
from distributed.sampler import get_distributed_sampler


class DistributedTrainer(Trainer):
    """
    Trainer with distributed training support.
    
    Automatically handles:
    - Model wrapping with DDP
    - Distributed data loading
    - Gradient synchronization
    - Metric aggregation across processes
    - Checkpointing from master process only
    
    Args:
        model: Model to train
        optimizer: Optimizer
        backend: Distributed backend ('nccl', 'gloo')
        find_unused_parameters: DDP parameter (set True if dynamic graph)
        **kwargs: Additional Trainer arguments
        
    Example:
        >>> trainer = DistributedTrainer(
        ...     model=model,
        ...     optimizer=optimizer,
        ...     backend='nccl'
        ... )
        >>> trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        backend: str = 'nccl',
        find_unused_parameters: bool = False,
        **kwargs
    ):
        # Initialize distributed if not already done
        if not is_distributed():
            world_size = int(os.environ.get('WORLD_SIZE', 1))
            if world_size > 1:
                setup_distributed(backend=backend)
        
        # Set device based on local rank
        if torch.cuda.is_available() and is_distributed():
            device = f'cuda:{get_local_rank()}'
            torch.cuda.set_device(device)
        else:
            device = kwargs.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        kwargs['device'] = device
        
        # Move model to device before DDP wrapping
        model = model.to(device)
        
        # Wrap model with DDP if distributed
        if is_distributed():
            model = DDP(
                model,
                device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters
            )
            print(f"[Rank {get_rank()}] Model wrapped with DDP")
        
        # Initialize parent Trainer
        super().__init__(model, optimizer, **kwargs)
        
        self.backend = backend
        self.is_distributed = is_distributed()
    
    def _create_dataloader(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        **kwargs
    ) -> DataLoader:
        """
        Create DataLoader with distributed sampler.
        
        Args:
            dataset: Dataset
            batch_size: Batch size PER PROCESS
            shuffle: Whether to shuffle
            **kwargs: Additional DataLoader arguments
            
        Returns:
            DataLoader with DistributedSampler if distributed
        """
        # Get distributed sampler if needed
        sampler = get_distributed_sampler(dataset, shuffle=shuffle)
        
        # If using distributed sampler, don't use shuffle in DataLoader
        if sampler is not None:
            shuffle = False
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            **kwargs
        )
    
    def train_epoch(self, train_loader: DataLoader, epoch: int):
        # Set epoch for distributed sampler
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        # Get metrics from parent
        metrics = super().train_epoch(train_loader, epoch)

        # Synchronize after epoch
        if self.is_distributed:
            barrier()

        return metrics

    
    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """
        Validate one epoch with distributed aggregation.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch
            
        Returns:
            Aggregated metrics across all processes
        """
        # Call parent validate_epoch
        metrics = super().validate_epoch(val_loader, epoch)
        
        # Aggregate metrics across processes
        if self.is_distributed:
            metrics = self._aggregate_metrics(metrics)
        
        return metrics
    
    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        """
        Aggregate metrics across all processes.
        
        Args:
            metrics: Local metrics
            
        Returns:
            Aggregated metrics (averaged across processes)
        """
        if not self.is_distributed:
            return metrics
        
        # Convert to tensors
        metric_tensors = {
            k: torch.tensor(v, device=self.device)
            for k, v in metrics.items()
        }
        
        # All-reduce and average
        aggregated = all_reduce_dict(metric_tensors)
        
        return {k: float(v) for k, v in aggregated.items()}
    
    def save_checkpoint(self, path: str, epoch: int):
        """
        Save checkpoint (only from master process).
        """
        if not is_master():
            if self.is_distributed:
                barrier()
            return

        # Unwrap DDP model if needed
        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": self.state.to_dict() if hasattr(self.state, "to_dict") else None,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        if self.is_distributed:
            barrier()

    
    def load_checkpoint(self, path: str):
        """
        Load checkpoint (all processes load).
        
        Args:
            path: Checkpoint path
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)
        
        # Unwrap DDP model if needed
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        
        # Load states
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Synchronize
        if self.is_distributed:
            barrier()
    
    def cleanup(self):
        """Cleanup distributed training"""
        if self.is_distributed:
            cleanup_distributed()


def launch_distributed(
    train_fn,
    nprocs: int = None,
    backend: str = 'nccl',
    **kwargs
):
    """
    Launch distributed training using torch.multiprocessing.
    
    Args:
        train_fn: Training function that takes (rank, world_size, **kwargs)
        nprocs: Number of processes (defaults to # GPUs)
        backend: Distributed backend
        **kwargs: Additional arguments for train_fn
    """
    import torch.multiprocessing as mp
    
    if nprocs is None:
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if nprocs <= 1:
        print("Single process training")
        train_fn(0, 1, **kwargs)
        return
    
    print(f"Launching {nprocs} processes for distributed training")
    
    # Set environment variables
    os.environ['WORLD_SIZE'] = str(nprocs)
    os.environ['MASTER_ADDR'] = 'localhost'
    
    # Find free port
    from frameworm.distributed.utils import find_free_port
    os.environ['MASTER_PORT'] = str(find_free_port())
    
    # Spawn processes
    mp.spawn(
        train_fn,
        args=(nprocs,) + tuple(kwargs.values()),
        nprocs=nprocs,
        join=True
    )