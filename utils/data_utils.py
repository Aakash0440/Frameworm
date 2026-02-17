"""
Optimized data loading utilities.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable, Any
import os
import numpy as np


def create_optimized_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
    drop_last: bool = False
) -> DataLoader:
    """
    Create DataLoader with optimized settings for maximum throughput.
    
    Auto-configures num_workers, pin_memory, and prefetch_factor
    based on system capabilities.
    
    Args:
        dataset: Dataset to load from
        batch_size: Batch size
        shuffle: Shuffle data
        num_workers: Workers for parallel loading (None = auto)
        pin_memory: Pin memory for faster GPU transfer (None = auto)
        prefetch_factor: Batches to prefetch per worker
        persistent_workers: Keep workers alive between epochs
        drop_last: Drop last incomplete batch
        
    Returns:
        Optimized DataLoader
        
    Example:
        >>> loader = create_optimized_dataloader(
        ...     dataset=train_dataset,
        ...     batch_size=128
        ... )
    """
    # Auto-configure num_workers
    if num_workers is None:
        cpu_count = os.cpu_count() or 1
        # Leave some CPUs for main process and other tasks
        num_workers = min(cpu_count - 2, 8)
        num_workers = max(num_workers, 0)
    
    # Auto-configure pin_memory
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    
    # Don't use persistent_workers if no workers
    if num_workers == 0:
        persistent_workers = False
        prefetch_factor = None
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=drop_last
    )
    
    return loader


class CachedDataset(Dataset):
    """
    Dataset with in-memory caching.
    
    Loads entire dataset to RAM for faster access.
    Useful for small datasets that fit in memory.
    
    Args:
        dataset: Underlying dataset to cache
        
    Example:
        >>> original = MyDataset(root='data')
        >>> cached = CachedDataset(original)
        >>> # Second epoch is much faster
    """
    
    def __init__(self, dataset: Dataset):
        self.cache = {}
        
        print(f"Caching {len(dataset)} samples to memory...")
        
        for i in range(len(dataset)):
            self.cache[i] = dataset[i]
        
        print(f"✓ Cached {len(self.cache)} samples")
    
    def __len__(self):
        return len(self.cache)
    
    def __getitem__(self, idx):
        return self.cache[idx]


class PrefetchLoader:
    """
    Wraps DataLoader with GPU prefetching.
    
    Loads next batch to GPU while processing current batch.
    Hides data transfer overhead.
    
    Args:
        loader: DataLoader to wrap
        device: Device to prefetch to
        
    Example:
        >>> prefetch_loader = PrefetchLoader(loader, device='cuda')
        >>> for batch in prefetch_loader:
        ...     # batch is already on GPU
        ...     train_step(batch)
    """
    
    def __init__(self, loader: DataLoader, device: str = 'cuda'):
        self.loader = loader
        self.device = device
    
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        stream = torch.cuda.Stream() if self.device == 'cuda' else None
        first = True
        
        for next_batch in self.loader:
            with torch.cuda.stream(stream) if stream else nullcontext():
                if isinstance(next_batch, (tuple, list)):
                    next_batch = tuple(
                        b.to(self.device, non_blocking=True) 
                        for b in next_batch
                    )
                else:
                    next_batch = next_batch.to(self.device, non_blocking=True)
            
            if not first:
                yield current_batch
            
            if stream:
                torch.cuda.current_stream().wait_stream(stream)
            
            current_batch = next_batch
            first = False
        
        if not first:
            yield current_batch


def benchmark_dataloader_configs(dataset: Dataset, batch_size: int = 128):
    """
    Find optimal DataLoader configuration.
    
    Tests different num_workers settings to find fastest config.
    
    Returns:
        Optimal num_workers value
    """
    import time
    
    max_workers = min(os.cpu_count() or 1, 8)
    configs = [0, 1, 2, 4, max_workers]
    
    print("Benchmarking DataLoader configurations:")
    print(f"{'num_workers':<14} {'Batches/sec':<14} {'Throughput'}")
    print("-"*50)
    
    best_config = 0
    best_throughput = 0
    
    for num_workers in configs:
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
        
        start = time.perf_counter()
        total_samples = 0
        
        for i, batch in enumerate(loader):
            if isinstance(batch, (tuple, list)):
                total_samples += len(batch[0])
            else:
                total_samples += len(batch)
            
            if i >= 20:  # Test 20 batches
                break
        
        elapsed = time.perf_counter() - start
        batches_per_sec = 21 / elapsed
        throughput = total_samples / elapsed
        
        print(f"{num_workers:<14} {batches_per_sec:<14.1f} {throughput:.0f} samples/sec")
        
        if throughput > best_throughput:
            best_throughput = throughput
            best_config = num_workers
    
    print(f"\n✓ Optimal num_workers: {best_config}")
    
    return best_config