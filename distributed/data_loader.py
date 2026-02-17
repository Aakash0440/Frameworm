"""
Optimized data loading for distributed training.
"""

import torch
from torch.utils.data import DataLoader, Dataset
from typing import Optional, Callable
import multiprocessing as mp

from distributed.sampler import get_distributed_sampler


class OptimizedDataLoader:
    """
    Factory for creating optimized DataLoaders.

    Automatically configures:
    - Distributed sampler
    - Number of workers
    - Pin memory
    - Prefetch factor
    - Persistent workers
    """

    @staticmethod
    def create(
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        num_workers: Optional[int] = None,
        pin_memory: Optional[bool] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = True,
        drop_last: bool = False,
        **kwargs,
    ) -> DataLoader:
        """
        Create optimized DataLoader.

        Args:
            dataset: Dataset
            batch_size: Batch size per process
            shuffle: Whether to shuffle
            num_workers: Number of workers (auto if None)
            pin_memory: Pin memory (auto if None)
            prefetch_factor: Batches to prefetch
            persistent_workers: Keep workers alive
            drop_last: Drop last incomplete batch
            **kwargs: Additional DataLoader arguments

        Returns:
            Optimized DataLoader
        """
        # Auto-configure num_workers
        if num_workers is None:
            num_workers = OptimizedDataLoader._auto_num_workers()

        # Auto-configure pin_memory
        if pin_memory is None:
            pin_memory = torch.cuda.is_available()

        # Get distributed sampler
        sampler = get_distributed_sampler(dataset, shuffle=shuffle)

        # If using sampler, don't shuffle in DataLoader
        if sampler is not None:
            shuffle = False

        # Persistent workers only makes sense with num_workers > 0
        if num_workers == 0:
            persistent_workers = False

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
            **kwargs,
        )

    @staticmethod
    def _auto_num_workers() -> int:
        """
        Auto-configure number of workers.

        Returns:
            Optimal number of workers
        """
        # Rule of thumb: 2-4 workers per GPU
        try:
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            cpu_count = mp.cpu_count()

            # 4 workers per GPU, but cap at CPU count
            workers = min(4 * num_gpus, cpu_count)

            # At least 1, at most 16
            workers = max(1, min(workers, 16))

            return workers
        except:
            return 4  # Safe default


def benchmark_data_loading(
    dataset: Dataset, batch_size: int, num_workers_list: list = [0, 2, 4, 8], num_batches: int = 100
):
    """
    Benchmark different data loading configurations.

    Args:
        dataset: Dataset to benchmark
        batch_size: Batch size
        num_workers_list: List of worker counts to test
        num_batches: Number of batches to time
    """
    import time

    print(f"\nBenchmarking Data Loading (batch_size={batch_size}):")
    print("=" * 60)

    results = []

    for num_workers in num_workers_list:
        loader = OptimizedDataLoader.create(
            dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
        )

        # Warmup
        for i, batch in enumerate(loader):
            if i >= 10:
                break

        # Time
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        elapsed = time.time() - start

        batches_per_sec = num_batches / elapsed
        results.append((num_workers, batches_per_sec))

        print(f"  Workers={num_workers}: {batches_per_sec:.1f} batches/sec")

    # Find best
    best = max(results, key=lambda x: x[1])
    print(f"\n  Best: {best[0]} workers ({best[1]:.1f} batches/sec)")
    print("=" * 60)

    return results
