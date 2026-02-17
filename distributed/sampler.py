"""
Distributed data samplers.
"""

import torch
from torch.utils.data import Sampler, Dataset
from typing import Iterator, Optional
import math


class DistributedSampler(Sampler):
    """
    Sampler that splits data across processes.

    Each process gets a subset of the data. Ensures all processes
    process the same number of samples (padding if necessary).

    Args:
        dataset: Dataset to sample from
        num_replicas: Number of processes (world size)
        rank: Current process rank
        shuffle: Whether to shuffle
        seed: Random seed for shuffling
        drop_last: Whether to drop last incomplete batch

    Example:
        >>> sampler = DistributedSampler(dataset, num_replicas=4, rank=0)
        >>> loader = DataLoader(dataset, sampler=sampler)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            from distributed.utils import get_world_size

            num_replicas = get_world_size()

        if rank is None:
            from distributed.utils import get_rank

            rank = get_rank()

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last

        # Calculate samples per replica
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # Drop last incomplete batch
            self.num_samples = math.floor(len(self.dataset) / self.num_replicas)
        else:
            # Pad to make evenly divisible
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)

        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self) -> Iterator[int]:
        """Generate indices for this process"""
        # Shuffle or not
        if self.shuffle:
            # Deterministic shuffling based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # Add padding if needed
        if not self.drop_last:
            padding_size = self.total_size - len(indices)
            if padding_size > 0:
                indices += indices[:padding_size]
        else:
            # Drop last to make evenly divisible
            indices = indices[: self.total_size]

        assert len(indices) == self.total_size

        # Subsample for this rank
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self) -> int:
        """Number of samples for this process"""
        return self.num_samples

    def set_epoch(self, epoch: int):
        """
        Set epoch for shuffling.

        Call this before each epoch to ensure proper shuffling
        across processes.

        Args:
            epoch: Current epoch number
        """
        self.epoch = epoch


def get_distributed_sampler(
    dataset: Dataset, shuffle: bool = True, seed: int = 0
) -> Optional[DistributedSampler]:
    """
    Get distributed sampler if in distributed mode.

    Args:
        dataset: Dataset
        shuffle: Whether to shuffle
        seed: Random seed

    Returns:
        DistributedSampler if distributed, None otherwise
    """
    from distributed.utils import is_distributed

    if is_distributed():
        return DistributedSampler(dataset, shuffle=shuffle, seed=seed)
    return None
