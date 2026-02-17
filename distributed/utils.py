"""
Distributed training utilities.
"""

import os
import pickle
import socket
from typing import Any, Optional

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    """Check if running in distributed mode"""
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    """Get total number of processes"""
    if is_distributed():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get process rank"""
    if is_distributed():
        return dist.get_rank()
    return 0


def get_local_rank() -> int:
    """Get local rank (GPU ID on current machine)"""
    if is_distributed():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def is_master() -> bool:
    """Check if this is the master process (rank 0)"""
    return get_rank() == 0


def setup_distributed(
    backend: str = "nccl",
    init_method: Optional[str] = None,
    world_size: Optional[int] = None,
    rank: Optional[int] = None,
):
    """
    Initialize distributed training.

    Args:
        backend: Communication backend ('nccl', 'gloo', 'mpi')
        init_method: Initialization method URL
        world_size: Total number of processes
        rank: Current process rank
    """
    if is_distributed():
        print("Distributed already initialized")
        return

    # Get from environment if not provided
    if world_size is None:
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    if rank is None:
        rank = int(os.environ.get("RANK", 0))

    if world_size == 1:
        print("Single process training (no distributed)")
        return

    # Default init method
    if init_method is None:
        # Try to find free port
        if rank == 0:
            port = find_free_port()
            os.environ["MASTER_PORT"] = str(port)
        else:
            port = int(os.environ.get("MASTER_PORT", 29500))

        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        init_method = f"tcp://{master_addr}:{port}"

    # Initialize process group
    dist.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank
    )

    print(f"Distributed initialized: rank={rank}/{world_size}, backend={backend}")


def cleanup_distributed():
    """Cleanup distributed training"""
    if is_distributed():
        dist.destroy_process_group()


def find_free_port() -> int:
    """Find a free port for distributed training"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


def barrier():
    """Synchronize all processes"""
    if is_distributed():
        dist.barrier()


def all_reduce(tensor: torch.Tensor, op=dist.ReduceOp.SUM):
    """
    All-reduce tensor across processes.

    Args:
        tensor: Tensor to reduce
        op: Reduction operation (SUM, PRODUCT, MIN, MAX)
    """
    if is_distributed():
        dist.all_reduce(tensor, op=op)
    return tensor


def all_gather(tensor: torch.Tensor) -> list:
    """
    Gather tensor from all processes.

    Args:
        tensor: Tensor to gather

    Returns:
        List of tensors from all processes
    """
    if not is_distributed():
        return [tensor]

    world_size = get_world_size()
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)

    return tensor_list


def broadcast(tensor: torch.Tensor, src: int = 0):
    """
    Broadcast tensor from source to all processes.

    Args:
        tensor: Tensor to broadcast
        src: Source rank
    """
    if is_distributed():
        dist.broadcast(tensor, src=src)
    return tensor


def all_reduce_dict(data: dict) -> dict:
    """
    All-reduce dictionary of tensors.

    Args:
        data: Dictionary with tensor values

    Returns:
        Dictionary with reduced tensors
    """
    if not is_distributed():
        return data

    # Convert to tensor
    keys = list(data.keys())
    values = torch.tensor([data[k].item() if torch.is_tensor(data[k]) else data[k] for k in keys])

    # Reduce
    all_reduce(values)

    # Average
    values = values / get_world_size()

    # Convert back
    return {k: v.item() for k, v in zip(keys, values)}


def synchronize():
    """Synchronize all CUDA devices"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    barrier()


class DistributedContext:
    """
    Context manager for distributed training.

    Example:
        >>> with DistributedContext(backend='nccl'):
        ...     # Distributed training code
        ...     pass
    """

    def __init__(
        self,
        backend: str = "nccl",
        init_method: Optional[str] = None,
        world_size: Optional[int] = None,
        rank: Optional[int] = None,
    ):
        self.backend = backend
        self.init_method = init_method
        self.world_size = world_size
        self.rank = rank

    def __enter__(self):
        setup_distributed(
            backend=self.backend,
            init_method=self.init_method,
            world_size=self.world_size,
            rank=self.rank,
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        cleanup_distributed()
        return False
