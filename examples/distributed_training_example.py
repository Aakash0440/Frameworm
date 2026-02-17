"""
Example: Distributed Training

Demonstrates:
- Single-GPU training
- Multi-GPU with DataParallel
- Multi-GPU with DistributedDataParallel
- Multi-node training setup
"""

import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core import Config, get_model
from distributed import cleanup_distributed, get_rank, get_world_size, is_master, setup_distributed
from distributed.data_parallel import DataParallelTrainer
from distributed.trainer import DistributedTrainer


def get_mnist_loaders(batch_size=128):
    """Get MNIST loaders"""
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def example_single_gpu():
    """Example 1: Single-GPU training (baseline)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: SINGLE-GPU TRAINING")
    print("=" * 60)

    # Config
    config = Config("configs/models/vae/vanilla.yaml")
    config.training.epochs = 3
    config.training.batch_size = 128

    # Data
    train_loader, val_loader = get_mnist_loaders(config.training.batch_size)

    # Model
    vae = get_model("vae")(config)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    # Train
    from training import Trainer

    trainer = Trainer(
        model=vae, optimizer=optimizer, device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print("\nTraining on single GPU...")
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)

    print("\n✓ Single-GPU training complete")


def example_data_parallel():
    """Example 2: DataParallel (simple multi-GPU)"""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: DATAPARALLEL (MULTI-GPU)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping")
        return

    if torch.cuda.device_count() < 2:
        print("⚠️  Need at least 2 GPUs for DataParallel, skipping")
        return

    # Config
    config = Config("configs/models/vae/vanilla.yaml")
    config.training.epochs = 3
    config.training.batch_size = 128

    # Data
    train_loader, val_loader = get_mnist_loaders(config.training.batch_size)

    # Model with DataParallel
    vae = get_model("vae")(config)
    vae = DataParallelTrainer.wrap(vae)  # Wrap with DataParallel

    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    # Train
    from training import Trainer

    trainer = Trainer(model=vae, optimizer=optimizer, device="cuda:0")

    print(f"\nTraining with DataParallel on {torch.cuda.device_count()} GPUs...")
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)

    print("\n✓ DataParallel training complete")


def train_ddp(rank: int, world_size: int):
    """
    Training function for DDP.

    Args:
        rank: Process rank
        world_size: Total number of processes
    """
    print(f"[Rank {rank}] Starting DDP training")

    # Setup distributed
    setup_distributed(backend="nccl")

    # Config
    config = Config("configs/models/vae/vanilla.yaml")
    config.training.epochs = 3
    config.training.batch_size = 128  # Per-process batch size

    # Data
    train_loader, val_loader = get_mnist_loaders(config.training.batch_size)

    # Model
    vae = get_model("vae")(config)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    # Distributed trainer
    trainer = DistributedTrainer(model=vae, optimizer=optimizer, backend="nccl")

    if is_master():
        print(f"\nTraining with DDP on {world_size} GPUs...")

    trainer.train(train_loader, val_loader, epochs=config.training.epochs)

    if is_master():
        print("\n✓ DDP training complete")

    # Cleanup
    cleanup_distributed()


def example_ddp():
    """Example 3: DistributedDataParallel"""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: DISTRIBUTEDDATAPARALLEL (DDP)")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("⚠️  CUDA not available, skipping")
        return

    n_gpus = torch.cuda.device_count()

    if n_gpus < 2:
        print("⚠️  Need at least 2 GPUs for DDP, skipping")
        return

    # Launch distributed training
    from distributed.trainer import launch_distributed

    print(f"Launching DDP training on {n_gpus} GPUs...")
    launch_distributed(train_fn=train_ddp, nprocs=n_gpus, backend="nccl")


def example_multi_node_setup():
    """Example 4: Multi-node training setup"""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: MULTI-NODE TRAINING SETUP")
    print("=" * 60)

    print("""
Multi-node training requires:

1. Set environment variables on each node:

   # Node 0 (master):
   export MASTER_ADDR=<node0_ip>
   export MASTER_PORT=29500
   export WORLD_SIZE=8  # 2 nodes × 4 GPUs
   export RANK=0
   python train.py

   # Node 1:
   export MASTER_ADDR=<node0_ip>
   export MASTER_PORT=29500
   export WORLD_SIZE=8
   export RANK=4  # Start from 4 (node0 has ranks 0-3)
   python train.py

2. Use DistributedTrainer as normal:

   trainer = DistributedTrainer(model, optimizer)
   trainer.train(train_loader, val_loader)

3. The framework handles the rest automatically!
    """)


def main():
    print("Distributed Training Examples")
    print("=" * 60)

    # Example 1: Single-GPU (always works)
    example_single_gpu()

    # Example 2: DataParallel (if multiple GPUs)
    example_data_parallel()

    # Example 3: DDP (if multiple GPUs)
    example_ddp()

    # Example 4: Multi-node setup (informational)
    example_multi_node_setup()

    print("\n" + "=" * 60)
    print("Examples complete!")


if __name__ == "__main__":
    main()
