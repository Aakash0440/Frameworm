# Distributed Training

## Overview

Frameworm supports distributed training for faster training on multiple GPUs and machines.

## Training Modes

### 1. Single-GPU (Baseline)
```python
from frameworm.training import Trainer

trainer = Trainer(model, optimizer, device='cuda:0')
trainer.train(train_loader, val_loader)
```

### 2. DataParallel (Simple Multi-GPU)
```python
from frameworm.distributed import DataParallelTrainer

# Wrap model
model = DataParallelTrainer.wrap(model)

# Train normally
trainer = Trainer(model, optimizer, device='cuda:0')
trainer.train(train_loader, val_loader)
```

**Pros:** Simple, single process
**Cons:** Less efficient, GIL bottleneck

### 3. DistributedDataParallel (Recommended)
```python
from frameworm.distributed import DistributedTrainer

# Automatically handles DDP
trainer = DistributedTrainer(model, optimizer, backend='nccl')
trainer.train(train_loader, val_loader)
```

**Pros:** Efficient, scales well
**Cons:** Multi-process setup

## Single-Machine Multi-GPU

### Automatic Launch
```python
from frameworm.distributed.trainer import launch_distributed

def train_fn(rank, world_size):
    # Setup
    setup_distributed()
    
    # Train
    trainer = DistributedTrainer(model, optimizer)
    trainer.train(train_loader, val_loader)
    
    # Cleanup
    cleanup_distributed()

# Launch on all GPUs
launch_distributed(train_fn, nprocs=torch.cuda.device_count())
```

### Manual Launch with torchrun
```bash
# Using torchrun (PyTorch 1.9+)
torchrun --nproc_per_node=4 train.py

# Or older torch.distributed.launch
python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## Multi-Node Training

### Node Configuration

**Node 0 (master):**
```bash
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export WORLD_SIZE=8  # Total processes
export RANK=0
python train.py
```

**Node 1:**
```bash
export MASTER_ADDR=192.168.1.1
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4  # Offset by node0's GPU count
python train.py
```

### Training Script
```python
from frameworm.distributed import setup_distributed, DistributedTrainer

# Setup from environment variables
setup_distributed()

# Train
trainer = DistributedTrainer(model, optimizer)
trainer.train(train_loader, val_loader)
```

## Data Loading

### Distributed Sampler

Automatically used by DistributedTrainer:
```python
# Manual usage
from frameworm.distributed import DistributedSampler

sampler = DistributedSampler(dataset, shuffle=True)
loader = DataLoader(dataset, sampler=sampler, batch_size=64)

# Set epoch for proper shuffling
for epoch in range(epochs):
    sampler.set_epoch(epoch)
    for batch in loader:
        # Training...
```

### Batch Size

**Important:** Batch size is PER PROCESS.
```python
# Effective batch size = batch_size × world_size
batch_size = 32  # Per GPU
world_size = 4   # 4 GPUs
# Total effective batch size = 128
```

## Checkpointing

Only master process saves:
```python
trainer = DistributedTrainer(model, optimizer)
trainer.train(train_loader, val_loader)

# Automatically saves only from rank 0
trainer.save_checkpoint('checkpoint.pt')

# All processes load
trainer.load_checkpoint('checkpoint.pt')
```

## Metric Aggregation

Metrics automatically averaged across processes:
```python
# Each process computes local metrics
# DistributedTrainer averages across all processes
trainer.train(train_loader, val_loader)

# Final metrics are averaged
print(trainer.state.val_metrics)  # Already aggregated
```

## Communication Backends

### NCCL (NVIDIA)
```python
trainer = DistributedTrainer(model, optimizer, backend='nccl')
```
- **Best for:** NVIDIA GPUs
- **Supports:** CUDA only
- **Performance:** Fastest

### Gloo
```python
trainer = DistributedTrainer(model, optimizer, backend='gloo')
```
- **Best for:** CPU or mixed CPU/GPU
- **Supports:** CPU and CUDA
- **Performance:** Good

### MPI
```python
trainer = DistributedTrainer(model, optimizer, backend='mpi')
```
- **Best for:** HPC clusters
- **Supports:** CPU and CUDA
- **Performance:** Good

## Best Practices

1. **Use NCCL for GPUs** - Fastest backend
2. **Set epoch in sampler** - For proper shuffling
3. **Scale learning rate** - lr × world_size for large batches
4. **Gradient accumulation** - For even larger effective batches
5. **Warmup** - Help large batch training converge

## Troubleshooting

### NCCL Timeout
```python
import os
os.environ['NCCL_TIMEOUT'] = '1800'  # 30 minutes
```

### Find Unused Parameters Error
```python
trainer = DistributedTrainer(
    model,
    optimizer,
    find_unused_parameters=True  # For dynamic graphs
)
```

### Hangs at Initialization
```bash
# Check firewall
sudo ufw allow 29500

# Use different port
export MASTER_PORT=29501
```

## Examples

See `examples/distributed_training_example.py` for complete examples.