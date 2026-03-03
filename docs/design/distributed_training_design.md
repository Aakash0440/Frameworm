# Distributed Training System Design

## Goals
1. Single-GPU training (baseline)
2. Multi-GPU training (DataParallel)
3. Distributed Data Parallel (DDP)
4. Multi-node support
5. Gradient synchronization
6. Fault tolerance

## Distributed Training Strategies

### 1. DataParallel (DP)
Simple multi-GPU on single machine.

**Pros:**
- Easy to use
- Single process
- Good for small models

**Cons:**
- GIL bottleneck
- Inefficient communication
- Limited scalability

### 2. DistributedDataParallel (DDP)
Multi-process distributed training.

**Pros:**
- Efficient communication
- Scales to multi-node
- Better performance

**Cons:**
- More complex setup
- Multi-process management

### 3. Model Parallel
Split model across devices (for huge models).

**Note:** Day 13-14 focus on Data Parallel

## Architecture
```python
from frameworm.distributed import DistributedTrainer

# Automatic distributed setup
trainer = DistributedTrainer(
    model=model,
    optimizer=optimizer,
    world_size=4,  # 4 GPUs
    backend='nccl'
)

# Train normally
trainer.train(train_loader, val_loader)
```

## Communication Backends

- **NCCL**: Best for NVIDIA GPUs
- **Gloo**: CPU and GPU, cross-platform
- **MPI**: Multi-node clusters

## Key Concepts

### World Size
Total number of processes (= # GPUs in data parallel)

### Rank
Process ID (0 to world_size - 1)

### Local Rank
GPU ID on current machine (0 to # GPUs per machine - 1)

### Master Process
Rank 0 - coordinates training, saves checkpoints

## Implementation Plan

1. **Environment Setup** - Detect GPUs, setup environment
2. **Process Management** - Launch distributed processes
3. **Model Wrapping** - Wrap model with DDP
4. **Data Distribution** - DistributedSampler for data
5. **Synchronization** - Barrier, all_reduce
6. **Checkpointing** - Save from master, load to all