# Distributed Training Guide

Scale training to multiple GPUs and machines.

---

## Single-Machine Multi-GPU

### DataParallel (Simplest)
```python
from frameworm import Trainer

trainer = Trainer(model, optimizer)
trainer.enable_data_parallel(device_ids=[0, 1, 2, 3])
trainer.train(train_loader, val_loader)
```

**Pros:**
- One line of code
- No process management

**Cons:**
- GIL bottleneck
- Unbalanced GPU usage
- Slower than DDP

**Use when:** Quick prototyping, < 4 GPUs

---

### DistributedDataParallel (Recommended)
```python
import torch.multiprocessing as mp
from frameworm.distributed import DDPTrainer, setup_ddp

def train_worker(rank, world_size):
    # Setup DDP
    setup_ddp(rank, world_size)
    
    # Create model and trainer
    trainer = DDPTrainer(model, optimizer, rank, world_size)
    
    # Train
    trainer.train(train_loader, val_loader, epochs=100)
    
    # Cleanup
    trainer.cleanup()

if __name__ == '__main__':
    world_size = 4  # Number of GPUs
    mp.spawn(train_worker, args=(world_size,), nprocs=world_size, join=True)
```

**Pros:**
- Fastest for multi-GPU
- Balanced GPU usage
- Scales to 100s of GPUs

**Cons:**
- More complex setup
- Multi-process

**Use when:** > 4 GPUs, production training

---

## Multi-Machine Training

### Setup

**Node 0 (master):**
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12355
export RANK=0
export WORLD_SIZE=8  # Total GPUs across all nodes

frameworm train --config config.yaml --distributed
```

**Node 1 (worker):**
```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=12355
export RANK=4  # Offset by number of GPUs on previous nodes
export WORLD_SIZE=8

frameworm train --config config.yaml --distributed
```

---

## Performance Optimization

### Mixed Precision
```python
trainer.enable_mixed_precision()
```

**Speedup:** 2-3x on modern GPUs (V100, A100)

### Gradient Compression
```python
from frameworm.distributed import enable_gradient_compression

enable_gradient_compression(ddp_model, rank)
```

**Bandwidth reduction:** ~10x

### Optimal Batch Size

| GPUs | Batch Size | Effective Batch Size |
|------|-----------|---------------------|
| 1 | 128 | 128 |
| 4 | 128 | 512 |
| 8 | 128 | 1024 |
| 16 | 64 | 1024 |

**Rule:** Keep effective batch size constant, adjust learning rate.

---

## Troubleshooting

**NCCL Timeout:**
```bash
export NCCL_TIMEOUT=1800  # 30 minutes
```

**Unbalanced GPUs:**
- Check data loading is distributed
- Verify batch sizes are equal
- Monitor with `nvidia-smi`

**Out of Memory:**
- Reduce batch size
- Enable gradient checkpointing
- Use gradient accumulation