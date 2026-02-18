# Distributed Training Guide

Scale FRAMEWORM training across multiple GPUs and machines.

**What you'll learn:**
- DataParallel vs DistributedDataParallel
- Multi-machine setup
- Gradient compression
- Mixed precision at scale

**Hardware:** 2+ GPUs  
**Difficulty:** Advanced

---

## Single Machine, Multiple GPUs

### DataParallel (Simple, lower performance)
```python
import torch
from frameworm import Config, get_model, Trainer

# Create model
config = Config.from_file('config.yaml')
model = get_model('vae')(config)

# Wrap in DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    print(f"Using {torch.cuda.device_count()} GPUs")

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
trainer = Trainer(model, optimizer, device='cuda')

trainer.train(train_loader, val_loader, epochs=50)
```

---

### DistributedDataParallel (Recommended)
```python
# train_distributed.py

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from frameworm import Config, get_model, Trainer


def setup(rank, world_size):
    """Initialize distributed training"""
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up distributed training"""
    dist.destroy_process_group()


def train_ddp(rank, world_size, config):
    """Train with DDP"""
    setup(rank, world_size)
    
    # Create model on current GPU
    model = get_model('vae')(config).to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Distributed sampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=128 // world_size,  # Split batch across GPUs
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Trainer
    trainer = Trainer(model, optimizer, device=rank)
    trainer.train(train_loader, val_loader, epochs=50)
    
    cleanup()


if __name__ == '__main__':
    import torch.multiprocessing as mp
    
    world_size = torch.cuda.device_count()
    config = Config.from_file('config.yaml')
    
    # Spawn processes
    mp.spawn(
        train_ddp,
        args=(world_size, config),
        nprocs=world_size,
        join=True
    )
```

Launch:
```bash
# Single machine, 4 GPUs
python train_distributed.py
```

---

## Multi-Machine Training

### Setup on each machine:
```bash
# Machine 1 (master):
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export WORLD_SIZE=8  # Total GPUs across all machines
export RANK=0        # Machine rank (0 for master)

python train_distributed.py

# Machine 2 (worker):
export MASTER_ADDR=192.168.1.10
export MASTER_PORT=29500
export WORLD_SIZE=8
export RANK=4        # Start from 4 (if machine 1 has 4 GPUs)

python train_distributed.py
```

---

## Mixed Precision + Gradient Compression
```python
from torch.cuda.amp import autocast, GradScaler

def train_optimized(rank, world_size, config):
    setup(rank, world_size)
    
    model = get_model('vae')(config).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # For mixed precision
    
    # Enable gradient compression (PowerSGD)
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook
    state = powerSGD_hook.PowerSGDState(
        process_group=None,
        matrix_approximation_rank=1
    )
    model.register_comm_hook(state, powerSGD_hook.powerSGD_hook)
    
    for epoch in range(50):
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Mixed precision forward
            with autocast():
                output = model(batch)
                loss = output['loss']
            
            # Scaled backward
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
    
    cleanup()
```

---

## Performance Tips

### 1. Gradient Accumulation for Large Batches
```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    output = model(batch)
    loss = output['loss'] / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 2. Optimal Batch Size per GPU
```python
# Rule of thumb: Largest batch that fits in memory
# For V100 16GB: batch_size=128 for 64x64 images
# For A100 40GB: batch_size=256
```

### 3. Communication Overlap
```python
# DDP automatically overlaps computation and communication
# Ensure small model layers (reduces communication overhead)
```

---

## Monitoring

### Track training across all GPUs:
```python
import torch.distributed as dist

# Synchronize metrics across GPUs
if dist.is_initialized():
    # Reduce loss across all processes
    loss_tensor = torch.tensor([loss.item()]).to(rank)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
    avg_loss = loss_tensor.item() / world_size
    
    if rank == 0:
        print(f"Average loss across all GPUs: {avg_loss:.4f}")
```

---

## Troubleshooting

**NCCL timeout errors:**
```bash
export NCCL_TIMEOUT=3600  # Increase timeout
export NCCL_DEBUG=INFO    # Enable debug logs
```

**OOM on some GPUs:**
```python
# Reduce batch size or enable gradient checkpointing
trainer.enable_gradient_checkpointing()
```

**Slow synchronization:**
```python
# Use gradient compression
# Enable FP16
# Reduce communication frequency
```