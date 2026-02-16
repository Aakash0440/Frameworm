# Distributed Training Optimization

## Performance Optimization Guide

### 1. Mixed Precision Training

**Speedup:** 2-3x faster
**Memory:** 50% reduction
```python
trainer = DistributedTrainer(
    model,
    optimizer,
    use_amp=True  # Enable mixed precision
)
```

**When to use:**
- NVIDIA GPU with Tensor Cores (Volta, Turing, Ampere)
- Large models
- Want faster training

### 2. Gradient Accumulation

**Effect:** Larger effective batch size
**Memory:** Reduced memory usage
```python
trainer = DistributedTrainer(
    model,
    optimizer,
    gradient_accumulation_steps=4
)

# Effective batch size = batch_size × world_size × 4
```

**When to use:**
- Model doesn't fit with desired batch size
- Simulating larger batches
- Limited GPU memory

### 3. Optimized Data Loading
```python
from frameworm.distributed import OptimizedDataLoader

loader = OptimizedDataLoader.create(
    dataset,
    batch_size=128,
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    prefetch_factor=2,  # Prefetch batches
    persistent_workers=True  # Reuse workers
)
```

**Auto-configuration:**
```python
# Automatically picks optimal settings
loader = OptimizedDataLoader.create(dataset, batch_size=128)
```

### 4. Efficient Batch Size

**Rule of thumb:**
- Start with largest batch size that fits
- Typical: 32-256 per GPU
- Use gradient accumulation for larger effective batch

**Find optimal batch size:**
```python
# Binary search for max batch size
for batch_size in [32, 64, 128, 256, 512]:
    try:
        trainer.train(get_loader(batch_size), epochs=1)
        print(f"Batch size {batch_size}: OK")
    except RuntimeError:  # OOM
        print(f"Batch size {batch_size}: Too large")
        break
```

## Profiling Performance

### Basic Profiling
```python
from frameworm.distributed import PerformanceProfiler

profiler = PerformanceProfiler()

for batch in train_loader:
    with profiler.profile('step'):
        with profiler.profile('forward'):
            loss = model(batch)
        
        with profiler.profile('backward'):
            loss.backward()
        
        with profiler.profile('optimizer'):
            optimizer.step()
    
    profiler.record_gpu_memory()

profiler.print_summary()
```

### Identify Bottlenecks

**Data loading bottleneck:**
Data Loading: 50ms
Percent of step: 40%  # Too high!

**Solution:**
- Increase num_workers
- Use pin_memory
- Simplify data transforms

**GPU underutilization:**
Step Time: 100ms
Forward: 30ms
Backward: 30ms
Optimizer: 5ms
Data: 5ms
30ms unaccounted = communication overhead

**Solution:**
- Use gradient bucketing
- Enable gradient compression
- Check network bandwidth

## Communication Optimization

### Gradient Bucketing

Automatically enabled in DDP. Adjustable:
```python
from torch.nn.parallel import DistributedDataParallel as DDP

model = DDP(
    model,
    bucket_cap_mb=25,  # Default: 25MB
)
```

### Overlap Computation and Communication

Automatically handled by DDP. Ensure:
- Use NCCL backend
- Contiguous parameter layout
- Reasonable model size

## Multi-Node Optimization

### Network Configuration

1. **InfiniBand:** Best performance
2. **10GbE+:** Good for small models
3. **1GbE:** Avoid for multi-node

### Reduce Communication
```python
# Use gradient accumulation
trainer = DistributedTrainer(
    model,
    optimizer,
    gradient_accumulation_steps=8  # Fewer syncs
)
```

## Performance Targets

| Configuration | Expected Speedup |
|---------------|------------------|
| Single GPU | 1.0x (baseline) |
| + Mixed Precision | 2-3x |
| + 2 GPUs (DDP) | 1.8-1.9x per GPU |
| + 4 GPUs (DDP) | 3.5-3.8x total |
| + 8 GPUs (DDP) | 7.0-7.5x total |

## Troubleshooting

### Slow Training

**Check:**
1. GPU utilization (`nvidia-smi`)
2. Data loading time (profiler)
3. Batch size (too small?)
4. Number of workers

### Out of Memory

**Solutions:**
1. Reduce batch size
2. Enable gradient accumulation
3. Use mixed precision
4. Enable gradient checkpointing

### Poor Scaling

**Check:**
1. Communication overhead (profiler)
2. Batch size (too small for multi-GPU?)
3. Model size (too small = overhead dominates)
4. Network bandwidth (multi-node)

## Best Practices

1. **Always profile first** - Know your bottleneck
2. **Use mixed precision** - Almost always beneficial
3. **Optimize data loading** - Prevent GPU starvation
4. **Scale batch size** - With number of GPUs
5. **Monitor utilization** - GPU should be >90%
6. **Benchmark** - Test before production runs

## Example: Optimal Configuration
```python
from frameworm.distributed import DistributedTrainer, OptimizedDataLoader

# Data
train_loader = OptimizedDataLoader.create(
    dataset,
    batch_size=128,  # Per GPU
    num_workers=4,
    pin_memory=True
)

# Model
trainer = DistributedTrainer(
    model,
    optimizer,
    backend='nccl',
    use_amp=True,                    # 2-3x speedup
    gradient_accumulation_steps=2,   # 2x effective batch
    find_unused_parameters=False     # Faster
)

# Train
trainer.train(train_loader, val_loader)

# Expected: ~10x faster than single GPU baseline
```