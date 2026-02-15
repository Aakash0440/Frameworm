# Advanced Training

## Gradient Accumulation

Simulate larger batch sizes:
```python
trainer = Trainer(model, optimizer)
trainer.enable_gradient_accumulation(accumulation_steps=4)
# Effective batch size = actual_batch_size * 4
```

## Gradient Clipping

Prevent exploding gradients:
```python
trainer.enable_gradient_clipping(max_norm=1.0)
```

## Mixed Precision Training

Faster training with FP16:
```python
trainer.enable_mixed_precision()  # Requires CUDA
```

Benefits:
- 2-3x faster training
- 50% less memory
- Automatic loss scaling

## Exponential Moving Average (EMA)

Better generalization:
```python
trainer.enable_ema(decay=0.999)

# After training, use EMA for inference
trainer.ema.apply_shadow()
predictions = model(inputs)
trainer.ema.restore()
```

## TensorBoard Logging
```python
from frameworm.training.loggers import TensorBoardLogger

trainer.add_logger(TensorBoardLogger('runs/experiment'))

# View logs
# tensorboard --logdir runs
```

## Weights & Biases
```python
from frameworm.training.loggers import WandbLogger

trainer.add_logger(WandbLogger(
    project='my-project',
    name='experiment-1',
    config={'lr': 0.001, 'batch_size': 128}
))
```

## Complete Example
```python
# Create trainer
trainer = Trainer(model, optimizer, device='cuda')

# Enable all features
trainer.enable_gradient_accumulation(4)
trainer.enable_gradient_clipping(1.0)
trainer.enable_ema(0.999)
trainer.enable_mixed_precision()

# Add scheduler
from frameworm.training.schedulers import WarmupCosineScheduler
scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=5,
    total_epochs=100
)
trainer.set_scheduler(scheduler)

# Add logging
trainer.add_logger(TensorBoardLogger('runs/exp'))

# Add callbacks
from frameworm.training.callbacks import ModelCheckpoint
trainer.add_callback(ModelCheckpoint('best.pt', monitor='val_loss'))

# Train
trainer.train(train_loader, val_loader, epochs=100)
```

## Best Practices

1. **Gradient Accumulation** - Use when GPU memory is limited
2. **Gradient Clipping** - Essential for RNNs, helpful for all models
3. **EMA** - Almost always improves generalization
4. **Mixed Precision** - Use on modern GPUs (Volta+)
5. **Logging** - Log everything for debugging
6. **Early Stopping** - Save compute time

## Troubleshooting

### Mixed Precision Issues

If you see NaN losses with mixed precision:
```python
# Increase loss scaling
trainer.grad_scaler = torch.cuda.amp.GradScaler(init_scale=2.**10)
```

### EMA Memory

EMA doubles memory usage. Disable if OOM:
```python
# Don't enable EMA
# trainer.enable_ema(0.999)
```