# Training Guide

## Basic Training

### Simple Training Loop
```python
from frameworm.training import Trainer
from frameworm.core import Config, get_model
import torch.nn as nn
from torch.utils.data import DataLoader

# Create model
model = get_model("vae")(config)

# Create optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Create trainer
trainer = Trainer(
    model=model,
    optimizer=optimizer,
    device='cuda'
)

# Train
trainer.train(train_loader, val_loader, epochs=100)
```

## Learning Rate Scheduling

### Built-in Schedulers
```python
from torch.optim.lr_scheduler import CosineAnnealingLR

scheduler = CosineAnnealingLR(optimizer, T_max=100)
trainer.set_scheduler(scheduler)
```

### Custom Schedulers
```python
from frameworm.training.schedulers import WarmupCosineScheduler

scheduler = WarmupCosineScheduler(
    optimizer,
    warmup_epochs=10,
    total_epochs=100,
    min_lr=1e-6
)
trainer.set_scheduler(scheduler)
```

## Callbacks

### CSV Logging
```python
from frameworm.training.callbacks import CSVLogger

trainer.add_callback(CSVLogger('training.csv'))
```

### Model Checkpointing
```python
from frameworm.training.callbacks import ModelCheckpoint

trainer.add_callback(ModelCheckpoint(
    'model_{epoch}.pt',
    monitor='val_loss',
    mode='min',
    save_best_only=True
))
```

### Custom Callbacks
```python
from frameworm.training.callbacks import Callback

class MyCallback(Callback):
    def on_epoch_end(self, epoch, metrics, trainer):
        print(f"Custom action at epoch {epoch}")

trainer.add_callback(MyCallback())
```

## Early Stopping
```python
trainer.set_early_stopping(patience=10, min_delta=0.001)
```

## Resuming Training
```python
# Train
trainer.train(train_loader, val_loader, epochs=100)

# Resume from checkpoint
trainer.train(
    train_loader,
    val_loader,
    epochs=200,
    resume_from='checkpoints/latest.pt'
)
```

## Advanced Features

### GAN Training

See `examples/train_dcgan.py` for complete GAN training example.

### Multiple Optimizers

For models requiring multiple optimizers (like GANs), subclass Trainer
and override `train_epoch()`.

### Custom Loss Functions
```python
def custom_loss(outputs, targets):
    return ((outputs - targets) ** 2).mean()

trainer = Trainer(model, optimizer, criterion=custom_loss)
```

## Best Practices

1. **Start with small learning rate** - Use 1e-4 or 1e-3
2. **Use warmup** - Especially for large models
3. **Monitor validation** - Watch for overfitting
4. **Save checkpoints** - Enable resume
5. **Use callbacks** - Log everything
6. **Early stopping** - Save time
7. **LR scheduling** - Improve convergence

## Troubleshooting

### Loss is NaN

- Reduce learning rate
- Add gradient clipping (Day 8)
- Check data normalization
- Use mixed precision (Day 8)

### Not Converging

- Increase learning rate
- Try different scheduler
- Check model architecture
- Verify data quality

### Out of Memory

- Reduce batch size
- Use gradient accumulation (Day 8)
- Enable mixed precision (Day 8)