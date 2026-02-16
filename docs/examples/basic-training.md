# Basic Training Example

Simple training loop with FRAMEWORM.

---

## Complete Example
```python
"""
Basic VAE Training on MNIST

This example shows the minimal code needed to train a VAE.
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from frameworm import Trainer, Config, get_model

# Configuration
config = Config.from_dict({
    'model': {
        'type': 'vae',
        'latent_dim': 128,
        'hidden_dim': 256
    },
    'training': {
        'epochs': 50,
        'batch_size': 128,
        'lr': 0.001
    }
})

# Data
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST('data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# Model
model = get_model('vae')(config)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
trainer = Trainer(model, optimizer, device='cuda')
trainer.train(train_loader, val_loader, epochs=50)

print("Training complete!")
```

**Output:**
Epoch 1/50: train_loss=152.34, val_loss=145.67
Epoch 2/50: train_loss=128.45, val_loss=125.89
...
Epoch 50/50: train_loss=87.23, val_loss=88.91
Training complete!

---

## Key Points

1. **Minimal Setup** - Just 30 lines of code
2. **Automatic Logging** - Metrics tracked automatically
3. **GPU Support** - Set `device='cuda'`
4. **Checkpointing** - Best model saved automatically

---

## Variations

### With Experiment Tracking
```python
from frameworm.experiment import Experiment

with Experiment(name='vae-mnist', config=config) as exp:
    trainer.set_experiment(exp)
    trainer.train(train_loader, val_loader, epochs=50)
```

### With Multi-GPU
```python
trainer.enable_data_parallel(device_ids=[0, 1, 2, 3])
trainer.train(train_loader, val_loader, epochs=50)
```

### With Mixed Precision
```python
trainer.enable_mixed_precision()
trainer.train(train_loader, val_loader, epochs=50)
```