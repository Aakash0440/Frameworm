# Quick Start

Get started with FRAMEWORM in 5 minutes.

---

## Installation
```bash
pip install frameworm
```

!!! tip "Virtual Environment"
    It's recommended to use a virtual environment:
```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install frameworm
```

---

## Your First Model

### 1. Initialize Project
```bash
frameworm init my-first-model --template vae
cd my-first-model
```

This creates:
my-first-model/
├── configs/
│   └── config.yaml      # Configuration
├── data/                # Dataset directory
├── experiments/         # Experiment tracking
├── checkpoints/         # Model checkpoints
└── README.md

### 2. Prepare Data

Download MNIST dataset:
```python
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
```

### 3. Train Model

=== "CLI"
```bash
    frameworm train --config configs/config.yaml --gpus 0
```

=== "Python"
```python
    from frameworm import Trainer, Config, get_model
    import torch.optim as optim
    
    # Load config
    config = Config('configs/config.yaml')
    
    # Create model
    model = get_model('vae')(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, val_loader, epochs=100)
```

### 4. Monitor Training

Launch the dashboard:
```bash
frameworm dashboard --port 8080
```

Open http://localhost:8080 to see real-time training progress.

### 5. Export & Deploy
```bash
# Export to ONNX
frameworm export checkpoints/best.pt --format onnx

# Serve model
frameworm serve exported/model.pt --port 8000
```

Test the API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"data": [[...]]}'
```

---

## Next Steps

- [Configuration Guide](../user-guide/configuration.md) - Customize your setup
- [Training Guide](../user-guide/training.md) - Advanced training techniques
- [Tutorials](../tutorials/vae-tutorial.md) - Step-by-step tutorials
- [API Reference](../api-reference/core.md) - Complete API documentation

---

## Common Issues

??? question "CUDA out of memory"
    Reduce batch size in `config.yaml`:
```yaml
    training:
      batch_size: 64  # Try smaller values
```

??? question "Model not converging"
    Try adjusting learning rate:
```yaml
    training:
      lr: 0.0001  # Lower learning rate
```

??? question "Slow training"
    Enable multi-GPU:
```bash
    frameworm train --config config.yaml --gpus 0,1,2,3
```