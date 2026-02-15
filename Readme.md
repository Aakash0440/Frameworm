# FRAMEWORM 🐛

> Advanced Generative AI Framework with Plugin System and Dependency Graphs

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

**Status:** ✅ Day 4 Complete - Error System + 2 Models Ready

## Features

- ✅ **Config System** - YAML with inheritance, validation, templates
- ✅ **Type System** - Protocols, type guards, validation
- ✅ **Plugin System** - Auto-discovery, decorators, namespaces
- ✅ **Error System** - Helpful messages with suggestions & context
- ✅ **Base Classes** - Enhanced Model/Pipeline/Trainer
- ✅ **Models** - DCGAN & VAE implemented
- 🚧 **Dependency Graphs** - Coming Day 5-6
- ⏳ **Training Infrastructure** - Coming Week 2
- ⏳ **Hyperparameter Search** - Coming Week 3
- ⏳ **Benchmark Suite** - Coming Week 3

## Quick Start

### Installation
```bash
pip install -e ".[dev]"
```

### Generate Images with DCGAN
```python
from frameworm.core import Config, get_model
import torch

# Load model
cfg = Config.from_template('gan')
dcgan = get_model("dcgan")(cfg)

# Generate images
images = dcgan(batch_size=16)  # (16, 3, 64, 64)
```

### Reconstruct with VAE
```python
# Load VAE
vae = get_model("vae")(Config.from_template('vae'))

# Reconstruct
reconstructed = vae.reconstruct(images)

# Sample new images
samples = vae.sample(16)
```

### Create Custom Model
```python
# plugins/my_model.py
from frameworm.models import BaseModel
from frameworm.core import register_model

@register_model("my-model")
class MyModel(BaseModel):
    def forward(self, x):
        return x

# Auto-discovered and ready to use!
```

## Available Models

| Model | Type | Description | Config |
|-------|------|-------------|--------|
| **DCGAN** | GAN | Deep Convolutional GAN | `configs/models/gan/dcgan.yaml` |
| **VAE** | VAE | Variational Autoencoder | `configs/models/vae/vanilla.yaml` |

More models coming in Week 2!

## Project Structure
frameworm/
├── core/           # Config, registry, types, errors
├── models/         # DCGAN, VAE, (more coming)
├── trainers/       # Training logic
├── pipelines/      # Workflows
├── data/           # Data utilities
├── optimization/   # Hyperparameter search (coming)
├── experiment/     # Tracking (coming)
└── benchmark/      # Benchmarks (coming)
configs/            # YAML configurations
tests/              # Comprehensive test suite
docs/               # Full documentation
examples/           # Usage examples

## Documentation

- [User Guide](docs/user_guide/) - How to use Frameworm
- [Models](docs/user_guide/models.md) - Available models
- [Plugins](docs/user_guide/plugins.md) - Create custom components
- [Error Handling](docs/user_guide/error_handling.md) - Understanding errors
- [Architecture](docs/architecture/) - System design
- [API Reference](docs/api_reference.md) - Complete API

## Error Messages That Actually Help
```python
# Instead of cryptic errors, get:

DimensionMismatchError: Tensor dimension mismatch

Details:
  Expected shape: (4, 100, 1, 1)
  Received shape: (4, 100)

Likely Causes:
  1. Input has 2 fewer dimension(s) than expected

Suggested Fixes:
  → Add spatial dimensions: x.unsqueeze(-1).unsqueeze(-1)
  → Or reshape: x.view(batch, channels, 1, 1)
```

## Development
```bash
# Run tests
pytest

# With coverage
pytest --cov=frameworm --cov-report=html

# Format code
black frameworm tests

# Lint
flake8 frameworm
```

## Testing

Test coverage: **92%+**
```bash
pytest -v  # All tests
pytest tests/unit/test_registry.py  # Specific module
```
## Training
```python
from frameworm.training import Trainer
from frameworm.training.callbacks import CSVLogger, ModelCheckpoint

# Create trainer
trainer = Trainer(model, optimizer, device='cuda')

# Add callbacks
trainer.add_callback(CSVLogger('training.csv'))
trainer.add_callback(ModelCheckpoint('best.pt', monitor='val_loss'))

# Train
trainer.train(train_loader, val_loader, epochs=100)
```
## Advanced Training Features

### Gradient Accumulation
```python
trainer.enable_gradient_accumulation(accumulation_steps=4)
```

### Mixed Precision (FP16)
```python
trainer.enable_mixed_precision()  # 2-3x faster on modern GPUs
```

### Exponential Moving Average
```python
trainer.enable_ema(decay=0.999)  # Better generalization
```

### TensorBoard Logging
```python
from frameworm.training.loggers import TensorBoardLogger
trainer.add_logger(TensorBoardLogger('runs/experiment'))
```

## Roadmap

### ✅ Completed (Week 1 - Days 1-4)
- Config system with inheritance & validation
- Type system with protocols
- Plugin system with auto-discovery
- Error explanation engine
- DCGAN & VAE models
- Comprehensive testing (92% coverage)
- Full documentation

### 🚧 In Progress (Week 1 - Days 5-7)
- Dependency graph engine
- Pipeline execution system
- More model implementations

### ⏳ Upcoming
- **Week 2**: Training infrastructure, experiment tracking
- **Week 3**: Hyperparameter search, benchmarking
- **Week 4**: CLI wizard, final polish, launch

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT License - see [LICENSE](LICENSE).

---


**Current Progress:** Day 4/28 (14% complete)
**Hours Invested:** 40/280
**Commits:** 17
**Models:** 2 (DCGAN, VAE)
**Test Coverage:** 92%