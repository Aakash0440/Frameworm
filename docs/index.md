# FRAMEWORM

**Complete Machine Learning Framework for Production**

[![PyPI](https://img.shields.io/pypi/v/frameworm)](https://pypi.org/project/frameworm/)
[![Tests](https://img.shields.io/github/workflow/status/Aakash0440/frameworm/tests)](https://github.com/Aakash0440/frameworm/actions)
[![Coverage](https://img.shields.io/codecov/c/github/Aakash0440/frameworm)](https://codecov.io/gh/Aakash0440/frameworm)
[![License](https://img.shields.io/github/license/Aakash0440/frameworm)](https://github.com/Aakash0440/Aakash0440/blob/main/LICENSE)

---

## What is FRAMEWORM?

FRAMEWORM is a complete machine learning framework that provides everything you need to:

- **Train** deep learning models with ease
- **Track** experiments automatically
- **Search** hyperparameters efficiently
- **Deploy** models to production
- **Scale** to multiple GPUs and machines

All with a simple, unified API.

---

## Quick Example

=== "Training"
```python
    from frameworm import Trainer, Config, get_model
    
    # Load configuration
    config = Config('config.yaml')
    
    # Create model
    model = get_model('vae')(config)
    
    # Train
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, val_loader, epochs=100)
```

=== "CLI"
```bash
    # Initialize project
    frameworm init my-project --template vae
    
    # Train model
    frameworm train --config config.yaml --gpus 0,1,2,3
    
    # Deploy
    frameworm export best.pt --format onnx
    frameworm serve model.pt --port 8000
```

=== "Dashboard"
```bash
    # Launch web dashboard
    frameworm dashboard --port 8080
```

---

## Key Features

### 🚀 Easy to Use
Simple API that gets out of your way. Train state-of-the-art models in minutes.

### 📊 Experiment Tracking
Automatic experiment tracking with SQLite. No external servers needed.

### 🔍 Hyperparameter Search
Grid, random, and Bayesian optimization built-in. Find optimal hyperparameters efficiently.

### 🎯 Production Ready
Export to TorchScript/ONNX, serve with FastAPI, deploy with Docker/Kubernetes.

### ⚡ Fast & Scalable
Multi-GPU training, distributed training, mixed precision. Linear scaling to 100s of GPUs.

### 🎨 Beautiful Dashboard
Web UI for experiment tracking, model management, and training monitoring.

---

## Installation
```bash
pip install frameworm
```

See [Installation Guide](getting-started/quickstart.md) for more options.

---

## Quick Links

- [Quick Start Guide](getting-started/quickstart.md)
- [User Guide](user_guide/configuration.md)
- [Tutorials](tutorials/vae-tutorial.md)
- [API Reference](api-reference/core.md)
- [GitHub](https://github.com/Aakash0440/frameworm)

---

## Why FRAMEWORM?

| Feature | FRAMEWORM | PyTorch Lightning | Hugging Face |
|---------|-----------|-------------------|--------------|
| **Training** | ✅ Complete | ✅ Complete | ⚠️ Limited |
| **Experiment Tracking** | ✅ Built-in | ⚠️ External | ⚠️ External |
| **Hyperparameter Search** | ✅ Built-in | ❌ | ⚠️ Limited |
| **Model Deployment** | ✅ Built-in | ❌ | ⚠️ Limited |
| **Web Dashboard** | ✅ Built-in | ❌ | ❌ |
| **CLI Tool** | ✅ Complete | ⚠️ Basic | ⚠️ Basic |
| **Complexity** | **Low** | **Medium** | **High** |

---

## Community

- [GitHub Discussions](https://github.com/Aakash0440/frameworm/discussions)
- [Discord](https://discord.gg/frameworm)
- [Twitter](https://twitter.com/frameworm)

---

## License

FRAMEWORM is released under the MIT License. See [LICENSE](https://github.com/Aakash0440/frameworm/LICENSE) for details.