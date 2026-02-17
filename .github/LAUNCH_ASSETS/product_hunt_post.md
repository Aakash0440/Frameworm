# FRAMEWORM - Complete ML Framework for Production 🐛

**Tagline:** Train → Track → Search → Deploy in one unified framework

**Description:**

I spent 200 hours building FRAMEWORM - a complete machine learning framework that gives you everything you need in one package.

## The Problem

Every ML project involves stitching together:
- PyTorch Lightning (training)
- MLflow or W&B (experiments)  
- Optuna (hyperparameter search)
- FastAPI + Docker (deployment)
- 10+ config files

It's exhausting.

## The Solution

FRAMEWORM puts it all in one place with zero configuration:
```bash
pip install frameworm
frameworm init my-project
frameworm train --config config.yaml --gpus 0,1,2,3
frameworm search --method bayesian --trials 50
frameworm export best.pt --format onnx
frameworm serve model.pt --port 8000
```

## What's Inside

- 🚀 **Training**: Callbacks, schedulers, mixed precision, EMA
- 📊 **Experiments**: Automatic SQLite tracking, Git integration
- 🔍 **Search**: Grid, Random, Bayesian optimization
- ⚡ **Distributed**: DataParallel, DDP, multi-machine
- 🎯 **Deploy**: TorchScript, ONNX, FastAPI, Docker/K8s
- 🎨 **Dashboard**: Beautiful web UI for monitoring
- 💻 **CLI**: 10+ commands for complete workflow

## Stats

- 25,000+ lines of code
- 420+ tests, 90%+ coverage
- 80+ commits over 20 days
- 25+ documentation pages

## Comparison

| | FRAMEWORM | Lightning | HuggingFace |
|--|--|--|--|
| Training | ✅ | ✅ | ⚠️ |
| Experiment Tracking | ✅ Built-in | ❌ | ❌ |
| HP Search | ✅ Built-in | ❌ | ❌ |
| Deployment | ✅ Built-in | ❌ | ⚠️ |
| Web UI | ✅ Built-in | ❌ | ❌ |

🌟 **GitHub**: github.com/Aakash0440/frameworm
📚 **Docs**: frameworm.readthedocs.io
