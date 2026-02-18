**Title:** Show HN / r/MachineLearning: FRAMEWORM - Complete ML Framework I built in 20 days

**Body:**

Hey everyone! I spent 20 days (~200 hours) building FRAMEWORM, a complete ML framework that puts everything you need in one place.

**The core idea:** Stop stitching together 5+ tools. Get training, experiment tracking, hyperparameter search, AND deployment all in one.
```bash
pip install frameworm

# Full workflow
frameworm init my-project
frameworm train --config config.yaml --gpus 0,1,2,3
frameworm search --method bayesian --trials 50
frameworm export best.pt --format onnx
frameworm serve model.pt --port 8000
```

**What's included:**
- Training with callbacks, mixed precision, EMA, gradient accumulation
- Built-in experiment tracking (SQLite, no server needed)
- Grid/Random/Bayesian hyperparameter search
- DataParallel + DDP + multi-machine distributed training
- Model export (TorchScript, ONNX) + quantization
- FastAPI serving + Docker/K8s templates
- Web dashboard for monitoring
- CLI with 10+ commands

**Stats:**
- 25,000+ lines of code
- 420+ tests, 90%+ coverage
- 84 git commits
- 25+ documentation pages

GitHub: github.com/Aakash0440/frameworm
Docs: frameworm.readthedocs.io

Would love feedback from the community!