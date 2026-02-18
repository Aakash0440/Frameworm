# Hacker News - Show HN

## Title

Show HN: FRAMEWORM – Production-ready framework for generative AI

## Post

Hi HN,

I built FRAMEWORM over the past 28 days (280 hours) to solve a problem I kept running into: deploying generative models to production requires stitching together 10+ tools, each taking days to integrate.

**What it is:**
A batteries-included deep learning framework focused on generative models (VAE, GAN, Diffusion) with production deployment built-in from day one.

**Why it exists:**
Try deploying a DCGAN to production with PyTorch alone. You'll need:
- PyTorch (training)
- W&B or MLflow (tracking)
- S3/GCS (storage)
- FastAPI (serving)
- Docker (containerization)
- Kubernetes (orchestration)
- Prometheus + Grafana (monitoring)
- Custom code for A/B testing, drift detection, model registry...

Each integration takes 2-3 days. Total: ~3 weeks before you can deploy.

**With FRAMEWORM:**
```bash
pip install frameworm
frameworm train --config config.yaml
frameworm serve model.pt --port 8000
```

Deploy-ready in 5 minutes.

**Technical details:**
- 6 built-in models (VAE, DCGAN, DDPM, VQ-VAE-2, ViT-GAN, CFG-DDPM)
- Full MLOps: experiment tracking, model registry, drift detection, A/B testing
- Production features: health checks, graceful shutdown, rate limiting, auth
- 10+ integrations (W&B, MLflow, S3, GCS, Azure, PostgreSQL, Slack)
- Plugin system for extensibility
- Distributed training (DDP + gradient compression)
- CLI with 10+ commands
- React dashboard for monitoring
- 450+ tests, >90% coverage

**Architecture:**
Built on PyTorch, but abstracts away boilerplate while keeping full control.
- YAML configs (no magic Python dataclasses)
- Explicit > implicit (you see what happens)
- Plugin hooks at 15+ lifecycle points
- Compatible with Lightning models (mostly)

**Comparison to alternatives:**
- vs Lightning: More opinionated, production-focused, includes models
- vs Fast.ai: Production-grade, less "magic", MLOps built-in
- vs Accelerate: Higher-level, includes training loop and deployment

**Performance:**
Benchmarked on RTX 3090:
- Training: 2,500 samples/sec (DCGAN, 64x64)
- Inference: 45ms latency (batch=1), 3,200 samples/sec (batch=64)
- Memory: ~6GB for typical GAN training

Comparable to Lightning (~5% faster due to optimized data pipeline).

**Open source:**
- MIT License
- 30,000+ LOC
- 35+ doc pages
- 10+ complete examples
- Video tutorials

**Links:**
- GitHub: https://github.com/yourusername/frameworm
- Docs: https://frameworm.readthedocs.io
- PyPI: https://pypi.org/project/frameworm/

**What I learned:**
1. MLOps is 70% of deploying models (training is easy part)
2. Good docs > features (spent 3 days on docs alone)
3. Integration with existing tools (W&B, MLflow) crucial for adoption
4. Kubernetes is unavoidable for serious production

**Known limitations:**
- Focused on generative models (not for BERT/ResNet)
- Some rough edges in plugin system
- Dashboard could be prettier
- No TorchServe integration (yet)

**Roadmap:**
- AutoML (architecture search)
- Federated learning
- More models (StyleGAN, Stable Diffusion)
- Mobile deployment (Core ML, TFLite)
- Better dashboard UI

I'm here all day to answer questions about design decisions, implementation details, or anything else!

Try it: `pip install frameworm`