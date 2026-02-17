# Changelog

All notable changes to FRAMEWORM will be documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] - 2024-11-01

### Added

#### Core Framework
- Config system with YAML inheritance and environment variable support
- Type system with protocols and runtime validation
- Plugin registry with auto-discovery
- Dependency graph system (DAG execution, caching, visualization)
- Parallel execution (Thread and Process pools)
- Error explanation system with actionable messages

#### Models
- DCGAN (Deep Convolutional GAN)
- VAE (Variational Autoencoder) with VQ-VAE variant
- DDPM (Denoising Diffusion Probabilistic Model)

#### Training
- Complete training loop with validation
- 5 built-in callbacks (EarlyStopping, Checkpoint, LRScheduler, TensorBoard, W&B)
- 3 LR schedulers (StepLR, CosineAnnealing, ReduceOnPlateau)
- Gradient accumulation
- Gradient clipping
- Exponential Moving Average (EMA)
- Mixed precision training (FP16) with 2-3x speedup
- TensorBoard and W&B logging

#### Experiment Tracking
- Automatic experiment tracking with SQLite
- Git integration (commit hash, dirty status)
- Config versioning
- Artifact tracking
- CLI for experiment management
- Visualization tools

#### Metrics
- FID (Fréchet Inception Distance)
- Inception Score (IS)
- LPIPS (Learned Perceptual Similarity)
- Unified MetricEvaluator API

#### Hyperparameter Search
- Grid Search
- Random Search
- Bayesian Optimization (requires scikit-optimize)
- Search analysis tools (convergence plots, parameter importance)
- Early stopping for search

#### Distributed Training
- DataParallel (single-machine multi-GPU)
- DistributedDataParallel (multi-machine)
- Mixed precision with DDP
- Gradient compression (PowerSGD)
- Distributed checkpointing

#### Deployment
- TorchScript export
- ONNX export
- Dynamic quantization (4x size reduction)
- FastAPI model serving
- Docker containerization support
- Kubernetes deployment templates

#### CLI
- `frameworm init` - Project initialization with templates
- `frameworm train` - Training from CLI
- `frameworm evaluate` - Model evaluation
- `frameworm search` - Hyperparameter search
- `frameworm export` - Model export
- `frameworm serve` - Model serving
- `frameworm pipeline` - Workflow automation
- `frameworm monitor` - Real-time training monitoring
- `frameworm dashboard` - Web UI launcher
- `frameworm config` - Configuration management

#### Web Dashboard
- FastAPI backend with REST + WebSocket API
- React frontend with 4 pages
- Dashboard with system stats
- Experiment tracking UI
- Model management UI
- Real-time training monitor
- Live metrics charts

#### Performance Tools
- Training profiler (per-phase timing)
- Inference profiler (batch size benchmark)
- Optimized DataLoader factory
- GPU prefetch loader
- Dataset caching
- Memory monitor and estimator
- Gradient checkpointing

#### Documentation
- Complete MkDocs documentation with Material theme
- 25+ user guide pages
- 5+ step-by-step tutorials
- Complete API reference
- 10+ code examples
- Contributing guide, FAQ

### Fixed
- N/A (first release)

### Changed
- N/A (first release)

### Removed
- N/A (first release)

---

## Unreleased

### Planned
- TorchServe integration
- AutoML features
- Federated learning support
- MLOps integrations