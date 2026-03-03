# Experiment Tracking System Design

## Goals
1. Track every experiment with unique ID
2. Version control for experiments
3. Compare experiment results
4. Reproduce any experiment
5. Search and filter experiments
6. Export results

## Architecture

### Experiment Class
```python
experiment = Experiment(
    name="vae-experiment-1",
    config=config,
    tags=["vae", "mnist", "baseline"]
)

with experiment:
    trainer.train(train_loader, val_loader)
    
# Automatically logs:
# - Config
# - Metrics
# - Code version (git hash)
# - Environment (packages, versions)
# - Artifacts (checkpoints, logs)
```

### Storage Structure
experiments/
├── exp_001_vae_baseline/
│   ├── config.yaml
│   ├── metrics.json
│   ├── metadata.json
│   ├── checkpoints/
│   ├── logs/
│   └── artifacts/
├── exp_002_vae_beta4/
└── experiments.db  # SQLite database

## Features
1. **Automatic tracking** - Context manager
2. **Git integration** - Track code version
3. **Config versioning** - Track all hyperparameters
4. **Metric history** - All training metrics
5. **Artifact storage** - Checkpoints, images, etc.
6. **Comparison tools** - Compare multiple experiments
7. **Search/filter** - Query experiments
8. **Reproducibility** - Recreate any experiment

## Integration with Trainer
```python
trainer = Trainer(model, optimizer)
trainer.set_experiment(experiment)
# Metrics automatically logged to experiment
```