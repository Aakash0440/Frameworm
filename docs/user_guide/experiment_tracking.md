# Experiment Tracking

## Overview

Frameworm provides comprehensive experiment tracking to manage, compare, and reproduce experiments.

## Basic Usage

### Creating an Experiment
```python
from frameworm.experiment import Experiment

exp = Experiment(
    name="my-experiment",
    config=config,
    description="Testing new architecture",
    tags=["vae", "mnist", "baseline"],
    root_dir="experiments"
)

with exp:
    # Your training code
    trainer.set_experiment(exp)
    trainer.train(train_loader, val_loader)
    
# Automatically tracks:
# - Config
# - Metrics
# - Git version
# - Artifacts
```

### Integration with Trainer
```python
trainer = Trainer(model, optimizer)
trainer.set_experiment(exp)

# Metrics automatically logged
trainer.train(train_loader, val_loader)
```

## Managing Experiments

### List Experiments
```python
from frameworm.experiment import ExperimentManager

manager = ExperimentManager('experiments')

# List all experiments
df = manager.list_experiments()

# Filter by status
df = manager.list_experiments(status='completed')

# Filter by tags
df = manager.list_experiments(tags=['vae', 'mnist'])
```

### Compare Experiments
```python
# Compare multiple experiments
comparison = manager.compare_experiments(
    ['exp_001', 'exp_002', 'exp_003'],
    metrics=['loss', 'val_loss']
)

print(comparison[['name', 'loss', 'val_loss']])
```

### Search Experiments
```python
# Search by config
results = manager.search_experiments(
    config_filter={'training.lr': 0.001}
)

# Search by metrics
results = manager.search_experiments(
    metric_filter={'val_loss': ('<=', 0.5)}
)
```

## Visualization

### Plot Metric Comparison
```python
from frameworm.experiment.visualization import plot_metric_comparison

plot_metric_comparison(
    manager,
    ['exp_001', 'exp_002', 'exp_003'],
    'loss',
    save_path='comparison.png'
)
```

### Plot Multiple Metrics
```python
from frameworm.experiment.visualization import plot_multiple_metrics

plot_multiple_metrics(
    manager,
    'exp_001',
    ['loss', 'val_loss', 'kl_div'],
    save_path='metrics.png'
)
```

## CLI Interface

### List Experiments
```bash
python -m frameworm.experiment.cli list
python -m frameworm.experiment.cli list --status completed
```

### Show Experiment
```bash
python -m frameworm.experiment.cli show exp_001
```

### Compare Experiments
```bash
python -m frameworm.experiment.cli compare exp_001 exp_002 exp_003
```

### Plot Metrics
```bash
python -m frameworm.experiment.cli plot exp_001 exp_002 loss --output loss.png
```

### Delete Experiment
```bash
python -m frameworm.experiment.cli delete exp_001
```

## Best Practices

1. **Use Descriptive Names** - Easy to identify later
2. **Add Tags** - Enable filtering and organization
3. **Track Everything** - Config, metrics, git version
4. **Compare Often** - Find what works
5. **Clean Up** - Delete failed/test experiments

## Storage Structure
experiments/
├── exp_001_vae_baseline/
│   ├── config.yaml          # Saved configuration
│   ├── metadata.json        # Experiment metadata
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/               # Training logs
│   └── artifacts/          # Additional files
└── experiments.db          # SQLite database

## Reproducibility

Every experiment automatically tracks:
- Complete configuration
- Git commit hash
- Git dirty status (uncommitted changes)
- All hyperparameters
- Environment info
- Metrics history
- Artifacts

This enables perfect reproducibility of any experiment.