# Hyperparameter Search System Design

## Goals
1. Systematic hyperparameter exploration
2. Multiple search strategies (grid, random, Bayesian)
3. Parallel execution of trials
4. Integration with experiment tracking
5. Automatic best model selection
6. Resource-aware scheduling

## Search Strategies

### Grid Search
Exhaustive search over specified parameter values.
```python
search_space = {
    'lr': [0.001, 0.0001, 0.00001],
    'batch_size': [64, 128, 256],
    'hidden_dim': [128, 256, 512]
}
# Total trials: 3 × 3 × 3 = 27
```

### Random Search
Sample random combinations from distributions.
```python
search_space = {
    'lr': LogUniform(1e-5, 1e-2),
    'batch_size': Choice([64, 128, 256]),
    'hidden_dim': Uniform(128, 512)
}
# Sample N trials randomly
```

### Bayesian Optimization (Day 12)
Use Gaussian processes to model objective function.

## Architecture
```python
from frameworm.search import GridSearch, RandomSearch

# Define search space
search_space = {
    'training.lr': [0.001, 0.0001],
    'training.batch_size': [64, 128],
    'model.hidden_dim': [128, 256]
}

# Create search
search = GridSearch(
    base_config=config,
    search_space=search_space,
    metric='val_loss',
    mode='min'
)

# Run search
best_config, best_score = search.run(
    train_fn=train_function,
    n_jobs=4  # Parallel trials
)
```

## Features

1. **Multiple strategies** - Grid, Random, Bayesian
2. **Parallel execution** - Run multiple trials simultaneously
3. **Early stopping** - Stop unpromising trials early
4. **Experiment tracking** - Each trial is an experiment
5. **Resource management** - GPU/CPU allocation
6. **Resume capability** - Continue interrupted searches
7. **Visualization** - Plot parameter importance

## Integration
```python
# Automatic experiment tracking
search.run(
    train_fn=train_function,
    experiment_root='experiments/search',
    track_experiments=True
)

# Results automatically logged
# Best model automatically saved
# All trials tracked and comparable
```