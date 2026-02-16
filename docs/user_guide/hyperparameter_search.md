# Hyperparameter Search

## Overview

Frameworm provides systematic hyperparameter search with multiple strategies.

## Search Spaces

### Define Search Space
```python
from frameworm.search.space import Real, Integer, Categorical

search_space = {
    'training.lr': Real(1e-5, 1e-2, log=True),
    'training.batch_size': Integer(32, 256, log=True),
    'optimizer': Categorical(['adam', 'sgd', 'rmsprop']),
    'model.hidden_dim': Integer(128, 512)
}
```

### Space Types

**Categorical** - Discrete choices:
```python
Categorical(['adam', 'sgd', 'rmsprop'])
```

**Integer** - Integer values:
```python
Integer(32, 256, log=True)  # Sample in log space
Integer(1, 10, log=False)   # Sample linearly
```

**Real** - Continuous values:
```python
Real(1e-5, 1e-2, log=True)  # Sample in log space
Real(0.0, 1.0, log=False)   # Sample linearly
```

## Grid Search

Exhaustive search over all combinations.
```python
from frameworm.search import GridSearch

search = GridSearch(
    base_config=config,
    search_space={
        'training.lr': [0.001, 0.0001, 0.00001],
        'training.batch_size': [64, 128, 256]
    },
    metric='val_loss',
    mode='min'
)

best_config, best_score = search.run(train_fn)
```

**When to use:**
- Small search space (< 100 combinations)
- Want exhaustive coverage
- Discrete parameters

## Random Search

Sample random configurations.
```python
from frameworm.search import RandomSearch
from frameworm.search.space import Real, Integer

search = RandomSearch(
    base_config=config,
    search_space={
        'training.lr': Real(1e-5, 1e-2, log=True),
        'training.batch_size': Integer(32, 256, log=True)
    },
    metric='val_loss',
    mode='min',
    n_trials=50
)

best_config, best_score = search.run(train_fn)
```

**When to use:**
- Large search space
- Continuous parameters
- Limited compute budget
- Often more efficient than grid search

## Training Function

Define a function that takes Config and returns metrics:
```python
def train_fn(config: Config) -> dict:
    # Get data
    train_loader, val_loader = get_data(config.training.batch_size)
    
    # Create model
    model = get_model("vae")(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.lr)
    
    # Train
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, val_loader, epochs=config.training.epochs)
    
    # Return metrics
    return {
        'val_loss': trainer.state.val_metrics['loss'][-1],
        'train_loss': trainer.state.train_metrics['loss'][-1]
    }
```

## Parallel Execution

Run multiple trials in parallel:
```python
best_config, best_score = search.run(
    train_fn=train_fn,
    n_jobs=4  # Use 4 processes
)

# Use all CPUs
search.run(train_fn, n_jobs=-1)
```

## Experiment Tracking

Automatically track all trials:
```python
search.run(
    train_fn=train_fn,
    experiment_root='experiments/search'
)

# Each trial becomes an experiment
# Compare with ExperimentManager
from frameworm.experiment import ExperimentManager
manager = ExperimentManager('experiments/search')
df = manager.list_experiments(tags=['grid_search'])
```

## Analysis

### Analyze Results
```python
from frameworm.search import SearchAnalyzer

analyzer = SearchAnalyzer(search.results)

# Print summary
analyzer.print_summary()

# Plot convergence
analyzer.plot_convergence(save_path='convergence.png')

# Plot parameter importance
analyzer.plot_parameter_importance(save_path='importance.png')

# Plot parameter vs score
analyzer.plot_parameter_vs_score('training.lr', save_path='lr_vs_score.png')
```

### Get Best Configurations
```python
# Top 5 configs
top5 = analyzer.get_best_n(5)
print(top5)
```

## Save/Load Results
```python
# Save
search.save_results('search_results.json')

# Load
search2 = GridSearch(base_config, search_space)
search2.load_results('search_results.json')
```

## Best Practices

1. **Start with random search** - More efficient for most cases
2. **Use log scale** - For learning rates, batch sizes
3. **Run enough trials** - At least 20-50 for random search
4. **Reduce training time** - Use fewer epochs during search
5. **Track experiments** - Easy to compare later
6. **Analyze results** - Understand parameter importance
7. **Validate best config** - Train with full epochs

## Examples

See `examples/hyperparameter_search_example.py` for complete example.