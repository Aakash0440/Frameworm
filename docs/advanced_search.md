# Advanced Search Features

## Early Stopping

### Median Stopping
Stop trials performing worse than median:
```python
from frameworm.search.early_stopping import MedianStopper

stopper = MedianStopper(percentile=50, min_trials=5)
```

### Improvement Stopping
Stop if no improvement:
```python
from frameworm.search.early_stopping import ImprovementStopper

stopper = ImprovementStopper(patience=10, min_delta=0.001)
```

### Threshold Stopping
Stop when goal reached:
```python
from frameworm.search.early_stopping import ThresholdStopper

stopper = ThresholdStopper(threshold=0.5, mode='min')
```

## Hyperband

Adaptive resource allocation (skeleton implementation):
```python
from frameworm.search.hyperband import Hyperband

hyperband = Hyperband(
    base_config=config,
    search_space=search_space,
    max_resource=81
)
```

Note: Full implementation requires training loop integration.

## Comparison of Methods

| Method | Sample Efficiency | Parallelizable | Best For |
|--------|------------------|----------------|----------|
| Grid Search | Low | ✅ | Small discrete spaces |
| Random Search | Medium | ✅ | Large continuous spaces |
| Bayesian Optimization | High | ❌ | Expensive evaluations |
| Hyperband | High | ⚠️ | Variable training budgets |

## When to Use What

**Grid Search:**
- < 100 configurations
- Discrete parameters
- Want exhaustive coverage

**Random Search:**
- > 100 configurations
- Continuous parameters
- Limited compute budget

**Bayesian Optimization:**
- Expensive training (>10 min per trial)
- < 100 dimensions
- Sequential evaluation okay

**Hyperband:**
- Can stop training early
- Many configurations to evaluate
- Training time varies with epochs