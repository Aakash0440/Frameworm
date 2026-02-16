# Bayesian Optimization Design

## Overview

Bayesian Optimization uses a probabilistic model (Gaussian Process) to guide the search towards promising regions of the hyperparameter space.

## How It Works

1. **Surrogate Model**: Fit Gaussian Process to observed results
2. **Acquisition Function**: Determine next point to evaluate
3. **Update**: Observe result, update surrogate model
4. **Repeat**: Until budget exhausted or convergence

## Advantages Over Random/Grid Search

- **Sample Efficient**: Needs fewer evaluations
- **Intelligent**: Focuses on promising regions
- **Handles Expensive Functions**: Good for costly training runs
- **Balances Exploration/Exploitation**

## Acquisition Functions

### Expected Improvement (EI)
Probability of improvement × magnitude of improvement

### Upper Confidence Bound (UCB)
Mean + β × std (balance exploration/exploitation)

### Probability of Improvement (PI)
Probability that point is better than current best

## Implementation
```python
from frameworm.search import BayesianSearch

search = BayesianSearch(
    base_config=config,
    search_space={
        'training.lr': Real(1e-5, 1e-2, log=True),
        'training.batch_size': Integer(32, 256, log=True)
    },
    metric='val_loss',
    mode='min',
    n_trials=50,
    acquisition='ei'  # Expected Improvement
)

best_config, best_score = search.run(train_fn)
```

## Libraries

Using `scikit-optimize` for Bayesian Optimization:
- Gaussian Process surrogate model
- Multiple acquisition functions
- Handles different parameter types
- Robust and well-tested
