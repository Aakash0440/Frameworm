"""
Hyperparameter search command.
"""

from core import Config
from search import GridSearch, RandomSearch
import yaml
from click import echo


def run_search(
    config_path: str,
    space_path: str,
    method: str = 'random',
    n_trials: int = 50,
    n_jobs: int = 1
):
    """Run hyperparameter search from CLI"""
    
    # Load config
    config = Config(config_path)
    
    # Load search space
    with open(space_path) as f:
        search_space = yaml.safe_load(f)
    
    echo(f"Search space: {search_space}")
    echo(f"Method: {method}")
    echo(f"Trials: {n_trials}")
    
    # Create search
    if method == 'grid':
        search = GridSearch(
            base_config=config,
            search_space=search_space,
            metric='val_loss',
            mode='min'
        )
        best_config, best_score = search.run(
            train_fn=None,  # Placeholder
            n_jobs=n_jobs
        )
    
    elif method == 'random':
        search = RandomSearch(
            base_config=config,
            search_space=search_space,
            metric='val_loss',
            mode='min',
            n_trials=n_trials
        )
        best_config, best_score = search.run(
            train_fn=None,
            n_jobs=n_jobs
        )
    
    elif method == 'bayesian':
        try:
            from frameworm.search import BayesianSearch
            search = BayesianSearch(
                base_config=config,
                search_space=search_space,
                metric='val_loss',
                mode='min',
                n_trials=n_trials
            )
            best_config, best_score = search.run(train_fn=None)
        except ImportError:
            echo("✗ Bayesian search requires scikit-optimize")
            return
    
    echo(f"\n✓ Search complete!")
    echo(f"Best score: {best_score}")
    echo(f"Best config: {best_config}")