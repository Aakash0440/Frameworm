"""
Bayesian optimization for hyperparameter search.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

try:
    from skopt import gp_minimize
    from skopt.space import Real as SkoptReal, Integer as SkoptInteger, Categorical as SkoptCategorical
    from skopt.utils import use_named_args
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

from search.space import expand_search_space, SearchSpace, Real, Integer, Categorical
from core import Config
from experiment import Experiment


class BayesianSearch:
    """
    Bayesian optimization for hyperparameter search.
    
    Uses Gaussian Processes to model the objective function and
    intelligently select promising configurations to evaluate.
    
    Args:
        base_config: Base configuration
        search_space: Dictionary of parameters to search
        metric: Metric to optimize
        mode: 'min' or 'max'
        n_trials: Number of trials
        n_initial_points: Number of random points before GP
        acquisition: Acquisition function ('ei', 'lcb', 'pi')
        random_state: Random seed
        
    Example:
        >>> from frameworm.search.space import Real, Integer
        >>> search = BayesianSearch(
        ...     base_config=config,
        ...     search_space={
        ...         'training.lr': Real(1e-5, 1e-2, log=True),
        ...         'training.batch_size': Integer(32, 256, log=True)
        ...     },
        ...     metric='val_loss',
        ...     mode='min',
        ...     n_trials=50
        ... )
        >>> best_config, best_score = search.run(train_fn)
    """
    
    def __init__(
        self,
        base_config: Config,
        search_space: Dict[str, Any],
        metric: str = 'val_loss',
        mode: str = 'min',
        n_trials: int = 50,
        n_initial_points: int = 10,
        acquisition: str = 'ei',
        random_state: Optional[int] = None
    ):
        if not SKOPT_AVAILABLE:
            raise ImportError(
                "scikit-optimize is required for Bayesian optimization. "
                "Install with: pip install scikit-optimize"
            )
        
        self.base_config = base_config
        self.search_space = expand_search_space(search_space)
        self.metric = metric
        self.mode = mode
        self.n_trials = n_trials
        self.n_initial_points = n_initial_points
        self.acquisition = acquisition
        self.random_state = random_state
        
        # Results
        self.results: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
        
        # Convert to skopt space
        self.skopt_space, self.param_names = self._create_skopt_space()
    
    def _create_skopt_space(self) -> Tuple[List, List[str]]:
        """
        Convert search space to scikit-optimize format.
        
        Returns:
            (skopt_space, param_names)
        """
        skopt_space = []
        param_names = []
        
        for name, space_obj in self.search_space.items():
            param_names.append(name)
            
            if isinstance(space_obj, Categorical):
                skopt_space.append(
                    SkoptCategorical(space_obj.choices, name=name)
                )
            elif isinstance(space_obj, Integer):
                if space_obj.log:
                    skopt_space.append(
                        SkoptInteger(space_obj.low, space_obj.high, prior='log-uniform', name=name)
                    )
                else:
                    skopt_space.append(
                        SkoptInteger(space_obj.low, space_obj.high, name=name)
                    )
            elif isinstance(space_obj, Real):
                if space_obj.log:
                    skopt_space.append(
                        SkoptReal(space_obj.low, space_obj.high, prior='log-uniform', name=name)
                    )
                else:
                    skopt_space.append(
                        SkoptReal(space_obj.low, space_obj.high, name=name)
                    )
            else:
                raise ValueError(f"Unsupported space type: {type(space_obj)}")
        
        return skopt_space, param_names
    
    def _apply_config(self, config_updates: Dict[str, Any]) -> Config:
        """Apply parameter updates to base config"""
        config = Config.from_dict(self.base_config.to_dict())
        
        for key, value in config_updates.items():
            parts = key.split('.')
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        
        return config
    
    def _objective_function(
        self,
        params: List[Any],
        train_fn: Callable,
        experiment_root: Optional[str],
        verbose: bool
    ) -> float:
        """
        Objective function for Bayesian optimization.
        
        Args:
            params: Parameter values (ordered by param_names)
            train_fn: Training function
            experiment_root: Experiment directory
            verbose: Print progress
            
        Returns:
            Score (negated if mode='max' since skopt minimizes)
        """
        # Convert params to config dict
        config_updates = dict(zip(self.param_names, params))
        
        # Apply configuration
        config = self._apply_config(config_updates)
        
        # Get trial index
        trial_idx = len(self.results)
        
        # Create experiment if requested
        if experiment_root:
            exp = Experiment(
                name=f"bayes_opt_trial_{trial_idx}",
                config=config.to_dict(),
                description=f"Bayesian optimization trial with {config_updates}",
                tags=['bayesian_optimization'],
                root_dir=experiment_root
            )
            
            with exp:
                metrics = train_fn(config)
                
                for metric_name, value in metrics.items():
                    exp.log_metric(metric_name, value, epoch=0, metric_type='final')
        else:
            metrics = train_fn(config)
        
        # Get target metric
        score = metrics.get(self.metric)
        
        if score is None:
            raise ValueError(f"Metric '{self.metric}' not found in results: {metrics}")
        
        # Store result
        result = {
            'config': config_updates,
            'metrics': metrics,
            'score': score,
            'trial_idx': trial_idx
        }
        self.results.append(result)
        
        # Print progress
        if verbose:
            best_so_far = min(self.results, key=lambda x: x['score'])['score'] if self.mode == 'min' else max(self.results, key=lambda x: x['score'])['score']
            is_best = score == best_so_far
            print(f"Trial {trial_idx}: {self.metric}={score:.4f}{' ✓ (new best)' if is_best else ''}")
        
        # Return score (negate if maximizing)
        return score if self.mode == 'min' else -score
    
    def run(
        self,
        train_fn: Callable[[Config], Dict[str, float]],
        experiment_root: Optional[str] = None,
        verbose: bool = True
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run Bayesian optimization.
        
        Args:
            train_fn: Training function
            experiment_root: Root directory for experiments
            verbose: Print progress
            
        Returns:
            (best_config, best_score)
        """
        if verbose:
            print(f"Starting Bayesian optimization with {self.n_trials} trials")
            print(f"Optimizing: {self.metric} ({self.mode})")
            print(f"Acquisition function: {self.acquisition}")
            print(f"Initial random points: {self.n_initial_points}")
        
        # Create objective function wrapper
        @use_named_args(self.skopt_space)
        def objective(**params):
            param_values = [params[name] for name in self.param_names]
            return self._objective_function(param_values, train_fn, experiment_root, verbose)
        
        # Run optimization
        result = gp_minimize(
            objective,
            self.skopt_space,
            n_calls=self.n_trials,
            n_initial_points=self.n_initial_points,
            acq_func=self.acquisition,
            random_state=self.random_state,
            verbose=False  # We handle verbosity ourselves
        )
        
        # Extract best configuration
        best_params = result.x
        self.best_config = dict(zip(self.param_names, best_params))
        self.best_score = result.fun if self.mode == 'min' else -result.fun
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Best configuration found:")
            for key, value in self.best_config.items():
                print(f"  {key}: {value}")
            print(f"Best {self.metric}: {self.best_score:.4f}")
            print(f"{'='*60}")
        
        return self.best_config, self.best_score
    
    def save_results(self, path: str):
        """Save search results"""
        results_data = {
            'search_space': {k: str(v) for k, v in self.search_space.items()},
            'metric': self.metric,
            'mode': self.mode,
            'n_trials': self.n_trials,
            'n_initial_points': self.n_initial_points,
            'acquisition': self.acquisition,
            'random_state': self.random_state,
            'best_config': self.best_config,
            'best_score': self.best_score,
            'all_results': self.results
        }
        
        with open(path, 'w') as f:
            json.dump(results_data, f, indent=2)
    
    def load_results(self, path: str):
        """Load search results"""
        with open(path, 'r') as f:
            results_data = json.load(f)
        
        self.best_config = results_data['best_config']
        self.best_score = results_data['best_score']
        self.results = results_data['all_results']
        self.n_trials = results_data.get('n_trials', len(self.results))