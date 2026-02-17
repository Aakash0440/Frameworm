"""
Grid search implementation.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
from itertools import product
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from joblib import Parallel, delayed

from search.space import expand_search_space, SearchSpace
from core import Config
from experiment import Experiment


class GridSearch:
    """
    Exhaustive grid search over hyperparameters.

    Args:
        base_config: Base configuration
        search_space: Dictionary of parameters to search
        metric: Metric to optimize
        mode: 'min' or 'max'

    Example:
        >>> search = GridSearch(
        ...     base_config=config,
        ...     search_space={
        ...         'training.lr': [0.001, 0.0001],
        ...         'training.batch_size': [64, 128]
        ...     },
        ...     metric='val_loss',
        ...     mode='min'
        ... )
        >>> best_config, best_score = search.run(train_fn)
    """

    def __init__(
        self,
        base_config: Config,
        search_space: Dict[str, Any],
        metric: str = "val_loss",
        mode: str = "min",
    ):
        self.base_config = base_config
        self.search_space = expand_search_space(search_space)
        self.metric = metric
        self.mode = mode

        # Results
        self.results: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate all possible configurations.

        Returns:
            List of configuration dictionaries
        """
        # Get grid values for each parameter
        param_names = list(self.search_space.keys())
        param_grids = []

        for name in param_names:
            space = self.search_space[name]
            grid_values = space.get_grid_values()
            param_grids.append(grid_values)

        # Generate all combinations
        configurations = []
        for values in product(*param_grids):
            config = dict(zip(param_names, values))
            configurations.append(config)

        return configurations

    def _apply_config(self, config_updates: Dict[str, Any]) -> Config:
        """
        Apply parameter updates to base config.

        Args:
            config_updates: Dictionary of parameter updates

        Returns:
            Updated configuration
        """
        # Create copy of base config
        config = Config.from_dict(self.base_config.to_dict())

        # Apply updates
        for key, value in config_updates.items():
            parts = key.split(".")
            obj = config
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)

        return config

    def _evaluate_configuration(
        self,
        config_updates: Dict[str, Any],
        train_fn: Callable,
        trial_idx: int,
        experiment_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a single configuration.

        Args:
            config_updates: Parameter updates
            train_fn: Training function (config) -> metrics
            trial_idx: Trial index
            experiment_root: Root directory for experiments

        Returns:
            Dictionary with config and results
        """
        # Apply configuration
        config = self._apply_config(config_updates)

        # Create experiment if requested
        if experiment_root:
            exp = Experiment(
                name=f"grid_search_trial_{trial_idx}",
                config=config.to_dict(),
                description=f"Grid search trial with {config_updates}",
                tags=["grid_search"],
                root_dir=experiment_root,
            )

            with exp:
                # Train
                metrics = train_fn(config)

                # Log final metrics
                for metric_name, value in metrics.items():
                    exp.log_metric(metric_name, value, epoch=0, metric_type="final")
        else:
            # Train without experiment tracking
            metrics = train_fn(config)

        # Get target metric
        score = metrics.get(self.metric)

        if score is None:
            raise ValueError(f"Metric '{self.metric}' not found in results: {metrics}")

        return {
            "config": config_updates,
            "metrics": metrics,
            "score": score,
            "trial_idx": trial_idx,
        }

    def run(
        self,
        train_fn: Callable[[Config], Dict[str, float]],
        n_jobs: int = 1,
        experiment_root: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run grid search.

        Args:
            train_fn: Training function that takes Config and returns metrics dict
            n_jobs: Number of parallel jobs (-1 for all CPUs)
            experiment_root: Root directory for experiment tracking
            verbose: Print progress

        Returns:
            (best_config, best_score)
        """
        # Generate configurations
        configurations = self._generate_configurations()

        if verbose:
            print(f"Starting grid search with {len(configurations)} configurations")
            print(f"Optimizing: {self.metric} ({self.mode})")
            print(f"Parallel jobs: {n_jobs}")

        # Evaluate configurations
        if n_jobs == 1:
            # Sequential execution
            results = []
            for i, config in enumerate(
                tqdm(configurations, desc="Grid search") if verbose else configurations
            ):
                result = self._evaluate_configuration(config, train_fn, i, experiment_root)
                results.append(result)

                if verbose:
                    print(f"Trial {i}: {self.metric}={result['score']:.4f}")
        else:
            # Parallel execution
            results = Parallel(n_jobs=n_jobs)(
                delayed(self._evaluate_configuration)(config, train_fn, i, experiment_root)
                for i, config in enumerate(configurations)
            )

        # Store results
        self.results = results

        # Find best configuration
        if self.mode == "min":
            best_result = min(results, key=lambda x: x["score"])
        else:
            best_result = max(results, key=lambda x: x["score"])

        self.best_config = best_result["config"]
        self.best_score = best_result["score"]

        if verbose:
            print(f"\n{'='*60}")
            print(f"Best configuration found:")
            for key, value in self.best_config.items():
                print(f"  {key}: {value}")
            print(f"Best {self.metric}: {self.best_score:.4f}")
            print(f"{'='*60}")

        return self.best_config, self.best_score

    def save_results(self, path: str):
        """Save search results to file"""
        results_data = {
            "search_space": {k: str(v) for k, v in self.search_space.items()},
            "metric": self.metric,
            "mode": self.mode,
            "best_config": self.best_config,
            "best_score": self.best_score,
            "all_results": self.results,
        }

        with open(path, "w") as f:
            json.dump(results_data, f, indent=2)

    def load_results(self, path: str):
        """Load search results from file"""
        with open(path, "r") as f:
            results_data = json.load(f)

        self.best_config = results_data["best_config"]
        self.best_score = results_data["best_score"]
        self.results = results_data["all_results"]
