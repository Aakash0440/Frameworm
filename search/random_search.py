"""
Random search implementation.
"""

from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from joblib import Parallel, delayed

from search.space import expand_search_space, sample_configuration, SearchSpace
from core import Config
from experiment import Experiment


class RandomSearch:
    """
    Random search over hyperparameters.

    Often more efficient than grid search for high-dimensional spaces.

    Args:
        base_config: Base configuration
        search_space: Dictionary of parameters to search
        metric: Metric to optimize
        mode: 'min' or 'max'
        n_trials: Number of random trials
        random_state: Random seed for reproducibility

    Example:
        >>> from frameworm.search.space import Real, Integer
        >>> search = RandomSearch(
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
        metric: str = "val_loss",
        mode: str = "min",
        n_trials: int = 50,
        random_state: Optional[int] = None,
    ):
        self.base_config = base_config
        self.search_space = expand_search_space(search_space)
        self.metric = metric
        self.mode = mode
        self.n_trials = n_trials
        self.random_state = random_state

        # Results
        self.results: List[Dict[str, Any]] = []
        self.best_config: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None

    def _generate_configurations(self) -> List[Dict[str, Any]]:
        """
        Generate random configurations.

        Returns:
            List of configuration dictionaries
        """
        configurations = []
        rng = np.random.RandomState(self.random_state)

        for i in range(self.n_trials):
            # Sample configuration
            config = sample_configuration(self.search_space, random_state=rng.randint(0, 2**31))
            configurations.append(config)

        return configurations

    def _apply_config(self, config_updates: Dict[str, Any]) -> Config:
        """Apply parameter updates to base config"""
        config = Config.from_dict(self.base_config.to_dict())

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
        """Evaluate a single configuration"""
        # Apply configuration
        config = self._apply_config(config_updates)

        # Create experiment if requested
        if experiment_root:
            exp = Experiment(
                name=f"random_search_trial_{trial_idx}",
                config=config.to_dict(),
                description=f"Random search trial with {config_updates}",
                tags=["random_search"],
                root_dir=experiment_root,
            )

            with exp:
                metrics = train_fn(config)

                for metric_name, value in metrics.items():
                    exp.log_metric(metric_name, value, epoch=0, metric_type="final")
        else:
            metrics = train_fn(config)

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
        early_stopping_patience: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], float]:
        """
        Run random search.

        Args:
            train_fn: Training function
            n_jobs: Number of parallel jobs
            experiment_root: Root directory for experiments
            verbose: Print progress
            early_stopping_patience: Stop if no improvement for N trials

        Returns:
            (best_config, best_score)
        """
        # Generate configurations
        configurations = self._generate_configurations()

        if verbose:
            print(f"Starting random search with {len(configurations)} trials")
            print(f"Optimizing: {self.metric} ({self.mode})")
            print(f"Parallel jobs: {n_jobs}")

        # Track best score for early stopping
        best_score_so_far = float("inf") if self.mode == "min" else float("-inf")
        trials_without_improvement = 0

        # Evaluate configurations
        results = []

        if n_jobs == 1:
            # Sequential execution with early stopping support
            for i, config in enumerate(
                tqdm(configurations, desc="Random search") if verbose else configurations
            ):
                result = self._evaluate_configuration(config, train_fn, i, experiment_root)
                results.append(result)

                # Check for improvement
                improved = (self.mode == "min" and result["score"] < best_score_so_far) or (
                    self.mode == "max" and result["score"] > best_score_so_far
                )

                if improved:
                    best_score_so_far = result["score"]
                    trials_without_improvement = 0
                    if verbose:
                        print(f"Trial {i}: {self.metric}={result['score']:.4f} ✓ (new best)")
                else:
                    trials_without_improvement += 1
                    if verbose:
                        print(f"Trial {i}: {self.metric}={result['score']:.4f}")

                # Early stopping
                if (
                    early_stopping_patience
                    and trials_without_improvement >= early_stopping_patience
                ):
                    if verbose:
                        print(
                            f"\nEarly stopping after {i+1} trials (no improvement for {early_stopping_patience} trials)"
                        )
                    break
        else:
            # Parallel execution (no early stopping support)
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
            print(f"Best configuration found (out of {len(results)} trials):")
            for key, value in self.best_config.items():
                print(f"  {key}: {value}")
            print(f"Best {self.metric}: {self.best_score:.4f}")
            print(f"{'='*60}")

        return self.best_config, self.best_score

    def save_results(self, path: str):
        """Save search results"""
        results_data = {
            "search_space": {k: str(v) for k, v in self.search_space.items()},
            "metric": self.metric,
            "mode": self.mode,
            "n_trials": self.n_trials,
            "random_state": self.random_state,
            "best_config": self.best_config,
            "best_score": self.best_score,
            "all_results": self.results,
        }

        with open(path, "w") as f:
            json.dump(results_data, f, indent=2)

    def load_results(self, path: str):
        """Load search results"""
        with open(path, "r") as f:
            results_data = json.load(f)

        self.best_config = results_data["best_config"]
        self.best_score = results_data["best_score"]
        self.results = results_data["all_results"]
        self.n_trials = results_data.get("n_trials", len(self.results))
