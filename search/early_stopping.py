"""
Early stopping mechanisms for hyperparameter search.
"""

from typing import Any, Callable, Dict, Optional

import numpy as np


class EarlyStopper:
    """
    Base class for early stopping in search.
    """

    def should_stop(self, trial_idx: int, score: float, config: Dict[str, Any]) -> bool:
        """
        Check if search should stop early.

        Args:
            trial_idx: Current trial index
            score: Current score
            config: Current configuration

        Returns:
            True if should stop
        """
        raise NotImplementedError


class MedianStopper(EarlyStopper):
    """
    Stop trial if performing worse than median at same stage.

    Inspired by Google Vizier's median stopping rule.

    Args:
        percentile: Percentile threshold (50 = median)
        min_trials: Minimum trials before stopping
    """

    def __init__(self, percentile: float = 50, min_trials: int = 5):
        self.percentile = percentile
        self.min_trials = min_trials
        self.history: list = []

    def should_stop(self, trial_idx: int, score: float, config: Dict[str, Any]) -> bool:
        """Check if trial should stop based on median performance"""
        self.history.append(score)

        if len(self.history) < self.min_trials:
            return False

        # Compare to percentile
        threshold = np.percentile(self.history, self.percentile)
        return score > threshold


class ImprovementStopper(EarlyStopper):
    """
    Stop if no improvement for N trials.

    Args:
        patience: Number of trials without improvement
        min_delta: Minimum improvement to count
    """

    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float("inf")
        self.trials_without_improvement = 0

    def should_stop(self, trial_idx: int, score: float, config: Dict[str, Any]) -> bool:
        """Check if no improvement for patience trials"""
        if score < self.best_score - self.min_delta:
            self.best_score = score
            self.trials_without_improvement = 0
        else:
            self.trials_without_improvement += 1

        return self.trials_without_improvement >= self.patience


class ThresholdStopper(EarlyStopper):
    """
    Stop if score crosses threshold.

    Args:
        threshold: Score threshold
        mode: 'min' or 'max'
    """

    def __init__(self, threshold: float, mode: str = "min"):
        self.threshold = threshold
        self.mode = mode

    def should_stop(self, trial_idx: int, score: float, config: Dict[str, Any]) -> bool:
        """Check if threshold reached"""
        if self.mode == "min":
            return score <= self.threshold
        else:
            return score >= self.threshold


class BudgetStopper(EarlyStopper):
    """
    Stop after budget exhausted.

    Args:
        max_trials: Maximum number of trials
        max_time: Maximum time in seconds (optional)
    """

    def __init__(self, max_trials: int, max_time: Optional[float] = None):
        self.max_trials = max_trials
        self.max_time = max_time
        self.start_time: Optional[float] = None

    def should_stop(self, trial_idx: int, score: float, config: Dict[str, Any]) -> bool:
        """Check if budget exhausted"""
        import time

        if self.start_time is None:
            self.start_time = time.time()

        # Check trial budget
        if trial_idx >= self.max_trials:
            return True

        # Check time budget
        if self.max_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.max_time:
                return True

        return False
