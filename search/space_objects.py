"""
Hyperparameter search space objects.
"""

from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np


class SearchSpace(ABC):
    """Base class for search space dimensions."""

    @abstractmethod
    def sample(self, random_state=None) -> Any:
        """Sample a value from this dimension."""
        pass

    @abstractmethod
    def get_grid_values(self, num_values: int = 5) -> List[Any]:
        """Return grid values for grid search."""
        pass


class Categorical(SearchSpace):
    """Categorical parameter."""

    def __init__(self, choices: List[Any]):
        if not isinstance(choices, list):
            raise TypeError("choices must be a list")
        self.choices = choices

    def sample(self, random_state=None) -> Any:
        rng = np.random.RandomState(random_state)
        return rng.choice(self.choices)

    def get_grid_values(self, num_values: int = 5) -> List[Any]:
        """Return all choices (ignores num_values)"""
        return self.choices

    def __repr__(self):
        return f"Categorical({self.choices})"


class Integer(SearchSpace):
    """Integer parameter."""

    def __init__(self, low: int, high: int, log: bool = False):
        self.low = low
        self.high = high
        self.log = log

    def sample(self, random_state=None) -> int:
        rng = np.random.RandomState(random_state)
        if self.log:
            value = np.exp(rng.uniform(np.log(self.low), np.log(self.high)))
            return int(round(value))
        else:
            return rng.randint(self.low, self.high + 1)

    def get_grid_values(self, num_values: int = 5) -> List[int]:
        if self.log:
            return list(np.unique(np.geomspace(self.low, self.high, num_values).astype(int)))
        else:
            return list(range(self.low, self.high + 1))

    def __repr__(self):
        return f"Integer({self.low}, {self.high}, log={self.log})"


class Real(SearchSpace):
    """Real-valued parameter."""

    def __init__(self, low: float, high: float, log: bool = False):
        self.low = low
        self.high = high
        self.log = log

    def sample(self, random_state=None) -> float:
        rng = np.random.RandomState(random_state)
        if self.log:
            return float(np.exp(rng.uniform(np.log(self.low), np.log(self.high))))
        else:
            return float(rng.uniform(self.low, self.high))

    def get_grid_values(self, num_values: int = 5) -> List[float]:
        if self.log:
            return list(np.geomspace(self.low, self.high, num_values))
        else:
            return list(np.linspace(self.low, self.high, num_values))

    def __repr__(self):
        return f"Real({self.low}, {self.high}, log={self.log})"
