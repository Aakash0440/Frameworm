"""
Hyperparameter search space definitions.
"""

from typing import Any, List, Union, Dict
import numpy as np
from abc import ABC, abstractmethod


class SearchSpace(ABC):
    """Base class for search space dimensions."""
    
    @abstractmethod
    def sample(self, random_state=None):
        """Sample a value from this dimension."""
        pass
    
    @abstractmethod
    def get_grid_values(self):
        """Get grid values for grid search."""
        pass


class Categorical(SearchSpace):
    """
    Categorical parameter.
    
    Args:
        choices: List of possible values
        
    Example:
        >>> param = Categorical(['adam', 'sgd', 'rmsprop'])
        >>> value = param.sample()
    """
    
    def __init__(self, choices: List[Any]):
        self.choices = choices
    
    def sample(self, random_state=None):
        """Sample random choice"""
        rng = np.random.RandomState(random_state)
        return rng.choice(self.choices)
    
    def get_grid_values(self):
        """Return all choices"""
        return self.choices
    
    def __repr__(self):
        return f"Categorical({self.choices})"


class Integer(SearchSpace):
    """
    Integer parameter.
    
    Args:
        low: Minimum value (inclusive)
        high: Maximum value (inclusive)
        log: If True, sample in log space
        
    Example:
        >>> param = Integer(64, 512, log=True)
        >>> value = param.sample()
    """
    
    def __init__(self, low: int, high: int, log: bool = False):
        self.low = low
        self.high = high
        self.log = log
    
    def sample(self, random_state=None):
        """Sample random integer"""
        rng = np.random.RandomState(random_state)
        
        if self.log:
            # Sample in log space
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            value = np.exp(rng.uniform(log_low, log_high))
            return int(np.round(value))
        else:
            return rng.randint(self.low, self.high + 1)
    
    def get_grid_values(self):
        """Return range of values"""
        if self.log:
            # Geometric progression
            num_values = min(10, self.high - self.low + 1)
            return np.unique(np.geomspace(self.low, self.high, num_values).astype(int))
        else:
            # Linear progression
            return list(range(self.low, self.high + 1))
    
    def __repr__(self):
        return f"Integer({self.low}, {self.high}, log={self.log})"


class Real(SearchSpace):
    """
    Real-valued parameter.
    
    Args:
        low: Minimum value
        high: Maximum value
        log: If True, sample in log space
        
    Example:
        >>> param = Real(0.0001, 0.1, log=True)
        >>> value = param.sample()
    """
    
    def __init__(self, low: float, high: float, log: bool = False):
        self.low = low
        self.high = high
        self.log = log
    
    def sample(self, random_state=None):
        """Sample random float"""
        rng = np.random.RandomState(random_state)
        
        if self.log:
            # Sample in log space
            log_low = np.log(self.low)
            log_high = np.log(self.high)
            return np.exp(rng.uniform(log_low, log_high))
        else:
            return rng.uniform(self.low, self.high)
    
    def get_grid_values(self, num_values: int = 5):
        """Return grid of values"""
        if self.log:
            return np.geomspace(self.low, self.high, num_values)
        else:
            return np.linspace(self.low, self.high, num_values)
    
    def __repr__(self):
        return f"Real({self.low}, {self.high}, log={self.log})"


def expand_search_space(
    search_space: Dict[str, Union[SearchSpace, List]]
) -> Dict[str, SearchSpace]:
    """
    Convert search space to SearchSpace objects.
    
    Args:
        search_space: Dictionary mapping parameter names to values/SearchSpace objects
        
    Returns:
        Dictionary mapping parameter names to SearchSpace objects
        
    Example:
        >>> space = expand_search_space({
        ...     'lr': [0.001, 0.0001],  # Becomes Categorical
        ...     'batch_size': Integer(64, 256, log=True)
        ... })
    """
    expanded = {}
    
    for name, spec in search_space.items():
        if isinstance(spec, SearchSpace):
            expanded[name] = spec
        elif isinstance(spec, (list, tuple)):
            # Convert to Categorical
            expanded[name] = Categorical(list(spec))
        else:
            raise ValueError(f"Invalid search space specification for {name}: {spec}")
    
    return expanded


def sample_configuration(
    search_space: Dict[str, SearchSpace],
    random_state=None
) -> Dict[str, Any]:
    """
    Sample a configuration from search space.
    
    Args:
        search_space: Dictionary of SearchSpace objects
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary of sampled values
    """
    config = {}
    
    for name, space in search_space.items():
        config[name] = space.sample(random_state)
    
    return config