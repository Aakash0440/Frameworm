"""
Hyperparameter search space definitions.
"""

from typing import Any, Dict, List
import numpy as np
from search.space_objects import Real, Integer, Categorical, SearchSpace


def expand_search_space(search_space: Dict[str, Any]) -> Dict[str, SearchSpace]:
    """
    Convert user-provided search space specs into SearchSpace objects.
    
    Args:
        search_space: Dictionary of name -> spec
    
    Returns:
        Dictionary of name -> SearchSpace object
    """
    expanded = {}
    for name, spec in search_space.items():
        if isinstance(spec, (Real, Integer, Categorical, SearchSpace)):
            expanded[name] = spec
        elif isinstance(spec, (list, tuple)):
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
    """
    config = {}
    for name, space in search_space.items():
        config[name] = space.sample(random_state)
    return config
