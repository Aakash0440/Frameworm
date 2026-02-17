"""Hyperparameter search"""

from search.analysis import SearchAnalyzer
from search.grid_search import GridSearch
from search.random_search import RandomSearch
from search.space_objects import Categorical, Integer, Real

from .space import expand_search_space, sample_configuration

__all__ = [
    "SearchSpace",
    "Categorical",
    "Integer",
    "Real",
    "expand_search_space",
    "sample_configuration",
    "GridSearch",
    "RandomSearch",
    "SearchAnalyzer",
]
try:
    from search.bayesian_search import BayesianSearch

    __all__.append("BayesianSearch")
except ImportError:
    pass  # scikit-optimize not available
