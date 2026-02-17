"""Frameworm package"""

# Search
from search import (
    GridSearch,
    RandomSearch,
    SearchAnalyzer,
    Categorical,
    Integer,
    Real
)

# Distributed
from distributed import (
    DistributedTrainer,
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    get_world_size,
    get_rank
)

from distributed.data_loader import OptimizedDataLoader
from distributed.profiler import PerformanceProfiler

try:
    from __version__ import __version__, __author__, __email__
except ImportError:
    __version__ = "0.1.0"
    __author__ = "Unknown"
    __email__ = "unknown@example.com"

__all__ = [
    '__version__',
    '__author__',
    '__email__',
    'GridSearch',
    'RandomSearch',
    'SearchAnalyzer',
    'Categorical',
    'Integer',
    'Real',
    'DistributedTrainer',
    'setup_distributed',
    'cleanup_distributed',
    'is_distributed',
    'get_world_size',
    'get_rank',
    'OptimizedDataLoader',
    'PerformanceProfiler',
]
