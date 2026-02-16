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
    DataParallelTrainer,
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    get_world_size,
    get_rank
)

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
]