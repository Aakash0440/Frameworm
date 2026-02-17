"""
FRAMEWORM - Complete Machine Learning Framework

Quick start:
    from frameworm import Trainer, Config, get_model

    config = Config('config.yaml')
    model = get_model('vae')(config)
    trainer = Trainer(model, optimizer)
    trainer.train(train_loader, val_loader)
"""
import models  # triggers all @register_model decorators

# Distributed
from distributed import (
    DistributedTrainer,
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_distributed,
    setup_distributed,
)
from distributed.data_loader import OptimizedDataLoader
from distributed.profiler import PerformanceProfiler
from experiment.experiment import Experiment
from experiment.manager import ExperimentManager

# Search
from search import Categorical, GridSearch, Integer, RandomSearch, Real, SearchAnalyzer

try:
    from __version__ import __author__, __email__, __version__
except ImportError:

    __version__ = "0.1.0"
    __author__ = "Unknown"
    __email__ = "unknown@example.com"

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "GridSearch",
    "RandomSearch",
    "SearchAnalyzer",
    "Categorical",
    "Integer",
    "Real",
    "DistributedTrainer",
    "setup_distributed",
    "cleanup_distributed",
    "is_distributed",
    "get_world_size",
    "get_rank",
    "OptimizedDataLoader",
    "PerformanceProfiler",
    "PerformanceProfiler",
    "Experiment",          # ADD
    "ExperimentManager",
]
