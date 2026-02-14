"""Core framework components"""

from core.config import Config, ConfigNode, create_model_from_config
from core.types import *
from core.registry import (
    # Registries
    Registry,
    ModelRegistry,
    TrainerRegistry,
    PipelineRegistry,
    DatasetRegistry,
    # Decorators
    register_model,
    register_trainer,
    register_pipeline,
    register_dataset,
    # Getters
    get_model,
    get_trainer,
    get_pipeline,
    get_dataset,
    # Listers
    list_models,
    list_trainers,
    list_pipelines,
    list_datasets,
    # Checkers
    has_model,
    has_trainer,
    has_pipeline,
    has_dataset,
)

__all__ = [
    # Config
    'Config',
    'ConfigNode',
    # Registry
    'Registry',
    'ModelRegistry',
    'TrainerRegistry',
    'PipelineRegistry',
    'DatasetRegistry',
    'register_model',
    'register_trainer',
    'register_pipeline',
    'register_dataset',
    'get_model',
    'get_trainer',
    'get_pipeline',
    'get_dataset',
    'list_models',
    'list_trainers',
    'list_pipelines',
    'list_datasets',
    'has_model',
    'has_trainer',
    'has_pipeline',
    'has_dataset',
    'discover_plugins',
    'auto_discover',
    'reset_discovery',
    'set_auto_discover',
    'search_models',
    'search_trainers',
    'print_registry_summary',
    'create_model_from_config',
]
