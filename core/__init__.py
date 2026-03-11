"""Core framework components"""

from core.config import Config, ConfigNode, create_model_from_config

# Import exceptions
from core.exceptions import (
    ConfigInheritanceError,
    ConfigNotFoundError,
    ConfigurationError,
    ConfigValidationError,
    ConvergenceError,
    DimensionMismatchError,
    FramewormError,
    ModelError,
    ModelNotFoundError,
    PluginError,
    PluginValidationError,
    TrainingError,
)
from core.registry import has_trainer  # ADD THIS
from core.registry import (  # Registries; Decorators; Getters; Listers; Checkers
    DatasetRegistry,
    ModelRegistry,
    PipelineRegistry,
    Registry,
    TrainerRegistry,
    get_dataset,
    get_model,
    get_model_metadata,
    get_pipeline,
    get_trainer,
    has_dataset,
    has_model,
    has_pipeline,
    list_datasets,
    list_models,
    list_pipelines,
    list_trainers,
    register_dataset,
    register_model,
    register_pipeline,
    register_trainer,
)
from core.types import *

__all__ = [
    # Config
    "Config",
    "ConfigNode",
    # Registry
    "Registry",
    "ModelRegistry",
    "TrainerRegistry",
    "PipelineRegistry",
    "DatasetRegistry",
    "register_model",
    "register_trainer",
    "register_pipeline",
    "register_dataset",
    "get_model",
    "get_trainer",
    "get_pipeline",
    "get_dataset",
    "list_models",
    "list_trainers",
    "list_pipelines",
    "list_datasets",
    "has_model",
    "has_trainer",
    "has_pipeline",
    "has_dataset",
    "discover_plugins",
    "auto_discover",
    "reset_discovery",
    "set_auto_discover",
    "search_models",
    "search_trainers",
    "print_registry_summary",
    "create_model_from_config",
]
