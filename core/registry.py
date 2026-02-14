"""
Plugin registry system for Frameworm.

Provides a flexible plugin architecture where users can register custom
models, trainers, pipelines, and other components.

Example:
    >>> @register_model("my-gan")
    >>> class MyGAN(BaseModel):
    ...     def forward(self, x):
    ...         return x
    >>> 
    >>> model_class = get_model("my-gan")
    >>> model = model_class(config)
"""

from typing import Type, Dict, Any, Optional, List, Callable
from abc import ABC
import importlib
import inspect
from pathlib import Path
import warnings
import sys
import os


class Registry:
    """
    Base registry class for managing plugins.
    
    Each namespace (models, trainers, pipelines) has its own Registry instance.
    
    Example:
        >>> registry = Registry(name="models")
        >>> registry.register("my-model", MyModel)
        >>> model_class = registry.get("my-model")
    """
    
    def __init__(self, name: str):
        """
        Initialize registry.
        
        Args:
            name: Registry namespace (e.g., "models", "trainers")
        """
        self.name = name
        self._registry: Dict[str, Type] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}
    
    def register(
        self,
        name: str,
        cls: Type,
        override: bool = False,
        **metadata
    ) -> Type:
        """
        Register a class in the registry.
        
        Args:
            name: Name to register under
            cls: Class to register
            override: If True, allow overriding existing registration
            **metadata: Additional metadata about the plugin
            
        Returns:
            The registered class (for use as decorator)
            
        Raises:
            ValueError: If name already registered and override=False
            TypeError: If cls doesn't meet requirements
        """
        # Check for duplicate registration
        if name in self._registry and not override:
            raise ValueError(
                f"'{name}' is already registered in {self.name} registry. "
                f"Use override=True to replace it."
            )
        
        # Validate class
        self._validate(cls)
        
        # Register
        self._registry[name] = cls
        self._metadata[name] = {
            'module': cls.__module__,
            'qualname': cls.__qualname__,
            **metadata
        }
        
        return cls
    
    def get(self, name: str) -> Type:
        """
        Get a registered class by name.
        
        Args:
            name: Registered name
            
        Returns:
            Registered class
            
        Raises:
            KeyError: If name not found
        """
        if name not in self._registry:
            available = ', '.join(self.list())
            raise KeyError(
                f"'{name}' not found in {self.name} registry. "
                f"Available: {available}"
            )
        
        return self._registry[name]
    
    def has(self, name: str) -> bool:
        """
        Check if name is registered.
        
        Args:
            name: Name to check
            
        Returns:
            True if registered
        """
        return name in self._registry
    
    def list(self) -> List[str]:
        """
        List all registered names.
        
        Returns:
            List of registered names
        """
        return sorted(self._registry.keys())
    
    def remove(self, name: str):
        """
        Remove a registration.
        
        Args:
            name: Name to remove
            
        Raises:
            KeyError: If name not found
        """
        if name not in self._registry:
            raise KeyError(f"'{name}' not found in {self.name} registry")
        
        del self._registry[name]
        del self._metadata[name]
    
    def clear(self):
        """Clear all registrations"""
        self._registry.clear()
        self._metadata.clear()
    
    def get_metadata(self, name: str) -> Dict[str, Any]:
        """
        Get metadata for a registered item.
        
        Args:
            name: Registered name
            
        Returns:
            Metadata dictionary
        """
        if name not in self._metadata:
            raise KeyError(f"'{name}' not found in {self.name} registry")
        
        return self._metadata[name].copy()
    
    def _validate(self, cls: Type):
        """
        Validate a class before registration.
        
        Override in subclasses for namespace-specific validation.
        
        Args:
            cls: Class to validate
            
        Raises:
            TypeError: If validation fails
        """
        if not inspect.isclass(cls):
            raise TypeError(f"Expected a class, got {type(cls)}")
    
    def __len__(self) -> int:
        """Number of registered items"""
        return len(self._registry)
    
    def __contains__(self, name: str) -> bool:
        """Check if name is registered"""
        return name in self._registry
    
    def __repr__(self) -> str:
        """String representation"""
        return f"Registry(name='{self.name}', items={len(self)})"


class ModelRegistry(Registry):
    """
    Registry for models.
    
    Validates that registered classes inherit from BaseModel.
    """
    
    def __init__(self):
        super().__init__(name="models")
    
    def _validate(self, cls: Type):
        """Validate model class"""
        super()._validate(cls)
        
        # Check for required methods
        required_methods = ['forward']
        for method in required_methods:
            if not hasattr(cls, method):
                raise TypeError(
                    f"Model class {cls.__name__} must implement {method}() method"
                )


class TrainerRegistry(Registry):
    """
    Registry for trainers.
    
    Validates that registered classes inherit from BaseTrainer.
    """
    
    def __init__(self):
        super().__init__(name="trainers")
    
    def _validate(self, cls: Type):
        """Validate trainer class"""
        super()._validate(cls)
        
        # Check for required methods
        required_methods = ['training_step', 'validation_step']
        for method in required_methods:
            if not hasattr(cls, method):
                raise TypeError(
                    f"Trainer class {cls.__name__} must implement {method}() method"
                )


class PipelineRegistry(Registry):
    """
    Registry for pipelines.
    
    Validates that registered classes inherit from BasePipeline.
    """
    
    def __init__(self):
        super().__init__(name="pipelines")
    
    def _validate(self, cls: Type):
        """Validate pipeline class"""
        super()._validate(cls)
        
        # Check for required methods
        required_methods = ['run']
        for method in required_methods:
            if not hasattr(cls, method):
                raise TypeError(
                    f"Pipeline class {cls.__name__} must implement {method}() method"
                )


class DatasetRegistry(Registry):
    """Registry for datasets"""
    
    def __init__(self):
        super().__init__(name="datasets")
    
    def _validate(self, cls: Type):
        """Validate dataset class"""
        super()._validate(cls)
        
        # Check for required methods
        required_methods = ['__len__', '__getitem__']
        for method in required_methods:
            if not hasattr(cls, method):
                raise TypeError(
                    f"Dataset class {cls.__name__} must implement {method}() method"
                )


# Create global registry instances
_MODEL_REGISTRY = ModelRegistry()
_TRAINER_REGISTRY = TrainerRegistry()
_PIPELINE_REGISTRY = PipelineRegistry()
_DATASET_REGISTRY = DatasetRegistry()


# ==================== Decorator Functions ====================

def register_model(name: str, **metadata):
    """
    Decorator to register a model.
    
    Args:
        name: Name to register under
        **metadata: Additional metadata
        
    Example:
        >>> @register_model("my-gan")
        >>> class MyGAN(BaseModel):
        ...     def forward(self, x):
        ...         return x
    """
    def decorator(cls: Type) -> Type:
        _MODEL_REGISTRY.register(name, cls, **metadata)
        return cls
    return decorator


def register_trainer(name: str, **metadata):
    """
    Decorator to register a trainer.
    
    Args:
        name: Name to register under
        **metadata: Additional metadata
        
    Example:
        >>> @register_trainer("my-trainer")
        >>> class MyTrainer(BaseTrainer):
        ...     def training_step(self, batch, idx):
        ...         return {'loss': 0.0}
    """
    def decorator(cls: Type) -> Type:
        _TRAINER_REGISTRY.register(name, cls, **metadata)
        return cls
    return decorator


def register_pipeline(name: str, **metadata):
    """
    Decorator to register a pipeline.
    
    Args:
        name: Name to register under
        **metadata: Additional metadata
    """
    def decorator(cls: Type) -> Type:
        _PIPELINE_REGISTRY.register(name, cls, **metadata)
        return cls
    return decorator


def register_dataset(name: str, **metadata):
    """
    Decorator to register a dataset.
    
    Args:
        name: Name to register under
        **metadata: Additional metadata
    """
    def decorator(cls: Type) -> Type:
        _DATASET_REGISTRY.register(name, cls, **metadata)
        return cls
    return decorator

_AUTO_DISCOVER = True  # Global flag for auto-discovery

def set_auto_discover(enabled: bool):
    """
    Enable/disable automatic plugin discovery.
    
    Args:
        enabled: If True, auto-discover plugins on first access
    """
    global _AUTO_DISCOVER
    _AUTO_DISCOVER = enabled

# ==================== Getter Functions ====================

def get_model(name: str, auto_discover: Optional[bool] = None) -> Type:
    """Get a registered model class."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()  # call the plugin discovery function
    
    return _MODEL_REGISTRY.get(name)


def get_trainer(name: str, auto_discover: Optional[bool] = None) -> Type:
    """Get a registered trainer class."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _TRAINER_REGISTRY.get(name)


def get_pipeline(name: str, auto_discover: Optional[bool] = None) -> Type:
    """Get a registered pipeline class."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _PIPELINE_REGISTRY.get(name)


def get_dataset(name: str, auto_discover: Optional[bool] = None) -> Type:
    """Get a registered dataset class."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _DATASET_REGISTRY.get(name)


# ==================== List Functions ====================

def list_models(auto_discover: Optional[bool] = None) -> List[str]:
    """List all registered models."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _MODEL_REGISTRY.list()


def list_trainers(auto_discover: Optional[bool] = None) -> List[str]:
    """List all registered trainers."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _TRAINER_REGISTRY.list()


def list_pipelines(auto_discover: Optional[bool] = None) -> List[str]:
    """List all registered pipelines."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _PIPELINE_REGISTRY.list()


def list_datasets(auto_discover: Optional[bool] = None) -> List[str]:
    """List all registered datasets."""
    should_discover = auto_discover if auto_discover is not None else _AUTO_DISCOVER
    if should_discover:
        _auto_discover_plugins()
    
    return _DATASET_REGISTRY.list()

# ==================== Has Functions ====================

def has_model(name: str) -> bool:
    """Check if model is registered"""
    return _MODEL_REGISTRY.has(name)


def has_trainer(name: str) -> bool:
    """Check if trainer is registered"""
    return _TRAINER_REGISTRY.has(name)


def has_pipeline(name: str) -> bool:
    """Check if pipeline is registered"""
    return _PIPELINE_REGISTRY.has(name)


def has_dataset(name: str) -> bool:
    """Check if dataset is registered"""
    return _DATASET_REGISTRY.has(name)

# ==================== Plugin Discovery ====================

_PLUGINS_DISCOVERED = False
_DISCOVERY_CACHE = set()


def discover_plugins(
    plugins_dir: PathLike = "plugins",
    recursive: bool = True,
    force: bool = False
) -> Dict[str, List[str]]:
    """
    Discover and import plugins from directory.
    
    Scans the plugins directory for Python files and imports them.
    Registration happens automatically via decorators.
    
    Args:
        plugins_dir: Directory to scan for plugins
        recursive: If True, scan subdirectories
        force: If True, re-discover even if already done
        
    Returns:
        Dictionary mapping registry names to newly discovered items
        
    Example:
        >>> discovered = discover_plugins("plugins")
        >>> print(f"Found {len(discovered['models'])} new models")
    """
    global _PLUGINS_DISCOVERED, _DISCOVERY_CACHE
    
    if _PLUGINS_DISCOVERED and not force:
        return {
            'models': [],
            'trainers': [],
            'pipelines': [],
            'datasets': []
        }
    
    plugins_path = Path(plugins_dir)
    
    if not plugins_path.exists():
        warnings.warn(f"Plugins directory not found: {plugins_path}")
        return {
            'models': [],
            'trainers': [],
            'pipelines': [],
            'datasets': []
        }
    
    # Get counts before discovery
    before_counts = {
        'models': len(_MODEL_REGISTRY),
        'trainers': len(_TRAINER_REGISTRY),
        'pipelines': len(_PIPELINE_REGISTRY),
        'datasets': len(_DATASET_REGISTRY),
    }
    
    # Find all Python files
    if recursive:
        python_files = list(plugins_path.rglob("*.py"))
    else:
        python_files = list(plugins_path.glob("*.py"))
    
    # Filter out __init__.py and already discovered
    python_files = [
        f for f in python_files
        if f.name != '__init__.py' and str(f) not in _DISCOVERY_CACHE
    ]
    
    # Import each file
    for py_file in python_files:
        _import_plugin_file(py_file, plugins_path)
        _DISCOVERY_CACHE.add(str(py_file))
    
    # Mark as discovered
    _PLUGINS_DISCOVERED = True
    
    # Get counts after discovery
    after_counts = {
        'models': len(_MODEL_REGISTRY),
        'trainers': len(_TRAINER_REGISTRY),
        'pipelines': len(_PIPELINE_REGISTRY),
        'datasets': len(_DATASET_REGISTRY),
    }
    
    # Calculate newly discovered
    newly_discovered = {
        key: _get_new_items(key, before_counts[key])
        for key in before_counts.keys()
    }
    
    return newly_discovered


def _import_plugin_file(file_path: Path, base_path: Path):
    """
    Import a plugin file.
    
    Args:
        file_path: Path to Python file
        base_path: Base plugins directory
    """
    import importlib
    import sys
    import warnings
    
    try:
        # Convert to absolute path first
        abs_file_path = file_path.resolve()
        abs_cwd = Path.cwd().resolve()
        
        # Calculate module name from path relative to CWD (project root)
        relative_path = abs_file_path.relative_to(abs_cwd)
        module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
        module_name = '.'.join(module_parts)
        
        # Import module
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        
    except Exception as e:
        warnings.warn(
            f"Failed to import plugin {file_path}: {e}",
            ImportWarning
        )


def _get_new_items(registry_name: str, old_count: int) -> List[str]:
    """Get newly registered items in a registry"""
    registry_map = {
        'models': _MODEL_REGISTRY,
        'trainers': _TRAINER_REGISTRY,
        'pipelines': _PIPELINE_REGISTRY,
        'datasets': _DATASET_REGISTRY,
    }
    
    registry = registry_map[registry_name]
    all_items = registry.list()
    
    if len(all_items) > old_count:
        # Return the new ones (assumes they're at the end when sorted)
        return all_items[old_count:]
    return []


def reset_discovery():
    """
    Reset plugin discovery state.
    
    Useful for testing or when plugins directory changes.
    """
    global _PLUGINS_DISCOVERED, _DISCOVERY_CACHE
    _PLUGINS_DISCOVERED = False
    _DISCOVERY_CACHE.clear()


def auto_discover():
    """
    Automatically discover plugins if not already done.
    
    This is called automatically when getting/listing plugins
    if auto-discovery is enabled.
    """
    discover_plugins()

def _auto_discover_plugins():
    """Internal helper to discover plugins"""
    discover_plugins()

def get_model_metadata(name: str) -> Dict[str, Any]:
    """Get metadata for a registered model"""
    return _MODEL_REGISTRY.get_metadata(name)


def get_trainer_metadata(name: str) -> Dict[str, Any]:
    """Get metadata for a registered trainer"""
    return _TRAINER_REGISTRY.get_metadata(name)


def search_models(query: str) -> List[str]:
    """
    Search for models by name.
    
    Args:
        query: Search query (case-insensitive substring match)
        
    Returns:
        List of matching model names
    """
    auto_discover()
    all_models = _MODEL_REGISTRY.list()
    query_lower = query.lower()
    return [name for name in all_models if query_lower in name.lower()]


def search_trainers(query: str) -> List[str]:
    """Search for trainers by name"""
    auto_discover()
    all_trainers = _TRAINER_REGISTRY.list()
    query_lower = query.lower()
    return [name for name in all_trainers if query_lower in name.lower()]


def print_registry_summary():
    """
    Print a summary of all registries.
    
    Useful for debugging and seeing what's available.
    """
    auto_discover()
    
    print("\nFrameworm Registry Summary")
    print("=" * 60)
    
    print(f"\nModels ({len(_MODEL_REGISTRY)}):")
    for name in list_models(auto_discover=False):
        metadata = _MODEL_REGISTRY.get_metadata(name)
        print(f"  - {name} ({metadata['module']})")
    
    print(f"\nTrainers ({len(_TRAINER_REGISTRY)}):")
    for name in list_trainers(auto_discover=False):
        metadata = _TRAINER_REGISTRY.get_metadata(name)
        print(f"  - {name} ({metadata['module']})")
    
    print(f"\nPipelines ({len(_PIPELINE_REGISTRY)}):")
    for name in list_pipelines(auto_discover=False):
        metadata = _PIPELINE_REGISTRY.get_metadata(name)
        print(f"  - {name} ({metadata['module']})")
    
    print(f"\nDatasets ({len(_DATASET_REGISTRY)}):")
    for name in list_datasets(auto_discover=False):
        metadata = _DATASET_REGISTRY.get_metadata(name)
        print(f"  - {name} ({metadata['module']})")
    
    print("=" * 60 + "\n")


def create_model_from_config(config) -> Any:
    """
    Create a model instance from config.
    
    The config must have a 'type' field specifying the model name.
    
    Args:
        config: Config object with 'type' field
        
    Returns:
        Model instance
        
    Example:
        >>> config = Config.from_template('gan')
        >>> config.model.type = 'my-registered-model'
        >>> model = create_model_from_config(config)
    """
    if not hasattr(config, 'model') or not hasattr(config.model, 'type'):
        raise ValueError(
            "Config must have 'model.type' field specifying model name"
        )
    
    model_name = config.model.type
    model_class = get_model(model_name)
    return model_class(config)