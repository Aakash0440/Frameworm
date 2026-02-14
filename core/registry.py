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


# ==================== Getter Functions ====================

def get_model(name: str) -> Type:
    """
    Get a registered model class.
    
    Args:
        name: Registered model name
        
    Returns:
        Model class
        
    Example:
        >>> model_class = get_model("my-gan")
        >>> model = model_class(config)
    """
    return _MODEL_REGISTRY.get(name)


def get_trainer(name: str) -> Type:
    """Get a registered trainer class"""
    return _TRAINER_REGISTRY.get(name)


def get_pipeline(name: str) -> Type:
    """Get a registered pipeline class"""
    return _PIPELINE_REGISTRY.get(name)


def get_dataset(name: str) -> Type:
    """Get a registered dataset class"""
    return _DATASET_REGISTRY.get(name)


# ==================== List Functions ====================

def list_models() -> List[str]:
    """List all registered models"""
    return _MODEL_REGISTRY.list()


def list_trainers() -> List[str]:
    """List all registered trainers"""
    return _TRAINER_REGISTRY.list()


def list_pipelines() -> List[str]:
    """List all registered pipelines"""
    return _PIPELINE_REGISTRY.list()


def list_datasets() -> List[str]:
    """List all registered datasets"""
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