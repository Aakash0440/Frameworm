"""
Type definitions and protocols for Frameworm.

This module provides:
- Type aliases for common types
- Protocols for structural typing
- Type guards and checking utilities
- Generic types for containers
"""

from typing import (
    Any, Dict, List, Union, Optional, Protocol, TypeVar, Generic,
    Callable, Sequence, Mapping, runtime_checkable
)
from pathlib import Path
import torch

# ==================== Basic Type Aliases ====================

PathLike = Union[str, Path]
"""Type for file paths (string or Path object)"""

ConfigDict = Dict[str, Any]
"""Type for configuration dictionaries"""

TensorLike = Union[torch.Tensor, 'np.ndarray']
"""Type for tensor-like objects"""

DeviceType = Union[str, torch.device]
"""Type for device specifications"""

MetricDict = Dict[str, Union[float, int, torch.Tensor]]
"""Type for metrics dictionaries"""

# ==================== Generic Types ====================

T = TypeVar('T')
"""Generic type variable"""

K = TypeVar('K')
"""Generic key type"""

V = TypeVar('V')
"""Generic value type"""

TensorT = TypeVar('TensorT', bound=torch.Tensor)
"""Type variable bound to Tensor"""

# ==================== Protocols (Structural Typing) ====================

@runtime_checkable
class ModelProtocol(Protocol):
    """
    Protocol for model-like objects.
    
    Any object implementing these methods is considered model-like.
    """
    
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass"""
        ...
    
    def parameters(self):
        """Get parameters"""
        ...
    
    def train(self, mode: bool = True):
        """Set training mode"""
        ...
    
    def eval(self):
        """Set evaluation mode"""
        ...


@runtime_checkable
class ConfigProtocol(Protocol):
    """Protocol for config-like objects"""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        ...
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value with default"""
        ...


@runtime_checkable
class TrainerProtocol(Protocol):
    """Protocol for trainer-like objects"""
    
    def fit(self, train_loader, val_loader=None):
        """Training loop"""
        ...
    
    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, torch.Tensor]:
        """Single training step"""
        ...


@runtime_checkable
class PipelineProtocol(Protocol):
    """Protocol for pipeline-like objects"""
    
    def run(self, *args, **kwargs) -> Any:
        """Execute pipeline"""
        ...
    
    def setup(self):
        """Setup resources"""
        ...
    
    def teardown(self):
        """Cleanup resources"""
        ...


# ==================== Type Guards ====================

def is_model(obj: Any) -> bool:
    """
    Check if object implements ModelProtocol.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is model-like
    """
    return isinstance(obj, ModelProtocol)


def is_config(obj: Any) -> bool:
    """
    Check if object implements ConfigProtocol.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is config-like
    """
    return isinstance(obj, ConfigProtocol)


def is_trainer(obj: Any) -> bool:
    """
    Check if object implements TrainerProtocol.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is trainer-like
    """
    return isinstance(obj, TrainerProtocol)


def is_pipeline(obj: Any) -> bool:
    """
    Check if object implements PipelineProtocol.
    
    Args:
        obj: Object to check
        
    Returns:
        True if object is pipeline-like
    """
    return isinstance(obj, PipelineProtocol)


def is_tensor_like(obj: Any) -> bool:
    """
    Check if object is tensor-like.
    
    Args:
        obj: Object to check
        
    Returns:
        True if tensor-like
    """
    return isinstance(obj, torch.Tensor) or (
        hasattr(obj, 'shape') and hasattr(obj, 'dtype')
    )


# ==================== Generic Containers ====================

class TypedDict(Generic[K, V]):
    """
    Generic typed dictionary.
    
    Usage:
        >>> metrics: TypedDict[str, float] = TypedDict()
    """
    
    def __init__(self):
        self._data: Dict[K, V] = {}
    
    def __setitem__(self, key: K, value: V):
        self._data[key] = value
    
    def __getitem__(self, key: K) -> V:
        return self._data[key]
    
    def __contains__(self, key: K) -> bool:
        return key in self._data
    
    def keys(self):
        return self._data.keys()
    
    def values(self):
        return self._data.values()
    
    def items(self):
        return self._data.items()


# ==================== Callback Types ====================

StepCallback = Callable[[int], None]
"""Callback called at each step: callback(step)"""

EpochCallback = Callable[[int], None]
"""Callback called at each epoch: callback(epoch)"""

MetricCallback = Callable[[Dict[str, float]], None]
"""Callback called with metrics: callback(metrics)"""

ProgressCallback = Callable[[str, int, int], None]
"""Progress callback: callback(step_name, current, total)"""


# ==================== Validation Helpers ====================

def validate_device(device: DeviceType) -> torch.device:
    """
    Validate and convert device specification.
    
    Args:
        device: Device spec (string or torch.device)
        
    Returns:
        torch.device object
        
    Raises:
        ValueError: If device is invalid
    """
    if isinstance(device, torch.device):
        return device
    
    if isinstance(device, str):
        if device in ['cpu', 'cuda', 'mps']:
            return torch.device(device)
        elif device.startswith('cuda:'):
            return torch.device(device)
        else:
            raise ValueError(f"Invalid device: {device}")
    
    raise TypeError(f"Device must be str or torch.device, got {type(device)}")


def validate_path(path: PathLike, must_exist: bool = False) -> Path:
    """
    Validate and convert path.
    
    Args:
        path: Path specification
        must_exist: If True, raise error if path doesn't exist
        
    Returns:
        Path object
        
    Raises:
        FileNotFoundError: If must_exist and path doesn't exist
    """
    path = Path(path)
    
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    
    return path


def validate_config(config: Any) -> ConfigDict:
    """
    Validate and convert config to dictionary.
    
    Args:
        config: Config object or dict
        
    Returns:
        Configuration dictionary
        
    Raises:
        TypeError: If config cannot be converted
    """
    if isinstance(config, dict):
        return config
    
    if hasattr(config, 'to_dict'):
        return config.to_dict()
    
    if is_config(config):
        return config.to_dict()
    
    raise TypeError(f"Cannot convert {type(config)} to config dict")


# ==================== Type Hints for Common Patterns ====================

# Model factory type
ModelFactory = Callable[[ConfigDict], ModelProtocol]

# Data loader type
DataLoader = Any  # torch.utils.data.DataLoader

# Optimizer type
Optimizer = Any  # torch.optim.Optimizer

# Scheduler type
Scheduler = Any  # torch.optim.lr_scheduler._LRScheduler


# ==================== Export __all__ ====================

__all__ = [
    # Basic types
    'PathLike',
    'ConfigDict',
    'TensorLike',
    'DeviceType',
    'MetricDict',
    
    # Generic types
    'T', 'K', 'V', 'TensorT',
    
    # Protocols
    'ModelProtocol',
    'ConfigProtocol',
    'TrainerProtocol',
    'PipelineProtocol',
    
    # Type guards
    'is_model',
    'is_config',
    'is_trainer',
    'is_pipeline',
    'is_tensor_like',
    
    # Generic containers
    'TypedDict',
    
    # Callback types
    'StepCallback',
    'EpochCallback',
    'MetricCallback',
    'ProgressCallback',
    
    # Validation
    'validate_device',
    'validate_path',
    'validate_config',
    
    # Common types
    'ModelFactory',
    'DataLoader',
    'Optimizer',
    'Scheduler',
]