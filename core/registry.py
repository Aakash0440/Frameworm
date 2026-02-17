"""
Comprehensive plugin registry system for models, trainers, pipelines, datasets.
"""

import warnings
from collections import OrderedDict


# ------------------- Base Registry -------------------
class Registry:
    def __init__(self, name):
        self._name = name
        self._items = OrderedDict()
        self._metadata = {}

    def register(self, key, cls, override=False, **metadata):
        if not isinstance(cls, type):
            raise TypeError(f"{key} must be a class")
        if key in self._items and not override:
            raise ValueError(f"{key} already registered in {self._name}")
        self._items[key] = cls
        self._metadata[key] = metadata

    def get(self, key):
        if key not in self._items:
            raise KeyError(f"{key} not found in {self._name}")
        return self._items[key]

    def remove(self, key):
        self._items.pop(key, None)
        self._metadata.pop(key, None)

    def clear(self):
        self._items.clear()
        self._metadata.clear()

    def has(self, key):
        return key in self._items

    def list(self):
        return list(self._items.keys())

    def get_metadata(self, key):
        return self._metadata.get(key, {})

    def __len__(self):
        return len(self._items)


# ------------------- Specific Registries -------------------


class ModelRegistry(Registry):
    def register(self, key, cls, override=False, **metadata):
        # Use cls.__dict__ (not hasattr) so inherited methods don't satisfy
        # the check — the subclass must explicitly define forward().
        if "forward" not in cls.__dict__ or not callable(cls.__dict__["forward"]):
            from core.exceptions import PluginValidationError

            raise PluginValidationError(
                f"{cls.__name__} must implement forward",
                plugin_name=cls.__name__,
                missing_methods=["forward"],
            )
        super().register(key, cls, override, **metadata)

    def get(self, key):
        if key not in self._items:
            from core.exceptions import ModelNotFoundError

            raise ModelNotFoundError(key, available=self.list())
        return self._items[key]


class TrainerRegistry(Registry):
    def register(self, key, cls, override=False, **metadata):
        # Use cls.__dict__ so inherited training_step() doesn't count.
        if "training_step" not in cls.__dict__ or not callable(cls.__dict__["training_step"]):
            from core.exceptions import PluginValidationError

            raise PluginValidationError(
                f"{cls.__name__} must implement training_step",
                plugin_name=cls.__name__,
                missing_methods=["training_step"],
            )
        super().register(key, cls, override, **metadata)


class PipelineRegistry(Registry):
    pass


class DatasetRegistry(Registry):
    pass


# Instantiate subclass registries so validation actually runs
_MODEL_REGISTRY = ModelRegistry("models")
_TRAINER_REGISTRY = TrainerRegistry("trainers")
_PIPELINE_REGISTRY = PipelineRegistry("pipelines")
_DATASET_REGISTRY = DatasetRegistry("datasets")


# ------------------- Decorators -------------------
def register_model(key, **metadata):
    def decorator(cls):
        _MODEL_REGISTRY.register(key, cls, **metadata)
        return cls

    return decorator


def register_trainer(key, **metadata):
    def decorator(cls):
        _TRAINER_REGISTRY.register(key, cls, **metadata)
        return cls

    return decorator


def register_pipeline(key, **metadata):
    def decorator(cls):
        _PIPELINE_REGISTRY.register(key, cls, **metadata)
        return cls

    return decorator


def register_dataset(key, **metadata):
    def decorator(cls):
        _DATASET_REGISTRY.register(key, cls, **metadata)
        return cls

    return decorator


# ------------------- Accessors: Models -------------------
def get_model(key, auto_discover=True):
    if auto_discover and _auto_discover:
        discover_plugins()
    return _MODEL_REGISTRY.get(key)


def list_models(auto_discover=True):
    if auto_discover and _auto_discover:
        discover_plugins()
    return _MODEL_REGISTRY.list()


def has_model(key):
    return _MODEL_REGISTRY.has(key)


def get_model_metadata(key):
    return _MODEL_REGISTRY.get_metadata(key)


# ------------------- Accessors: Trainers -------------------
def get_trainer(key):
    return _TRAINER_REGISTRY.get(key)


def list_trainers():
    return _TRAINER_REGISTRY.list()


def has_trainer(key):
    return _TRAINER_REGISTRY.has(key)


# ------------------- Accessors: Pipelines -------------------
def get_pipeline(key):
    return _PIPELINE_REGISTRY.get(key)


def list_pipelines():
    return _PIPELINE_REGISTRY.list()


def has_pipeline(key):
    return _PIPELINE_REGISTRY.has(key)


# ------------------- Accessors: Datasets -------------------
def get_dataset(key):
    return _DATASET_REGISTRY.get(key)


def list_datasets():
    return _DATASET_REGISTRY.list()


def has_dataset(key):
    return _DATASET_REGISTRY.has(key)


# ------------------- Search & Summary -------------------
def search_models(query: str):
    query_lower = query.lower()
    return [k for k in _MODEL_REGISTRY.list() if query_lower in k.lower()]


def print_registry_summary():
    print("Models:", list_models(auto_discover=False))
    print("Trainers:", list_trainers())
    print("Pipelines:", list_pipelines())
    print("Datasets:", list_datasets())


# ------------------- Plugin Discovery -------------------
_auto_discover = True
_discovered_paths = set()
_extra_plugin_paths: list = []


def set_auto_discover(value: bool):
    global _auto_discover
    _auto_discover = value


def reset_discovery():
    global _discovered_paths
    _discovered_paths.clear()


def add_plugin_path(path) -> None:
    """Register an additional directory to scan during auto-discovery."""
    from pathlib import Path

    resolved = Path(path).resolve()
    if resolved not in _extra_plugin_paths:
        _extra_plugin_paths.append(resolved)


def remove_plugin_path(path) -> None:
    """Remove a previously registered extra plugin path."""
    from pathlib import Path

    resolved = Path(path).resolve()
    if resolved in _extra_plugin_paths:
        _extra_plugin_paths.remove(resolved)


def clear_plugin_paths() -> None:
    """Remove all extra plugin paths."""
    _extra_plugin_paths.clear()


def _discover_one(path, recursive: bool = False, force: bool = False) -> dict:
    """
    Internal: discover plugins in a single directory.
    Returns dict of newly registered keys per category.
    """
    import importlib.util
    import sys
    import time
    from pathlib import Path

    path = Path(path).resolve()

    if force:
        _discovered_paths.discard(path)

    if path in _discovered_paths:
        return {"models": [], "trainers": [], "pipelines": [], "datasets": []}

    if not path.exists():
        return {"models": [], "trainers": [], "pipelines": [], "datasets": []}

    # Ensure project root is on sys.path so plugin imports resolve correctly.
    cwd = str(Path.cwd())
    inserted = False
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
        inserted = True

    py_files = list(path.rglob("*.py") if recursive else path.glob("*.py"))

    # Snapshot registry keys BEFORE loading so we can diff afterwards.
    # On force re-discovery the models are already registered, so we use an
    # EMPTY baseline — everything in the registry after loading is "discovered".
    # On normal discovery we diff against what was there before.
    if force:
        before = {
            "models": set(),
            "trainers": set(),
            "pipelines": set(),
            "datasets": set(),
        }
    else:
        before = {
            "models": set(_MODEL_REGISTRY.list()),
            "trainers": set(_TRAINER_REGISTRY.list()),
            "pipelines": set(_PIPELINE_REGISTRY.list()),
            "datasets": set(_DATASET_REGISTRY.list()),
        }

    for f in py_files:
        if f.name == "__init__.py":
            continue
        # On force, use a time-based suffix so Python's module cache never
        # returns a stale module — the file always re-executes fresh.
        if force:
            unique_name = f"_plugin_{f.stem}_{abs(hash(str(f)))}_{time.time_ns()}"
        else:
            unique_name = f"_plugin_{f.stem}_{abs(hash(str(f)))}"
        try:
            spec = importlib.util.spec_from_file_location(unique_name, f)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except ValueError:
            # Duplicate registration on non-force run: already registered,
            # will be captured by the diff below — nothing to do.
            pass
        except Exception as e:
            warnings.warn(f"Failed to import {f}: {e}", ImportWarning)

    if inserted:
        sys.path.remove(cwd)

    _discovered_paths.add(path)

    return {
        "models": [k for k in _MODEL_REGISTRY.list() if k not in before["models"]],
        "trainers": [k for k in _TRAINER_REGISTRY.list() if k not in before["trainers"]],
        "pipelines": [k for k in _PIPELINE_REGISTRY.list() if k not in before["pipelines"]],
        "datasets": [k for k in _DATASET_REGISTRY.list() if k not in before["datasets"]],
    }


def discover_plugins(path=None, recursive=False, force=False):
    """
    Discover plugin Python files and import them.

    If `path` is given, only that directory is scanned.
    If `path` is None (auto-discovery), scans:
      1. <cwd>/plugins
      2. Any paths registered via add_plugin_path()

    Returns a dict with keys 'models', 'trainers', 'pipelines', 'datasets',
    each containing the registry keys newly registered in this call.
    """
    from pathlib import Path

    if path is not None:
        return _discover_one(path, recursive=recursive, force=force)

    combined = {"models": [], "trainers": [], "pipelines": [], "datasets": []}
    for p in [Path.cwd() / "plugins"] + list(_extra_plugin_paths):
        result = _discover_one(p, recursive=recursive, force=force)
        for category in combined:
            combined[category].extend(result[category])

    return combined
