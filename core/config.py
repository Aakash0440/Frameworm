"""
Configuration system with YAML loading and inheritance.

Features:
- Dot notation access (cfg.model.name)
- Config inheritance via _base_ key
- Type validation
- Immutable after loading
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import yaml
from pydantic import BaseModel, Field, validator

from core.exceptions import ConfigInheritanceError
from core.registry import get_model


class ConfigNode(dict):
    """
    Enhanced dictionary that supports dot notation access.

    Example:
        >>> cfg = ConfigNode({'model': {'name': 'gan', 'dim': 128}})
        >>> print(cfg.model.name)  # 'gan'
        >>> print(cfg.model.dim)   # 128
    """

    def __init__(self, data: Dict[str, Any] = None):
        """Initialize ConfigNode from dictionary"""
        if data is None:
            data = {}
        super().__init__(data)

        # Convert nested dicts to ConfigNodes
        for key, value in data.items():
            if isinstance(value, dict):
                self[key] = ConfigNode(value)
            elif isinstance(value, list):
                # Convert list items that are dicts
                self[key] = [ConfigNode(item) if isinstance(item, dict) else item for item in value]
            else:
                self[key] = value

    def __getattr__(self, key: str) -> Any:
        """
        Enable dot notation access: cfg.model instead of cfg['model']

        Args:
            key: Attribute name

        Returns:
            Value at key

        Raises:
            AttributeError: If key doesn't exist
        """
        try:
            return self[key]
        except KeyError:
            raise AttributeError(
                f"ConfigNode has no attribute '{key}'. " f"Available keys: {list(self.keys())}"
            )

    def __setattr__(self, key: str, value: Any):
        """Enable dot notation assignment: cfg.model = value"""
        self[key] = value

    def __delattr__(self, key: str):
        """Enable dot notation deletion: del cfg.model"""
        try:
            del self[key]
        except KeyError:
            raise AttributeError(f"ConfigNode has no attribute '{key}'")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert ConfigNode back to regular Python dict.

        Returns:
            Regular dictionary with nested ConfigNodes also converted
        """
        result = {}
        for key, value in self.items():
            if isinstance(value, ConfigNode):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, ConfigNode) else item for item in value
                ]
            else:
                result[key] = value
        return result

    def validate(self, schema: Type[BaseModel]) -> BaseModel:
        return schema(**self.to_dict())


class Config:
    """
    Main configuration class with YAML loading and inheritance.

    Features:
    - Load configs from YAML files
    - Support for config inheritance via _base_ key
    - Recursive config merging
    - Validation support
    - Freeze configs after loading

    Example:
        >>> cfg = Config('configs/model.yaml')
        >>> print(cfg.model.name)
        >>> cfg.freeze()
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Config.

        Args:
            config_path: Optional path to YAML config file to load immediately
        """
        self._data = ConfigNode({})
        self._frozen = False
        self._config_path = None

        if config_path:
            self.load(config_path)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Config":
        """
        Create Config from a Python dictionary.
        """
        cfg = cls()
        cfg._data = ConfigNode(data)
        return cfg

    def load(self, config_path: Union[str, Path]) -> "Config":
        """
        Load configuration from YAML file.

        Supports inheritance via _base_ key. If a config has _base_: path/to/base.yaml,
        it will first load the base config, then merge the current config on top.

        Args:
            config_path: Path to YAML config file

        Returns:
            self for chaining

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML is invalid
        """
        if self._frozen:
            raise RuntimeError("Cannot load into frozen config")

        config_path = Path(config_path).resolve()
        self._config_path = config_path

        if not config_path.exists():
            from core.exceptions import ConfigNotFoundError

            raise ConfigNotFoundError(str(config_path))

        # Load YAML
        with open(config_path, "r") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(f"Failed to parse YAML in {config_path}:\n{e}")

        if data is None:
            data = {}

        # Handle inheritance
        if "_base_" in data:
            base_path = self._resolve_base_path(config_path, data["_base_"])

            # Recursively load base config
            base_config = Config(base_path)

            # Merge: base config + current config
            merged = self._merge_configs(base_config._data.to_dict(), data)
            merged = self._interpolate_env_vars(merged)
            self._data = ConfigNode(merged)
        else:
            self._data = ConfigNode(data)

        return self

    def _resolve_base_path(self, current_path: Path, base: str) -> Path:
        """
        Resolve the path to a base config.
        Raises ConfigInheritanceError if base config is missing.
        """
        if base.startswith("/"):
            base_path = Path(base)
        else:
            base_path = (current_path.parent / base).resolve()

        if not base_path.exists():
            from core.exceptions import ConfigInheritanceError

            raise ConfigInheritanceError(f"Base config not found: {base}", base_config=str(base))

        return base_path

    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge two config dictionaries.

        Override values take precedence over base values.
        For nested dicts, merge recursively.
        For other types, override completely replaces base.

        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary

        Returns:
            Merged configuration dictionary
        """
        result = base.copy()

        for key, value in override.items():
            # Skip the _base_ key itself
            if key == "_base_":
                continue

            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Both are dicts - merge recursively
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override completely
                result[key] = value

        return result

    def _interpolate_env_vars(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace ${VAR_NAME} with environment variable values.

        Args:
            data: Configuration dictionary

        Returns:
            Dictionary with env vars interpolated
        """
        import os
        import re

        def interpolate_value(value: Any) -> Any:
            if isinstance(value, str):
                # Find ${VAR_NAME} patterns
                pattern = r"\$\{([^}]+)\}"
                matches = re.findall(pattern, value)

                for var_name in matches:
                    env_value = os.environ.get(var_name)
                    if env_value is None:
                        raise ValueError(f"Environment variable ${{{var_name}}} not set")
                    value = value.replace(f"${{{var_name}}}", env_value)
            elif isinstance(value, dict):
                return {k: interpolate_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [interpolate_value(item) for item in value]

            return value

        return interpolate_value(data)

    def freeze(self):
        """
        Make configuration immutable.

        After freezing, no modifications can be made to the config.
        """
        self._frozen = True

    def is_frozen(self) -> bool:
        """Check if config is frozen"""
        return self._frozen

    def __getattr__(self, key: str) -> Any:
        """Enable dot notation: cfg.model"""
        return getattr(self._data, key)

    def __getitem__(self, key: str) -> Any:
        """Enable dict-style access: cfg['model']"""
        return self._data[key]

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value with default.

        Args:
            key: Key to look up
            default: Default value if key doesn't exist

        Returns:
            Value at key or default
        """
        return self._data.get(key, default)

    def keys(self) -> List[str]:
        """Get all top-level keys"""
        return list(self._data.keys())

    def items(self):
        """Get all top-level items"""
        return self._data.items()

    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration as regular Python dict.

        Returns:
            Dictionary representation of config
        """
        return self._data.to_dict()

    def dump(self, output_path: Union[str, Path]):
        """
        Save merged configuration to YAML file.

        Useful for seeing the final merged config after inheritance.

        Args:
            output_path: Where to save the config
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    def __repr__(self) -> str:
        """String representation"""
        frozen_str = " (frozen)" if self._frozen else ""
        return f"Config({len(self.keys())} keys{frozen_str})"

    @staticmethod
    def from_cli_args(base_config: Union[str, Path], overrides: List[str]) -> "Config":
        """
        Create config from file with CLI overrides.

        Args:
            base_config: Path to base config file
            overrides: List of override strings like 'model.dim=256' or 'training.epochs=500'

        Returns:
            Config with overrides applied

        Example:
            >>> cfg = Config.from_cli_args(
            ...     'config.yaml',
            ...     ['model.dim=256', 'training.epochs=500']
            ... )
        """
        cfg = Config(base_config)

        for override in overrides:
            if "=" not in override:
                raise ValueError(f"Invalid override '{override}'. Expected format: key=value")

            key_path, value = override.split("=", 1)
            keys = key_path.split(".")

            # Try to convert value to appropriate type
            try:
                # Try int
                value = int(value)
            except ValueError:
                try:
                    # Try float
                    value = float(value)
                except ValueError:
                    # Try bool
                    if value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    # Otherwise keep as string

            # Navigate to the right nested dict and set value
            current = cfg._data
            for key in keys[:-1]:
                if key not in current:
                    current[key] = ConfigNode({})
                current = current[key]

            current[keys[-1]] = value

        return cfg


class ConfigSchema(BaseModel):
    """
    Base class for config validation schemas.

    Inherit from this to define validation schemas for your configs.
    """

    class Config:
        # Allow extra fields not in schema
        extra = "allow"
        # Allow arbitrary types
        arbitrary_types_allowed = True

    # Add this method to the Config class
    # Place it after the dump() method

    def validate(self, schema: Type[BaseModel]) -> BaseModel:
        """
        Validate configuration against a Pydantic schema.

        Args:
            schema: Pydantic BaseModel class defining the schema

        Returns:
            Validated config as schema instance

        Raises:
            ValidationError: If config doesn't match schema

        Example:
            >>> class ModelConfig(BaseModel):
            ...     name: str
            ...     dim: int
            >>> cfg.validate(ModelConfig)
        """
        return schema(**self.to_dict())


def create_model_from_config(cfg: "Config"):
    """
    Create a model instance from configuration.

    Expected format:

    model:
        type: model_name
        other_params: ...

    Returns:
        Instantiated model
    """
    if not hasattr(cfg, "model"):
        raise ValueError("Config must contain a 'model' section")

    model_cfg = cfg.model

    if not hasattr(model_cfg, "type"):
        raise ValueError("Config.model must contain a 'type' field")

    model_type = model_cfg.type

    # Get model class from registry
    model_cls = get_model(model_type)

    # IMPORTANT: pass full config object to model
    return model_cls(model_cfg)
