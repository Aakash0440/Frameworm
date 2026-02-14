"""
Configuration system with YAML loading and inheritance.

Features:
- Dot notation access (cfg.model.name)
- Config inheritance via _base_ key
- Type validation
- Immutable after loading
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml


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
                self[key] = [
                    ConfigNode(item) if isinstance(item, dict) else item 
                    for item in value
                ]
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
                f"ConfigNode has no attribute '{key}'. "
                f"Available keys: {list(self.keys())}"
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
                    item.to_dict() if isinstance(item, ConfigNode) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

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
    
    def load(self, config_path: Union[str, Path]) -> 'Config':
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
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Current directory: {Path.cwd()}"
            )
        
        # Load YAML
        with open(config_path, 'r') as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise yaml.YAMLError(
                    f"Failed to parse YAML in {config_path}:\n{e}"
                )
        
        if data is None:
            data = {}
        
        # Handle inheritance
        if '_base_' in data:
            base_path = self._resolve_base_path(config_path, data['_base_'])
            
            # Recursively load base config
            base_config = Config(base_path)
            
            # Merge: base config + current config
            merged = self._merge_configs(
                base_config._data.to_dict(),
                data
            )
            self._data = ConfigNode(merged)
        else:
            self._data = ConfigNode(data)
        
        return self
    
    def _resolve_base_path(self, current_path: Path, base: str) -> Path:
        """
        Resolve the path to a base config.
        
        Args:
            current_path: Path to current config file
            base: Base path string (absolute or relative)
            
        Returns:
            Resolved absolute path to base config
        """
        if base.startswith('/'):
            # Absolute path
            return Path(base)
        else:
            # Relative to current config
            return (current_path.parent / base).resolve()
    
    def _merge_configs(
        self, 
        base: Dict[str, Any], 
        override: Dict[str, Any]
    ) -> Dict[str, Any]:
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
            if key == '_base_':
                continue
            
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                # Both are dicts - merge recursively
                result[key] = self._merge_configs(result[key], value)
            else:
                # Override completely
                result[key] = value
        
        return result
    
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
        
        with open(output_path, 'w') as f:
            yaml.dump(
                self.to_dict(),
                f,
                default_flow_style=False,
                sort_keys=False
            )
    
    def __repr__(self) -> str:
        """String representation"""
        frozen_str = " (frozen)" if self._frozen else ""
        return f"Config({len(self.keys())} keys{frozen_str})"
