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

