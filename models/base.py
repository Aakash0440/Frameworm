
"""Base classes for all models"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn
from core import Config


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all models.
    
    All models in Frameworm should inherit from this class and implement
    the required abstract methods.
    
    Args:
        config: Configuration object or dict
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self._is_built = False
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """
        Forward pass of the model.
        
        Must be implemented by subclasses.
        """
        pass
    
    def build(self):
        """
        Build the model architecture.
        
        Override this to lazily construct model layers.
        """
        self._is_built = True
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Configuration dictionary
        """
        return self.config.to_dict() if hasattr(self.config, 'to_dict') else dict(self.config)
    
    def summary(self):
        """
        Print model summary.
        """
        print(f"\n{self.__class__.__name__} Summary")
        print("=" * 60)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Non-trainable parameters: {total_params - trainable_params:,}")
        print("=" * 60)
        print()
    
    def save(self, path: str):
        """
        Save model weights.
        
        Args:
            path: Path to save weights
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': self.get_config(),
        }, path)
    
    def load(self, path: str):
        """
        Load model weights.
        
        Args:
            path: Path to load weights from
        """
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
    
    def __repr__(self) -> str:
        """String representation"""
        num_params = sum(p.numel() for p in self.parameters())
        return f"{self.__class__.__name__}(params={num_params:,})"