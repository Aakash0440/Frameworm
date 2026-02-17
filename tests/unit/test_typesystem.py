"""Tests for frameworm type system and validations"""

import tempfile
from pathlib import Path

import pytest
import torch

from core import Config
from core.types import *
from models import BaseModel


class TestTypeSystem:
    """Test type guards, validation, and tensor checks"""

    def test_type_guards(self):
        """Check is_model and is_config guards"""

        class TestModel(BaseModel):
            def __init__(self, cfg):
                super().__init__(cfg)

            def forward(self, x):
                return x

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value")
            cfg_path = f.name

        cfg = Config(cfg_path)
        model = TestModel(cfg)

        assert is_model(model)
        assert is_config(cfg)
        assert not is_model(cfg)

        # Clean up
        Path(cfg_path).unlink()

    def test_device_validation(self):
        """Check validate_device returns torch.device"""
        device = validate_device("cpu")
        assert isinstance(device, torch.device)

    def test_path_validation(self):
        """Check validate_path returns a Path object"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value")
            cfg_path = f.name

        path = validate_path(cfg_path, must_exist=True)
        assert isinstance(path, Path)

        # Clean up
        Path(cfg_path).unlink()

    def test_config_validation(self):
        """Check validate_config returns a dict"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value")
            cfg_path = f.name

        cfg = Config(cfg_path)
        config_dict = validate_config(cfg)
        assert isinstance(config_dict, dict)

        # Clean up
        Path(cfg_path).unlink()

    def test_tensor_like_check(self):
        """Check is_tensor_like works for torch tensors"""
        tensor = torch.randn(3, 3)
        assert is_tensor_like(tensor)
