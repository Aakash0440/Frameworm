"""Final verification tests for frameworm core functionality"""

import tempfile
from pathlib import Path
import pytest

from core import Config, ConfigNode
from core.types import *
from models import BaseModel
from pipelines import BasePipeline, PipelineStatus
from trainers import BaseTrainer


class TestFinalVerification:
    """Comprehensive Day 2 verification"""

    def test_imports(self):
        """All core imports should succeed"""
        assert Config is not None
        assert ConfigNode is not None
        assert BaseModel is not None
        assert BasePipeline is not None
        assert PipelineStatus is not None
        assert BaseTrainer is not None

    def test_config_system_basic(self):
        """Config system should load YAML and coerce types"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write('value: "123"\nenabled: "true"')
            cfg_path = f.name

        cfg = Config(cfg_path)
        # Assuming your Config system coerces types automatically
        assert cfg.value == 123
        assert cfg.enabled is True

        # Clean up
        Path(cfg_path).unlink()

    def test_type_system(self):
        """Type system checks"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("key: value")
            cfg_path = f.name

        cfg = Config(cfg_path)
        assert is_config(cfg)
        assert not is_model(cfg)  # Example check

        Path(cfg_path).unlink()

    def test_template_system(self):
        """Config.from_template should return a valid Config"""
        cfg = Config.from_template("gan")
        assert isinstance(cfg, Config)
        assert hasattr(cfg, "model") or hasattr(cfg, "training")
