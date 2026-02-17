"""Integration tests for plugin system"""

import pytest
from pathlib import Path
import tempfile
import shutil
from core.config import Config
from core.registry import (
    register_model,
    get_model,
    discover_plugins,
    reset_discovery,
)
from models.base import BaseModel


class TestPluginWorkflow:
    """Test complete plugin workflow"""

    def setup_method(self):
        """Setup test environment"""
        self.test_dir = Path(tempfile.mkdtemp())
        self.plugins_dir = self.test_dir / "plugins"
        self.plugins_dir.mkdir()
        (self.plugins_dir / "__init__.py").write_text("")

        reset_discovery()

    def teardown_method(self):
        """Cleanup"""
        shutil.rmtree(self.test_dir)
        reset_discovery()

    def test_end_to_end_plugin_usage(self):
        """Test complete workflow from plugin to usage"""
        # 1. Create plugin
        plugin_code = """
from models.base import BaseModel
from core.registry import register_model
import torch.nn as nn

@register_model("integration-test-model")
class IntegrationTestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.net = nn.Linear(10, 10)
    
    def forward(self, x):
        return self.net(x)
"""
        (self.plugins_dir / "test_model.py").write_text(plugin_code)

        # 2. Discover plugins
        discovered = discover_plugins(self.plugins_dir)
        assert "integration-test-model" in discovered.get("models", [])

        # 3. Get model class
        model_class = get_model("integration-test-model", auto_discover=False)

        # 4. Create config
        config_file = self.test_dir / "config.yaml"
        config_file.write_text("""
model:
  type: integration-test-model
  latent_dim: 100
training:
  device: cpu
""")

        cfg = Config(config_file)

        # 5. Instantiate model
        model = model_class(cfg)

        # 6. Use model
        import torch

        x = torch.randn(4, 10)
        output = model(x)

        assert output.shape == (4, 10)

    def test_multiple_plugins_interaction(self):
        """Test multiple plugins working together"""
        # Create multiple plugins
        model_plugin = """
from models.base import BaseModel
from core.registry import register_model

@register_model("multi-test-model")
class MultiTestModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        return x
"""
        (self.plugins_dir / "model.py").write_text(model_plugin)

        trainer_plugin = """
from trainers.base import BaseTrainer
from core.registry import register_trainer

@register_trainer("multi-test-trainer")
class MultiTestTrainer(BaseTrainer):
    def training_step(self, batch, idx):
        return {'loss': 0.0}
    def validation_step(self, batch, idx):
        return {'loss': 0.0}
"""
        (self.plugins_dir / "trainer.py").write_text(trainer_plugin)

        # Discover all
        discovered = discover_plugins(self.plugins_dir)

        assert len(discovered["models"]) > 0
        assert len(discovered["trainers"]) > 0


class TestConfigIntegration:
    """Test config integration with plugins"""

    def test_create_model_from_config(self):
        """Test creating model from config"""
        from core import create_model_from_config

        # Register a test model
        @register_model("config-test-model")
        class ConfigTestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x):
                return x

        # Create config
        config_file = Path(tempfile.mktemp(suffix=".yaml"))
        config_file.write_text("""
model:
  type: config-test-model
  latent_dim: 128
""")

        cfg = Config(config_file)

        # Create model from config
        model = create_model_from_config(cfg)

        assert isinstance(model, ConfigTestModel)

        # Cleanup
        config_file.unlink()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
