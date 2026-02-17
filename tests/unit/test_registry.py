"""Comprehensive tests for plugin registry system"""

import pytest
from pathlib import Path
import tempfile
import shutil
from core.registry import (
    Registry,
    ModelRegistry,
    TrainerRegistry,
    PipelineRegistry,
    DatasetRegistry,
    register_model,
    register_trainer,
    register_pipeline,
    register_dataset,
    get_model,
    get_trainer,
    list_models,
    list_trainers,
    has_model,
    has_trainer,
    discover_plugins,
    reset_discovery,
    set_auto_discover,
    search_models,
    print_registry_summary,
    _MODEL_REGISTRY,
    _TRAINER_REGISTRY,
)
from core.exceptions import PluginValidationError  # FIX: import for validation tests
from models.base import BaseModel
from trainers.base import BaseTrainer
from pipelines.base import BasePipeline


class TestRegistry:
    """Test base Registry class"""

    def setup_method(self):
        self.registry = Registry("test")

    def test_registration(self):
        class TestClass:
            pass

        self.registry.register("test-item", TestClass)
        assert self.registry.has("test-item")
        assert self.registry.get("test-item") == TestClass

    def test_duplicate_registration(self):
        class TestClass1:
            pass

        class TestClass2:
            pass

        self.registry.register("test-item", TestClass1)
        with pytest.raises(ValueError, match="already registered"):
            self.registry.register("test-item", TestClass2)

    def test_override_registration(self):
        class TestClass1:
            pass

        class TestClass2:
            pass

        self.registry.register("test-item", TestClass1)
        self.registry.register("test-item", TestClass2, override=True)
        assert self.registry.get("test-item") == TestClass2

    def test_get_nonexistent(self):
        with pytest.raises(KeyError, match="not found"):
            self.registry.get("nonexistent")

    def test_list(self):
        class TestClass1:
            pass

        class TestClass2:
            pass

        self.registry.register("item1", TestClass1)
        self.registry.register("item2", TestClass2)
        items = self.registry.list()
        assert "item1" in items
        assert "item2" in items
        assert len(items) == 2

    def test_remove(self):
        class TestClass:
            pass

        self.registry.register("test-item", TestClass)
        assert self.registry.has("test-item")
        self.registry.remove("test-item")
        assert not self.registry.has("test-item")

    def test_clear(self):
        class TestClass:
            pass

        self.registry.register("item1", TestClass)
        self.registry.register("item2", TestClass)
        assert len(self.registry) == 2
        self.registry.clear()
        assert len(self.registry) == 0

    def test_metadata(self):
        class TestClass:
            pass

        self.registry.register("test-item", TestClass, version="1.0", author="test")
        metadata = self.registry.get_metadata("test-item")
        assert metadata["version"] == "1.0"
        assert metadata["author"] == "test"


class TestModelRegistry:
    """Test ModelRegistry validation"""

    def setup_method(self):
        _MODEL_REGISTRY.clear()
        reset_discovery()

    def test_valid_model_registration(self):
        @register_model("test-model")
        class TestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x):
                return x

        assert has_model("test-model")
        assert get_model("test-model") == TestModel

    def test_invalid_model_no_forward(self):
        """Should reject model without forward()"""
        # FIX: registry now raises PluginValidationError, not TypeError
        with pytest.raises(PluginValidationError, match="must implement forward"):

            @register_model("invalid-model")
            class InvalidModel(BaseModel):
                def __init__(self, config):
                    super().__init__(config)

                # Missing forward()

    def test_model_metadata(self):
        @register_model("test-model", version="2.0", author="tester")
        class TestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x):
                return x

        from core.registry import get_model_metadata

        metadata = get_model_metadata("test-model")
        assert metadata["version"] == "2.0"
        assert metadata["author"] == "tester"


class TestTrainerRegistry:
    """Test TrainerRegistry validation"""

    def setup_method(self):
        _TRAINER_REGISTRY.clear()
        reset_discovery()

    def test_valid_trainer_registration(self):
        @register_trainer("test-trainer")
        class TestTrainer(BaseTrainer):
            def __init__(self, model, config):
                super().__init__(model, config)

            def training_step(self, batch, idx):
                return {"loss": 0.0}

            def validation_step(self, batch, idx):
                return {"loss": 0.0}

        assert has_trainer("test-trainer")
        from core import get_trainer

        assert get_trainer("test-trainer") == TestTrainer

    def test_invalid_trainer_no_training_step(self):
        """Should reject trainer without training_step()"""
        # FIX: registry now raises PluginValidationError, not TypeError
        with pytest.raises(PluginValidationError, match="must implement training_step"):

            @register_trainer("invalid-trainer")
            class InvalidTrainer(BaseTrainer):
                def __init__(self, model, config):
                    super().__init__(model, config)

                def validation_step(self, batch, idx):
                    return {}

                # Missing training_step()


class TestPluginDiscovery:
    """Test plugin discovery system"""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.plugins_dir = self.test_dir / "test_plugins"
        self.plugins_dir.mkdir()
        (self.plugins_dir / "__init__.py").write_text("")
        _MODEL_REGISTRY.clear()
        reset_discovery()
        set_auto_discover(False)

    def teardown_method(self):
        shutil.rmtree(self.test_dir)
        reset_discovery()

    def test_discover_single_plugin(self):
        plugin_code = """
from models.base import BaseModel
from core.registry import register_model

@register_model("discovered-model")
class DiscoveredModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        return x
"""
        (self.plugins_dir / "test_model.py").write_text(plugin_code)
        discovered = discover_plugins(self.plugins_dir)
        assert len(discovered["models"]) > 0
        assert has_model("discovered-model")

    def test_discover_multiple_plugins(self):
        for i in range(3):
            plugin_code = f"""
from models.base import BaseModel
from core.registry import register_model

@register_model("model-{i}")
class Model{i}(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        return x
"""
            (self.plugins_dir / f"model_{i}.py").write_text(plugin_code)
        discover_plugins(self.plugins_dir)
        assert has_model("model-0")
        assert has_model("model-1")
        assert has_model("model-2")

    def test_discover_recursive(self):
        subdir = self.plugins_dir / "subdir"
        subdir.mkdir()
        (subdir / "__init__.py").write_text("")
        plugin_code = """
from models.base import BaseModel
from core.registry import register_model

@register_model("nested-model")
class NestedModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        return x
"""
        (subdir / "nested.py").write_text(plugin_code)
        discover_plugins(self.plugins_dir, recursive=True)
        assert has_model("nested-model")

    def test_import_error_handling(self):
        bad_plugin = """
from nonexistent_module import something
"""
        (self.plugins_dir / "bad_plugin.py").write_text(bad_plugin)
        with pytest.warns(ImportWarning):
            discover_plugins(self.plugins_dir)

    def test_discovery_caching(self):
        plugin_code = """
from models.base import BaseModel
from core.registry import register_model

@register_model("cached-model")
class CachedModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        return x
"""
        (self.plugins_dir / "cached.py").write_text(plugin_code)

        # First discovery
        discovered1 = discover_plugins(self.plugins_dir)
        assert len(discovered1["models"]) > 0

        # Second discovery (should use cache)
        discovered2 = discover_plugins(self.plugins_dir)
        assert len(discovered2["models"]) == 0  # Already discovered

        # Force re-discovery
        reset_discovery()
        discovered3 = discover_plugins(self.plugins_dir, force=True)
        assert len(discovered3["models"]) > 0


class TestAutoDiscovery:
    """Test automatic plugin discovery"""

    def setup_method(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.plugins_dir = self.test_dir / "plugins"
        self.plugins_dir.mkdir()
        (self.plugins_dir / "__init__.py").write_text("")
        plugin_code = """
from models.base import BaseModel
from core.registry import register_model

@register_model("auto-discovered-model")
class AutoModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
    def forward(self, x):
        return x
"""
        (self.plugins_dir / "auto.py").write_text(plugin_code)
        _MODEL_REGISTRY.clear()
        reset_discovery()
        set_auto_discover(True)

    def teardown_method(self):
        shutil.rmtree(self.test_dir)
        reset_discovery()
        set_auto_discover(False)

    def test_auto_discover_on_get(self):
        discover_plugins(self.plugins_dir)
        model_class = get_model("auto-discovered-model", auto_discover=False)
        assert model_class is not None

    def test_auto_discover_on_list(self):
        discover_plugins(self.plugins_dir)
        models = list_models(auto_discover=False)
        assert "auto-discovered-model" in models


class TestSearchAndMetadata:
    """Test search and metadata features"""

    def setup_method(self):
        _MODEL_REGISTRY.clear()
        reset_discovery()

        @register_model("gan-basic")
        class GANBasic(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x):
                return x

        @register_model("gan-advanced")
        class GANAdvanced(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x):
                return x

        @register_model("diffusion-ddpm", version="1.0", author="test")
        class DiffusionDDPM(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            def forward(self, x):
                return x

    def test_search_models(self):
        results = search_models("gan")
        assert len(results) == 2
        assert "gan-basic" in results
        assert "gan-advanced" in results

    def test_search_case_insensitive(self):
        results = search_models("GAN")
        assert len(results) == 2

    def test_search_partial_match(self):
        results = search_models("diff")
        assert "diffusion-ddpm" in results

    def test_metadata_retrieval(self):
        from core import get_model_metadata

        metadata = get_model_metadata("diffusion-ddpm")
        assert metadata["version"] == "1.0"
        assert metadata["author"] == "test"


class TestEdgeCases:
    """Test edge cases and error conditions"""

    def setup_method(self):
        _MODEL_REGISTRY.clear()
        reset_discovery()

    def test_register_non_class(self):
        registry = Registry("test")
        with pytest.raises(TypeError):
            registry.register("not-a-class", "string value")

    def test_empty_registry(self):
        assert len(list_models(auto_discover=False)) == 0

    def test_registry_summary_empty(self):
        print_registry_summary()


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=frameworm.core.registry"])
