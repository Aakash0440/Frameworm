"""Tests for frameworm registry system"""

import pytest
from frameworm.core.registry import (
    Registry, ModelRegistry, register_model, get_model,
    list_models, has_model
)
from frameworm.models import BaseModel


class TestRegistryCore:
    """Tests for Registry, ModelRegistry, and decorators"""

    def test_basic_registry(self):
        """Basic registration, retrieval, and presence checks"""
        registry = Registry("test")

        class TestClass:
            pass

        registry.register("test-item", TestClass)
        assert registry.has("test-item")
        assert registry.get("test-item") == TestClass

        # List and length
        items = registry.list()
        assert "test-item" in items
        assert len(registry) == 1

    def test_model_registration_decorator(self):
        """Test @register_model decorator and retrieval"""
        @register_model("test-model")
        class TestModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def forward(self, x):
                return x

        # Registry checks
        assert has_model("test-model")
        model_class = get_model("test-model")
        assert model_class == TestModel

    def test_duplicate_model_registration(self):
        """Registering same model name should raise ValueError"""
        @register_model("dup-model")
        class FirstModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)
            def forward(self, x):
                return x

        with pytest.raises(ValueError) as e:
            @register_model("dup-model")
            class AnotherModel(BaseModel):
                def __init__(self, config):
                    super().__init__(config)
                def forward(self, x):
                    return x
        assert "already registered" in str(e.value)

    def test_model_validation(self):
        """Non-BaseModel or missing forward() should raise TypeError"""
        with pytest.raises(TypeError) as e:
            @register_model("invalid-model")
            class InvalidModel:  # Missing forward
                pass
        assert "must implement forward()" in str(e.value)

    def test_list_models(self):
        """list_models should include registered models"""
        models = list_models()
        assert "test-model" in models
