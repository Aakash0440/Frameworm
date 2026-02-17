import pytest

from core import Config, get_model, register_model
from core.exceptions import ConfigNotFoundError, ModelNotFoundError, PluginValidationError
from models import BaseModel


def test_config_not_found():
    """Test ConfigNotFoundError is raised properly"""
    with pytest.raises(ConfigNotFoundError) as excinfo:
        cfg = Config("nonexistent_config.yaml")
    assert "Suggested Fixes" in str(excinfo.value)


def test_model_not_found():
    """Test ModelNotFoundError is raised properly"""
    with pytest.raises(ModelNotFoundError) as excinfo:
        model_class = get_model("nonexistent-model")
    assert "Available models" in str(excinfo.value)


def test_plugin_validation_error():
    """Test PluginValidationError for invalid model registration"""
    with pytest.raises(PluginValidationError) as excinfo:

        @register_model("invalid-test-model")
        class InvalidModel(BaseModel):
            def __init__(self, config):
                super().__init__(config)

            # Missing forward() method!

    assert "Missing methods" in str(excinfo.value)
    assert "forward" in str(excinfo.value)
