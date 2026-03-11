"""Tests for frameworm plugin discovery system"""

import os
import shutil
from pathlib import Path

import pytest

from core.registry import (
    _MODEL_REGISTRY,
    add_plugin_path,
    discover_plugins,
    get_model,
    list_models,
    remove_plugin_path,
    reset_discovery,
    set_auto_discover,
)

# Plugin directory inside tests/unit
TEST_PLUGIN_DIR = Path(__file__).parent / "test_plugin_models"


@pytest.fixture(scope="session", autouse=True)
def setup_test_plugins():
    """Set up a fake plugin directory with a model - runs once for all tests"""
    # Create directory
    os.makedirs(TEST_PLUGIN_DIR, exist_ok=True)

    # Create __init__.py
    init_file = TEST_PLUGIN_DIR / "__init__.py"
    init_file.write_text("")

    # Create the plugin model file
    plugin_file = TEST_PLUGIN_DIR / "test_model.py"
    plugin_file.write_text("""
from models.base import BaseModel
from core.registry import register_model

@register_model("test-plugin-model")
class TestPluginModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, x):
        return x
""")

    print(f"\n[FIXTURE] Created test plugin directory: {TEST_PLUGIN_DIR}")
    print(f"[FIXTURE] Plugin file exists: {plugin_file.exists()}")

    # Register the test plugin directory so auto-discovery finds it
    add_plugin_path(TEST_PLUGIN_DIR)

    yield

    # Cleanup
    remove_plugin_path(TEST_PLUGIN_DIR)
    if TEST_PLUGIN_DIR.exists():
        shutil.rmtree(TEST_PLUGIN_DIR, ignore_errors=True)
        print(f"\n[FIXTURE] Cleaned up test plugin directory")


def test_discover_plugins():
    """Test that plugin models are discovered"""
    print(f"\n[TEST] Looking for plugins in: {TEST_PLUGIN_DIR}")
    print(f"[TEST] Directory exists: {TEST_PLUGIN_DIR.exists()}")
    print(f"[TEST] Files in directory: {list(TEST_PLUGIN_DIR.glob('*'))}")

    reset_discovery()
    discovered = discover_plugins(str(TEST_PLUGIN_DIR))

    print(f"[TEST] Discovery result: {discovered}")

    assert discovered, "No plugins discovered"

    models = list_models(auto_discover=False)
    print(f"[TEST] Models in registry: {models}")

    assert "test-plugin-model" in models, f"Expected 'test-plugin-model' in {models}"

    model_class = get_model("test-plugin-model", auto_discover=False)
    assert model_class.__name__ == "TestPluginModel"

    print("[TEST] ✓ Plugin discovery test passed!")


def test_auto_discover():
    """Test auto-discovery on first access"""
    print(f"\n[TEST] Auto-discover test starting...")

    _MODEL_REGISTRY.clear()
    reset_discovery()
    set_auto_discover(True)

    models = list_models()
    print(f"[TEST] Models after auto-discover: {models}")

    assert "test-plugin-model" in models, f"Expected 'test-plugin-model' in {models}"

    print("[TEST] ✓ Auto-discover test passed!")
