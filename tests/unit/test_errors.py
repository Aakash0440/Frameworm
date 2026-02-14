from core import Config, get_model, register_model
from core.exceptions import ConfigNotFoundError, ModelNotFoundError, PluginValidationError
from models import BaseModel

print("Testing Integrated Errors:")
print("="*60)

# Test 1: Config not found
print("\n1. Testing ConfigNotFoundError:")
try:
    cfg = Config("nonexistent_config.yaml")
except ConfigNotFoundError as e:
    print("✓ Caught ConfigNotFoundError")
    # Show a bit of the formatted error
    error_lines = str(e).split('\n')
    print(error_lines[0])  # Just the header
    assert "Suggested Fixes" in str(e), "Error message missing suggested fixes"

# Test 2: Model not found
print("\n2. Testing ModelNotFoundError:")
try:
    model_class = get_model("nonexistent-model")
except ModelNotFoundError as e:
    print("✓ Caught ModelNotFoundError")
    error_lines = str(e).split('\n')
    print(error_lines[0])  # Show first line of the error
    assert "Available models" in str(e), "Error message missing available models list"

# Test 3: Plugin validation error
print("\n3. Testing Plugin Validation:")
try:
    @register_model("invalid-test-model")
    class InvalidModel(BaseModel):
        def __init__(self, config):
            super().__init__(config)
        # Missing forward() method!
except PluginValidationError as e:
    print("✓ Caught PluginValidationError")
    assert "Missing methods" in str(e), "Error message missing missing methods info"
    assert "forward" in str(e), "Missing method name not reported"

print("\n" + "="*60)
print("✅ Error integration working!")
