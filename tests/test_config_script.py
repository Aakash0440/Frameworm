from core.config import Config

# Test 1: Load config
cfg = Config("test_configs/test.yaml")
assert cfg.model.name == "test-model"
assert cfg.model.dim == 128
assert cfg.training.epochs == 100
print("✓ Test 1 passed: Basic config loading")

# Test 2: Dict-style access
assert cfg["model"]["name"] == "test-model"
print("✓ Test 2 passed: Dict-style access")

# Test 3: get() with default
assert cfg.get("nonexistent", "default") == "default"
print("✓ Test 3 passed: get() with default")

# Test 4: to_dict()
d = cfg.to_dict()
assert d["model"]["name"] == "test-model"
print("✓ Test 4 passed: to_dict()")

# Test 5: freeze
cfg.freeze()
assert cfg.is_frozen()
print("✓ Test 5 passed: freeze")

print("\n✓ All Config tests passed!")
