from core.config import ConfigNode

# Test 1: Basic dict access
node = ConfigNode({"key": "value"})
assert node["key"] == "value"
print("✓ Test 1 passed: Basic dict access")

# Test 2: Dot notation
node = ConfigNode({"model": {"name": "gan"}})
assert node.model.name == "gan"
print("✓ Test 2 passed: Dot notation")

# Test 3: Nested conversion
node = ConfigNode({"a": {"b": {"c": 123}}})
assert node.a.b.c == 123
print("✓ Test 3 passed: Nested access")

# Test 4: to_dict
d = node.to_dict()
assert isinstance(d, dict)
assert d["a"]["b"]["c"] == 123
print("✓ Test 4 passed: to_dict conversion")

print("\n✓ All ConfigNode tests passed!")
