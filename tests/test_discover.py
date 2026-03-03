"""
Test discover_plugins to see if it's finding files
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.getcwd())

TEST_PLUGIN_DIR = "test_plugins"  # Not tests/unit/test_plugins

print("=" * 70)
print("TESTING discover_plugins STEP BY STEP")
print("=" * 70)

# Step 1: Check directory
plugins_path = Path(TEST_PLUGIN_DIR)
print(f"\n1. Plugins directory: {plugins_path}")
print(f"   Exists: {plugins_path.exists()}")

if not plugins_path.exists():
    print("   ERROR: Directory doesn't exist!")
    sys.exit(1)

# Step 2: Find Python files manually
print("\n2. Finding Python files...")
recursive = True

if recursive:
    python_files = list(plugins_path.rglob("*.py"))
else:
    python_files = list(plugins_path.glob("*.py"))

print(f"   Found {len(python_files)} Python files:")
for f in python_files:
    print(f"   - {f}")

# Step 3: Filter out __init__.py
filtered = [f for f in python_files if f.name != "__init__.py"]
print(f"\n3. After filtering __init__.py: {len(filtered)} files")
for f in filtered:
    print(f"   - {f}")

# Step 4: Now test the actual discover_plugins function
print("\n4. Testing discover_plugins()...")
from core.registry import discover_plugins, list_models, reset_discovery

reset_discovery()

# Call discover_plugins with verbose output
result = discover_plugins(TEST_PLUGIN_DIR)

print(f"\n   Result: {result}")
print(f"   Models discovered: {result.get('models', [])}")

models = list_models(auto_discover=False)
print(f"\n5. Models in registry: {models}")

if "test-plugin-model" in models:
    print("\n   ✓ SUCCESS!")
else:
    print("\n   ✗ FAILED - Model not registered")
    print("\n   This means _import_plugin_file was never called")
    print("   OR it was called but the import failed silently")
