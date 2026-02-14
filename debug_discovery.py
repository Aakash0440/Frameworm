"""
Debug why discover_plugins isn't importing files
"""
import sys
import os
from pathlib import Path

sys.path.insert(0, os.getcwd())

TEST_PLUGIN_DIR = Path(__file__).parent / "tests/unit/test_plugin_models"

print("=" * 70)
print("DEBUGGING discover_plugins")
print("=" * 70)

print(f"\nPlugin directory: {TEST_PLUGIN_DIR}")
print(f"Exists: {TEST_PLUGIN_DIR.exists()}")

if TEST_PLUGIN_DIR.exists():
    print(f"\nFiles in directory:")
    for f in TEST_PLUGIN_DIR.rglob("*.py"):
        print(f"  - {f}")

# Now trace through what discover_plugins does
from core.registry import discover_plugins, reset_discovery

# Monkey-patch _import_plugin_file to see if it's called
import core.registry as registry_module

original_import = registry_module._import_plugin_file

def debug_import(file_path, base_path):
    print(f"\n[DEBUG _import_plugin_file] Called with:")
    print(f"  file_path: {file_path}")
    print(f"  base_path: {base_path}")
    result = original_import(file_path, base_path)
    print(f"  Result: {result}")
    return result

registry_module._import_plugin_file = debug_import

# Now test
print("\n" + "=" * 70)
print("Calling discover_plugins...")
print("=" * 70)

reset_discovery()
result = discover_plugins(str(TEST_PLUGIN_DIR))

print(f"\nResult: {result}")

from core.registry import list_models
models = list_models(auto_discover=False)
print(f"Models: {models}")