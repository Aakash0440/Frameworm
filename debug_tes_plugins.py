"""
Test plugin discovery with detailed logging
"""
import sys
import os
from pathlib import Path

# Add project to path
sys.path.insert(0, os.getcwd())

print("=" * 70)
print("DETAILED PLUGIN DISCOVERY TEST")
print("=" * 70)

TEST_PLUGIN_DIR = "tests/unit/test_plugin_data"

# Step 1: Verify directory exists
print(f"\n1. Plugin directory: {TEST_PLUGIN_DIR}")
print(f"   Exists: {os.path.exists(TEST_PLUGIN_DIR)}")

if os.path.exists(TEST_PLUGIN_DIR):
    files = list(Path(TEST_PLUGIN_DIR).rglob("*.py"))
    print(f"   Python files found: {len(files)}")
    for f in files:
        print(f"   - {f}")

# Step 2: Test the _import_plugin_file logic manually
print("\n2. Testing module path calculation...")

from pathlib import Path
plugin_file = Path(TEST_PLUGIN_DIR) / "plugin_model.py"
base_path = Path(TEST_PLUGIN_DIR)

print(f"   Plugin file: {plugin_file}")
print(f"   Base path: {base_path}")
print(f"   CWD: {Path.cwd()}")

# What the fixed code should do
relative_path = plugin_file.relative_to(Path.cwd())
module_parts = list(relative_path.parts[:-1]) + [relative_path.stem]
module_name = '.'.join(module_parts)

print(f"   Relative path: {relative_path}")
print(f"   Module parts: {module_parts}")
print(f"   Module name: {module_name}")

# Step 3: Try importing manually
print(f"\n3. Testing manual import of '{module_name}'...")
try:
    import importlib
    import sys
    
    if module_name in sys.modules:
        print(f"   Module already in sys.modules, reloading...")
        importlib.reload(sys.modules[module_name])
    else:
        print(f"   Importing module for first time...")
        mod = importlib.import_module(module_name)
        print(f"   ✓ Import successful: {mod}")
    
    from core.registry import list_models
    models = list_models(auto_discover=False)
    print(f"   Models after import: {models}")
    
    if "test-plugin-model" in models:
        print("   ✓✓ SUCCESS: Model registered!")
    else:
        print("   ✗ Model not registered even after import")
        
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Test discover_plugins function
print("\n4. Testing discover_plugins() function...")
try:
    from core.registry import discover_plugins, reset_discovery, list_models
    
    reset_discovery()
    print("   Registry reset")
    
    # Enable verbose mode by monkey-patching
    discovered = discover_plugins(TEST_PLUGIN_DIR)
    print(f"   Discovered result: {discovered}")
    
    models = list_models(auto_discover=False)
    print(f"   Models in registry: {models}")
    
    if "test-plugin-model" in models:
        print("   ✓✓ SUCCESS: discover_plugins() works!")
    else:
        print("   ✗ discover_plugins() didn't register the model")
        print("   This means _import_plugin_file is failing silently")
        
except Exception as e:
    print(f"   ✗ Error: {e}")
    import traceback
    traceback.print_exc()

# Step 5: Add verbose logging to see what's happening
print("\n5. Checking if warnings are being suppressed...")
print("   The _import_plugin_file function uses warnings.warn()")
print("   If there are errors, they might be hidden")
print("\n   Try running with: python -W all debug_test_plugins.py")

print("\n" + "=" * 70)
print("DIAGNOSIS:")
print("=" * 70)
print("""
If manual import works but discover_plugins() doesn't:
- The _import_plugin_file function is catching exceptions silently
- Check that Path.cwd() fix was applied correctly
- The function might be using Path.cwd() but pytest changes the CWD

SOLUTION:
Add debug prints inside _import_plugin_file to see what's happening.
Or check if there's a try/except swallowing the error.
""")