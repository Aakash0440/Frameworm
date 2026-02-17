"""
Pre-launch checklist verification.
"""

import subprocess
import sys
import os
from pathlib import Path
import traceback

checks_passed = 0
checks_failed = 0


def check(name, fn):
    global checks_passed, checks_failed
    try:
        result = fn()
        if result:
            print(f"  [OK]  {name}")
            checks_passed += 1
        else:
            print(f"  [FAIL]  {name}")
            print(f"     -> Check returned False")
            checks_failed += 1
    except Exception as e:
        print(f"  [FAIL]  {name}: {e}")
        traceback.print_exc()
        checks_failed += 1


print("\nPRE-LAUNCH CHECKLIST")
print("=" * 60)

# ---------------- Package ----------------
print("\nPackage")
check("pyproject.toml exists", lambda: Path("pyproject.toml").exists())
check("LICENSE exists", lambda: Path("LICENSE").exists())
check("CHANGELOG.md exists", lambda: Path("CHANGELOG.md").exists())
check("README.md exists", lambda: Path("README.md").exists())

# ---------------- Tests ----------------
print("\nTests")


def run_pytest():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--tb=short", "-v"],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
    )
    
    # Always print a clean summary
    lines = result.stdout.split('\n')
    
    # Show only failed/error lines for quick diagnosis
    failures = [l for l in lines if 'FAILED' in l or 'ERROR' in l]
    passed = [l for l in lines if 'passed' in l or 'failed' in l]
    
    if failures:
        print("\n  ❌ FAILING TESTS:")
        for line in failures:
            print(f"     {line.strip()}")
    
    if passed:
        print(f"  📊 {passed[-1].strip()}")
    
    # Show full output only if something failed
    if result.returncode != 0:
        print("\n  📋 FULL FAILURE DETAILS:")
        print("  " + "\n  ".join(lines))
    
    if result.stderr:
        print("\n  ⚠ STDERR:")
        print("  " + result.stderr[:500])
    
    return result.returncode == 0


check("All tests pass", run_pytest)
check("Tests exist", lambda: len(list(Path("tests").rglob("test_*.py"))) > 0)

# ---------------- Documentation ----------------
print("\nDocumentation")
check("docs/index.md exists", lambda: Path("docs/index.md").exists())
check("mkdocs.yml exists", lambda: Path("mkdocs.yml").exists())
check("Getting started guide exists", lambda: Path("docs/getting-started/quickstart.md").exists())

# ---------------- CLI ----------------
print("\nCLI")
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
cli_result = subprocess.run(
    [sys.executable, "cli/main.py", "--help"],
    capture_output=True,
    encoding="utf-8",
    errors="replace",
    env=env,
)
print("CLI stdout:\n", cli_result.stdout)
print("CLI stderr:\n", cli_result.stderr)
check("CLI works", lambda: cli_result.returncode == 0)
check("CLI has train command", lambda: "train" in cli_result.stdout)
check("CLI has serve command", lambda: "serve" in cli_result.stdout)

# ---------------- Core Features ----------------
print("\nCore Features")
try:
    from core import Config
    from core.registry import get_model
    check("Config imports", lambda: True)
except Exception as e:
    print("Error importing core:", e)
    traceback.print_exc()
    checks_failed += 1

try:
    # FIX 3: explicitly import the VAE module so @register_model("vae")
    # decorator fires before we call get_model("vae").
    # We do this by adding the project root to sys.path and importing directly.
    project_root = str(Path(__file__).parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import importlib
    importlib.import_module("models.vae.vanilla")

    check("Model registry works", lambda: get_model("vae", auto_discover=False) is not None)
except Exception as e:
    print("Error in model registry check:", e)
    traceback.print_exc()
    checks_failed += 1

try:
    from training import Trainer
    check("Trainer imports", lambda: True)
except Exception as e:
    print("Error importing Trainer:", e)
    traceback.print_exc()
    check("Trainer imports", lambda: False)

try:
    from experiment import Experiment, ExperimentManager
    check("Experiment tracking imports", lambda: True)
except Exception as e:
    print("Error importing Experiment:", e)
    traceback.print_exc()
    check("Experiment tracking imports", lambda: False)

try:
    from search import GridSearch, RandomSearch
    check("Search imports", lambda: True)
except Exception as e:
    print("Error importing search:", e)
    traceback.print_exc()
    check("Search imports", lambda: False)

try:
    from deployment import ModelExporter
    check("Deployment imports", lambda: True)
except Exception as e:
    print("Error importing ModelExporter:", e)
    traceback.print_exc()
    check("Deployment imports", lambda: False)

# ---------------- Community ----------------
print("\nCommunity")
check(".github/ISSUE_TEMPLATE exists", lambda: Path(".github/ISSUE_TEMPLATE").exists())
check("CONTRIBUTING.md exists", lambda: Path("CONTRIBUTING.md").exists())

# ---------------- Summary ----------------
print("\n" + "=" * 60)
total = checks_passed + checks_failed
print(f"Results: {checks_passed}/{total} checks passed")

if checks_failed == 0:
    print("\nALL CHECKS PASSED! Ready to launch!")
else:
    print(f"\n{checks_failed} check(s) failed. Fix before launching.")
    sys.exit(1)