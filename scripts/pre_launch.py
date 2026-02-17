"""
Pre-launch checklist verification (debug mode).
"""

import subprocess
import sys
from pathlib import Path
import traceback

checks_passed = 0
checks_failed = 0


def check(name, fn):
    """
    Run a check function, print pass/fail, and show detailed errors if it fails.
    """
    global checks_passed, checks_failed
    try:
        result = fn()
        if result:
            print(f"  ✅ {name}")
            checks_passed += 1
        else:
            print(f"  ❌ {name}")
            print(f"     → Check returned False")
            checks_failed += 1
    except Exception as e:
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()
        checks_failed += 1


print("\n🚀 PRE-LAUNCH CHECKLIST")
print("=" * 60)

# ---------------- Package ----------------
print("\n📦 Package")
check("pyproject.toml exists", lambda: Path("pyproject.toml").exists())
check("LICENSE exists", lambda: Path("LICENSE").exists())
check("CHANGELOG.md exists", lambda: Path("CHANGELOG.md").exists())
check("README.md exists", lambda: Path("README.md").exists())

# ---------------- Tests ----------------
print("\n🧪 Tests")


def run_pytest():
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q", "--tb=no"], capture_output=True, text=True
    )
    print("pytest stdout:\n", result.stdout)
    print("pytest stderr:\n", result.stderr)
    return result.returncode == 0


check("All tests pass", run_pytest)
check("Tests exist", lambda: len(list(Path("tests").rglob("test_*.py"))) > 0)

# ---------------- Documentation ----------------
print("\n📚 Documentation")
check("docs/index.md exists", lambda: Path("docs/index.md").exists())
check("mkdocs.yml exists", lambda: Path("mkdocs.yml").exists())
check("Getting started guide exists", lambda: Path("docs/getting-started/quickstart.md").exists())

# ---------------- CLI ----------------
print("\n💻 CLI")


def check_cli():
    result = subprocess.run(
        [sys.executable, "cli/main.py", "--help"], capture_output=True, text=True
    )
    print("CLI stdout:\n", result.stdout)
    print("CLI stderr:\n", result.stderr)
    return result.returncode == 0


cli_result = subprocess.run(
    [sys.executable, "cli/main.py", "--help"], capture_output=True, text=True
)
print("CLI stdout:\n", cli_result.stdout)
print("CLI stderr:\n", cli_result.stderr)
check("CLI works", lambda: cli_result.returncode == 0)
check("CLI has train command", lambda: "train" in cli_result.stdout)
check("CLI has serve command", lambda: "serve" in cli_result.stdout)

# ---------------- Core Features ----------------
print("\n🔧 Core Features")
try:
    from core import Config
    from core.registry import get_model

    check("Config imports", lambda: True)
    try:
        check("Model registry works", lambda: get_model("vae") is not None)
    except Exception as e:
        print("Error in get_model:", e)
        traceback.print_exc()
        check("Model registry works", lambda: False)
except Exception as e:
    print("Error importing core:", e)
    traceback.print_exc()
    check("Core imports", lambda: False)

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
print("\n🌐 Community")
check(".github/ISSUE_TEMPLATE exists", lambda: Path(".github/ISSUE_TEMPLATE").exists())
check("CONTRIBUTING.md exists", lambda: Path("CONTRIBUTING.md").exists())

# ---------------- Summary ----------------
print("\n" + "=" * 60)
total = checks_passed + checks_failed
print(f"Results: {checks_passed}/{total} checks passed")

if checks_failed == 0:
    print("\n🎉 ALL CHECKS PASSED! Ready to launch!")
else:
    print(f"\n⚠️  {checks_failed} check(s) failed. Fix before launching.")
    sys.exit(1)
