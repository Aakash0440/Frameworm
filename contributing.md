# Contributing to FRAMEWORM

Thank you for contributing! 🎉

---

## Quick Start
```bash
# Fork and clone
git clone https://github.com/YOURNAME/frameworm.git
cd frameworm

# Create virtual environment
python -m venv venv && source venv/bin/activate

# Install in dev mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest tests/unit/ -q
```

---

## Development Workflow

1. **Create issue** describing your change
2. **Create branch** `git checkout -b feat/my-feature`
3. **Make changes** with tests
4. **Run tests** `pytest tests/ -q`
5. **Format code** `black frameworm/ && isort frameworm/`
6. **Submit PR**

---

## Code Style

We use **black** for formatting (100 char line length).
```bash
black frameworm/ tests/
isort frameworm/ tests/
flake8 frameworm/ --max-line-length=100
```

---

## Writing Tests
```python
# tests/unit/test_my_feature.py

import pytest
from frameworm.my_module import MyClass

class TestMyClass:
    def test_basic_usage(self):
        obj = MyClass()
        result = obj.do_something()
        assert result == expected_value
    
    def test_edge_case(self):
        with pytest.raises(ValueError):
            MyClass(invalid_param=-1)
```

**Coverage requirement:** >90% for new code.

---

## Documentation

All public functions need:
```python
def my_function(arg1: str, arg2: int = 0) -> str:
    """
    One-line description.
    
    Longer description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg1 is empty
        
    Example:
        >>> result = my_function("hello", arg2=5)
        >>> print(result)
        "hello-5"
    """
```

---

## Release Process (Maintainers)
```bash
# Update version in pyproject.toml
# Update CHANGELOG.md
# Commit: git commit -m "chore: release v1.1.0"
# Tag: git tag v1.1.0
# Push: git push && git push --tags
# CI/CD handles PyPI publish automatically
```

---

## Getting Help

- [Discord](https://discord.gg/frameworm)
- [GitHub Discussions](https://github.com/Aakash0440/frameworm/discussions)
- [Email](Aakashali0440@example.com)

---

## Code of Conduct

Be kind, inclusive, and constructive. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).