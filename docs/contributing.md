# Contributing to FRAMEWORM

We love your input! We want to make contributing as easy as possible.

---

## Development Setup
```bash
# Clone repo
git clone https://github.com/yourusername/frameworm.git
cd frameworm

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

---

## Development Workflow

1. **Create a branch**
```bash
   git checkout -b feature/my-feature
```

2. **Make changes**
   - Write code
   - Add tests
   - Update docs

3. **Run tests**
```bash
   pytest tests/
```

4. **Format code**
```bash
   black frameworm/
   isort frameworm/
```

5. **Submit PR**
```bash
   git push origin feature/my-feature
```

---

## Code Style

We use:
- **black** for formatting
- **isort** for imports
- **mypy** for type checking
- **flake8** for linting

---

## Testing
```bash
# Run all tests
pytest

# Run specific test
pytest tests/test_training.py

# Run with coverage
pytest --cov=frameworm
```

---

## Documentation

- Use Google-style docstrings
- Add examples to docstrings
- Update user guides for new features

---

## Pull Request Process

1. Update README.md if needed
2. Add tests for new features
3. Update documentation
4. Ensure CI passes
5. Request review

---

## Code of Conduct

Be respectful and inclusive. See CODE_OF_CONDUCT.md.