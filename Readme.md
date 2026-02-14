# FRAMEWORM 🐛

> Advanced Generative AI Framework with Plugin System and Dependency Graphs

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

**Status:** ✅ Day 2 Complete - Core Architecture Ready

## Features

- ✅ **Config System** - YAML configs with inheritance, validation, templates
- ✅ **Type System** - Protocols, type guards, validation utilities
- ✅ **Base Classes** - Enhanced BaseModel, BasePipeline, BaseTrainer
- 🚧 **Plugin System** - Coming Day 3
- 🚧 **Dependency Graphs** - Coming Day 5-6
- ⏳ **Experiment Tracking** - Coming Day 10-11
- ⏳ **Hyperparameter Search** - Coming Day 15-17
- ⏳ **Benchmark Suite** - Coming Day 18-20

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/frameworm.git
cd frameworm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install
pip install -e ".[dev]"
```

### Basic Usage
```python
from frameworm.core import Config
from frameworm.models import BaseModel

# Load config
cfg = Config('configs/models/gan/dcgan.yaml')

# Or use template
cfg = Config.from_template('gan', **{'model.latent_dim': 256})

# Create model
class MyGAN(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # ... build architecture
    
    def forward(self, z):
        # ... generation logic
        return images

model = MyGAN(cfg)
model.to_device('cuda')
```

## Project Structure
frameworm/
├── core/           # Core utilities
│   ├── config.py   # Config system
│   ├── types.py    # Type system
│   └── registry.py # Plugin registry (coming)
├── models/         # Model implementations
│   └── base.py     # BaseModel
├── trainers/       # Training logic
│   └── base.py     # BaseTrainer
├── pipelines/      # Pipelines
│   └── base.py     # BasePipeline
├── data/           # Data utilities
├── optimization/   # Hyperparameter search
├── experiment/     # Experiment tracking
└── benchmark/      # Benchmark suite
configs/            # Configuration files
├── base.yaml
├── templates/      # Quick-start templates
└── models/         # Model configs
tests/              # Test suite
├── unit/
├── integration/
└── benchmark/
docs/               # Documentation
├── user_guide/
├── architecture/
└── developer_guide/


## Documentation

- [User Guide](docs/user_guide/) - How to use Frameworm
- [Architecture](docs/architecture/) - System design
- [Developer Guide](docs/developer_guide/) - Contributing

## Development
```bash
# Run tests
pytest

# With coverage
pytest --cov=frameworm --cov-report=html

# Format code
black frameworm tests

# Type check
mypy frameworm --ignore-missing-imports

# Lint
flake8 frameworm
```

## Testing

Test coverage: **95%+**
```bash
# Run all tests
pytest

# Run specific module
pytest tests/unit/test_config.py

# See coverage report
pytest --cov=frameworm --cov-report=html
open htmlcov/index.html
```

## Roadmap

### ✅ Completed (Days 1-2)
- Config system with inheritance
- Type system with protocols
- Enhanced base classes
- Comprehensive testing
- Documentation

### 🚧 In Progress (Week 1)
- Plugin registry system
- Dependency graph engine
- Error explanation system
- First model implementation

### ⏳ Upcoming
- **Week 2**: Training infrastructure, experiment tracking
- **Week 3**: Hyperparameter search, benchmarking
- **Week 4**: CLI wizard, documentation, launch

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](LICENSE) file.

---

**Built with ❤️ during a 4-week intensive project**

**Current Progress:** Day 2/28 (7% complete)
**Hours Invested:** 20/280
**Commits:** 10
