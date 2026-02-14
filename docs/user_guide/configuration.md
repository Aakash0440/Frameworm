# Configuration System

## Overview

Frameworm uses a powerful YAML-based configuration system with support for:
- Config inheritance
- Environment variable interpolation
- CLI overrides
- Pydantic validation

## Basic Usage

### Loading a Config
```python
from frameworm.core import Config

cfg = Config('configs/model.yaml')
print(cfg.model.name)  # Access with dot notation
print(cfg['model']['name'])  # Or dict-style
```

### Config Inheritance

Configs can inherit from other configs using the `_base_` key:
```yaml
# configs/base.yaml
training:
  epochs: 100
  batch_size: 32

# configs/gan.yaml
_base_: ./base.yaml
training:
  epochs: 200  # Override
model:
  type: gan  # Add new
```

Multiple levels of inheritance are supported.

### Environment Variables

Use `${VAR_NAME}` syntax for environment variables:
```yaml
paths:
  data_dir: ${DATA_ROOT}/images
  output_dir: ./outputs/${EXPERIMENT_NAME}
```

### CLI Overrides

Override config values from command line:
```python
cfg = Config.from_cli_args(
    'config.yaml',
    ['training.epochs=500', 'model.dim=256']
)
```

### Validation

Define validation schemas with Pydantic:
```python
from pydantic import BaseModel, Field

class ModelConfig(BaseModel):
    name: str
    dim: int = Field(gt=0)

cfg = Config('model.yaml')
validated = cfg.validate(ModelConfig)
```

## Best Practices

1. **Use inheritance** - Create a base config and extend it
2. **Validate configs** - Define schemas for important configs
3. **Environment variables** - For paths and secrets
4. **Freeze configs** - After loading to prevent accidental modifications
5. **Dump configs** - Save merged configs for reproducibility

## Examples

See `examples/config_examples.py` for more usage patterns.