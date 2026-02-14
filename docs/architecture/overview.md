# Frameworm Architecture

## Overview

Frameworm is built on three core principles:
1. **Flexibility** - Plugin system allows easy extension
2. **Reproducibility** - Config and experiment tracking ensure reproducible results
3. **Simplicity** - Clean abstractions make it easy to use

## Core Components

### Config System (`frameworm.core.config`)
- YAML-based configuration with inheritance
- Environment variable interpolation
- Type coercion and validation
- CLI overrides
- Template support

### Base Classes (`frameworm.models`, `frameworm.pipelines`, `frameworm.trainers`)
- `BaseModel` - Foundation for all models
- `BasePipeline` - Step-based execution pipelines
- `BaseTrainer` - Training loop abstraction

### Type System (`frameworm.core.types`)
- Protocols for structural typing
- Type guards for runtime checking
- Validation utilities
- Generic containers

## Design Patterns

### Configuration Pattern

configs/
├── base.yaml           # Shared defaults
├── models/
│   ├── gan/
│   │   ├── base_gan.yaml
│   │   └── dcgan.yaml  # Inherits from base_gan.yaml

### Plugin Pattern
```python
@register_model("my-model")
class MyModel(BaseModel):
    ...
```

### Pipeline Pattern
```python
pipeline = Pipeline(config)
pipeline.add_step('preprocess', preprocess_fn)
pipeline.add_step('train', train_fn, depends_on=['preprocess'])
pipeline.execute_steps()
```

## Dependency Graph
User Code
↓
frameworm.models (BaseModel)
↓
frameworm.core (Config, Registry, Types)
↓
torch


## Extension Points

1. **Models** - Inherit from `BaseModel`
2. **Trainers** - Inherit from `BaseTrainer`  
3. **Pipelines** - Inherit from `BasePipeline`
4. **Plugins** - Use `@register_*` decorators

See individual component docs for details.
EOF

cat > docs/architecture/config_system.md << 'EOF'
# Config System Architecture

## Design Goals

1. **Human-readable** - YAML format
2. **DRY** - Inheritance reduces duplication
3. **Flexible** - Multiple override mechanisms
4. **Validated** - Catch errors early

## Implementation

### ConfigNode
Dictionary with dot notation access:
```python
node = ConfigNode({'model': {'dim': 128}})
node.model.dim  # 128
```

### Config Class
Main config manager:
- Loads YAML files
- Resolves inheritance chain
- Applies type coercion
- Validates required fields

### Inheritance Resolution
dcgan.yaml
↓ (base: base_gan.yaml)
base_gan.yaml
↓ (base: ../../base.yaml)
base.yaml
↓
Merged Config

## Advanced Features

### Type Coercion
Automatic string → type conversion:
- "123" → 123 (int)
- "0.5" → 0.5 (float)
- "true" → True (bool)

### Required Fields
```yaml
_required:
  - model.name
  - training.epochs
```

### Templates
Quick-start configs:
```python
cfg = Config.from_template('gan', **overrides)
```

### Environment Variables
```yaml
data_dir: ${DATA_ROOT}/images
```

### CLI Overrides
```python
cfg = Config.from_cli_args(
    'config.yaml',
    ['training.epochs=500']
)
```

## Performance

- Lazy loading - configs loaded only when needed
- Caching - parsed configs cached in memory
- O(n) inheritance resolution where n = depth

## Testing

95%+ coverage including:
- Unit tests for each feature
- Integration tests for inheritance chains
- Edge case tests