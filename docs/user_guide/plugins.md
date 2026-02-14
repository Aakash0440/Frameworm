# Plugin System

## Overview

Frameworm's plugin system allows you to extend the framework with custom components without modifying the core code.

## Quick Start

### 1. Create a Plugin

Create a Python file in the `plugins/` directory:
```python
# plugins/my_model.py
from frameworm.models import BaseModel
from frameworm.core import register_model

@register_model("my-model")
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        # Your architecture here
    
    def forward(self, x):
        # Your forward pass here
        return x
```

### 2. Use Your Plugin
```python
from frameworm.core import get_model, Config

# Auto-discovered!
model_class = get_model("my-model")
model = model_class(config)
```

That's it! No imports, no registration code in your main script.

## Plugin Types

### Models
```python
@register_model("name", version="1.0", author="you")
class MyModel(BaseModel):
    def forward(self, x):
        ...
```

**Requirements:**
- Inherit from `BaseModel`
- Implement `forward()` method

### Trainers
```python
@register_trainer("name")
class MyTrainer(BaseTrainer):
    def training_step(self, batch, batch_idx):
        ...
    
    def validation_step(self, batch, batch_idx):
        ...
```

**Requirements:**
- Inherit from `BaseTrainer`
- Implement `training_step()` and `validation_step()`

### Pipelines
```python
@register_pipeline("name")
class MyPipeline(BasePipeline):
    def run(self, *args, **kwargs):
        ...
```

**Requirements:**
- Inherit from `BasePipeline`
- Implement `run()` method

### Datasets
```python
@register_dataset("name")
class MyDataset:
    def __len__(self):
        ...
    
    def __getitem__(self, idx):
        ...
```

**Requirements:**
- Implement `__len__()` and `__getitem__()`

## Auto-Discovery

Plugins are automatically discovered when:
- First access (get/list)
- Manual discovery

### Disable Auto-Discovery
```python
from frameworm.core import set_auto_discover

set_auto_discover(False)

# Now you must manually discover
from frameworm.core import discover_plugins
discover_plugins()
```

## Advanced Features

### Search
```python
from frameworm.core import search_models

# Find all GAN models
gan_models = search_models("gan")
```

### Metadata
```python
from frameworm.core import get_model_metadata

metadata = get_model_metadata("my-model")
print(f"Version: {metadata['version']}")
print(f"Author: {metadata['author']}")
```

### Registry Summary
```python
from frameworm.core import print_registry_summary

print_registry_summary()
```

## Best Practices

1. **One plugin per file**
2. **Descriptive names**
3. **Add metadata** (version, author, description)
4. **Include docstrings**
5. **Test your plugins**
6. **Keep dependencies minimal**

## Organization

Organize plugins in subdirectories:
plugins/
├── models/
│   ├── gans/
│   │   ├── stylegan.py
│   │   └── dcgan.py
│   └── diffusion/
│       └── ddpm.py
├── trainers/
│   └── custom_trainer.py
└── utils/
└── helpers.py

All `.py` files are discovered recursively.

## Troubleshooting

### Plugin Not Found
```python
# Check if plugin was discovered
from frameworm.core import list_models, discover_plugins

print(list_models())  # See all models

# Force re-discovery
from frameworm.core import reset_discovery
reset_discovery()
discover_plugins(force=True)
```

### Import Errors

Plugin files with import errors are skipped with a warning:
Warning: Failed to import plugin plugins/my_model.py: ModuleNotFoundError
Check your plugin file for errors.

### Validation Errors

If your plugin doesn't meet requirements:
```python
TypeError: Model class MyModel must implement forward() method
```

Ensure your plugin implements all required methods.

## Examples

See `examples/custom_plugin.py` for complete working examples.

See `plugins/README.md` for quick reference.