# Frameworm Plugins

This directory is for custom plugins (models, trainers, pipelines, datasets).

## Quick Start

### Creating a Custom Model
```python
# plugins/my_model.py
from frameworm.models import BaseModel
from frameworm.core import register_model
import torch.nn as nn

@register_model("my-custom-gan")
class MyCustomGAN(BaseModel):
    """My custom GAN implementation"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Build your architecture
        self.generator = nn.Sequential(
            nn.Linear(config.model.latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.generator(z)
```

### Using Your Plugin
```python
from frameworm.core import get_model, Config

# Your plugin is auto-discovered!
cfg = Config.from_template('gan')
model_class = get_model("my-custom-gan")
model = model_class(cfg)
```

## Plugin Types

### Models
- Must inherit from `BaseModel`
- Must implement `forward()` method
- Register with `@register_model("name")`

### Trainers  
- Must inherit from `BaseTrainer`
- Must implement `training_step()` and `validation_step()`
- Register with `@register_trainer("name")`

### Pipelines
- Must inherit from `BasePipeline`
- Must implement `run()` method
- Register with `@register_pipeline("name")`

### Datasets
- Must implement `__len__()` and `__getitem__()`
- Register with `@register_dataset("name")`

## Auto-Discovery

Plugins are automatically discovered when you:
- Call `get_model("name")`
- Call `list_models()`
- Or manually call `discover_plugins()`

## Organization

You can organize plugins in subdirectories:
plugins/
├── models/
│   ├── my_gan.py
│   └── my_vae.py
├── trainers/
│   └── my_trainer.py
└── utils/
└── helpers.py


All `.py` files are discovered recursively.

## Debugging
```python
from frameworm.core import print_registry_summary

# See all registered items
print_registry_summary()

# Search for specific models
from frameworm.core import search_models
models = search_models("gan")
```

## Best Practices

1. **One plugin per file** - Makes debugging easier
2. **Clear names** - Use descriptive registration names
3. **Add metadata** - Include version, author, description
4. **Document your plugin** - Add docstrings
5. **Test it** - Create tests in tests/plugins/

## Examples

See `examples/custom_plugin.py` for complete examples.
