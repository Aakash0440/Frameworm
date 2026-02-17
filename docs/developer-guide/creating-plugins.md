# Creating FRAMEWORM Plugins

Extend FRAMEWORM with custom models, callbacks, metrics, and integrations.

---

## Quick Start

### 1. Create Plugin Template
```bash
frameworm plugins create my-awesome-plugin
cd frameworm_plugins/my-awesome-plugin
```

This creates:
my-awesome-plugin/
├── plugin.yaml       # Metadata
└── init.py       # Entry point

### 2. Edit plugin.yaml
```yaml
name: my-awesome-plugin
version: 1.0.0
author: Your Name
description: My custom FRAMEWORM extension
entry_point: plugin:register
dependencies:
  - torch>=2.0.0
hooks:
  - model
  - callback
```

### 3. Implement Your Plugin
```python
# __init__.py

from frameworm.core import register_model
from frameworm.plugins.hooks import HookRegistry
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    
    def forward(self, x):
        return self.fc(x)
    
    def compute_loss(self, x, y=None):
        # Must return dict with 'loss' key
        output = self(x[0] if isinstance(x, (tuple, list)) else x)
        loss = nn.functional.mse_loss(output, y if y is not None else output)
        return {'loss': loss}

def register():
    """Called when plugin loads"""
    register_model('my_model', MyModel)
    
    @HookRegistry.on('on_epoch_end')
    def my_hook(trainer, epoch, metrics):
        print(f"Custom: Epoch {epoch} done!")
```

### 4. Load & Use
```bash
# Load plugin
frameworm plugins load my-awesome-plugin

# Use in training
frameworm train --config config.yaml --model my_model
```

Or programmatically:
```python
from frameworm.plugins.loader import load_plugins
from frameworm.core import get_model

load_plugins()  # Loads all discovered plugins

model = get_model('my_model')(config)
```

---

## Plugin Types

### Model Plugin

Register custom architectures:
```python
from frameworm.core import register_model

@register_model('my_transformer')
class TransformerModel(nn.Module):
    def __init__(self, config):
        # Your architecture
        pass
    
    def compute_loss(self, x, y=None):
        return {'loss': ...}
```

### Callback Plugin

Custom training callbacks:
```python
from frameworm.plugins.hooks import CallbackHook

class MyCallback(CallbackHook):
    def on_epoch_end(self, trainer, epoch, metrics):
        # Custom logic
        pass
    
    def on_train_end(self, trainer):
        # Cleanup
        pass

# Use:
callback = MyCallback()
callback.register()
```

### Metric Plugin

Custom evaluation metrics:
```python
from frameworm.core import register_metric

@register_metric('my_score')
class MyScoreMetric:
    def __call__(self, real_images, fake_images):
        # Compute score
        return score
```

---

## Hook Reference

### Training Hooks

- `on_train_begin(trainer)` - Before training starts
- `on_train_end(trainer)` - After training completes
- `on_epoch_begin(trainer, epoch)` - Start of epoch
- `on_epoch_end(trainer, epoch, metrics)` - End of epoch
- `on_batch_begin(trainer, batch_idx, batch)` - Before batch
- `on_batch_end(trainer, batch_idx, loss)` - After batch

### Gradient Hooks

- `on_backward_begin(trainer, loss)` - Before backward pass
- `on_backward_end(trainer)` - After backward pass
- `on_optimizer_step(trainer)` - Before optimizer.step()

### Validation Hooks

- `on_validation_begin(trainer)` - Start validation
- `on_validation_end(trainer, metrics)` - End validation

### Model Hooks

- `on_checkpoint_save(trainer, path)` - Before saving checkpoint
- `on_checkpoint_load(trainer, checkpoint)` - After loading
- `on_model_export(model, exporter, format)` - Before export

---

## Best Practices

1. **Fail gracefully** - Check dependencies, handle missing imports
2. **Don't modify core** - Use hooks, don't patch core classes
3. **Document well** - Clear docstrings, usage examples
4. **Version properly** - Semantic versioning (MAJOR.MINOR.PATCH)
5. **Test thoroughly** - Include unit tests in your plugin

---

## Example: MLflow Logger
```python
# mlflow_logger/__init__.py

from frameworm.plugins.hooks import CallbackHook
import mlflow

class MLflowLogger(CallbackHook):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
    
    def on_train_begin(self, trainer):
        mlflow.start_run(experiment_name=self.experiment_name)
    
    def on_epoch_end(self, trainer, epoch, metrics):
        mlflow.log_metrics(metrics, step=epoch)
    
    def on_train_end(self, trainer):
        mlflow.end_run()

def register():
    print("✓ MLflow logger available")
```

---

## Publishing Plugins

### Option 1: Local Plugin

Place in `frameworm_plugins/` directory.

### Option 2: PyPI Package

1. Create package: `frameworm_my_plugin`
2. Include `__plugin_metadata__` in `__init__.py`:
```python
__plugin_metadata__ = {
    'name': 'my-plugin',
    'version': '1.0.0',
    'author': 'Your Name',
    'description': 'My plugin',
    'entry_point': 'register'
}

def register():
    # Registration code
    pass
```

3. Publish: `pip install build && python -m build && twine upload dist/*`
4. Users install: `pip install frameworm_my_plugin`

---

## Debugging
```python
# List loaded plugins
from frameworm.plugins.loader import get_plugin_loader

loader = get_plugin_loader()
loader.print_plugins()

# Check registered hooks
from frameworm.plugins.hooks import HookRegistry

print(HookRegistry.list_hooks())

# Disable hooks temporarily
HookRegistry.disable()
# ... training ...
HookRegistry.enable()
```