# Core API

Core components of FRAMEWORM.

---

## Config

::: frameworm.core.Config
    options:
      show_source: true
      members:
        - __init__
        - from_dict
        - to_dict
        - to_yaml
        - get
        - update

### Usage
```python
from frameworm.core import Config

# Load from YAML
config = Config('config.yaml')

# Access values
lr = config.training.lr
batch_size = config.training.batch_size

# Update values
config.update({'training.lr': 0.0001})

# Save
config.to_yaml('updated_config.yaml')
```

---

## Model Registry

::: frameworm.core.get_model

### Built-in Models

- `vae` - Variational Autoencoder
- `dcgan` - Deep Convolutional GAN
- `ddpm` - Denoising Diffusion Probabilistic Model

### Usage
```python
from frameworm.core import get_model

# Get model class
VAE = get_model('vae')

# Create instance
model = VAE(config)
```

---

## Plugin System

::: frameworm.core.plugin.PluginRegistry

### Register Custom Model
```python
from frameworm.core import register_model
import torch.nn as nn

@register_model('my_model')
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Your model code
    
    def forward(self, x):
        # Forward pass
        return x

# Now available via get_model
model = get_model('my_model')(config)
```

---

## Type System

::: frameworm.core.types.ModelProtocol
::: frameworm.core.types.OptimizerProtocol
::: frameworm.core.types.DataLoaderProtocol

### Usage
```python
from frameworm.core.types import ModelProtocol
import torch.nn as nn

class MyModel(nn.Module, ModelProtocol):
    def compute_loss(self, inputs, targets):
        # Required by protocol
        return {'loss': loss_value}
```