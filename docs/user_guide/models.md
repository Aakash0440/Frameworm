# Models

## Available Models

### DCGAN

Deep Convolutional GAN for image generation.

**Usage:**
```python
from frameworm.core import Config, get_model

# Load config
cfg = Config('configs/models/gan/dcgan.yaml')

# Get model
model_class = get_model("dcgan")
model = model_class(cfg)

# Generate images
import torch
z = torch.randn(4, 100, 1, 1)
images = model(z)  # (4, 3, 64, 64)
```

**Config Options:**
```yaml
model:
  type: dcgan
  latent_dim: 100      # Latent vector dimension
  image_size: 64       # Output image size
  channels: 3          # Color channels
  ngf: 64             # Generator feature maps
  ndf: 64             # Discriminator feature maps
```

**Architecture:**
- Generator: 4 transposed conv layers
- Discriminator: 4 conv layers
- BatchNorm + ReLU/LeakyReLU

## Creating Custom Models

See [Plugin Guide](plugins.md) for creating custom models.

### Requirements

- Inherit from `BaseModel`
- Implement `forward()` method
- Register with `@register_model()`

### Example
```python
from frameworm.models import BaseModel
from frameworm.core import register_model
import torch.nn as nn

@register_model("my-model")
class MyModel(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Build architecture
        self.net = nn.Sequential(
            nn.Linear(config.model.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, config.model.output_dim)
        )
    
    def forward(self, x):
        return self.net(x)
```