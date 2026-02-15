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

### VAE (Variational Autoencoder)

Learn a latent representation and generate new samples.

**Usage:**
```python
from frameworm.core import Config, get_model
import torch

# Load config
cfg = Config('configs/models/vae/vanilla.yaml')

# Get model
vae = get_model("vae")(cfg)

# Reconstruct images
images = torch.rand(4, 3, 64, 64)
reconstructed = vae.reconstruct(images)

# Generate new samples
samples = vae.sample(16)

# Training
recon, mu, logvar = vae(images)
loss_dict = vae.compute_loss(images, recon, mu, logvar)
loss = loss_dict['loss']
```

**Config Options:**
```yaml
model:
  type: vae
  latent_dim: 128      # Latent space dimension
  image_size: 64       # Input image size
  channels: 3          # Color channels
  beta: 1.0           # β coefficient (>1 for β-VAE)
```

**β-VAE:**

Use β > 1 for better disentanglement:
```yaml
model:
  beta: 4.0  # Emphasizes KL divergence
```

**Architecture:**
- Encoder: 4 conv layers → latent distribution
- Reparameterization: z = μ + σ * ε  
- Decoder: 4 transposed conv layers → reconstruction

**Loss:**
- Reconstruction: MSE between input and output
- KL Divergence: Regularization to N(0, I)
- Total: recon_loss + β * kl_loss

### DDPM (Denoising Diffusion Probabilistic Model)

Generate images through iterative denoising.

**Usage:**
```python
from frameworm.core import Config, get_model
import torch

# Load config
cfg = Config('configs/models/diffusion/ddpm.yaml')

# Get model
ddpm = get_model("ddpm")(cfg)

# Generate samples (slow - 1000 steps)
samples = ddpm.sample(batch_size=8)

# Training
x = torch.rand(batch, 3, 64, 64)
loss_dict = ddpm.compute_loss(x)
loss = loss_dict['loss']
```

**Config Options:**
```yaml
model:
  type: ddpm
  timesteps: 1000      # Number of diffusion steps
  image_size: 64       # Output image size
  channels: 3          # Color channels
  base_channels: 128   # U-Net base channels
```

**Architecture:**
- U-Net with time embeddings
- 1000 denoising steps
- Linear beta schedule
- MSE loss on noise prediction

**Notes:**
- Generation is slow (~1000 forward passes)
- Requires substantial training
- Quality improves with more timesteps
- Use EMA for best results

**Advantages:**
- High-quality samples
- Stable training
- Principled framework

**Disadvantages:**
- Very slow sampling
- Computationally expensive