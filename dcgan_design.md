# DCGAN Implementation

## Architecture

Generator:
- Input: latent vector (100-dim)
- 4 transposed conv layers
- BatchNorm + ReLU
- Output: 64x64 image

Discriminator:
- Input: 64x64 image
- 4 conv layers
- BatchNorm + LeakyReLU
- Output: real/fake probability

## Config
```yaml
model:
  type: dcgan
  latent_dim: 100
  image_size: 64
  channels: 3
  ngf: 64  # generator features
  ndf: 64  # discriminator features
```