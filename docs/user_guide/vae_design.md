# Variational Autoencoder (VAE) Design

## Architecture

### Encoder
- Input: (B, C, H, W) image
- Conv layers to compress
- Output: mean (μ) and log_variance (log σ²)

### Reparameterization
- Sample ε ~ N(0, 1)
- z = μ + σ * ε

### Decoder  
- Input: z (latent vector)
- Transposed conv to reconstruct
- Output: (B, C, H, W) reconstructed image

## Loss Function
- Reconstruction loss (MSE or BCE)
- KL divergence: -0.5 * sum(1 + log(σ²) - μ² - σ²)

## Config
```yaml
model:
  type: vae
  latent_dim: 128
  image_size: 64
  channels: 3
  beta: 1.0  # β-VAE coefficient
```