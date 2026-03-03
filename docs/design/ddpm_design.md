# DDPM (Denoising Diffusion Probabilistic Model) Design

## Overview
Diffusion models gradually add noise to data (forward process)
then learn to reverse this process (reverse/denoising process).

## Forward Process (Training)
x_0 (data) → x_1 → x_2 → ... → x_T (pure noise)

At each step:
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise

## Reverse Process (Generation)
x_T (noise) → x_{T-1} → ... → x_1 → x_0 (data)

Learn to predict noise at each step.

## Architecture

### Noise Predictor (U-Net)
- Encoder: Downsampling with skip connections
- Bottleneck: Middle processing
- Decoder: Upsampling with skip connections
- Time embedding: Sinusoidal position encoding

## Config
```yaml
model:
  type: ddpm
  timesteps: 1000
  image_size: 64
  channels: 3
  base_channels: 128
```

## Key Components
1. Beta schedule (variance schedule)
2. Time embedding
3. U-Net architecture
4. Noise prediction loss