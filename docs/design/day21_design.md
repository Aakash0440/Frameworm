# Day 21: Advanced Model Architectures

## Three New Models

### 1. VQ-VAE-2 (Vector Quantized VAE)
- Hierarchical latent codes
- Much sharper reconstructions than vanilla VAE
- Two-level codebook (top + bottom)
- Used by DALL-E predecessor

### 2. ViT-GAN (Vision Transformer GAN)
- Transformer discriminator replaces CNN
- Self-attention captures global structure
- State-of-the-art image synthesis quality
- TransGAN / ViT-GAN architecture

### 3. Improved DDPM (Classifier-Free Guidance)
- Conditional generation with guidance scale
- Much better sample quality vs Day 1 DDPM
- Supports text/class conditioning
- Used by Stable Diffusion, DALL-E 2

## Integration
All models use the existing:
- Config system
- Training loop
- Experiment tracking
- Deployment pipeline