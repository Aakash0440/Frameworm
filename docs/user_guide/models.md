# Model Zoo

FRAMEWORM includes 6 production-ready generative models.

---

## VAE — Variational Autoencoder

Classic VAE with KL regularization.
```python
model = get_model('vae')(config)
```

**Best for:** Fast prototyping, smooth latent space

---

## DCGAN — Deep Convolutional GAN

Stable GAN training with spectral normalization.
```python
model = get_model('dcgan')(config)
```

**Best for:** Quick image generation

---

## DDPM — Denoising Diffusion

Original DDPM with linear noise schedule.
```python
model = get_model('ddpm')(config)
```

**Best for:** High quality unconditional generation

---

## VQ-VAE-2 — Vector Quantized VAE

Hierarchical codebook latents. Sharp reconstructions.
```python
model = get_model('vqvae2')(config)
```

**Best for:** Compression, tokenization, discrete representations

Key feature — two-level hierarchy:
- Bottom level: local details (64x64)
- Top level: global structure (16x16)
```python
z_top, z_bottom, vq_loss = model.encode(x)
recon = model.decode(z_top, z_bottom)
```

---

## ViT-GAN — Vision Transformer GAN

Transformer discriminator for global coherence.
```python
model = get_model('vitgan')(config)
```

**Best for:** High-resolution, globally coherent generation
```yaml
model:
  type: vitgan
  image_size: 64
  patch_size: 8
  vit_embed_dim: 384
  vit_depth: 6
```

---

## CFG-DDPM — Classifier-Free Guidance Diffusion

Conditional generation with guidance scale control.
```python
model = get_model('cfg_ddpm')(config)
```

**Best for:** Conditional generation, text-to-image, class-guided synthesis
```python
# Train conditionally
losses = model.compute_loss((x, class_labels))

# Sample with guidance
samples = model.sample(
    num_samples=4,
    class_label=3,          # Generate class 3
    guidance_scale=7.5       # Higher = more faithful but less diverse
)
```

**Guidance scale guide:**
- 1.0 = no guidance (unconditional)
- 3.0–5.0 = mild guidance
- 7.5 = recommended (Stable Diffusion default)
- 15+ = strong guidance, reduced diversity

---

## Comparison

| Model | Quality | Speed | Controllable | Memory |
|-------|---------|-------|-------------|--------|
| VAE | ⭐⭐ | ⭐⭐⭐⭐⭐ | ❌ | Low |
| DCGAN | ⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Low |
| DDPM | ⭐⭐⭐⭐ | ⭐⭐ | ❌ | Medium |
| VQ-VAE-2 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ❌ | Medium |
| ViT-GAN | ⭐⭐⭐⭐ | ⭐⭐⭐ | ❌ | High |
| CFG-DDPM | ⭐⭐⭐⭐⭐ | ⭐ | ✅ | High |