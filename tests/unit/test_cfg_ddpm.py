import torch
from core import Config
from models.cfg_ddpm import CFGDDPM

print("Testing CFG-DDPM:")
print("="*60)

config = Config.from_dict({
    'model': {
        'image_size': 16, 'image_channels': 1,
        'num_classes': 5, 'model_channels': 32,
        'timesteps': 50, 'p_uncond': 0.15
    }
})

model = CFGDDPM(config)
params = sum(p.numel() for p in model.parameters())
print(f"✓ CFG-DDPM: {params:,} params")

# Training
x = torch.randn(2, 1, 16, 16)
y = torch.randint(0, 5, (2,))
losses = model.compute_loss((x, y))
assert 'loss' in losses
print(f"✓ Training loss: {losses['loss'].item():.4f}")

# Unconditional
losses_uncond = model.compute_loss(x)
print(f"✓ Unconditional loss: {losses_uncond['loss'].item():.4f}")

# Sampling (very few steps for test)
model.timesteps = 5
samples = model.sample(num_samples=2, class_label=0, guidance_scale=3.0)
assert samples.shape == (2, 1, 16, 16)
print(f"✓ CFG sampling: {samples.shape}")

# Without guidance
samples_uncond = model.sample(num_samples=2, guidance_scale=1.0)
assert samples_uncond.shape == (2, 1, 16, 16)
print(f"✓ Unconditional sampling: {samples_uncond.shape}")

print("="*60)
print("✅ CFG-DDPM complete!")