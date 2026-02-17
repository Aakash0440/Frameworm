import torch
from core import Config
from models.vitgan import ViTGAN, ViTDiscriminator

print("Testing ViT-GAN:")
print("="*60)

config = Config.from_dict({
    'model': {
        'latent_dim': 64, 'image_size': 32, 'image_channels': 3,
        'gen_hidden': 64, 'vit_embed_dim': 128, 'vit_depth': 2,
        'vit_heads': 4, 'patch_size': 4
    }
})

model = ViTGAN(config)
g_params = sum(p.numel() for p in model.generator.parameters())
d_params = sum(p.numel() for p in model.discriminator.parameters())
print(f"✓ Generator: {g_params:,} params")
print(f"✓ Discriminator: {d_params:,} params")

real = torch.randn(2, 3, 32, 32)
losses = model.compute_loss(real)

assert 'loss' in losses and 'g_loss' in losses and 'd_loss' in losses
print(f"✓ Losses: g={losses['g_loss'].item():.3f}, d={losses['d_loss'].item():.3f}")

fake = model.generate(4)
assert fake.shape == (4, 3, 32, 32)
assert fake.min() >= -1.0 - 1e-4 and fake.max() <= 1.0 + 1e-4
print(f"✓ Generation: {fake.shape}")

print("="*60)
print("✅ ViT-GAN complete!")