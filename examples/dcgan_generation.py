"""
Example: Image Generation with DCGAN

Demonstrates loading DCGAN and generating images.
"""

from core import Config, get_model
import torch
import models.gan.dcgan

def main():
    print("DCGAN Image Generation Example")
    print("=" * 60)
    
    # Load config
    cfg = Config('configs/models/gan/dcgan.yaml')
    print(f"✓ Loaded config")
    
    # Get DCGAN model
    model_class = get_model("dcgan")
    model = model_class(cfg)
    model.to_device('cpu')
    print(f"✓ Created DCGAN model")
    
    # Show model summary
    model.summary()

    # Generate random images
    print("\nGenerating images...")
    batch_size = 16
    images = model(batch_size=batch_size)
    
    print(f"✓ Generated {batch_size} images")
    print(f"  Shape: {images.shape}")
    print(f"  Range: [{images.min():.2f}, {images.max():.2f}]")
    
    # Generate with specific latent vectors
    print("\nGenerating with custom latent vectors...")
    latent_dim = cfg.model.latent_dim  # dynamically get latent_dim from config
    z = torch.randn(8, latent_dim, 1, 1)  # use correct latent_dim
    custom_images = model(z)
    print(f"✓ Generated {len(custom_images)} images")
    
    # Use discriminator
    print("\nDiscriminating images...")
    probs = model.discriminate(images)
    print(f"✓ Discrimination probabilities: {probs.mean():.3f} ± {probs.std():.3f}")
    
    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == '__main__':
    main()