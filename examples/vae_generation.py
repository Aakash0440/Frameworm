"""
Example: VAE Image Generation and Reconstruction

Demonstrates using VAE for reconstruction and sampling.
"""

from core import Config, get_model
import torch
from core.registry import discover_plugins

# Import your model file so it registers itself
import models.vae.vanilla

# Force plugin discovery
discover_plugins(force=True)


def main():
    print("VAE Example")
    print("=" * 60)

    # Load config
    cfg = Config("configs/models/vae/vanilla.yaml")
    print("✓ Loaded config")

    # Create VAE
    vae = get_model("vae")(cfg)
    vae.to_device("cpu")
    print("✓ Created VAE model")

    # Show model summary
    vae.summary(detailed=True)

    # Generate random images to reconstruct
    print("\nReconstruction:")
    images = torch.rand(8, 3, 64, 64)
    reconstructed = vae.reconstruct(images)

    print(f"✓ Reconstructed {len(images)} images")
    print(f"  Original range: [{images.min():.2f}, {images.max():.2f}]")
    print(f"  Recon range: [{reconstructed.min():.2f}, {reconstructed.max():.2f}]")

    # Compute reconstruction quality
    recon_error = (images - reconstructed).pow(2).mean()
    print(f"  MSE: {recon_error:.4f}")

    # Generate new samples from prior
    print("\nGeneration:")
    samples = vae.sample(16)

    print(f"✓ Generated {len(samples)} new samples from prior N(0,I)")
    print(f"  Sample range: [{samples.min():.2f}, {samples.max():.2f}]")

    # Show latent space
    print("\nLatent Space:")
    mu, logvar = vae.encode(images)
    print(f"  Latent mean range: [{mu.min():.2f}, {mu.max():.2f}]")
    print(f"  Latent logvar range: [{logvar.min():.2f}, {logvar.max():.2f}]")

    # Compute loss
    recon, mu, logvar = vae(images)
    loss_dict = vae.compute_loss(images, recon, mu, logvar)

    print("\nLoss:")
    print(f"  Total: {loss_dict['loss']:.4f}")
    print(f"  Reconstruction: {loss_dict['recon_loss']:.4f}")
    print(f"  KL Divergence: {loss_dict['kl_loss']:.4f}")

    print("\n" + "=" * 60)
    print("Example complete!")


if __name__ == "__main__":
    main()
