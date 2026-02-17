"""
Example: Diffusion Model Image Generation

Demonstrates using DDPM for generation and interpolation.
"""

from core import Config, get_model
import torch


def main():
    print("DDPM Generation Example")
    print("=" * 60)

    # Load config
    cfg = Config("configs/models/diffusion/ddpm.yaml")
    print("✓ Loaded config")

    # Reduce timesteps for faster demo
    cfg.model.timesteps = 100  # Instead of 1000

    # Create DDPM
    ddpm = get_model("ddpm")(cfg)
    ddpm.to_device("cpu")
    print("✓ Created DDPM model")

    # Show model summary
    ddpm.summary(detailed=False)

    # Generate samples
    print("\nGenerating samples (100 denoising steps)...")
    print("This will take a minute...")

    with torch.no_grad():
        samples = ddpm.sample(batch_size=4, image_size=64, channels=3)

    print(f"✓ Generated {len(samples)} samples")
    print(f"  Shape: {samples.shape}")
    print(f"  Range: [{samples.min():.2f}, {samples.max():.2f}]")

    # Show diffusion process
    print("\nDiffusion Process:")
    x = torch.randn(1, 3, 64, 64)

    for t in [0, 250, 500, 750, 999]:
        if t < ddpm.timesteps:
            t_tensor = torch.tensor([t])
            x_noisy = ddpm.q_sample(x, t_tensor)
            print(f"  t={t:4d}: noise level = {x_noisy.std():.2f}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNote: For production use:")
    print("  - Use timesteps=1000 for best quality")
    print("  - Train on large dataset")
    print("  - Use exponential moving average (EMA)")


if __name__ == "__main__":
    main()
