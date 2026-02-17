"""
Config System Examples

This file demonstrates various config system features.
"""

from core.config import Config
from pydantic import BaseModel, Field
from pathlib import Path


def example_basic_loading():
    """Example 1: Basic config loading"""
    print("=" * 60)
    print("Example 1: Basic Config Loading")
    print("=" * 60)

    cfg = Config("configs/base.yaml")

    print(f"Training epochs: {cfg.training.epochs}")
    print(f"Batch size: {cfg.training.batch_size}")
    print(f"Optimizer: {cfg.optimizer.type}")
    print(f"Learning rate: {cfg.optimizer.lr}")
    print()


def example_inheritance():
    """Example 2: Config inheritance"""
    print("=" * 60)
    print("Example 2: Config Inheritance")
    print("=" * 60)

    cfg = Config("configs/models/gan/dcgan.yaml")

    print("Values from base.yaml (2 levels up):")
    print(f"  batch_size: {cfg.training.batch_size}")
    print()

    print("Values from base_gan.yaml (1 level up):")
    print(f"  image_size: {cfg.model.image_size}")
    print()

    print("Values from dcgan.yaml:")
    print(f"  model type: {cfg.model.type}")
    print(f"  latent_dim: {cfg.model.latent_dim}")
    print(f"  generator ngf: {cfg.model.generator.ngf}")
    print()


def example_validation():
    """Example 3: Config validation"""
    print("=" * 60)
    print("Example 3: Config Validation")
    print("=" * 60)

    class TrainingConfig(BaseModel):
        epochs: int = Field(gt=0)
        batch_size: int = Field(gt=0, le=512)
        device: str

    cfg = Config("configs/base.yaml")

    try:
        validated = cfg.validate(TrainingConfig)
        print("✓ Config is valid")
        print(f"  epochs: {validated.epochs}")
        print(f"  batch_size: {validated.batch_size}")
        print(f"  device: {validated.device}")
    except Exception as e:
        print(f"✗ Validation failed: {e}")
    print()


def example_cli_overrides():
    """Example 4: CLI overrides"""
    print("=" * 60)
    print("Example 4: CLI Overrides")
    print("=" * 60)

    cfg = Config.from_cli_args(
        "configs/base.yaml",
        ["training.epochs=500", "training.batch_size=128", "optimizer.lr=0.001"],
    )

    print("Original values:")
    original = Config("configs/base.yaml")
    print(f"  epochs: {original.training.epochs}")
    print(f"  batch_size: {original.training.batch_size}")
    print()

    print("After CLI overrides:")
    print(f"  epochs: {cfg.training.epochs}")
    print(f"  batch_size: {cfg.training.batch_size}")
    print(f"  lr: {cfg.optimizer.lr}")
    print()


def example_dump_config():
    """Example 5: Dump merged config"""
    print("=" * 60)
    print("Example 5: Dump Merged Config")
    print("=" * 60)

    cfg = Config("configs/models/gan/dcgan.yaml")

    # Dump to file
    output_path = Path("merged_config.yaml")
    cfg.dump(output_path)

    print(f"✓ Merged config saved to {output_path}")
    print(f"  File size: {output_path.stat().st_size} bytes")

    # Clean up
    output_path.unlink()
    print()


if __name__ == "__main__":
    example_basic_loading()
    example_inheritance()
    example_validation()
    example_cli_overrides()
    example_dump_config()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
