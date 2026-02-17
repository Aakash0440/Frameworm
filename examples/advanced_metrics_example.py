"""
Example: Advanced Metrics Evaluation

Demonstrates:
- FID (Fréchet Inception Distance)
- Inception Score
- LPIPS (Perceptual Similarity)
- Automated evaluation during training
"""

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core import Config, get_model
from experiment import Experiment
from metrics import FID, LPIPS, InceptionScore, MetricEvaluator
from training import Trainer


def get_mnist_loaders(batch_size=128):
    """Get MNIST data loaders"""
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST("data", train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


def main():
    print("Advanced Metrics Example")
    print("=" * 60)

    # Config
    cfg = Config("configs/models/vae/vanilla.yaml")
    cfg.training.epochs = 20
    cfg.training.batch_size = 128
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data
    train_loader, val_loader = get_mnist_loaders(cfg.training.batch_size)

    # Get real images for evaluation
    real_images = []
    for batch, _ in val_loader:
        real_images.append(batch)
    real_images = torch.cat(real_images, dim=0)
    print(f"✓ Loaded {len(real_images)} real images for evaluation")

    # Model
    vae = get_model("vae")(cfg)
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)

    # Create experiment
    exp = Experiment(
        name="vae-with-metrics",
        config=cfg.to_dict(),
        description="VAE training with comprehensive metrics",
        tags=["vae", "mnist", "metrics"],
        root_dir="experiments",
    )

    # Create metric evaluator
    evaluator = MetricEvaluator(
        metrics=["fid", "is", "lpips"],
        real_data=real_images,
        device=cfg.training.device,
        batch_size=100,
    )

    # Trainer
    trainer = Trainer(model=vae, optimizer=optimizer, device=cfg.training.device)
    trainer.set_experiment(exp)
    trainer.set_evaluator(evaluator, eval_every=5)

    # Train
    print("\nTraining with automatic evaluation every 5 epochs...")
    with exp:
        trainer.train(train_loader, val_loader, epochs=cfg.training.epochs)

    # Final comprehensive evaluation
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)

    final_results = evaluator.evaluate(vae, num_samples=10000, save_samples="generated_samples.pt")

    print("\nFinal Metrics:")
    print(f"  FID: {final_results['fid']:.2f}")
    print(f"  IS: {final_results['is']:.2f} ± {final_results['is_std']:.2f}")
    print(f"  LPIPS: {final_results['lpips']:.4f}")

    # Save results to experiment
    for metric_name, value in final_results.items():
        exp.log_metric(f"final_{metric_name}", value, epoch=cfg.training.epochs)

    print("\n" + "=" * 60)
    print("Example complete!")
    print(f"Results saved to: {exp.exp_dir}")


if __name__ == "__main__":
    main()
