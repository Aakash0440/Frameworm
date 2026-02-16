"""
Example: Complete Experiment Tracking Workflow

Demonstrates:
- Creating experiments
- Tracking metrics
- Comparing experiments
- Visualizing results
"""

from core import Config, get_model
from training.trainer import Trainer
from experiment import Experiment, ExperimentManager
from experiment.visualization import plot_metric_comparison
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_mnist_loaders(batch_size=128):
    """Get MNIST data loaders"""
    transform = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    return train_loader, val_loader


def run_experiment(name, config_updates, tags):
    """Run a single experiment"""
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"{'='*60}")
    
    # Load config
    cfg = Config('configs/models/vae/vanilla.yaml')
    cfg.training.epochs = 10
    cfg.training.batch_size = 128
    
    # Apply updates
    for key, value in config_updates.items():
        parts = key.split('.')
        obj = cfg
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)
    
    # Create experiment
    exp = Experiment(
        name=name,
        config=cfg.to_dict(),
        description=f"VAE experiment with {config_updates}",
        tags=tags,
        root_dir='experiments'
    )
    
    # Model
    vae = get_model("vae")(cfg)
    optimizer = torch.optim.Adam(vae.parameters(), lr=cfg.training.get('lr', 0.001))
    
    # Data
    train_loader, val_loader = get_mnist_loaders(cfg.training.batch_size)
    
    # Trainer
    trainer = Trainer(
        model=vae,
        optimizer=optimizer,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    trainer.set_experiment(exp)
    
    # Train
    with exp:
        trainer.train(train_loader, val_loader, epochs=cfg.training.epochs)
    
    print(f"✓ Experiment complete: {exp.experiment_id}")
    return exp.experiment_id


def main():
    print("Complete Experiment Tracking Example")
    print("=" * 60)
    
    # Run multiple experiments with different hyperparameters
    experiments = []
    
    # Experiment 1: Baseline
    exp_id = run_experiment(
        name="vae-baseline",
        config_updates={},
        tags=["vae", "mnist", "baseline"]
    )
    experiments.append(exp_id)
    
    # Experiment 2: Higher learning rate
    exp_id = run_experiment(
        name="vae-high-lr",
        config_updates={'training.lr': 0.002},
        tags=["vae", "mnist", "high-lr"]
    )
    experiments.append(exp_id)
    
    # Experiment 3: Beta-VAE
    exp_id = run_experiment(
        name="vae-beta",
        config_updates={'model.beta': 4.0},
        tags=["vae", "mnist", "beta-vae"]
    )
    experiments.append(exp_id)
    
    # Analyze experiments
    print("\n" + "=" * 60)
    print("EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    manager = ExperimentManager('experiments')
    
    # List all experiments
    print("\nAll Experiments:")
    df = manager.list_experiments(tags=['vae'], limit=10)
    print(df[['name', 'status', 'created_at']])
    
    # Compare experiments
    print("\nComparison:")
    comparison = manager.compare_experiments(experiments, metrics=['loss', 'val_loss'])
    print(comparison[['name', 'loss', 'val_loss']])
    
    # Plot comparison
    print("\nGenerating plots...")
    plot_metric_comparison(
        manager,
        experiments,
        'loss',
        save_path='vae_loss_comparison.png'
    )
    print("✓ Saved: vae_loss_comparison.png")
    
    plot_metric_comparison(
        manager,
        experiments,
        'val_loss',
        save_path='vae_val_loss_comparison.png'
    )
    print("✓ Saved: vae_val_loss_comparison.png")
    
    # Summary
    print("\n" + "=" * 60)
    print("Experiment tracking complete!")
    print(f"Total experiments run: {len(experiments)}")
    print(f"Results saved to: experiments/")
    print("=" * 60)


if __name__ == '__main__':
    main()