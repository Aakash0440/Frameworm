"""
Training command implementation.
"""

import torch
from pathlib import Path
from core import Config, get_model
from training import Trainer
from experiment import Experiment
from click import echo, progressbar


def run_training(
    config_path: str,
    gpu_ids: list = None,
    experiment_name: str = None,
    resume_from: str = None,
    debug: bool = False,
):
    """
    Run training from CLI.

    Args:
        config_path: Path to config file
        gpu_ids: List of GPU IDs to use
        experiment_name: Name for experiment
        resume_from: Checkpoint to resume from
        debug: Enable debug mode
    """
    # Load config
    echo("Loading configuration...")
    config = Config(config_path)

    # Device setup
    if gpu_ids is not None:
        device = f"cuda:{gpu_ids[0]}" if len(gpu_ids) > 0 else "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    echo(f"Device: {device}")

    # Create model
    echo("Creating model...")
    model = get_model(config.model.type)(config)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.get("lr", 0.001))

    # Create experiment
    if experiment_name:
        echo(f"Creating experiment: {experiment_name}")
        experiment = Experiment(
            name=experiment_name, config=config, tags=["cli", config.model.type]
        )
    else:
        experiment = None

    # Trainer
    trainer = Trainer(model=model, optimizer=optimizer, device=device)

    # Multi-GPU
    if gpu_ids and len(gpu_ids) > 1:
        echo(f"Enabling multi-GPU: {gpu_ids}")
        trainer.enable_data_parallel(device_ids=gpu_ids)

    # Set experiment
    if experiment:
        trainer.set_experiment(experiment)

    # Load data (you would implement this based on config)
    echo("Loading data...")
    # train_loader, val_loader = load_data(config)

    # Resume if checkpoint provided
    if resume_from:
        echo(f"Resuming from: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Train
    echo("\nStarting training...")
    echo("=" * 60)

    if experiment:
        with experiment:
            trainer.train(
                train_loader=None,  # Placeholder
                val_loader=None,  # Placeholder
                epochs=config.training.epochs,
            )
    else:
        trainer.train(train_loader=None, val_loader=None, epochs=config.training.epochs)

    echo("=" * 60)
    echo("✓ Training complete!")

    # Save final model
    final_path = Path("checkpoints") / "final.pt"
    final_path.parent.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config.to_dict(),
        },
        final_path,
    )

    echo(f"✓ Model saved: {final_path}")
