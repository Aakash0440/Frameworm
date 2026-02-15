"""
Example: Advanced Training with All Features

Demonstrates:
- Gradient accumulation
- Gradient clipping
- EMA
- Mixed precision
- TensorBoard logging
- Learning rate scheduling
"""

from core import Config, get_model
from training import Trainer
from training.callbacks import CSVLogger, ModelCheckpoint
from training.schedulers import WarmupCosineScheduler
from training.loggers import TensorBoardLogger
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=2)
    
    return train_loader, val_loader


def main():
    print("Advanced Training Example")
    print("=" * 60)
    
    # Config
    cfg = Config('configs/models/vae/vanilla.yaml')
    cfg.training.epochs = 20
    cfg.training.batch_size = 128
    cfg.training.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Device: {cfg.training.device}")
    
    # Model
    vae = get_model("vae")(cfg)
    print(f"✓ Loaded VAE ({vae.count_parameters():,} parameters)")
    
    # Optimizer
    optimizer = torch.optim.Adam(vae.parameters(), lr=0.001)
    
    # Data
    train_loader, val_loader = get_mnist_loaders(cfg.training.batch_size)
    print(f"✓ Loaded data: {len(train_loader)} train, {len(val_loader)} val batches")
    
    # Trainer
    trainer = Trainer(
        model=vae,
        optimizer=optimizer,
        device=cfg.training.device,
        checkpoint_dir='checkpoints/vae_advanced',
        log_every_n_steps=50
    )
    
    # Enable advanced features
    trainer.enable_gradient_accumulation(accumulation_steps=2)
    trainer.enable_gradient_clipping(max_norm=1.0)
    trainer.enable_ema(decay=0.999)
    
    if torch.cuda.is_available():
        trainer.enable_mixed_precision()
        print("✓ Mixed precision enabled")
    
    # LR Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=2,
        total_epochs=cfg.training.epochs,
        min_lr=1e-6
    )
    trainer.set_scheduler(scheduler)
    
    # Early stopping
    trainer.set_early_stopping(patience=5, min_delta=0.001)
    
    # Callbacks
    trainer.add_callback(CSVLogger('vae_training.csv'))
    trainer.add_callback(ModelCheckpoint(
        'vae_epoch{epoch}.pt',
        monitor='val_loss',
        mode='min',
        save_best_only=False
    ))
    
    # Logger
    try:
        trainer.add_logger(TensorBoardLogger('runs/vae_advanced'))
        print("✓ TensorBoard logger added")
        print("  View with: tensorboard --logdir runs")
    except ImportError:
        print("⚠ TensorBoard not available, skipping")
    
    # Train
    print(f"\nTraining for {cfg.training.epochs} epochs with:")
    print(f"  - Gradient accumulation (2 steps)")
    print(f"  - Gradient clipping (max_norm=1.0)")
    print(f"  - EMA (decay=0.999)")
    print(f"  - Warmup cosine LR schedule")
    print(f"  - Early stopping (patience=5)")
    
    trainer.train(train_loader, val_loader, epochs=cfg.training.epochs)
    
    # Use EMA for final evaluation
    if trainer.ema:
        print("\nEvaluating with EMA weights...")
        trainer.ema.apply_shadow()
        
        # Generate samples
        with torch.no_grad():
            samples = vae.sample(16)
            print(f"✓ Generated {len(samples)} samples with EMA model")
        
        trainer.ema.restore()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best epoch: {trainer.state.best_epoch}")
    print(f"Best metric: {trainer.state.best_metric:.4f}")


if __name__ == '__main__':
    main()