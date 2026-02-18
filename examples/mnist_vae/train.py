"""
Complete MNIST VAE training script.

Usage:
    python train.py --epochs 20 --lr 0.001
"""

import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from frameworm import Config, get_model, Trainer
from frameworm.training.callbacks import EarlyStopping, ModelCheckpoint
from frameworm.experiment import Experiment
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--latent-dim', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)
    
    # Config
    config = Config.from_dict({
        'model': {
            'type': 'vae',
            'in_channels': 1,
            'latent_dim': args.latent_dim,
            'hidden_dim': 128
        }
    })
    
    # Model
    model = get_model('vae')(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Trainer(model, optimizer, device=device)
    
    # Callbacks
    trainer.add_callback(EarlyStopping(monitor='val_loss', patience=5))
    trainer.add_callback(ModelCheckpoint(
        filepath='checkpoints/best.pt',
        monitor='val_loss',
        save_best_only=True
    ))
    
    # Track experiment
    exp = Experiment(
        name='mnist-vae',
        config=config.to_dict(),
        tags=['mnist', 'vae']
    )
    trainer.set_experiment(exp)
    
    # Train
    with exp:
        trainer.train(train_loader, test_loader, epochs=args.epochs)
    
    # Generate samples
    print("\nGenerating samples...")
    model.eval()
    with torch.no_grad():
        z = torch.randn(16, args.latent_dim).to(device)
        samples = model.decode(z)
        
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(16):
            ax = axes[i // 4, i % 4]
            ax.imshow(samples[i, 0].cpu(), cmap='gray')
            ax.axis('off')
        
        plt.savefig('generated_samples.png')
        print("✓ Saved generated_samples.png")
    
    print(f"\n✓ Training complete!")
    print(f"  Final val_loss: {trainer.state.val_metrics['loss'][-1]:.4f}")
    print(f"  Best checkpoint: checkpoints/best.pt")


if __name__ == '__main__':
    main()