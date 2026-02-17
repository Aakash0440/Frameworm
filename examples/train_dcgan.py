"""
Example: Training DCGAN on MNIST

Demonstrates complete training workflow with FRAMEWORM.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from core import Config, get_model
from core.config import Config
from training.callbacks import CSVLogger, ModelCheckpoint
from training.schedulers import WarmupCosineScheduler
from training.trainer import Trainer


def get_dataloaders(batch_size=128):
    """Create MNIST dataloaders"""
    transform = transforms.Compose(
        [transforms.Resize(64), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    train_dataset = datasets.MNIST("data", train=True, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    return train_loader


class GANTrainer(Trainer):
    """
    Custom trainer for GAN training.

    Handles alternating generator and discriminator updates.
    """

    def __init__(self, generator, discriminator, g_optimizer, d_optimizer, **kwargs):
        # Use generator as main model
        super().__init__(generator, g_optimizer, **kwargs)

        self.generator = generator
        self.discriminator = discriminator
        self.d_optimizer = d_optimizer
        self.criterion = nn.BCELoss()

    def train_epoch(self, train_loader, epoch):
        """Train GAN for one epoch"""
        self.generator.train()
        self.discriminator.train()
        self.train_tracker.epoch_start()

        for batch_idx, (real_images, _) in enumerate(train_loader):
            batch_size = real_images.size(0)
            real_images = real_images.to(self.device)

            # Labels
            real_labels = torch.ones(batch_size, device=self.device)
            fake_labels = torch.zeros(batch_size, device=self.device)

            # ==================
            # Train Discriminator
            # ==================
            self.d_optimizer.zero_grad()

            # Real images
            real_output = self.discriminator.discriminate(real_images)
            d_real_loss = self.criterion(real_output, real_labels)

            # Fake images
            z = self.generator.sample_z(batch_size)
            fake_images = self.generator(z)
            fake_output = self.discriminator.discriminate(fake_images.detach())
            d_fake_loss = self.criterion(fake_output, fake_labels)

            # Total discriminator loss
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            self.d_optimizer.step()

            # ==================
            # Train Generator
            # ==================
            self.optimizer.zero_grad()

            # Generate fake images
            z = self.generator.sample_z(batch_size)
            fake_images = self.generator(z)
            fake_output = self.discriminator.discriminate(fake_images)

            # Generator tries to fool discriminator
            g_loss = self.criterion(fake_output, real_labels)
            g_loss.backward()
            self.optimizer.step()

            # Update metrics
            metrics = {
                "g_loss": g_loss.item(),
                "d_loss": d_loss.item(),
                "d_real": real_output.mean().item(),
                "d_fake": fake_output.mean().item(),
            }
            self.train_tracker.update(metrics)

            # Log
            self.logger.log_batch(epoch + 1, batch_idx, len(train_loader), metrics)
            self.state.global_step += 1

        return self.train_tracker.epoch_end()


def main():
    print("DCGAN Training Example")
    print("=" * 60)

    # Config
    cfg = Config("configs/models/gan/dcgan.yaml")
    cfg.training.batch_size = 128
    cfg.training.epochs = 50
    cfg.training.device = "cuda" if torch.cuda.is_available() else "cpu"

    # Model
    dcgan = get_model("dcgan")(cfg)
    print(f"✓ Loaded DCGAN")

    # Optimizers
    g_optimizer = torch.optim.Adam(dcgan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    d_optimizer = torch.optim.Adam(dcgan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Data
    train_loader = get_dataloaders(cfg.training.batch_size)
    print(f"✓ Loaded data: {len(train_loader)} batches")

    # Trainer
    trainer = GANTrainer(
        generator=dcgan,
        discriminator=dcgan,
        g_optimizer=g_optimizer,
        d_optimizer=d_optimizer,
        device=cfg.training.device,
        checkpoint_dir="checkpoints/dcgan",
    )

    # Callbacks
    trainer.add_callback(CSVLogger("dcgan_training.csv"))
    trainer.add_callback(ModelCheckpoint("dcgan_best.pt", monitor="g_loss", mode="min"))

    # Train
    print(f"\nTraining for {cfg.training.epochs} epochs...")
    trainer.train(train_loader, epochs=cfg.training.epochs)

    print("\n" + "=" * 60)
    print("Training complete!")


if __name__ == "__main__":
    main()
