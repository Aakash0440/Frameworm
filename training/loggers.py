"""
Experiment logging integrations.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import torch


class Logger:
    """
    Base logger class.
    """

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log scalar values"""
        pass

    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images"""
        pass

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram"""
        pass

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters"""
        pass

    def close(self):
        """Close logger"""
        pass


class TensorBoardLogger(Logger):
    """
    TensorBoard logger.

    Args:
        log_dir: Directory for TensorBoard logs
    """

    def __init__(self, log_dir: str = "runs"):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            raise ImportError("TensorBoard not available. Install with: pip install tensorboard")

        self.writer = SummaryWriter(log_dir)
        self.log_dir = Path(log_dir)

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log scalar values"""
        for name, value in scalars.items():
            self.writer.add_scalar(name, value, step)

    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """
        Log images.

        Args:
            images: Dictionary of name -> image tensor (C, H, W) or (B, C, H, W)
            step: Global step
        """
        for name, img in images.items():
            # Handle batch of images
            if img.dim() == 4:
                self.writer.add_images(name, img, step)
            else:
                self.writer.add_image(name, img, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram of values"""
        self.writer.add_histogram(tag, values, step)

    def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float]):
        """Log hyperparameters and metrics"""
        self.writer.add_hparams(hparams, metrics)

    def log_graph(self, model: torch.nn.Module, input_tensor: torch.Tensor):
        """Log model graph"""
        self.writer.add_graph(model, input_tensor)

    def close(self):
        """Close writer"""
        self.writer.close()


class WandbLogger(Logger):
    """
    Weights & Biases logger.

    Args:
        project: W&B project name
        name: Run name
        config: Configuration dictionary
    """

    def __init__(self, project: str, name: Optional[str] = None, config: Optional[Dict] = None):
        try:
            import wandb
        except ImportError:
            raise ImportError("wandb not available. Install with: pip install wandb")

        self.wandb = wandb
        self.run = wandb.init(project=project, name=name, config=config)

    def log_scalars(self, scalars: Dict[str, float], step: int):
        """Log scalar values"""
        self.wandb.log(scalars, step=step)

    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        """Log images"""
        wandb_images = {}
        for name, img in images.items():
            # Convert to numpy and handle batch
            if img.dim() == 4:
                # Log first image in batch
                img = img[0]

            # Convert to numpy (H, W, C) format
            img_np = img.detach().cpu().numpy()
            if img_np.shape[0] in [1, 3]:  # (C, H, W)
                img_np = img_np.transpose(1, 2, 0)

            wandb_images[name] = self.wandb.Image(img_np)

        self.wandb.log(wandb_images, step=step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        """Log histogram"""
        self.wandb.log({tag: self.wandb.Histogram(values.detach().cpu().numpy())}, step=step)

    def close(self):
        """Finish run"""
        self.wandb.finish()


class LoggerList:
    """
    Container for multiple loggers.
    """

    def __init__(self, loggers=None):
        self.loggers = loggers or []

    def append(self, logger: Logger):
        """Add logger"""
        self.loggers.append(logger)

    def log_scalars(self, scalars: Dict[str, float], step: int):
        for logger in self.loggers:
            logger.log_scalars(scalars, step)

    def log_images(self, images: Dict[str, torch.Tensor], step: int):
        for logger in self.loggers:
            logger.log_images(images, step)

    def log_histogram(self, tag: str, values: torch.Tensor, step: int):
        for logger in self.loggers:
            logger.log_histogram(tag, step)

    def close_all(self):
        for logger in self.loggers:
            logger.close()
