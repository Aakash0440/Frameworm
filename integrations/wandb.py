"""
Weights & Biases integration.

Automatically logs metrics, artifacts, and system info to W&B.

Example:
    >>> from frameworm.integrations import WandBIntegration
    >>> 
    >>> integration = WandBIntegration(
    ...     project='my-project',
    ...     entity='my-team',
    ...     config=config.to_dict()
    ... )
    >>> 
    >>> trainer.add_callback(integration)
    >>> trainer.train(...)  # Auto-logs to W&B
"""

from typing import Optional, Dict, Any
from training.callbacks import Callback

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class WandBIntegration(Callback):
    """
    Weights & Biases logging integration.
    
    Logs training metrics, model artifacts, system info, and more to W&B.
    
    Args:
        project: W&B project name
        entity: W&B entity (team or username)
        name: Run name (optional)
        config: Training config to log
        log_model: Whether to log model checkpoints (default: True)
        log_frequency: Log every N batches (default: None = every batch)
        tags: List of tags for the run
        notes: Run description
        
    Example:
        >>> wandb_logger = WandBIntegration(
        ...     project='frameworm-vae',
        ...     entity='my-team',
        ...     config=config.to_dict(),
        ...     tags=['vae', 'mnist']
        ... )
        >>> trainer.add_callback(wandb_logger)
    """
    
    def __init__(
        self,
        project: str,
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict] = None,
        log_model: bool = True,
        log_frequency: Optional[int] = None,
        tags: Optional[list] = None,
        notes: Optional[str] = None
    ):
        if not WANDB_AVAILABLE:
            raise ImportError(
                "wandb not installed. Install: pip install wandb"
            )
        
        self.project = project
        self.entity = entity
        self.name = name
        self.config = config
        self.log_model = log_model
        self.log_frequency = log_frequency
        self.tags = tags
        self.notes = notes
        
        self.run = None
        self._step = 0
    
    def on_train_begin(self, trainer):
        """Initialize W&B run"""
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            config=self.config,
            tags=self.tags,
            notes=self.notes,
            reinit=True
        )
        
        # Log model architecture
        if hasattr(trainer.model, '__str__'):
            wandb.config.update({'model_summary': str(trainer.model)})
        
        print(f"✓ W&B run initialized: {self.run.url}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Log epoch metrics"""
        if not self.run:
            return
        
        log_dict = {'epoch': epoch}
        
        # Add train/val metrics
        for key, value in metrics.items():
            log_dict[key] = value
        
        # Add learning rate
        if trainer.optimizer and hasattr(trainer.optimizer, 'param_groups'):
            log_dict['learning_rate'] = trainer.optimizer.param_groups[0]['lr']
        
        wandb.log(log_dict, step=self._step)
        self._step += 1
    
    def on_batch_end(self, trainer, batch_idx, loss):
        """Log batch metrics (if log_frequency set)"""
        if not self.run:
            return
        
        if self.log_frequency and (batch_idx + 1) % self.log_frequency == 0:
            wandb.log({'batch_loss': loss}, step=self._step)
            self._step += 1
    
    def on_checkpoint_save(self, trainer, path):
        """Upload model checkpoint to W&B"""
        if not self.run or not self.log_model:
            return
        
        artifact = wandb.Artifact(
            name=f'model-{self.run.id}',
            type='model',
            description=f'Model checkpoint at epoch {trainer.state.current_epoch}'
        )
        artifact.add_file(path)
        self.run.log_artifact(artifact)
        
        print(f"✓ Uploaded checkpoint to W&B: {path}")
    
    def on_train_end(self, trainer):
        """Finish W&B run"""
        if not self.run:
            return
        
        # Log final metrics summary
        final_metrics = {}
        for key in trainer.state.train_metrics:
            values = trainer.state.train_metrics[key]
            if values:
                final_metrics[f'final_{key}'] = values[-1]
        
        wandb.summary.update(final_metrics)
        
        self.run.finish()
        print("✓ W&B run finished")


# Convenience function
def setup_wandb(
    project: str,
    entity: Optional[str] = None,
    config: Optional[Dict] = None,
    **kwargs
) -> WandBIntegration:
    """
    Quick setup for W&B logging.
    
    Example:
        >>> wandb_callback = setup_wandb('my-project', config=config)
        >>> trainer.add_callback(wandb_callback)
    """
    return WandBIntegration(
        project=project,
        entity=entity,
        config=config,
        **kwargs
    )