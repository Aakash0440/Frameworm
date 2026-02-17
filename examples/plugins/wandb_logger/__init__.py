"""
W&B integration plugin.

Automatically logs training metrics to Weights & Biases.
"""

from plugins.hooks import CallbackHook


class WandBLogger(CallbackHook):
    """
    Weights & Biases logger callback.
    
    Logs all training metrics to W&B dashboard.
    
    Args:
        project: W&B project name
        entity: W&B entity (username/team)
        config: Training config to log
    """
    
    def __init__(self, project: str, entity: str = None, config: dict = None):
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            print("⚠️  wandb not installed. Install: pip install wandb")
            self.wandb = None
        
        self.project = project
        self.entity = entity
        self.config = config
        self.run = None
    
    def on_train_begin(self, trainer):
        """Initialize W&B run"""
        if not self.wandb:
            return
        
        self.run = self.wandb.init(
            project=self.project,
            entity=self.entity,
            config=self.config
        )
        print(f"✓ W&B run started: {self.run.url}")
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Log metrics after each epoch"""
        if not self.wandb or not self.run:
            return
        
        # Log all metrics
        log_dict = {'epoch': epoch}
        log_dict.update(metrics)
        self.wandb.log(log_dict)
    
    def on_train_end(self, trainer):
        """Finish W&B run"""
        if not self.wandb or not self.run:
            return
        
        self.run.finish()
        print("✓ W&B run finished")


def register():
    """Plugin entry point"""
    print("✓ W&B logger plugin loaded (use WandBLogger callback in your training)")
    print("  Example: from wandb_logger import WandBLogger")
    print("           logger = WandBLogger(project='my-project')")
    print("           logger.register()")