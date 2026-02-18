"""
MLflow integration for experiment tracking.

Example:
    >>> from frameworm.integrations import MLflowIntegration
    >>> 
    >>> integration = MLflowIntegration(
    ...     experiment_name='my-experiment',
    ...     tracking_uri='http://localhost:5000'
    ... )
    >>> trainer.add_callback(integration)
"""

from typing import Optional, Dict, Any
from training.callbacks import Callback

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


class MLflowIntegration(Callback):
    """
    MLflow experiment tracking integration.
    
    Logs metrics, parameters, and artifacts to MLflow.
    
    Args:
        experiment_name: MLflow experiment name
        tracking_uri: MLflow tracking server URI (default: local)
        run_name: Name for this run
        tags: Dict of tags
        log_model: Whether to log model (default: True)
        
    Example:
        >>> mlflow_logger = MLflowIntegration(
        ...     experiment_name='vae-training',
        ...     tracking_uri='http://mlflow-server:5000'
        ... )
        >>> trainer.add_callback(mlflow_logger)
    """
    
    def __init__(
        self,
        experiment_name: str,
        tracking_uri: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        log_model: bool = True
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError(
                "mlflow not installed. Install: pip install mlflow"
            )
        
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = tags or {}
        self.log_model = log_model
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        self._run = None
        self._step = 0
    
    def on_train_begin(self, trainer):
        """Start MLflow run"""
        self._run = mlflow.start_run(run_name=self.run_name)
        
        # Log tags
        for key, value in self.tags.items():
            mlflow.set_tag(key, value)
        
        # Log parameters from config if available
        if hasattr(trainer, 'config'):
            config_dict = trainer.config.to_dict() if hasattr(trainer.config, 'to_dict') else {}
            mlflow.log_params(self._flatten_dict(config_dict))
        
        print(f"✓ MLflow run started: {self._run.info.run_id}")
    
    def _flatten_dict(self, d: Dict, parent_key: str = '', sep: str = '.') -> Dict:
        """Flatten nested dict for MLflow params"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def on_epoch_end(self, trainer, epoch, metrics):
        """Log metrics"""
        if not self._run:
            return
        
        # Log all metrics
        mlflow.log_metrics(metrics, step=epoch)
        
        # Log learning rate
        if trainer.optimizer and hasattr(trainer.optimizer, 'param_groups'):
            lr = trainer.optimizer.param_groups[0]['lr']
            mlflow.log_metric('learning_rate', lr, step=epoch)
    
    def on_checkpoint_save(self, trainer, path):
        """Log model artifact"""
        if not self._run or not self.log_model:
            return
        
        mlflow.log_artifact(path)
        print(f"✓ Logged checkpoint to MLflow: {path}")
    
    def on_train_end(self, trainer):
        """End MLflow run"""
        if self._run:
            mlflow.end_run()
            print("✓ MLflow run ended")


def setup_mlflow(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    **kwargs
) -> MLflowIntegration:
    """Quick MLflow setup"""
    return MLflowIntegration(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        **kwargs
    )
