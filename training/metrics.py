"""
Metrics tracking and logging.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict
import time


class MetricsTracker:
    """
    Track and aggregate metrics during training.
    
    Handles both batch-level and epoch-level metrics.
    """
    
    def __init__(self):
        self.batch_metrics: Dict[str, List[float]] = defaultdict(list)
        self.epoch_metrics: Dict[str, List[float]] = defaultdict(list)
        self.epoch_start_time: Optional[float] = None
    
    def update(self, metrics: Dict[str, float]):
        """
        Update batch metrics.
        
        Args:
            metrics: Dictionary of metric name -> value
        """
        for name, value in metrics.items():
            # Handle tensors
            if hasattr(value, 'item'):
                value = value.item()
            self.batch_metrics[name].append(value)
    
    def epoch_start(self):
        """Mark start of epoch"""
        self.batch_metrics.clear()
        self.epoch_start_time = time.time()
    
    def epoch_end(self) -> Dict[str, float]:
        """
        Compute epoch metrics.
        
        Returns:
            Dictionary of averaged metrics
        """
        epoch_metrics = {}
        
        for name, values in self.batch_metrics.items():
            if values:
                avg_value = sum(values) / len(values)
                epoch_metrics[name] = avg_value
                self.epoch_metrics[name].append(avg_value)
        
        # Add epoch duration
        if self.epoch_start_time:
            duration = time.time() - self.epoch_start_time
            epoch_metrics['epoch_time'] = duration
        
        return epoch_metrics
    
    def get_latest(self, metric_name: str) -> Optional[float]:
        """Get latest value of a metric"""
        if metric_name in self.epoch_metrics and self.epoch_metrics[metric_name]:
            return self.epoch_metrics[metric_name][-1]
        return None
    
    def get_history(self, metric_name: str) -> List[float]:
        """Get full history of a metric"""
        return self.epoch_metrics.get(metric_name, [])
    
    def reset(self):
        """Reset all metrics"""
        self.batch_metrics.clear()
        self.epoch_metrics.clear()
        self.epoch_start_time = None


class ProgressLogger:
    """
    Log training progress to console.
    """
    
    def __init__(self, total_epochs: int, log_every_n_steps: int = 100):
        self.total_epochs = total_epochs
        self.log_every_n_steps = log_every_n_steps
    
    def log_epoch_start(self, epoch: int):
        """Log epoch start"""
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{self.total_epochs}")
        print(f"{'='*60}")
    
    def log_batch(
        self,
        epoch: int,
        batch_idx: int,
        total_batches: int,
        metrics: Dict[str, float]
    ):
        """Log batch progress"""
        if batch_idx % self.log_every_n_steps != 0 and batch_idx != total_batches - 1:
            return
        
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(f"  [{batch_idx+1}/{total_batches}] {metrics_str}")
    
    def log_epoch_end(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log epoch end"""
        train_str = " | ".join([f"{k}: {v:.4f}" for k, v in train_metrics.items()])
        print(f"\nTrain: {train_str}")
        
        if val_metrics:
            val_str = " | ".join([f"{k}: {v:.4f}" for k, v in val_metrics.items()])
            print(f"Val:   {val_str}")
    
    def log_training_end(self, state: 'TrainingState'):
        """Log training completion"""
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"Best epoch: {state.best_epoch}")
        print(f"Best metric: {state.best_metric:.4f}")
        print(f"Total epochs: {state.current_epoch}")
        print(f"Total steps: {state.global_step}")
        print(f"{'='*60}\n")
