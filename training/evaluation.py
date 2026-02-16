"""
Model evaluation during training.
"""

from typing import Optional, Dict, Any
import torch
from metrics.evaluator import MetricEvaluator
from training.evaluation import EvaluationCallback

class EvaluationCallback:
    """
    Callback for evaluating model during training.
    
    Args:
        evaluator: MetricEvaluator instance
        eval_every: Evaluate every N epochs
        num_samples: Number of samples for evaluation
        
    Example:
        >>> eval_callback = EvaluationCallback(
        ...     evaluator=evaluator,
        ...     eval_every=5,
        ...     num_samples=5000
        ... )
        >>> trainer.add_callback(eval_callback)
    """
    
    def __init__(
        self,
        evaluator: MetricEvaluator,
        eval_every: int = 5,
        num_samples: int = 5000
    ):
        self.evaluator = evaluator
        self.eval_every = eval_every
        self.num_samples = num_samples
        self.best_fid = float('inf')
    
    def on_epoch_end(self, epoch: int, metrics: Dict[str, float], trainer):
        """Evaluate model at end of epoch"""
        # Check if should evaluate
        if (epoch + 1) % self.eval_every != 0:
            return
        
        print(f"\n{'='*60}")
        print(f"Evaluating model (Epoch {epoch + 1})")
        print(f"{'='*60}")
        
        # Evaluate
        results = self.evaluator.evaluate(
            trainer.model,
            num_samples=self.num_samples
        )
        
        # Log to experiment if available
        if trainer.experiment:
            for metric_name, value in results.items():
                trainer.experiment.log_metric(
                    f"eval_{metric_name}",
                    value,
                    epoch=epoch,
                    metric_type='eval'
                )
        
        # Track best FID
        if 'fid' in results:
            if results['fid'] < self.best_fid:
                self.best_fid = results['fid']
                print(f"\n🎉 New best FID: {self.best_fid:.2f}")
        
        print(f"{'='*60}\n")