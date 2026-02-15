"""
Main Trainer class for training models.
"""

from typing import Optional, Dict, Any, Callable
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from training.callbacks import CallbackList, Callback
from training.state import TrainingState
from training.metrics import MetricsTracker, ProgressLogger
from core.exceptions import FramewormError


class TrainingError(FramewormError):
    """Raised when training fails"""
    pass


class Trainer:
    """
    Main trainer for model training.
    
    Handles training loop, validation, checkpointing, and scheduling.
    
    Args:
        model: Model to train
        optimizer: Optimizer
        criterion: Loss function (if not using model.compute_loss)
        device: Device to train on
        
    Example:
        >>> trainer = Trainer(model, optimizer, criterion, device='cuda')
        >>> trainer.train(train_loader, val_loader, epochs=100)
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[Callable] = None,
        device: str = 'cuda',
        checkpoint_dir: str = 'checkpoints',
        log_every_n_steps: int = 100
        
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = CallbackList()
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.state = TrainingState()
        
        # Metrics
        self.train_tracker = MetricsTracker()
        self.val_tracker = MetricsTracker()
        self.logger = ProgressLogger(total_epochs=0, log_every_n_steps=log_every_n_steps)
        
        # Optional components
        self.scheduler: Optional[_LRScheduler] = None
        self.early_stopping_patience: Optional[int] = None
        self.early_stopping_min_delta: float = 0.0
    
    def set_scheduler(self, scheduler: _LRScheduler):
        """Set learning rate scheduler"""
        self.scheduler = scheduler

    def add_callback(self, callback: Callback):
        """Add a callback"""
        self.callbacks.append(callback)
    
    def set_early_stopping(self, patience: int, min_delta: float = 0.0):
        """
        Enable early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.early_stopping_patience = patience
        self.early_stopping_min_delta = min_delta
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from: Optional[str] = None
    ):
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            resume_from: Path to checkpoint to resume from
        """
        # Resume if requested
        if resume_from:
            self.load_checkpoint(resume_from)
        
        self.logger.total_epochs = epochs
        self.callbacks.on_train_begin(self)
        try:
            for epoch in range(self.state.current_epoch, epochs):
                self.state.current_epoch = epoch
                self.callbacks.on_epoch_begin(epoch, self)
                # Log epoch start
                self.logger.log_epoch_start(epoch + 1)
                
                # Training phase
                train_metrics = self.train_epoch(train_loader, epoch)
                self.state.update_train_metrics(train_metrics)
                
                # Validation phase
                val_metrics = None
                if val_loader is not None:
                    val_metrics = self.validate_epoch(val_loader, epoch)
                    self.state.update_val_metrics(val_metrics)
                
                # Log epoch end
                self.logger.log_epoch_end(epoch + 1, train_metrics, val_metrics)
                self.callbacks.on_epoch_end(epoch, train_metrics, self)
                # Check if best epoch
                if val_metrics and 'loss' in val_metrics:
                    is_best = self.state.is_best_epoch(val_metrics['loss'], mode='min')
                    
                    if is_best:
                        self.save_checkpoint('best.pt', epoch)
                        print(f"  → Saved best model (val_loss: {val_metrics['loss']:.4f})")
                
                # Save latest checkpoint
                self.save_checkpoint('latest.pt', epoch)
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_metrics and 'loss' in val_metrics:
                            self.scheduler.step(val_metrics['loss'])
                    else:
                        self.scheduler.step()
                    
                    # Track LR
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.state.lr_history.append(current_lr)
                
                # Early stopping
                if self._should_stop_early():
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break
        
        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_checkpoint('interrupted.pt', epoch)
        
        except Exception as e:
            print(f"\n\nTraining failed: {e}")
            self.save_checkpoint('failed.pt', epoch)
            raise TrainingError(f"Training failed: {e}")
        
        finally:
            # Log training end
            self.logger.log_training_end(self.state)
            
            # Save final state
            self.state.save(self.checkpoint_dir / 'training_state.json')
            self.callbacks.on_train_end(self)
    def train_epoch(
        self,
        train_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of averaged metrics
        """
        self.model.train()
        self.train_tracker.epoch_start()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            self.callbacks.on_batch_begin(batch_idx, self)
            # Forward pass
            if hasattr(self.model, 'compute_loss'):
                # Model has its own loss computation
                if isinstance(batch, (list, tuple)) and len(batch) == 1:
                    loss_dict = self.model.compute_loss(batch[0])
                else:
                    loss_dict = self.model.compute_loss(*batch if isinstance(batch, (list, tuple)) else batch)
                
                loss = loss_dict['loss']
                metrics = loss_dict
            else:
                # Use provided criterion
                if self.criterion is None:
                    raise ValueError("Model doesn't have compute_loss() and no criterion provided")
                
                # Unpack batch
                if isinstance(batch, (list, tuple)):
                    *inputs, targets = batch
                    outputs = self.model(*inputs)
                else:
                    outputs = self.model(batch)
                    targets = batch
                
                loss = self.criterion(outputs, targets)
                metrics = {'loss': loss}
            
            # Backward pass
            loss.backward()
            
            # Update weights
            self.optimizer.step()
            self.callbacks.on_batch_end(batch_idx, metrics, self)
            # Update metrics
            self.train_tracker.update(metrics)
            
            # Log batch
            self.logger.log_batch(
                epoch + 1,
                batch_idx,
                len(train_loader),
                metrics
            )
            
            # Update global step
            self.state.global_step += 1
        
        # Compute epoch metrics
        return self.train_tracker.epoch_end()
    
    def validate_epoch(
        self,
        val_loader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            epoch: Current epoch number
            
        Returns:
            Dictionary of averaged metrics
        """
        self.model.eval()
        self.val_tracker.epoch_start()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # Move batch to device
                if isinstance(batch, (list, tuple)):
                    batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                elif isinstance(batch, torch.Tensor):
                    batch = batch.to(self.device)
                
                # Forward pass
                if hasattr(self.model, 'compute_loss'):
                    if isinstance(batch, (list, tuple)) and len(batch) == 1:
                        loss_dict = self.model.compute_loss(batch[0])
                    else:
                        loss_dict = self.model.compute_loss(*batch if isinstance(batch, (list, tuple)) else batch)
                    metrics = loss_dict
                else:
                    if isinstance(batch, (list, tuple)):
                        *inputs, targets = batch
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(batch)
                        targets = batch
                    
                    loss = self.criterion(outputs, targets)
                    metrics = {'loss': loss}
                
                # Update metrics
                self.val_tracker.update(metrics)
        
        # Compute epoch metrics
        return self.val_tracker.epoch_end()
    
    def save_checkpoint(self, filename: str, epoch: int):
        """
        Save checkpoint.
        
        Args:
            filename: Checkpoint filename
            epoch: Current epoch
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.state.to_dict(),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.state = TrainingState.from_dict(checkpoint['training_state'])
        
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        print(f"Resumed from epoch {self.state.current_epoch}")
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        if self.early_stopping_patience is None:
            return False
        
        return self.state.patience_counter >= self.early_stopping_patience