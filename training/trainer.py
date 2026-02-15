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
from training.loggers import LoggerList, Logger
from training.advanced import (
    GradientAccumulator,
    GradientClipper,
    EMAModel,
    compute_gradient_norm
)

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
        self.loggers = LoggerList()
        # Move model to device
        self.model.to(self.device)
        self.gradient_accumulator: Optional[GradientAccumulator] = None
        self.gradient_clipper: Optional[GradientClipper] = None
        self.ema: Optional[EMAModel] = None
        self.mixed_precision: bool = False
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
        
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

    def add_logger(self, logger: Logger):
        """Add experiment logger"""
        self.loggers.append(logger)

    def enable_gradient_accumulation(self, accumulation_steps: int):
        """Enable gradient accumulation"""
        self.gradient_accumulator = GradientAccumulator(accumulation_steps)
    
    def enable_gradient_clipping(self, max_norm: float = 1.0):
        """Enable gradient clipping"""
        self.gradient_clipper = GradientClipper(max_norm)
    
    def enable_ema(self, decay: float = 0.999):
        """Enable exponential moving average"""
        self.ema = EMAModel(self.model, decay)
    
    def enable_mixed_precision(self):
        """Enable mixed precision training (FP16)"""
        if not torch.cuda.is_available():
            print("Warning: Mixed precision requires CUDA, skipping")
            return
        
        self.mixed_precision = True
        self.grad_scaler = torch.cuda.amp.GradScaler()

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
                combined_metrics = {f'train_{k}': v for k, v in train_metrics.items()}
                if val_metrics:
                    combined_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
            
                self.loggers.log_scalars(combined_metrics, self.state.global_step)
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
                self.state.current_epoch = epoch + 1
        
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
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch with advanced features"""
        self.model.train()
        self.train_tracker.epoch_start()
        
        # Reset gradient accumulator
        if self.gradient_accumulator:
            self.gradient_accumulator.reset()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
            elif isinstance(batch, torch.Tensor):
                batch = batch.to(self.device)
            
            # Callbacks
            self.callbacks.on_batch_begin(batch_idx, self)
            
            # Forward pass with optional mixed precision
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_loss(batch)
                    loss = loss_dict['loss']
                    
                    # Scale loss for gradient accumulation
                    if self.gradient_accumulator:
                        loss = self.gradient_accumulator.scale_loss(loss)
                
                # Backward with gradient scaling
                self.grad_scaler.scale(loss).backward()
                
                # Gradient clipping (unscale first)
                if self.gradient_clipper:
                    self.grad_scaler.unscale_(self.optimizer)
                    grad_norm = self.gradient_clipper.clip(self.model.parameters())
                    loss_dict['grad_norm'] = grad_norm
                
                # Optimizer step
                should_update = (
                    not self.gradient_accumulator or
                    self.gradient_accumulator.should_update()
                )
                
                if should_update:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()
                    
                    # Update EMA
                    if self.ema:
                        self.ema.update()
            else:
                # Standard training
                self.optimizer.zero_grad()
                
                loss_dict = self._compute_loss(batch)
                loss = loss_dict['loss']
                
                # Scale loss for gradient accumulation
                if self.gradient_accumulator:
                    loss = self.gradient_accumulator.scale_loss(loss)
                
                # Backward
                loss.backward()
                
                # Gradient clipping
                if self.gradient_clipper:
                    grad_norm = self.gradient_clipper.clip(self.model.parameters())
                    loss_dict['grad_norm'] = grad_norm
                
                # Optimizer step
                should_update = (
                    not self.gradient_accumulator or
                    self.gradient_accumulator.should_update()
                )
                
                if should_update:
                    self.optimizer.step()
                    
                    # Update EMA
                    if self.ema:
                        self.ema.update()
            
            # Update metrics
            self.train_tracker.update(loss_dict)
            
            # Callbacks
            self.callbacks.on_batch_end(batch_idx, loss_dict, self)
            
            # Log batch
            self.logger.log_batch(epoch + 1, batch_idx, len(train_loader), loss_dict)
            
            # Update global step
            self.state.global_step += 1
        
        return self.train_tracker.epoch_end()

    def _compute_loss(self, batch):
        """Compute loss for batch"""
        if hasattr(self.model, 'compute_loss'):
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                return self.model.compute_loss(batch[0])
            else:
                return self.model.compute_loss(*batch if isinstance(batch, (list, tuple)) else batch)
        else:
            if self.criterion is None:
                raise ValueError("Model doesn't have compute_loss() and no criterion provided")
            
            if isinstance(batch, (list, tuple)):
                *inputs, targets = batch
                outputs = self.model(*inputs)
            else:
                outputs = self.model(batch)
                targets = batch
            
            loss = self.criterion(outputs, targets)
            return {'loss': loss}
    
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
    
    def save_checkpoint(self, filename, epoch):
        """Save checkpoint with EMA if enabled"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.state.to_dict(),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.grad_scaler:
            checkpoint['grad_scaler_state_dict'] = self.grad_scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)

    def save_checkpoint(self, filename, epoch):
        """Save checkpoint with EMA if enabled"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_state': self.state.to_dict(),
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if self.ema:
            checkpoint['ema_state_dict'] = self.ema.state_dict()
        
        if self.grad_scaler:
            checkpoint['grad_scaler_state_dict'] = self.grad_scaler.state_dict()
        
        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
    
    def _should_stop_early(self) -> bool:
        """Check if training should stop early"""
        if self.early_stopping_patience is None:
            return False
        
        return self.state.patience_counter >= self.early_stopping_patience