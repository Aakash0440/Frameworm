"""
Main Trainer class for training models.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from core.exceptions import FramewormError
from training.advanced import EMAModel, GradientAccumulator, GradientClipper
from training.callbacks import Callback, CallbackList
from training.loggers import Logger, LoggerList
from training.metrics import MetricsTracker, ProgressLogger
from training.state import TrainingState

if TYPE_CHECKING:
    from experiment import Experiment

from metrics.evaluator import MetricEvaluator


class TrainingError(FramewormError):
    pass


class Trainer:
    """Trainer for PyTorch models with callbacks, EMA, mixed precision, and checkpointing."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: Optional[Callable] = None,
        device: str = "cuda",
        checkpoint_dir: str = "checkpoints",
        gradient_accumulation_steps: int = 1,
        log_every_n_steps: int = 100,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.callbacks = CallbackList()
        self.loggers = LoggerList()
        self.experiment: Optional["Experiment"] = None
        self.model.to(self.device)

        self.gradient_accumulator: Optional[GradientAccumulator] = None
        self.gradient_clipper: Optional[GradientClipper] = None
        self.ema: Optional[EMAModel] = None
        self.mixed_precision = False
        self.grad_scaler: Optional[torch.cuda.amp.GradScaler] = None
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Training state
        self.state = TrainingState()
        self.train_tracker = MetricsTracker()
        self.val_tracker = MetricsTracker()
        self.logger = ProgressLogger(total_epochs=0, log_every_n_steps=log_every_n_steps)
        self.scheduler: Optional[_LRScheduler] = None
        self.early_stopping_patience: Optional[int] = None
        self.early_stopping_min_delta: float = 0.0

    # ------------------ Setup Methods ------------------
    def set_scheduler(self, scheduler: _LRScheduler):
        self.scheduler = scheduler

    def enable_gradient_accumulation(self, accumulation_steps: int):
        self.gradient_accumulation_steps = accumulation_steps
        self.gradient_accumulator = GradientAccumulator(accumulation_steps)

    def enable_gradient_clipping(self, max_norm: float = 1.0):
        self.gradient_clipper = GradientClipper(max_norm)

    def enable_ema(self, decay: float = 0.999):
        self.ema = EMAModel(self.model, decay)

    def enable_mixed_precision(self):
        if not torch.cuda.is_available():
            print("Warning: Mixed precision requires CUDA, skipping")
            return
        self.mixed_precision = True
        self.grad_scaler = torch.cuda.amp.GradScaler()

    def set_experiment(self, experiment: "Experiment"):
        self.experiment = experiment
        if experiment:
            self.checkpoint_dir = experiment.checkpoint_dir

    def set_early_stopping(self, patience: int, min_delta: float = 0.0):
        self.early_stopping_patience = patience
        self.early_stopping_min_delta = min_delta

    def add_callback(self, callback: Callback):
        self.callbacks.append(callback)

    def add_logger(self, logger: Logger):
        self.loggers.append(logger)

    def set_evaluator(self, evaluator: "MetricEvaluator", eval_every: int = 5):
        from training.evaluation import EvaluationCallback

        callback = EvaluationCallback(evaluator=evaluator, eval_every=eval_every, num_samples=5000)
        self.add_callback(callback)

    # ------------------ Training Loop ------------------
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 100,
        resume_from: Optional[str] = None,
    ):
        if resume_from:
            self.load_checkpoint(resume_from)
        self.logger.total_epochs = epochs
        self.callbacks.on_train_begin(self)

        try:
            for epoch in range(self.state.current_epoch, epochs):
                self.state.current_epoch = epoch
                self.callbacks.on_epoch_begin(epoch, self)
                self.logger.log_epoch_start(epoch + 1)

                train_metrics = self.train_epoch(train_loader, epoch)
                self.state.update_train_metrics(train_metrics)

                val_metrics = None
                if val_loader:
                    val_metrics = self.validate_epoch(val_loader, epoch)
                    self.state.update_val_metrics(val_metrics)

                self.logger.log_epoch_end(epoch + 1, train_metrics, val_metrics)

                # FIX: build combined metrics with prefixed keys AND flat val keys
                # so callbacks like ModelCheckpoint can find 'val_loss' directly.
                combined_metrics = {f"train_{k}": v for k, v in train_metrics.items()}
                if val_metrics:
                    combined_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
                    # Also expose val metrics without prefix so monitors like
                    # 'val_loss' (unprefixed) work in ModelCheckpoint.
                    combined_metrics.update(val_metrics)

                # Pass combined metrics to callbacks so they see val_loss etc.
                self.callbacks.on_epoch_end(epoch, combined_metrics, self)

                self.loggers.log_scalars(combined_metrics, self.state.global_step)
                if self.experiment:
                    self.experiment.log_metrics(
                        train_metrics, epoch=epoch, step=self.state.global_step, metric_type="train"
                    )
                    if val_metrics:
                        self.experiment.log_metrics(
                            val_metrics, epoch=epoch, step=self.state.global_step, metric_type="val"
                        )

                # Best epoch saving
                if val_metrics and "loss" in val_metrics:
                    is_best = self.state.is_best_epoch(val_metrics["loss"], mode="min")
                    if is_best:
                        self.save_checkpoint("best.pt", epoch)

                # Latest checkpoint
                self.save_checkpoint("latest.pt", epoch)

                # LR scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if val_metrics and "loss" in val_metrics:
                            self.scheduler.step(val_metrics["loss"])
                    else:
                        self.scheduler.step()
                    self.state.lr_history.append(self.optimizer.param_groups[0]["lr"])

                # Early stopping
                if self._should_stop_early():
                    print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                    break

                self.state.current_epoch = epoch + 1

        except KeyboardInterrupt:
            print("\n\nTraining interrupted by user")
            self.save_checkpoint("interrupted.pt", epoch)
        except Exception as e:
            print(f"\n\nTraining failed: {e}")
            self.save_checkpoint("failed.pt", epoch)
            raise TrainingError(f"Training failed: {e}")
        finally:
            self.logger.log_training_end(self.state)
            self.state.save(self.checkpoint_dir / "training_state.json")
            self.callbacks.on_train_end(self)

    # ------------------ Epoch & Batch ------------------
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        self.train_tracker.epoch_start()
        if self.gradient_accumulator:
            self.gradient_accumulator.reset()

        for batch_idx, batch in enumerate(train_loader):
            batch = (
                [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                if isinstance(batch, (list, tuple))
                else batch.to(self.device) if isinstance(batch, torch.Tensor) else batch
            )

            self.callbacks.on_batch_begin(batch_idx, self)

            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    loss_dict = self._compute_loss(batch)
                    loss = loss_dict["loss"]
                    if self.gradient_accumulator:
                        loss = self.gradient_accumulator.scale_loss(loss)
                self.grad_scaler.scale(loss).backward()
                if self.gradient_clipper:
                    self.grad_scaler.unscale_(self.optimizer)
                    loss_dict["grad_norm"] = self.gradient_clipper.clip(self.model.parameters())
                should_update = (
                    not self.gradient_accumulator or self.gradient_accumulator.should_update()
                )
                if should_update:
                    self.grad_scaler.step(self.optimizer)
                    self.grad_scaler.update()
                    self.optimizer.zero_grad()
                    if self.ema:
                        self.ema.update()
            else:
                self.optimizer.zero_grad()
                loss_dict = self._compute_loss(batch)
                loss = loss_dict["loss"]
                if self.gradient_accumulator:
                    loss = self.gradient_accumulator.scale_loss(loss)
                loss.backward()
                if self.gradient_clipper:
                    loss_dict["grad_norm"] = self.gradient_clipper.clip(self.model.parameters())
                should_update = (
                    not self.gradient_accumulator or self.gradient_accumulator.should_update()
                )
                if should_update:
                    self.optimizer.step()
                    if self.ema:
                        self.ema.update()

            self.train_tracker.update(loss_dict)
            self.callbacks.on_batch_end(batch_idx, loss_dict, self)
            self.logger.log_batch(epoch + 1, batch_idx, len(train_loader), loss_dict)
            self.state.global_step += 1

        return self.train_tracker.epoch_end()

    def _compute_loss(self, batch):
        if hasattr(self.model, "compute_loss"):
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                return self.model.compute_loss(batch[0])
            return self.model.compute_loss(*batch if isinstance(batch, (list, tuple)) else batch)
        if self.criterion is None:
            raise ValueError("No criterion provided and model has no compute_loss")
        if isinstance(batch, (list, tuple)):
            *inputs, targets = batch
            outputs = self.model(*inputs)
        else:
            outputs = self.model(batch)
            targets = batch
        return {"loss": self.criterion(outputs, targets)}

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.eval()
        self.val_tracker.epoch_start()
        with torch.no_grad():
            for batch in val_loader:
                batch = (
                    [b.to(self.device) if isinstance(b, torch.Tensor) else b for b in batch]
                    if isinstance(batch, (list, tuple))
                    else batch.to(self.device) if isinstance(batch, torch.Tensor) else batch
                )

                if hasattr(self.model, "compute_loss"):
                    loss_dict = (
                        self.model.compute_loss(*batch)
                        if isinstance(batch, (list, tuple))
                        else self.model.compute_loss(batch)
                    )
                    metrics = loss_dict
                else:
                    if isinstance(batch, (list, tuple)):
                        *inputs, targets = batch
                        outputs = self.model(*inputs)
                    else:
                        outputs = self.model(batch)
                        targets = batch
                    metrics = {"loss": self.criterion(outputs, targets)}
                self.val_tracker.update(metrics)
        return self.val_tracker.epoch_end()

    # ------------------ Checkpoint ------------------
    def save_checkpoint(self, filename, epoch):
        self.state.current_epoch = epoch
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "training_state": self.state.to_dict(),
        }
        if self.scheduler:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()
        if self.grad_scaler:
            checkpoint["grad_scaler_state_dict"] = self.grad_scaler.state_dict()
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, path) -> Dict[str, Any]:
        """
        Load a checkpoint saved by save_checkpoint().
        Restores model, optimizer, training state, and any optional
        scheduler / EMA / grad-scaler states present in the file.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "training_state" in checkpoint:
            self.state = TrainingState.from_dict(checkpoint["training_state"])
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "ema_state_dict" in checkpoint and self.ema is not None:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])
        if "grad_scaler_state_dict" in checkpoint and self.grad_scaler is not None:
            self.grad_scaler.load_state_dict(checkpoint["grad_scaler_state_dict"])
        return checkpoint

    def _should_stop_early(self) -> bool:
        if self.early_stopping_patience is None:
            return False
        return self.state.patience_counter >= self.early_stopping_patience
