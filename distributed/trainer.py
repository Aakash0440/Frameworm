"""
Distributed training wrapper for Trainer.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from distributed.sampler import get_distributed_sampler
from distributed.utils import (
    all_reduce_dict,
    barrier,
    cleanup_distributed,
    get_local_rank,
    get_rank,
    is_distributed,
    is_master,
    setup_distributed,
)
from training import Trainer


class DistributedTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        backend: str = "nccl",
        find_unused_parameters: bool = False,
        gradient_accumulation_steps: int = 1,
        use_amp: bool = False,
        **kwargs,
    ):
        if not is_distributed():
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            if world_size > 1:
                setup_distributed(backend=backend)

        if torch.cuda.is_available() and is_distributed():
            device = f"cuda:{get_local_rank()}"
            torch.cuda.set_device(device)
        else:
            device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        kwargs["device"] = device
        model = model.to(device)

        if is_distributed():
            model = DDP(
                model,
                device_ids=[get_local_rank()] if torch.cuda.is_available() else None,
                find_unused_parameters=find_unused_parameters,
            )
            print(f"[Rank {get_rank()}] Model wrapped with DDP")

        super().__init__(model, optimizer, **kwargs)

        self.backend = backend
        self.is_distributed = is_distributed()
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self._accumulation_counter = 0
        self.use_amp = use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        self.gradient_clipper = None  # added for safety

        if self.use_amp:
            print("Mixed precision (AMP) enabled")

    def _create_dataloader(
        self, dataset, batch_size: int, shuffle: bool = True, **kwargs
    ) -> DataLoader:
        sampler = get_distributed_sampler(dataset, shuffle=shuffle)
        if sampler is not None:
            shuffle = False

        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, **kwargs
        )

    def train_epoch(self, train_loader: DataLoader, epoch: int):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        self.model.train()
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, (list, tuple)):
                batch = [b.to(self.device) if torch.is_tensor(b) else b for b in batch]

            with autocast(enabled=self.use_amp):
                loss_dict = self.model.compute_loss(*batch)
                loss = loss_dict["loss"]
                if self.gradient_accumulation_steps > 1:
                    loss = loss / self.gradient_accumulation_steps

            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self._accumulation_counter += 1

            if self._accumulation_counter % self.gradient_accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)

                if self.gradient_clipper:
                    self.gradient_clipper.clip(self.model.parameters())

                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()
                self.state.global_step += 1

        if self.is_distributed:
            barrier(force=True)

    def enable_gradient_accumulation(self, steps: int):
        self.gradient_accumulation_steps = steps
        print(f"Gradient accumulation enabled: {steps} steps")
        print(f"Effective batch size = batch_size × world_size × {steps}")

    def validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        metrics = super().validate_epoch(val_loader, epoch)
        if self.is_distributed:
            metrics = self._aggregate_metrics(metrics)
        return metrics

    def _aggregate_metrics(self, metrics: Dict[str, float]) -> Dict[str, float]:
        if not self.is_distributed:
            return metrics
        metric_tensors = {k: torch.tensor(v, device=self.device) for k, v in metrics.items()}
        aggregated = all_reduce_dict(metric_tensors)
        return {k: float(v) for k, v in aggregated.items()}

    def save_checkpoint(self, path: str, epoch: int):
        if not is_master():
            if self.is_distributed:
                barrier(force=True)
            return

        model_to_save = self.model.module if isinstance(self.model, DDP) else self.model

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_to_save.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "state": self.state.to_dict() if hasattr(self.state, "to_dict") else None,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        if self.is_distributed:
            barrier(force=True)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        model_to_load = self.model.module if isinstance(self.model, DDP) else self.model
        model_to_load.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "state" in checkpoint and checkpoint["state"] is not None:
            if hasattr(self.state, "from_dict"):
                self.state.from_dict(checkpoint["state"])

        if self.is_distributed:
            barrier(force=True)

    def cleanup(self):
        if self.is_distributed:
            cleanup_distributed()


def launch_distributed(train_fn, nprocs: int = None, backend: str = "nccl", **kwargs):
    import torch.multiprocessing as mp

    if nprocs is None:
        nprocs = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if nprocs <= 1:
        print("Single process training")
        train_fn(0, 1, **kwargs)
        return

    print(f"Launching {nprocs} processes for distributed training")
    os.environ["WORLD_SIZE"] = str(nprocs)
    os.environ["MASTER_ADDR"] = "localhost"
    from distributed.utils import find_free_port

    os.environ["MASTER_PORT"] = str(find_free_port())
    mp.spawn(train_fn, args=(nprocs,) + tuple(kwargs.values()), nprocs=nprocs, join=True)
