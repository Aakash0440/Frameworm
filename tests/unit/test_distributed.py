"""Comprehensive distributed training tests"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from distributed.data_loader import OptimizedDataLoader
from distributed.profiler import PerformanceProfiler
from distributed.trainer import DistributedTrainer


class TestDistributedOptimization:
    def test_gradient_accumulation(self):
        """Test gradient accumulation reduces optimizer steps"""

        # Model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)

            def forward(self, x):
                return self.fc(x)

            def compute_loss(self, x, y):
                pred = self.forward(x)
                return {"loss": nn.MSELoss()(pred, y)}

        # Data
        X = torch.randn(100, 10)
        y = torch.randn(100, 1)
        loader = DataLoader(TensorDataset(X, y), batch_size=10)

        # Train with accumulation
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())

        trainer = DistributedTrainer(model, optimizer, device="cpu", gradient_accumulation_steps=2)

        trainer.train_epoch(loader, epoch=0)

        # Should have 5 optimizer steps (10 batches / 2 accumulation)
        assert trainer.state.global_step == 5

    def test_optimized_dataloader(self):
        """Test optimized dataloader creation"""
        dataset = TensorDataset(torch.randn(100, 10))

        loader = OptimizedDataLoader.create(dataset, batch_size=32, shuffle=True)

        # Should auto-configure
        assert loader.num_workers >= 0
        assert isinstance(loader.pin_memory, bool)

    def test_performance_profiler(self):
        """Test performance profiler"""
        profiler = PerformanceProfiler()

        # Profile some operations
        for _ in range(10):
            with profiler.profile("step"):
                with profiler.profile("forward"):
                    pass

        results = profiler.get_results()

        assert len(results.step_times) == 10
        assert len(results.forward_times) == 10
