"""Tests for distributed training"""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

from distributed.utils import (
    is_distributed,
    get_world_size,
    get_rank,
    is_master
)
from distributed.sampler import DistributedSampler
from distributed.trainer import DistributedTrainer


class TestDistributedUtils:
    def test_single_process(self):
        """Test utilities in single-process mode"""
        assert not is_distributed()
        assert get_world_size() == 1
        assert get_rank() == 0
        assert is_master()


class TestDistributedSampler:
    def test_sampler_split(self):
        """Test data split across processes"""
        dataset = TensorDataset(torch.arange(100))
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=4,
            rank=0,
            shuffle=False
        )
        
        indices = list(sampler)
        assert len(indices) == 25
    
    def test_sampler_disjoint(self):
        """Test different ranks get different data"""
        dataset = TensorDataset(torch.arange(100))
        
        sampler_0 = DistributedSampler(dataset, num_replicas=2, rank=0, shuffle=False)
        sampler_1 = DistributedSampler(dataset, num_replicas=2, rank=1, shuffle=False)
        
        indices_0 = set(sampler_0)
        indices_1 = set(sampler_1)
        
        # Should be disjoint
        assert len(indices_0 & indices_1) == 0


class TestDistributedTrainer:
    def test_single_process_training(self):
        """Test DistributedTrainer in single-process mode"""
        # Dummy model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
            
            def forward(self, x):
                return self.fc(x)
            
            def compute_loss(self, x, y):
                pred = self.forward(x)
                loss = nn.MSELoss()(pred, y)
                return {'loss': loss}
        
        # Dummy data
        from torch.utils.data import DataLoader
        
        X = torch.randn(50, 10)
        y = torch.randn(50, 1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=10)
        
        # Train
        model = SimpleModel()
        optimizer = torch.optim.Adam(model.parameters())
        
        trainer = DistributedTrainer(model, optimizer, device='cpu')
        trainer.train(loader, epochs=2)
        
        assert trainer.state.current_epoch == 2
