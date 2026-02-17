# (Run with: jupyter nbconvert --to notebook --execute)

"""
FRAMEWORM Demo: Complete ML Workflow in Under 50 Lines

This demo shows the complete power of FRAMEWORM:
train → track → search → deploy in one unified system.
"""

# ============================================================
# SECTION 1: INSTALLATION
# ============================================================
# pip install frameworm

# ============================================================
# SECTION 2: BASIC TRAINING
# ============================================================

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from frameworm import Config, Trainer, get_model
from frameworm.experiment import Experiment

# 1. Configuration
config = Config.from_dict(
    {
        "model": {"type": "vae", "latent_dim": 64, "hidden_dim": 128},
        "training": {"epochs": 5, "batch_size": 64, "lr": 0.001},
    }
)

# 2. Dummy dataset (replace with real data)
X = torch.randn(500, 1, 32, 32)
dataset = TensorDataset(X)
train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X[:100]), batch_size=64)

# 3. Create model
model = get_model("vae")(config)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 4. Train with experiment tracking
with Experiment(name="demo-vae", config=config, tags=["demo"]) as exp:
    trainer = Trainer(model, optimizer, device="cpu")
    trainer.set_experiment(exp)
    trainer.train(train_loader, val_loader, epochs=5)

print(f"✓ Training complete! Experiment: {exp.experiment_id}")

# ============================================================
# SECTION 3: HYPERPARAMETER SEARCH
# ============================================================

from frameworm.search import RandomSearch
from frameworm.search.space import Integer, Real


def train_fn(search_config):
    """Training function for search"""
    model = get_model("vae")(search_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=search_config.training.lr)
    trainer = Trainer(model, optimizer, device="cpu")
    trainer.train(train_loader, val_loader, epochs=2)
    return {"val_loss": trainer.state.val_metrics["loss"][-1]}


search = RandomSearch(
    base_config=config,
    search_space={
        "training.lr": Real(1e-4, 1e-2, log=True),
        "model.latent_dim": Integer(32, 128, log=True),
    },
    metric="val_loss",
    mode="min",
    n_trials=5,
    random_state=42,
)

best_config, best_score = search.run(train_fn, verbose=True)
print(f"✓ Best config: lr={best_config['training.lr']:.5f}, score={best_score:.4f}")

# ============================================================
# SECTION 4: EXPORT & DEPLOY
# ============================================================

from frameworm.deployment import ModelExporter

# Export to TorchScript
example_input = torch.randn(1, 1, 32, 32)
exporter = ModelExporter(model, example_input)
exporter.to_torchscript("demo_model.pt")

# Serve (would run server in production)
# from frameworm.deployment import ModelServer
# server = ModelServer('demo_model.pt')
# server.run(port=8000)

print("✓ Model exported and ready to serve!")
print("\n🎉 Complete ML pipeline in ~50 lines!")

import os

os.remove("demo_model.pt")
