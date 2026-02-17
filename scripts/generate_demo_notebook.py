# Converts demo script to Jupyter notebook format

import json

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# FRAMEWORM Demo 🐛\n",
                "\n",
                "Complete ML workflow: **Train → Track → Search → Deploy**\n",
                "\n",
                "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yourusername/frameworm/blob/main/examples/demo.ipynb)\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Install FRAMEWORM\n",
                "!pip install frameworm -q"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 1. Setup & Data"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import torch\n",
                "from torch.utils.data import DataLoader, TensorDataset\n",
                "from frameworm import Trainer, Config, get_model\n",
                "\n",
                "config = Config.from_dict({\n",
                "    'model': {'type': 'vae', 'latent_dim': 64, 'hidden_dim': 128},\n",
                "    'training': {'epochs': 5, 'batch_size': 64, 'lr': 0.001}\n",
                "})\n",
                "\n",
                "X = torch.randn(500, 1, 32, 32)\n",
                "train_loader = DataLoader(TensorDataset(X), batch_size=64, shuffle=True)\n",
                "val_loader = DataLoader(TensorDataset(X[:100]), batch_size=64)\n",
                "\n",
                "print('✓ Setup complete')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 2. Train with Experiment Tracking"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from frameworm.experiment import Experiment\n",
                "\n",
                "model = get_model('vae')(config)\n",
                "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)\n",
                "\n",
                "with Experiment(name='demo-vae', config=config) as exp:\n",
                "    trainer = Trainer(model, optimizer, device='cpu')\n",
                "    trainer.set_experiment(exp)\n",
                "    trainer.train(train_loader, val_loader, epochs=5)\n",
                "\n",
                "print(f'✓ Experiment ID: {exp.experiment_id}')\n",
                "print(f'✓ Final val_loss: {trainer.state.val_metrics[\"loss\"][-1]:.4f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 3. Hyperparameter Search"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from frameworm.search import RandomSearch\n",
                "from frameworm.search.space import Real, Integer\n",
                "\n",
                "def train_fn(cfg):\n",
                "    m = get_model('vae')(cfg)\n",
                "    opt = torch.optim.Adam(m.parameters(), lr=cfg.training.lr)\n",
                "    t = Trainer(m, opt, device='cpu')\n",
                "    t.train(train_loader, val_loader, epochs=2)\n",
                "    return {'val_loss': t.state.val_metrics['loss'][-1]}\n",
                "\n",
                "search = RandomSearch(\n",
                "    base_config=config,\n",
                "    search_space={'training.lr': Real(1e-4, 1e-2, log=True)},\n",
                "    metric='val_loss', mode='min', n_trials=5\n",
                ")\n",
                "\n",
                "best_config, best_score = search.run(train_fn)\n",
                "print(f'✓ Best score: {best_score:.4f}')\n",
                "print(f'✓ Best lr: {best_config[\"training.lr\"]:.5f}')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": ["## 4. Export & Deploy"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "from frameworm.deployment import ModelExporter\n",
                "\n",
                "example_input = torch.randn(1, 1, 32, 32)\n",
                "exporter = ModelExporter(model, example_input)\n",
                "exporter.to_torchscript('model.pt')\n",
                "\n",
                "print('✓ Model exported!')\n",
                "print('✓ Run: frameworm serve model.pt --port 8000')"
            ]
        }
    ]
}

with open('examples/demo.ipynb', 'w') as f:
    json.dump(notebook, f, indent=2)

print("✓ Demo notebook created: examples/demo.ipynb")