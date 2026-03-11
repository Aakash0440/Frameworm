"""
Hyperparameter search command.
"""

import yaml
import torch
from click import echo
from core import Config
from search import GridSearch, RandomSearch


def make_train_fn(config_path):
    """Create a training function for hyperparameter search."""

    def train_fn(trial_config):
        # trial_config is a Config object passed by random_search.py
        cfg = trial_config

        from core.registry import get_model
        import models.gan.dcgan

        try:
            model = get_model(cfg.get("model.type", "dcgan"))(cfg)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("optimizer.lr", 0.0002))

            from torch.utils.data import DataLoader, TensorDataset

            dummy = TensorDataset(torch.randn(64, 3, 64, 64))
            loader = DataLoader(dummy, batch_size=32)

            from training.trainer import Trainer
            import tempfile

            checkpoint_dir = tempfile.mkdtemp()
            trainer = Trainer(
                model=model,
                optimizer=optimizer,
                device="cpu",
                checkpoint_dir=checkpoint_dir,
            )

            # Run just 3 epochs for search
            trainer.train(train_loader=loader, epochs=3)

            # train_metrics is populated by state.update_train_metrics()
            # after each epoch — grab whatever key exists
            metrics = trainer.state.train_metrics
            echo(f"  Trial metrics: {metrics}")

            # Try common loss key names
            raw = (
                metrics.get("loss")
                or metrics.get("g_loss")
                or metrics.get("total_loss")
                or metrics.get("train_loss")
            )
            loss = raw[-1] if isinstance(raw, list) else raw

            if loss is None:
                echo(f"  Warning: no loss key found in {list(metrics.keys())}, defaulting to 9999")
                loss = 9999.0

            return {"val_loss": float(loss)}

        except Exception as e:
            echo(f"  Trial failed: {e}")
            return {"val_loss": 9999.0}

    return train_fn


def run_search(
    config_path: str, space_path: str, method: str = "random", n_trials: int = 50, n_jobs: int = 1
):
    """Run hyperparameter search from CLI"""
    config = Config(config_path)

    with open(space_path) as f:
        search_space = yaml.safe_load(f)

    echo(f"Search space: {search_space}")
    echo(f"Method: {method}")
    echo(f"Trials: {n_trials}")

    train_fn = make_train_fn(config_path)

    if method == "grid":
        search = GridSearch(
            base_config=config, search_space=search_space, metric="val_loss", mode="min"
        )
        best_config, best_score = search.run(train_fn=train_fn, n_jobs=n_jobs)
    elif method == "random":
        search = RandomSearch(
            base_config=config,
            search_space=search_space,
            metric="val_loss",
            mode="min",
            n_trials=n_trials,
        )
        best_config, best_score = search.run(train_fn=train_fn, n_jobs=n_jobs)
    elif method == "bayesian":
        try:
            from search import BayesianSearch

            search = BayesianSearch(
                base_config=config,
                search_space=search_space,
                metric="val_loss",
                mode="min",
                n_trials=n_trials,
            )
            best_config, best_score = search.run(train_fn=train_fn)
        except ImportError:
            echo("✗ Bayesian search requires scikit-optimize")
            return

    echo(f"\n✓ Search complete!")
    echo(f"Best score: {best_score}")
    echo(f"Best config: {best_config}")
