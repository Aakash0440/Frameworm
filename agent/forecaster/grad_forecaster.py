"""
GradForecaster — LSTM model for gradient trajectory prediction.

Architecture:
    Input:  (batch, SEQ_LEN=100, N_FEATURES=8)
    LSTM:   2 layers, hidden=64, dropout=0.2
    Head:   Linear → (N_FAILURE_MODES=6, N_HORIZONS=3)
    Output: P(failure_mode) at each horizon — sigmoid activated

Intentionally small — must run on CPU in ~0.5ms during training
without stealing GPU from the main training loop.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agent.forecaster.training_data import (N_FAILURE_MODES, N_FEATURES,
                                            N_HORIZONS, SEQ_LEN,
                                            ForecasterDataset)

logger = logging.getLogger(__name__)


@dataclass
class ForecasterConfig:
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    learning_rate: float = 1e-3
    max_epochs: int = 100
    patience: int = 10
    weight_decay: float = 1e-4
    device: str = "cpu"
    confidence_threshold: float = 0.80
    min_inference_steps: int = 100


class GradForecaster(nn.Module):
    """
    LSTM-based gradient trajectory forecaster.

    Usage (training):
        dataset = DataCollector().collect()
        model = GradForecaster()
        model.fit(dataset)
        model.save("agent/forecaster/weights/grad_forecaster.pt")

    Usage (inference during training):
        model = GradForecaster.load("agent/forecaster/weights/grad_forecaster.pt")
        probs = model.predict(feature_window)
        # probs: (N_FAILURE_MODES, N_HORIZONS)
    """

    def __init__(self, config: Optional[ForecasterConfig] = None) -> None:
        super().__init__()
        self.config = config or ForecasterConfig()
        cfg = self.config
        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.head = nn.Linear(cfg.hidden_size, N_FAILURE_MODES * N_HORIZONS)
        self._is_trained = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, SEQ_LEN, N_FEATURES)
        returns: (batch, N_FAILURE_MODES, N_HORIZONS) — sigmoid probs
        """
        out, _ = self.lstm(x)
        last = self.dropout(out[:, -1, :])
        logits = self.head(last)
        probs = torch.sigmoid(logits)
        return probs.view(-1, N_FAILURE_MODES, N_HORIZONS)

    def fit(self, dataset: ForecasterDataset) -> dict:
        """Train forecaster. Returns history dict with train/val losses."""
        cfg = self.config
        device = torch.device(cfg.device)
        self.to(device)

        train_ds, val_ds = dataset.split(val_frac=0.15)
        train_X, train_y = train_ds.to_arrays()
        val_X, val_y = val_ds.to_arrays()

        train_loader = DataLoader(
            TensorDataset(torch.tensor(train_X), torch.tensor(train_y)),
            batch_size=cfg.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(val_X), torch.tensor(val_y)),
            batch_size=cfg.batch_size * 2,
            shuffle=False,
        )

        optimizer = optim.AdamW(
            self.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
        criterion = nn.BCELoss()

        history = {"train_loss": [], "val_loss": [], "best_epoch": 0}
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        logger.info(f"Training GradForecaster: {len(train_ds)} train, {len(val_ds)} val samples")

        for epoch in range(cfg.max_epochs):
            self.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(self(X_batch), y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()
                train_losses.append(loss.item())

            self.eval()
            val_losses = []
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_losses.append(criterion(self(X_batch), y_batch).item())

            train_loss = float(np.mean(train_losses))
            val_loss = float(np.mean(val_losses))
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            scheduler.step()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{cfg.max_epochs} — train: {train_loss:.4f}, val: {val_loss:.4f}"
                )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.state_dict().items()}
                history["best_epoch"] = epoch + 1
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= cfg.patience:
                    logger.info(f"Early stopping at epoch {epoch+1}. Best val: {best_val_loss:.4f}")
                    break

        if best_state is not None:
            self.load_state_dict(best_state)

        self._is_trained = True
        self.to("cpu")
        return history

    @torch.no_grad()
    def predict(self, feature_window: np.ndarray) -> np.ndarray:
        """
        Run inference on a single feature window.
        feature_window: (SEQ_LEN, N_FEATURES)
        returns: (N_FAILURE_MODES, N_HORIZONS) probability array
        """
        self.eval()
        if len(feature_window) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(feature_window), N_FEATURES), dtype=np.float32)
            feature_window = np.vstack([pad, feature_window])
        else:
            feature_window = feature_window[-SEQ_LEN:]
        x = torch.tensor(feature_window, dtype=torch.float32).unsqueeze(0)
        return self(x).squeeze(0).numpy()

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.state_dict(),
                "config": self.config,
                "is_trained": self._is_trained,
            },
            path,
        )
        logger.info(f"GradForecaster saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "GradForecaster":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"No forecaster weights at {path}")
        torch.serialization.add_safe_globals([ForecasterConfig])
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(config=checkpoint.get("config", ForecasterConfig()))
        model.load_state_dict(checkpoint["state_dict"])
        model._is_trained = checkpoint.get("is_trained", True)
        model.eval()
        logger.info(f"GradForecaster loaded from {path}")
        return model

    @classmethod
    def load_or_init(cls, path: Path) -> "GradForecaster":
        """Load if weights exist, else return fresh untrained model."""
        try:
            return cls.load(path)
        except FileNotFoundError:
            logger.info(f"No weights at {path}. Starting untrained — call fit() after 5+ runs.")
            return cls()

    @property
    def is_ready(self) -> bool:
        return self._is_trained
