"""
Training data collection and formatting for GradForecaster.

The forecaster is a supervised model:
    Input:  sliding window of gradient + loss statistics (100 steps)
    Output: P(failure_type) at horizons [50, 100, 500] steps ahead

Training data comes from:
    1. experiments/experiments.db   — your existing SQLite run history
    2. experiments/*/logs/          — per-run JSON log files
    Fallback: synthetic data for cold start (no real runs needed)

Hooks into:
    experiment/manager.py       → load run history
    experiments/experiments.db  → raw metric logs
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Feature set — 8 features per step, same order always
FEATURE_NAMES = [
    "loss", "loss_ema", "loss_delta", "loss_z_score",
    "grad_norm", "grad_norm_var", "lr", "plateau_score",
]
N_FEATURES = len(FEATURE_NAMES)

# Prediction horizons (steps ahead)
HORIZONS = [50, 100, 500]
N_HORIZONS = len(HORIZONS)

# Failure mode labels — matches AnomalyType enum order (no HEALTHY)
FAILURE_MODES = [
    "gradient_explosion", "divergence", "loss_spike",
    "vanishing_grad", "oscillating", "plateau",
]
N_FAILURE_MODES = len(FAILURE_MODES)

SEQ_LEN = 100  # steps of history per sample


@dataclass
class ForecasterSample:
    """One training example for the GradForecaster."""
    features: np.ndarray      # (SEQ_LEN, N_FEATURES) float32
    labels: np.ndarray        # (N_FAILURE_MODES, N_HORIZONS) float32 in [0,1]
    run_id: str = ""
    start_step: int = 0


class ForecasterDataset:
    """
    In-memory dataset of ForecasterSamples.

    Usage:
        collector = DataCollector()
        dataset = collector.collect()
        X, y = dataset.to_arrays()
        # X: (N, SEQ_LEN, N_FEATURES), y: (N, N_FAILURE_MODES, N_HORIZONS)
    """

    def __init__(self) -> None:
        self._samples: List[ForecasterSample] = []

    def add(self, sample: ForecasterSample) -> None:
        self._samples.append(sample)

    def __len__(self) -> int:
        return len(self._samples)

    def __iter__(self) -> Iterator[ForecasterSample]:
        return iter(self._samples)

    def to_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self._samples:
            raise ValueError("Dataset is empty. Run DataCollector.collect() first.")
        X = np.stack([s.features for s in self._samples]).astype(np.float32)
        y = np.stack([s.labels for s in self._samples]).astype(np.float32)
        return X, y

    def split(self, val_frac: float = 0.15) -> Tuple["ForecasterDataset", "ForecasterDataset"]:
        n = len(self._samples)
        split_idx = int(n * (1 - val_frac))
        train, val = ForecasterDataset(), ForecasterDataset()
        train._samples = self._samples[:split_idx]
        val._samples = self._samples[split_idx:]
        return train, val

    def save(self, path: Path) -> None:
        if not self._samples:
            return
        X, y = self.to_arrays()
        np.savez_compressed(path, X=X, y=y)
        logger.info(f"Dataset saved to {path} ({len(self)} samples)")

    @classmethod
    def load(cls, path: Path) -> "ForecasterDataset":
        data = np.load(path)
        dataset = cls()
        X, y = data["X"], data["y"]
        for i in range(len(X)):
            dataset.add(ForecasterSample(features=X[i], labels=y[i]))
        logger.info(f"Dataset loaded from {path} ({len(dataset)} samples)")
        return dataset


class DataCollector:
    """
    Extracts ForecasterSamples from past FRAMEWORM training runs.

    Sources (in priority order):
        1. experiments/experiments.db   (your existing SQLite DB)
        2. experiments/*/logs/          (per-run log JSON files)
        Fallback: synthetic data when no real runs exist yet

    Args:
        experiments_dir:    Path to experiments/ directory
        db_path:            Path to experiments.db
        min_run_length:     Minimum steps to include a run
        stride:             Sliding window stride
    """

    def __init__(
        self,
        experiments_dir: Path = Path("experiments"),
        db_path: Optional[Path] = None,
        min_run_length: int = 300,
        stride: int = 10,
    ) -> None:
        self.experiments_dir = Path(experiments_dir)
        self.db_path = db_path or (self.experiments_dir / "experiments.db")
        self.min_run_length = min_run_length
        self.stride = stride

    def collect(self) -> ForecasterDataset:
        dataset = ForecasterDataset()

        db_samples = self._collect_from_db()
        for s in db_samples:
            dataset.add(s)
        logger.info(f"Collected {len(db_samples)} samples from experiments.db")

        log_samples = self._collect_from_logs()
        for s in log_samples:
            dataset.add(s)
        logger.info(f"Collected {len(log_samples)} samples from run logs")

        if len(dataset) == 0:
            logger.warning(
                "No training data found — generating synthetic data for cold start. "
                "Run FRAMEWORM experiments to accumulate real data."
            )
            synthetic = self._generate_synthetic_data(n_runs=10)
            for s in synthetic:
                dataset.add(s)
            logger.info(f"Generated {len(synthetic)} synthetic training samples")

        logger.info(f"Total forecaster dataset size: {len(dataset)} samples")
        return dataset

    def _collect_from_db(self) -> List[ForecasterSample]:
        if not self.db_path.exists():
            return []
        samples = []
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            if "metrics" in tables:
                samples = self._extract_from_metrics_table(cursor)
            elif "runs" in tables:
                samples = self._extract_from_runs_table(cursor)

            conn.close()
        except Exception as exc:
            logger.warning(f"DB collection failed: {exc}")
        return samples

    def _extract_from_metrics_table(self, cursor) -> List[ForecasterSample]:
        samples = []
        try:
            cursor.execute("SELECT DISTINCT run_id FROM metrics")
            run_ids = [row[0] for row in cursor.fetchall()]
            for run_id in run_ids:
                cursor.execute(
                    "SELECT step, loss, grad_norm, lr FROM metrics "
                    "WHERE run_id=? ORDER BY step ASC", (run_id,)
                )
                rows = cursor.fetchall()
                if len(rows) < self.min_run_length:
                    continue
                samples.extend(self._build_samples_from_rows(
                    rows, str(run_id), {"step": 0, "loss": 1, "grad_norm": 2, "lr": 3}
                ))
        except Exception as exc:
            logger.debug(f"metrics table extraction failed: {exc}")
        return samples

    def _extract_from_runs_table(self, cursor) -> List[ForecasterSample]:
        samples = []
        try:
            cursor.execute("SELECT id FROM runs")
            run_ids = [row[0] for row in cursor.fetchall()]
            for run_id in run_ids:
                try:
                    cursor.execute(
                        "SELECT step, loss, grad_norm, lr FROM run_metrics "
                        "WHERE run_id=? ORDER BY step", (run_id,)
                    )
                    rows = cursor.fetchall()
                    if len(rows) < self.min_run_length:
                        continue
                    samples.extend(self._build_samples_from_rows(
                        rows, str(run_id), {"step": 0, "loss": 1, "grad_norm": 2, "lr": 3}
                    ))
                except Exception:
                    pass
        except Exception as exc:
            logger.debug(f"runs table extraction failed: {exc}")
        return samples

    def _collect_from_logs(self) -> List[ForecasterSample]:
        samples = []
        if not self.experiments_dir.exists():
            return samples
        for run_dir in self.experiments_dir.iterdir():
            if not run_dir.is_dir():
                continue
            logs_dir = run_dir / "logs"
            if not logs_dir.exists():
                continue
            for log_name in ["training_log.json", "metrics.json", "history.json"]:
                log_file = logs_dir / log_name
                if log_file.exists():
                    try:
                        with open(log_file) as f:
                            data = json.load(f)
                        samples.extend(self._parse_log_json(data, run_dir.name))
                        break
                    except Exception as exc:
                        logger.debug(f"Could not parse {log_file}: {exc}")
        return samples

    def _parse_log_json(self, data, run_id: str) -> List[ForecasterSample]:
        rows = []
        if isinstance(data, list):
            for item in data:
                step = item.get("step", item.get("epoch", len(rows)))
                loss = item.get("loss", item.get("train_loss"))
                if loss is not None:
                    rows.append((
                        int(step),
                        float(loss),
                        float(item.get("grad_norm", 1.0)),
                        float(item.get("lr", item.get("learning_rate", 0.001))),
                    ))
        elif isinstance(data, dict):
            losses = data.get("loss", data.get("train_loss", []))
            grad_norms = data.get("grad_norm", [1.0] * len(losses))
            lrs = data.get("lr", [0.001] * len(losses))
            for i, loss in enumerate(losses):
                if loss is not None:
                    rows.append((
                        i, float(loss),
                        float(grad_norms[i]) if i < len(grad_norms) else 1.0,
                        float(lrs[i]) if i < len(lrs) else 0.001,
                    ))
        if len(rows) < self.min_run_length:
            return []
        return self._build_samples_from_rows(
            rows, run_id, {"step": 0, "loss": 1, "grad_norm": 2, "lr": 3}
        )

    def _build_samples_from_rows(self, rows, run_id, col_map) -> List[ForecasterSample]:
        if len(rows) < SEQ_LEN + max(HORIZONS):
            return []
        losses = np.array([r[col_map["loss"]] for r in rows], dtype=np.float32)
        grad_norms = np.array([r[col_map["grad_norm"]] for r in rows], dtype=np.float32)
        lrs = np.array([r[col_map["lr"]] for r in rows], dtype=np.float32)
        features_array = self._compute_features(losses, grad_norms, lrs)
        anomaly_labels = self._label_anomalies(losses, grad_norms)
        samples = []
        end = len(features_array) - max(HORIZONS)
        for start in range(0, end - SEQ_LEN, self.stride):
            seq_end = start + SEQ_LEN
            if seq_end + max(HORIZONS) > len(anomaly_labels):
                break
            samples.append(ForecasterSample(
                features=features_array[start:seq_end],
                labels=self._build_label_matrix(anomaly_labels, seq_end),
                run_id=run_id,
                start_step=start,
            ))
        return samples

    def _compute_features(self, losses, grad_norms, lrs) -> np.ndarray:
        n = len(losses)
        features = np.zeros((n, N_FEATURES), dtype=np.float32)
        alpha = 0.05
        ema = np.zeros(n)
        ema[0] = losses[0]
        for i in range(1, n):
            ema[i] = alpha * losses[i] + (1 - alpha) * ema[i - 1]
        w = 50
        rolling_mean = np.array([np.mean(losses[max(0,i-w):i+1]) for i in range(n)])
        rolling_std = np.array([np.std(losses[max(0,i-w):i+1]) + 1e-8 for i in range(n)])
        loss_delta = np.zeros(n)
        for i in range(10, n):
            loss_delta[i] = np.mean(losses[i-5:i]) - np.mean(losses[max(0,i-10):i-5])
        grad_var = np.array([np.var(grad_norms[max(0,i-w):i+1]) for i in range(n)])
        features[:, 0] = losses
        features[:, 1] = ema
        features[:, 2] = loss_delta
        features[:, 3] = (losses - rolling_mean) / rolling_std
        features[:, 4] = grad_norms
        features[:, 5] = grad_var
        features[:, 6] = lrs
        features[:, 7] = np.abs(loss_delta) / rolling_std
        for j in range(N_FEATURES):
            col = features[:, j]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min > 1e-8:
                features[:, j] = (col - col_min) / (col_max - col_min)
        return features

    def _label_anomalies(self, losses, grad_norms) -> np.ndarray:
        n = len(losses)
        labels = np.zeros((n, N_FAILURE_MODES), dtype=np.float32)
        w = 50
        rolling_mean = np.array([np.mean(losses[max(0,i-w):i+1]) for i in range(n)])
        rolling_std = np.array([np.std(losses[max(0,i-w):i+1]) + 1e-8 for i in range(n)])
        z_scores = (losses - rolling_mean) / rolling_std
        loss_delta = np.zeros(n)
        for i in range(10, n):
            loss_delta[i] = np.mean(losses[i-5:i]) - np.mean(losses[max(0,i-10):i-5])
        divergence = np.zeros(n)
        for i in range(10, n):
            diffs = np.diff(losses[max(0,i-20):i+1])
            divergence[i] = np.mean(diffs > 0) if len(diffs) > 0 else 0
        oscillation = np.zeros(n)
        for i in range(20, n):
            oscillation[i] = np.var(np.diff(losses[i-20:i+1]))
        plateau = np.zeros(n)
        counter = 0
        for i in range(1, n):
            abs_delta = abs(loss_delta[i]) / (rolling_std[i] + 1e-8)
            counter = counter + 1 if abs_delta < 0.05 else 0
            if counter >= 100:
                plateau[i] = 1.0
        labels[:, 0] = (grad_norms > 10.0).astype(np.float32)
        labels[:, 1] = (divergence > 0.75).astype(np.float32)
        labels[:, 2] = (z_scores > 3.0).astype(np.float32)
        labels[:, 3] = (grad_norms < 0.001).astype(np.float32)
        labels[:, 4] = (oscillation > 0.01).astype(np.float32)
        labels[:, 5] = plateau
        return labels

    def _build_label_matrix(self, anomaly_labels, from_step) -> np.ndarray:
        label_matrix = np.zeros((N_FAILURE_MODES, N_HORIZONS), dtype=np.float32)
        for h_idx, horizon in enumerate(HORIZONS):
            end = min(from_step + horizon, len(anomaly_labels))
            if end <= from_step:
                continue
            future = anomaly_labels[from_step:end]
            label_matrix[:, h_idx] = (future.sum(axis=0) > 0).astype(np.float32)
        return label_matrix

    def _generate_synthetic_data(self, n_runs: int = 10) -> List[ForecasterSample]:
        samples = []
        rng = np.random.default_rng(42)
        for run_idx in range(n_runs):
            n_steps = int(rng.integers(800, 2000))
            base = np.exp(-np.linspace(0, 3, n_steps)) + 0.1
            losses = (base + rng.normal(0, 0.02, n_steps)).clip(0.01).astype(np.float32)
            grad_norms = rng.exponential(2.0, size=n_steps).clip(0.01, 50.0).astype(np.float32)
            lrs = np.full(n_steps, 0.0002, dtype=np.float32)
            inject_at = int(rng.integers(n_steps // 3, 2 * n_steps // 3))
            failure_type = run_idx % N_FAILURE_MODES
            if failure_type == 0:
                grad_norms[inject_at:inject_at+20] = rng.uniform(15, 80, 20)
            elif failure_type == 1:
                for i in range(inject_at, min(inject_at+200, n_steps)):
                    losses[i] = losses[inject_at] * (1.002 ** (i - inject_at))
            elif failure_type == 2:
                losses[inject_at:inject_at+5] *= float(rng.uniform(3, 8))
            elif failure_type == 3:
                grad_norms[inject_at:] = rng.uniform(0.0001, 0.0009, n_steps - inject_at)
            elif failure_type == 4:
                for i in range(inject_at, n_steps):
                    losses[i] += 0.1 * np.sin(i * 0.5)
            elif failure_type == 5:
                losses[inject_at:] = losses[inject_at] + rng.normal(0, 0.001, n_steps - inject_at)
            features = self._compute_features(losses, grad_norms, lrs)
            labels = self._label_anomalies(losses, grad_norms)
            end = len(features) - max(HORIZONS)
            for start in range(0, end - SEQ_LEN, self.stride * 2):
                seq_end = start + SEQ_LEN
                if seq_end + max(HORIZONS) > len(labels):
                    break
                samples.append(ForecasterSample(
                    features=features[start:seq_end],
                    labels=self._build_label_matrix(labels, seq_end),
                    run_id=f"synthetic_{run_idx}",
                    start_step=start,
                ))
        return samples