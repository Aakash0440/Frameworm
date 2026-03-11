"""
Computes a statistical fingerprint of any dataset.
Runs on training data (reference) and production data (current).
"""

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class NumericalProfile:
    feature_name: str
    mean: float
    std: float
    min: float
    max: float
    median: float
    skewness: float
    kurtosis: float
    missing_rate: float
    percentiles: Dict[str, float]  # p5, p25, p75, p95
    histogram_counts: List[float]
    histogram_edges: List[float]
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "NumericalProfile":
        return cls(**d)


@dataclass
class CategoricalProfile:
    feature_name: str
    value_counts: Dict[str, int]  # category -> count
    top_k: List[str]  # top 10 most frequent
    entropy: float
    n_unique: int
    missing_rate: float
    n_samples: int

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CategoricalProfile":
        return cls(**d)


@dataclass
class DatasetProfile:
    """Full statistical fingerprint of a dataset."""

    numerical: Dict[str, NumericalProfile] = field(default_factory=dict)
    categorical: Dict[str, CategoricalProfile] = field(default_factory=dict)
    n_samples: int = 0
    n_features: int = 0
    feature_names: List[str] = field(default_factory=list)
    created_at: str = ""

    def to_dict(self) -> dict:
        return {
            "numerical": {k: v.to_dict() for k, v in self.numerical.items()},
            "categorical": {k: v.to_dict() for k, v in self.categorical.items()},
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "feature_names": self.feature_names,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DatasetProfile":
        profile = cls(
            n_samples=d["n_samples"],
            n_features=d["n_features"],
            feature_names=d["feature_names"],
            created_at=d.get("created_at", ""),
        )
        for k, v in d.get("numerical", {}).items():
            profile.numerical[k] = NumericalProfile.from_dict(v)
        for k, v in d.get("categorical", {}).items():
            profile.categorical[k] = CategoricalProfile.from_dict(v)
        return profile


class FeatureProfiler:
    """
    Computes DatasetProfile from numpy arrays or pandas DataFrames.

    Usage:
        profiler = FeatureProfiler()
        profile = profiler.profile(X_train, feature_names=["age", "income", ...])
    """

    N_HISTOGRAM_BINS = 20
    TOP_K_CATEGORIES = 10
    CATEGORICAL_THRESHOLD = (
        20  # if n_unique <= this AND n_unique/n_samples > 0.3, treat as categorical
    )

    def profile(self, data, feature_names: Optional[List[str]] = None) -> DatasetProfile:
        """
        Accepts:
            - numpy array (n_samples, n_features)
            - pandas DataFrame
        """
        import datetime

        # Normalise to numpy + column names
        X, col_names = self._normalise_input(data, feature_names)
        n_samples, n_features = X.shape

        result = DatasetProfile(
            n_samples=n_samples,
            n_features=n_features,
            feature_names=col_names,
            created_at=datetime.datetime.utcnow().isoformat(),
        )

        for i, name in enumerate(col_names):
            col = X[:, i]
            if self._is_categorical(col):
                result.categorical[name] = self._profile_categorical(col, name)
            else:
                result.numerical[name] = self._profile_numerical(col, name)

        return result

    # ------------------------------------------------------------------ helpers

    def _normalise_input(self, data, feature_names):
        try:
            import pandas as pd

            if isinstance(data, pd.DataFrame):
                names = list(data.columns) if feature_names is None else feature_names
                return data.values.astype(object), names
        except ImportError:
            pass

        arr = np.array(data, dtype=object)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
        return arr, feature_names

    def _is_categorical(self, col: np.ndarray) -> bool:
        try:
            numeric = col.astype(float)
            clean = numeric[~np.isnan(numeric)]
            n_unique = len(np.unique(clean))
            # Only treat as categorical if low cardinality AND not just a tiny sample
            if len(clean) == 0:
                return False
            ratio = n_unique / len(clean)
            return n_unique <= self.CATEGORICAL_THRESHOLD and ratio < 0.9
        except (ValueError, TypeError):
            return True  # non-numeric → definitely categorical

    def _profile_numerical(self, col: np.ndarray, name: str) -> NumericalProfile:
        try:
            vals = col.astype(float)
        except (ValueError, TypeError):
            vals = np.zeros(len(col))

        missing_mask = np.isnan(vals)
        missing_rate = float(missing_mask.mean())
        clean = vals[~missing_mask]

        if len(clean) == 0:
            clean = np.array([0.0])

        counts, edges = np.histogram(clean, bins=self.N_HISTOGRAM_BINS)

        # skewness and kurtosis (manual — no scipy dependency)
        mean = float(np.mean(clean))
        std = float(np.std(clean)) or 1e-8
        z = (clean - mean) / std
        skewness = float(np.mean(z**3))
        kurtosis = float(np.mean(z**4) - 3)

        return NumericalProfile(
            feature_name=name,
            mean=mean,
            std=std,
            min=float(np.min(clean)),
            max=float(np.max(clean)),
            median=float(np.median(clean)),
            skewness=skewness,
            kurtosis=kurtosis,
            missing_rate=missing_rate,
            percentiles={
                "p5": float(np.percentile(clean, 5)),
                "p25": float(np.percentile(clean, 25)),
                "p75": float(np.percentile(clean, 75)),
                "p95": float(np.percentile(clean, 95)),
            },
            histogram_counts=counts.tolist(),
            histogram_edges=edges.tolist(),
            n_samples=len(vals),
        )

    def _profile_categorical(self, col: np.ndarray, name: str) -> CategoricalProfile:
        str_col = [str(v) if v is not None and v == v else None for v in col]
        missing_rate = sum(1 for v in str_col if v is None) / max(len(str_col), 1)
        clean = [v for v in str_col if v is not None]

        value_counts: Dict[str, int] = {}
        for v in clean:
            value_counts[v] = value_counts.get(v, 0) + 1

        total = max(sum(value_counts.values()), 1)
        top_k = sorted(value_counts, key=lambda x: -value_counts[x])[: self.TOP_K_CATEGORIES]

        # Shannon entropy
        probs = np.array([c / total for c in value_counts.values()])
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        return CategoricalProfile(
            feature_name=name,
            value_counts=value_counts,
            top_k=top_k,
            entropy=entropy,
            n_unique=len(value_counts),
            missing_rate=missing_rate,
            n_samples=len(col),
        )
