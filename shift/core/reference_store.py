"""
Saves and loads reference (training) distribution profiles to disk.
Stored as .shift files (JSON under the hood) in experiments/shift_profiles/.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from shift.core.feature_profiles import DatasetProfile, FeatureProfiler

DEFAULT_STORE_DIR = Path("experiments/shift_profiles")


class ReferenceStore:
    """
    Profiles training data and persists it to a .shift file.

    Usage:
        store = ReferenceStore()
        store.save(X_train, name="my_model")
        profile = store.load("my_model")
    """

    def __init__(self, store_dir: Optional[str] = None):
        self.store_dir = Path(store_dir) if store_dir else DEFAULT_STORE_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._profiler = FeatureProfiler()

    # ------------------------------------------------------------------ public

    def save(
        self,
        data,
        name: str,
        feature_names=None,
        metadata: Optional[dict] = None,
    ) -> Path:
        """
        Profile `data` and save to <store_dir>/<name>.shift

        Args:
            data:          numpy array (n, f) or pandas DataFrame
            name:          identifier — use your model name e.g. "fraud_classifier"
            feature_names: optional list of column names
            metadata:      any extra info to embed (model version, dataset path, etc.)

        Returns:
            Path to the saved .shift file
        """
        profile = self._profiler.profile(data, feature_names)
        checksum = self._checksum(data)

        payload = {
            "profile": profile.to_dict(),
            "checksum": checksum,
            "metadata": metadata or {},
        }

        path = self._path(name)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

        print(f"[SHIFT] Reference profile saved → {path}")
        print(
            f"        {profile.n_samples} samples · "
            f"{len(profile.numerical)} numerical · "
            f"{len(profile.categorical)} categorical features"
        )
        return path

    def load(self, name_or_path: str) -> DatasetProfile:
        """
        Load a reference profile from a .shift file.

        Args:
            name_or_path: either a short name ("my_model") or a full file path
        """
        path = self._resolve(name_or_path)
        if not path.exists():
            raise FileNotFoundError(
                f"[SHIFT] No reference profile found at {path}. "
                f"Run ReferenceStore().save(X_train, '{name_or_path}') first."
            )
        with open(path) as f:
            payload = json.load(f)
        return DatasetProfile.from_dict(payload["profile"])

    def load_metadata(self, name_or_path: str) -> dict:
        path = self._resolve(name_or_path)
        with open(path) as f:
            payload = json.load(f)
        return payload.get("metadata", {})

    def exists(self, name_or_path: str) -> bool:
        return self._resolve(name_or_path).exists()

    def list_profiles(self):
        """List all saved .shift profiles."""
        profiles = list(self.store_dir.glob("*.shift"))
        if not profiles:
            print("[SHIFT] No profiles saved yet.")
            return []
        print(f"[SHIFT] Saved profiles in {self.store_dir}:")
        for p in profiles:
            size = p.stat().st_size // 1024
            print(f"  · {p.stem:<30} ({size} KB)")
        return [p.stem for p in profiles]

    # ------------------------------------------------------------------ private

    def _path(self, name: str) -> Path:
        if not name.endswith(".shift"):
            name += ".shift"
        return self.store_dir / name

    def _resolve(self, name_or_path: str) -> Path:
        p = Path(name_or_path)
        if p.suffix == ".shift" and (
            p.is_absolute() or "/" in name_or_path or "\\" in name_or_path
        ):
            return p
        # Also handle the case where a directory path is given without .shift extension
        # e.g. "C:\Users\...\mymodel" — treat parent as store_dir, stem as name
        p_as_path = Path(name_or_path)
        if p_as_path.parent != Path(".") and not p_as_path.suffix:
            # Looks like a directory/name combo — use parent as store, stem as name
            candidate = p_as_path.parent / (p_as_path.name + ".shift")
            if candidate.exists():
                return candidate
        return self._path(name_or_path)

    def _checksum(self, data) -> str:
        try:
            import numpy as np

            arr = np.array(data)
            return hashlib.md5(arr.tobytes()).hexdigest()
        except Exception:
            return "unavailable"


# ─── convenience shortcut ────────────────────────────────────────────────────


def save_reference(data, name: str, feature_names=None, **kwargs) -> Path:
    return ReferenceStore().save(data, name, feature_names, **kwargs)


def load_reference(name: str) -> DatasetProfile:
    return ReferenceStore().load(name)
