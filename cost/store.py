"""
CostStore: records and queries cost history.

In-memory by default. Pass a path for JSON persistence.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Optional

from cost.calculator import CostBreakdown


class CostStore:
    """
    Thread-safe store for cost records.

    Args:
        path:       Optional JSON file path for persistence
        max_memory: Max records to keep in memory (oldest dropped)
    """

    def __init__(
        self,
        path: Optional[str | Path] = None,
        max_memory: int = 10_000,
    ):
        self._path = Path(path) if path else None
        self._records: deque[dict] = deque(maxlen=max_memory)
        self._lock = threading.Lock()
        self._total_cost: float = 0.0
        self._total_requests: int = 0

        if self._path and self._path.exists():
            self._load()

    def record(self, cost: CostBreakdown) -> None:
        entry = {**cost.to_dict(), "timestamp": time.time()}
        with self._lock:
            self._records.append(entry)
            self._total_cost += cost.total_cost_usd
            self._total_requests += 1
        if self._path:
            self._save()

    def get_all(self) -> list[dict]:
        with self._lock:
            return list(self._records)

    def get_by_model(self, model_name: str) -> list[dict]:
        return [r for r in self.get_all() if r["model_name"] == model_name]

    def summary(self) -> dict:
        records = self.get_all()
        if not records:
            return {"total_requests": 0, "total_cost_usd": 0.0}

        costs = [r["total_cost_usd"] for r in records]
        latencies = [r["latency_ms"] for r in records]

        # Per model breakdown
        by_model: dict[str, list[float]] = {}
        for r in records:
            by_model.setdefault(r["model_name"], []).append(r["total_cost_usd"])

        return {
            "total_requests": len(records),
            "total_cost_usd": round(sum(costs), 6),
            "avg_cost_usd": round(sum(costs) / len(costs), 8),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2),
            "cost_per_1k_usd": round(sum(costs) / len(costs) * 1000, 4),
            "projected_monthly_10rps_usd": round(
                (sum(costs) / len(costs)) * 10 * 30 * 24 * 3600, 2
            ),
            "by_model": {
                name: {
                    "requests": len(vals),
                    "total_cost_usd": round(sum(vals), 6),
                    "avg_cost_usd": round(sum(vals) / len(vals), 8),
                }
                for name, vals in by_model.items()
            },
        }

    def hints(self) -> list[str]:
        """Return all unique optimization hints across recorded requests."""
        seen = set()
        out = []
        for r in self.get_all():
            h = r.get("optimization_hint")
            if h and h not in seen:
                seen.add(h)
                out.append(h)
        return out

    def _save(self) -> None:
        with self._lock:
            data = list(self._records)
        with open(self._path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        with open(self._path) as f:
            data = json.load(f)
        for entry in data:
            self._records.append(entry)
            self._total_cost += entry.get("total_cost_usd", 0)
            self._total_requests += 1
