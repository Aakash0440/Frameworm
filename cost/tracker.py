"""
CostTracker: context manager that times inference and computes cost.

Usage:
    tracker = CostTracker(model_name="dcgan-v1", architecture="dcgan")

    with tracker.track():
        output = model(input_tensor)

    print(tracker.last_cost.total_cost_usd)
    print(tracker.last_cost.optimization_hint)
"""

from __future__ import annotations
import time
from contextlib import contextmanager
from typing import Optional, Generator

from cost.calculator import CostCalculator, CostBreakdown
from cost.store import CostStore


class CostTracker:
    """
    Tracks inference cost per request.
    Thread-safe. Can be reused across requests.

    Args:
        model_name:           Human-readable model identifier
        architecture:         One of: vae, dcgan, ddpm, cfg_ddpm, vqvae2, vit_gan, unknown
        hardware:             One of: a10g, a100, t4, v100, cpu_small, cpu_medium, default
        parameters_millions:  Model param count in millions (optional, improves hints)
        memory_gb:            Model memory footprint in GB (optional)
        store:                CostStore instance. If None, creates an in-memory store.
        auto_store:           Whether to auto-save each tracked request
    """

    def __init__(
        self,
        model_name: str = "model",
        architecture: str = "unknown",
        hardware: str = "default",
        parameters_millions: float = 0.0,
        memory_gb: float = 0.0,
        store: Optional[CostStore] = None,
        auto_store: bool = True,
    ):
        self.model_name = model_name
        self.architecture = architecture
        self.auto_store = auto_store

        self.calculator = CostCalculator(
            hardware=hardware,
            architecture=architecture,
            model_name=model_name,
            parameters_millions=parameters_millions,
            memory_gb=memory_gb,
        )

        self.store = store or CostStore()
        self.last_cost: Optional[CostBreakdown] = None
        self._total_requests: int = 0
        self._total_cost: float = 0.0

    @contextmanager
    def track(self, batch_size: int = 1) -> Generator[None, None, None]:
        """
        Context manager that times the block and computes cost.

        with tracker.track():
            output = model(input)
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            latency_ms = (time.perf_counter() - start) * 1000
            cost = self.calculator.calculate(latency_ms, batch_size=batch_size)
            self.last_cost = cost
            self._total_requests += 1
            self._total_cost += cost.total_cost_usd

            if self.auto_store:
                self.store.record(cost)

    def track_call(self, fn, *args, batch_size: int = 1, **kwargs):
        """
        Functional wrapper alternative to context manager.

        result, cost = tracker.track_call(model.predict, input_data)
        """
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        cost = self.calculator.calculate(latency_ms, batch_size=batch_size)
        self.last_cost = cost
        self._total_requests += 1
        self._total_cost += cost.total_cost_usd
        if self.auto_store:
            self.store.record(cost)
        return result, cost

    @property
    def total_cost(self) -> float:
        return self._total_cost

    @property
    def total_requests(self) -> int:
        return self._total_requests

    @property
    def average_cost(self) -> float:
        if self._total_requests == 0:
            return 0.0
        return self._total_cost / self._total_requests

    def summary(self) -> dict:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "total_requests": self._total_requests,
            "total_cost_usd": round(self._total_cost, 6),
            "average_cost_usd": round(self.average_cost, 8),
            "cost_per_1k_usd": round(self.average_cost * 1000, 4),
            "projected_monthly_10rps_usd": round(
                self.average_cost * 10 * 30 * 24 * 3600, 2
            ),
        }
