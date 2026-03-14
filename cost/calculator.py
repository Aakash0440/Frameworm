"""
CostCalculator: converts inference time + hardware into $/request.

Pricing is based on on-demand cloud rates (configurable).
Default: equivalent to a single A10G GPU on AWS ($1.006/hr).

Formula:
    cost_per_request = (latency_seconds / 3600) * hourly_rate * utilization_factor
    + memory_cost_per_request
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import psutil

# ── Default hardware pricing ($/hr, on-demand) ───────────────────────────────
HARDWARE_RATES = {
    # GPU instances (AWS equivalent)
    "a10g": 1.006,  # g5.xlarge
    "a100": 3.928,  # p4d.24xlarge per GPU
    "t4": 0.526,  # g4dn.xlarge
    "v100": 3.06,  # p3.2xlarge
    # CPU instances
    "cpu_small": 0.096,  # c5.large
    "cpu_medium": 0.192,  # c5.xlarge
    "cpu_large": 0.384,  # c5.2xlarge
    # Default fallback
    "default": 0.526,
}

# ── Architecture complexity multipliers ───────────────────────────────────────
# Relative to a simple VAE (1.0x baseline)
ARCH_COMPLEXITY = {
    "vae": 1.0,
    "dcgan": 1.4,
    "ddpm": 8.5,  # diffusion is expensive
    "cfg_ddpm": 9.2,
    "vqvae2": 2.1,
    "vit_gan": 3.8,
    "transformer": 5.0,
    "resnet": 1.2,
    "unknown": 1.5,
}


@dataclass
class CostBreakdown:
    """Full cost breakdown for a single inference request."""

    model_name: str
    architecture: str
    latency_ms: float
    compute_cost_usd: float
    memory_cost_usd: float
    total_cost_usd: float
    hardware: str
    batch_size: int = 1
    parameters_millions: float = 0.0
    optimization_hint: Optional[str] = None

    @property
    def cost_per_1k_requests(self) -> float:
        return self.total_cost_usd * 1000

    @property
    def monthly_cost_at_rps(self) -> dict[str, float]:
        """Estimated monthly cost at various request rates."""
        month_seconds = 30 * 24 * 3600
        rps_rates = [1, 10, 100, 1000]
        return {f"{rps}_rps": self.total_cost_usd * rps * month_seconds for rps in rps_rates}

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "architecture": self.architecture,
            "latency_ms": round(self.latency_ms, 3),
            "compute_cost_usd": round(self.compute_cost_usd, 8),
            "memory_cost_usd": round(self.memory_cost_usd, 8),
            "total_cost_usd": round(self.total_cost_usd, 8),
            "cost_per_1k_requests": round(self.cost_per_1k_requests, 6),
            "hardware": self.hardware,
            "batch_size": self.batch_size,
            "parameters_millions": self.parameters_millions,
            "optimization_hint": self.optimization_hint,
        }


class CostCalculator:
    """
    Converts inference latency + model metadata into dollar cost per request.

    Example:
        calc = CostCalculator(hardware="t4", architecture="dcgan")
        cost = calc.calculate(latency_ms=38.0, batch_size=1)
        print(f"${cost.total_cost_usd:.6f} per request")
    """

    def __init__(
        self,
        hardware: str = "default",
        architecture: str = "unknown",
        model_name: str = "model",
        parameters_millions: float = 0.0,
        memory_gb: float = 0.0,
        memory_cost_per_gb_hr: float = 0.016,  # $/GB/hr (approx AWS)
    ):
        self.hardware = hardware.lower()
        self.architecture = architecture.lower()
        self.model_name = model_name
        self.parameters_millions = parameters_millions
        self.memory_gb = memory_gb
        self.memory_cost_per_gb_hr = memory_cost_per_gb_hr

        self.hourly_rate = HARDWARE_RATES.get(self.hardware, HARDWARE_RATES["default"])
        self.arch_multiplier = ARCH_COMPLEXITY.get(self.architecture, ARCH_COMPLEXITY["unknown"])

    def calculate(
        self,
        latency_ms: float,
        batch_size: int = 1,
    ) -> CostBreakdown:
        """Calculate cost for a single inference call."""
        latency_sec = latency_ms / 1000.0

        # Compute cost: time on hardware × rate × architecture weight
        compute_cost = (latency_sec / 3600.0) * self.hourly_rate * self.arch_multiplier

        # Memory cost: GB held in memory × time × rate
        memory_cost = (latency_sec / 3600.0) * self.memory_gb * self.memory_cost_per_gb_hr

        # Per-item cost if batched
        total = (compute_cost + memory_cost) / max(batch_size, 1)

        hint = self._optimization_hint(latency_ms, batch_size, total)

        return CostBreakdown(
            model_name=self.model_name,
            architecture=self.architecture,
            latency_ms=latency_ms,
            compute_cost_usd=compute_cost / max(batch_size, 1),
            memory_cost_usd=memory_cost / max(batch_size, 1),
            total_cost_usd=total,
            hardware=self.hardware,
            batch_size=batch_size,
            parameters_millions=self.parameters_millions,
            optimization_hint=hint,
        )

    def _optimization_hint(
        self, latency_ms: float, batch_size: int, cost_per_req: float
    ) -> Optional[str]:
        """Return actionable optimization hint if cost is high."""
        monthly_at_10rps = cost_per_req * 10 * 30 * 24 * 3600

        if latency_ms > 200 and self.architecture in ("ddpm", "cfg_ddpm"):
            return (
                "Diffusion model latency is high. Consider DDIM sampling "
                "(10-50 steps vs 1000) — up to 50x speedup with minimal quality loss."
            )
        if batch_size == 1 and latency_ms > 50:
            return (
                f"Single-item inference. Batching 8-16 requests could reduce "
                f"cost/request by 60-80%. Est. monthly saving at 10 rps: "
                f"${monthly_at_10rps * 0.7:,.0f}."
            )
        if self.parameters_millions > 100 and latency_ms > 100:
            return (
                "Large model with high latency. INT8 quantization could give "
                "2-4x speedup with <1% accuracy loss on most architectures."
            )
        if monthly_at_10rps > 1000:
            return (
                f"At 10 req/s this model costs ~${monthly_at_10rps:,.0f}/month. "
                f"Consider model distillation or a smaller architecture."
            )
        return None

    def compare_architectures(
        self, latency_ms: float, architectures: list[str] | None = None
    ) -> list[CostBreakdown]:
        """Compare cost across architectures at the same latency."""
        archs = architectures or list(ARCH_COMPLEXITY.keys())
        results = []
        for arch in archs:
            calc = CostCalculator(
                hardware=self.hardware,
                architecture=arch,
                model_name=self.model_name,
                parameters_millions=self.parameters_millions,
                memory_gb=self.memory_gb,
            )
            results.append(calc.calculate(latency_ms))
        return sorted(results, key=lambda x: x.total_cost_usd)
