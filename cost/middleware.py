"""
CostMiddleware: FastAPI middleware that tracks cost on every request.

Usage:
    from cost import CostMiddleware

    app = FastAPI()
    app.add_middleware(
        CostMiddleware,
        model_name="my-dcgan",
        architecture="dcgan",
        hardware="t4",
    )

    # Cost data available at GET /cost/summary and GET /cost/records
"""

from __future__ import annotations
import time
from typing import Optional

try:
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.requests import Request
    from starlette.responses import Response, JSONResponse
except ImportError:
    BaseHTTPMiddleware = object  # fallback so class definition doesn't fail

from cost.calculator import CostCalculator
from cost.store import CostStore


class CostMiddleware(BaseHTTPMiddleware):
    """
    Drop-in FastAPI middleware for per-request cost tracking.

    Automatically tracks cost for all POST requests to endpoints
    containing 'predict' or 'infer' in the path.

    Adds response headers:
        X-Inference-Cost-USD: 0.00000412
        X-Inference-Latency-Ms: 38.4
        X-Cost-Per-1K-USD: 0.00412

    Exposes endpoints:
        GET /cost/summary   — aggregate cost stats
        GET /cost/records   — raw per-request records
        GET /cost/hints     — optimization recommendations
    """

    def __init__(
        self,
        app,
        model_name: str = "model",
        architecture: str = "unknown",
        hardware: str = "default",
        parameters_millions: float = 0.0,
        memory_gb: float = 0.0,
        track_paths: Optional[list[str]] = None,
        persist_path: Optional[str] = None,
        add_cost_routes: bool = True,
    ):
        super().__init__(app)
        self.calculator = CostCalculator(
            hardware=hardware,
            architecture=architecture,
            model_name=model_name,
            parameters_millions=parameters_millions,
            memory_gb=memory_gb,
        )
        self.store = CostStore(path=persist_path)
        self.track_paths = track_paths or ["predict", "infer", "inference"]
        self.add_cost_routes = add_cost_routes
        self._model_name = model_name

    def _should_track(self, path: str) -> bool:
        path_lower = path.lower()
        return any(p in path_lower for p in self.track_paths)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Serve cost endpoints
        if request.method == "GET":
            if request.url.path == "/cost/summary":
                return JSONResponse(self.store.summary())
            if request.url.path == "/cost/records":
                records = self.store.get_all()
                return JSONResponse({"records": records[-100:]})  # last 100
            if request.url.path == "/cost/hints":
                return JSONResponse({"hints": self.store.hints()})

        # Track inference requests
        if self._should_track(request.url.path):
            start = time.perf_counter()
            response = await call_next(request)
            latency_ms = (time.perf_counter() - start) * 1000

            cost = self.calculator.calculate(latency_ms)
            self.store.record(cost)

            # Inject cost headers
            response.headers["X-Inference-Cost-USD"] = f"{cost.total_cost_usd:.8f}"
            response.headers["X-Inference-Latency-Ms"] = f"{latency_ms:.2f}"
            response.headers["X-Cost-Per-1K-USD"] = f"{cost.cost_per_1k_requests:.6f}"
            if cost.optimization_hint:
                response.headers["X-Cost-Hint"] = cost.optimization_hint[:200]
            return response

        return await call_next(request)
