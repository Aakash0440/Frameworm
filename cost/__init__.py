"""
FRAMEWORM-COST
Per-request ML inference cost tracking, alerting, and optimization.

Quick start:
    # Drop into FastAPI (zero config)
    from cost import CostMiddleware
    app.add_middleware(CostMiddleware, model_name="my-model", architecture="dcgan")

    # Wrap any inference call
    from cost import CostTracker
    tracker = CostTracker(model_name="my-model", architecture="dcgan")
    with tracker.track():
        output = model(input)

    # Dashboard at /cost/dashboard
    from cost.dashboard import mount_dashboard
    mount_dashboard(app, tracker.store, alerter)

    # CLI
    frameworm-cost estimate --arch dcgan --hardware t4 --latency 38
    frameworm-cost compare --latency 50 --hardware t4
    frameworm-cost report costs.json
"""

from cost.calculator import CostCalculator, CostBreakdown
from cost.tracker import CostTracker
from cost.store import CostStore
from cost.report import CostReport
from cost.alerter import CostAlerter, Alert

try:
    from cost.middleware import CostMiddleware
except ImportError:
    CostMiddleware = None

__version__ = "0.1.0"
__all__ = [
    "CostMiddleware",
    "CostTracker",
    "CostCalculator",
    "CostBreakdown",
    "CostStore",
    "CostReport",
    "CostAlerter",
    "Alert",
]
