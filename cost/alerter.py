"""
CostAlerter: fires alerts when cost or latency thresholds are breached.

Supports:
  - Slack webhooks
  - Generic HTTP webhooks
  - Console/log alerts (default)

Usage:
    alerter = CostAlerter(
        slack_webhook="https://hooks.slack.com/...",
        monthly_threshold_usd=500,
        cost_per_request_threshold_usd=0.001,
        latency_threshold_ms=200,
    )
    alerter.check(cost_breakdown)
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import Callable, Optional

from cost.calculator import CostBreakdown


@dataclass
class Alert:
    level: str  # "warning" | "critical"
    type: str  # "cost_spike" | "latency_spike" | "monthly_projection"
    message: str
    value: float
    threshold: float
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "type": self.type,
            "message": self.message,
            "value": round(self.value, 8),
            "threshold": round(self.threshold, 8),
            "timestamp": self.timestamp,
        }

    def to_slack_payload(self, model_name: str = "model") -> dict:
        emoji = "🔴" if self.level == "critical" else "🟡"
        return {
            "text": f"{emoji} *FRAMEWORM-COST ALERT* [{model_name}]",
            "attachments": [
                {
                    "color": "#cc4444" if self.level == "critical" else "#cc8800",
                    "fields": [
                        {"title": "Type", "value": self.type, "short": True},
                        {"title": "Level", "value": self.level.upper(), "short": True},
                        {"title": "Value", "value": str(round(self.value, 6)), "short": True},
                        {
                            "title": "Threshold",
                            "value": str(round(self.threshold, 6)),
                            "short": True,
                        },
                        {"title": "Message", "value": self.message, "short": False},
                    ],
                    "footer": "FRAMEWORM-COST",
                    "ts": int(self.timestamp),
                }
            ],
        }


class CostAlerter:
    """
    Monitors cost breakdowns and fires alerts on threshold breach.

    Args:
        slack_webhook:                    Slack incoming webhook URL
        webhook_url:                      Generic HTTP POST webhook
        monthly_threshold_usd:            Alert if projected monthly > this
        cost_per_request_threshold_usd:   Alert if single request > this
        latency_threshold_ms:             Alert if latency > this
        on_alert:                         Custom callback(alert: Alert)
        cooldown_seconds:                 Min seconds between same-type alerts
        model_name:                       Used in alert messages
    """

    def __init__(
        self,
        slack_webhook: Optional[str] = None,
        webhook_url: Optional[str] = None,
        monthly_threshold_usd: float = 1000.0,
        cost_per_request_threshold_usd: float = 0.01,
        latency_threshold_ms: float = 500.0,
        on_alert: Optional[Callable[[Alert], None]] = None,
        cooldown_seconds: float = 300.0,
        model_name: str = "model",
    ):
        self.slack_webhook = slack_webhook
        self.webhook_url = webhook_url
        self.monthly_threshold = monthly_threshold_usd
        self.cost_threshold = cost_per_request_threshold_usd
        self.latency_threshold = latency_threshold_ms
        self.on_alert = on_alert
        self.cooldown = cooldown_seconds
        self.model_name = model_name

        self._last_fired: dict[str, float] = {}
        self._alert_history: list[Alert] = []
        self._lock = threading.Lock()

    def check(self, cost: CostBreakdown) -> list[Alert]:
        """Check a cost breakdown against thresholds. Returns any alerts fired."""
        alerts = []

        # Per-request cost spike
        if cost.total_cost_usd > self.cost_threshold:
            alert = Alert(
                level="critical",
                type="cost_spike",
                message=(
                    f"Request cost ${cost.total_cost_usd:.6f} exceeds threshold "
                    f"${self.cost_threshold:.6f}. Architecture: {cost.architecture}, "
                    f"latency: {cost.latency_ms:.1f}ms."
                ),
                value=cost.total_cost_usd,
                threshold=self.cost_threshold,
            )
            if self._should_fire("cost_spike"):
                alerts.append(alert)

        # Latency spike
        if cost.latency_ms > self.latency_threshold:
            alert = Alert(
                level="warning",
                type="latency_spike",
                message=(
                    f"Latency {cost.latency_ms:.1f}ms exceeds threshold "
                    f"{self.latency_threshold:.0f}ms on {cost.model_name}."
                ),
                value=cost.latency_ms,
                threshold=self.latency_threshold,
            )
            if self._should_fire("latency_spike"):
                alerts.append(alert)

        # Monthly projection
        monthly = cost.total_cost_usd * 10 * 30 * 24 * 3600
        if monthly > self.monthly_threshold:
            alert = Alert(
                level="warning",
                type="monthly_projection",
                message=(
                    f"At current rate, projected monthly cost is ${monthly:,.2f} "
                    f"(threshold: ${self.monthly_threshold:,.2f}) for {cost.model_name}."
                ),
                value=monthly,
                threshold=self.monthly_threshold,
            )
            if self._should_fire("monthly_projection"):
                alerts.append(alert)

        for alert in alerts:
            self._dispatch(alert)

        return alerts

    def _should_fire(self, alert_type: str) -> bool:
        with self._lock:
            last = self._last_fired.get(alert_type, 0)
            if time.time() - last > self.cooldown:
                self._last_fired[alert_type] = time.time()
                return True
            return False

    def _dispatch(self, alert: Alert) -> None:
        with self._lock:
            self._alert_history.append(alert)

        # Custom callback
        if self.on_alert:
            try:
                self.on_alert(alert)
            except Exception:
                pass

        # Slack
        if self.slack_webhook:
            self._post_webhook(self.slack_webhook, alert.to_slack_payload(self.model_name))

        # Generic webhook
        if self.webhook_url:
            self._post_webhook(
                self.webhook_url,
                {
                    "source": "frameworm-cost",
                    "model": self.model_name,
                    "alert": alert.to_dict(),
                },
            )

        # Always log to console
        level_str = "🔴 CRITICAL" if alert.level == "critical" else "🟡 WARNING"
        print(f"[FRAMEWORM-COST] {level_str} | {alert.type} | {alert.message}")

    def _post_webhook(self, url: str, payload: dict) -> None:
        try:
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"[FRAMEWORM-COST] Webhook failed: {e}")

    @property
    def alert_history(self) -> list[Alert]:
        with self._lock:
            return list(self._alert_history)

    def summary(self) -> dict:
        history = self.alert_history
        by_type: dict[str, int] = {}
        for a in history:
            by_type[a.type] = by_type.get(a.type, 0) + 1
        return {
            "total_alerts": len(history),
            "by_type": by_type,
            "thresholds": {
                "monthly_usd": self.monthly_threshold,
                "per_request_usd": self.cost_threshold,
                "latency_ms": self.latency_threshold,
            },
        }
