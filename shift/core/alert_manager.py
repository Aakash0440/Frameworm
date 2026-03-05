"""
Fires alerts when drift is detected.
Reuses FRAMEWORM's existing Slack integration.
Supports: Slack, webhook URL, log file, stdout.
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from shift.core.drift_engine import DriftResult, DriftSeverity


logger = logging.getLogger("frameworm.shift")


class AlertManager:
    """
    Sends drift alerts through one or more channels.

    Channels: slack | webhook | log | stdout
    Configurable via configs/shift_config.yaml or constructor args.

    Usage:
        manager = AlertManager(channels=["slack", "log"])
        manager.alert_if_needed(drift_result, model_name="fraud_classifier")
    """

    # Alert when severity >= this level
    DEFAULT_ALERT_SEVERITY = DriftSeverity.MEDIUM

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        slack_webhook: Optional[str] = None,
        webhook_url: Optional[str] = None,
        log_path: Optional[str] = None,
        min_severity: DriftSeverity = DEFAULT_ALERT_SEVERITY,
    ):
        self.channels = channels or self._channels_from_config()
        self.slack_webhook = slack_webhook or os.getenv("FRAMEWORM_SLACK_WEBHOOK")
        self.webhook_url = webhook_url or os.getenv("FRAMEWORM_SHIFT_WEBHOOK")
        self.log_path = Path(log_path) if log_path else Path("experiments/shift_logs")
        self.log_path.mkdir(parents=True, exist_ok=True)
        self.min_severity = min_severity

    def alert_if_needed(
        self,
        result: DriftResult,
        model_name: str = "unknown",
    ) -> bool:
        """
        Fire alerts if drift severity meets threshold.
        Returns True if alerts were fired.
        """
        severity_order = [
            DriftSeverity.NONE,
            DriftSeverity.LOW,
            DriftSeverity.MEDIUM,
            DriftSeverity.HIGH,
        ]

        if severity_order.index(result.overall_severity) < severity_order.index(self.min_severity):
            return False

        payload = self._build_payload(result, model_name)

        for channel in self.channels:
            try:
                if channel == "slack":
                    self._send_slack(payload, result)
                elif channel == "webhook":
                    self._send_webhook(payload)
                elif channel == "log":
                    self._write_log(payload)
                elif channel == "stdout":
                    result.print_summary()
            except Exception as e:
                logger.warning(f"[SHIFT] Alert channel '{channel}' failed: {e}")

        return True

    # ────────────────────────────────────────────────────────────── channels

    def _send_slack(self, payload: dict, result: DriftResult):
        """
        Reuses FRAMEWORM's Slack integration pattern.
        Falls back to direct webhook POST if integration unavailable.
        """
        import urllib.request

        webhook = self.slack_webhook
        if not webhook:
            logger.warning("[SHIFT] Slack webhook not configured. "
                           "Set FRAMEWORM_SLACK_WEBHOOK env var.")
            return

        severity_emoji = {
            "NONE":   ":white_check_mark:",
            "LOW":    ":warning:",
            "MEDIUM": ":orange_circle:",
            "HIGH":   ":red_circle:",
        }
        sev = payload["overall_severity"]
        emoji = severity_emoji.get(sev, ":warning:")

        drifted = payload["drifted_features"]
        feature_lines = "\n".join(
            f"  • `{f}`  p={payload['features'][f]['p_value']:.4f}  "
            f"({payload['features'][f]['severity']})"
            for f in drifted[:10]
        )
        if len(drifted) > 10:
            feature_lines += f"\n  _...and {len(drifted)-10} more_"

        message = {
            "text": f"{emoji} *FRAMEWORM SHIFT — Drift Detected*",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"{emoji} *Drift Alert* — `{payload['model_name']}`\n"
                            f"Severity: *{sev}*  |  "
                            f"{len(drifted)}/{payload['n_features_checked']} features drifted\n"
                            f"{payload['summary']}"
                        )
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Drifted features:*\n{feature_lines}" if drifted else "_No features drifted._"
                    }
                },
                {
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"Detected at {payload['detected_at']}"}]
                }
            ]
        }

        data = json.dumps(message).encode("utf-8")
        req = urllib.request.Request(
            webhook,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            logger.info(f"[SHIFT] Slack alert sent (status {resp.status})")

    def _send_webhook(self, payload: dict):
        import urllib.request
        if not self.webhook_url:
            logger.warning("[SHIFT] Webhook URL not configured.")
            return
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5):
            pass

    def _write_log(self, payload: dict):
        log_file = self.log_path / "drift_alerts.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(payload) + "\n")
        logger.info(f"[SHIFT] Drift alert logged → {log_file}")

    # ────────────────────────────────────────────────────────── helpers

    def _build_payload(self, result: DriftResult, model_name: str) -> dict:
        d = result.to_dict()
        d["model_name"] = model_name
        d["detected_at"] = datetime.utcnow().isoformat()
        return d

    def _channels_from_config(self) -> List[str]:
        """Read channels from configs/shift_config.yaml if available."""
        try:
            config_path = Path("configs/shift_config.yaml")
            if config_path.exists():
                import yaml
                with open(config_path) as f:
                    cfg = yaml.safe_load(f)
                return cfg.get("shift", {}).get("alert_on", ["stdout"])
        except Exception:
            pass
        return ["stdout"]

