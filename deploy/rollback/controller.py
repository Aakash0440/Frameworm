"""
Executes rollback when degradation is detected.
Swaps to the previous known-good version and alerts via Slack.
"""

import json
import logging
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger("frameworm.deploy.rollback")


class RollbackController:
    """
    Handles rollback from a degraded deployment to the previous good one.

    On rollback:
    1.  Looks up previous good version in model registry
    2.  Stops the current Docker container
    3.  Starts the previous version's container
    4.  Fires Slack alert with reason + metrics
    5.  Writes rollback event to deployment log

    Usage:
        controller = RollbackController(model_name="fraud_v2")
        controller.rollback("p95 latency exceeded threshold")
    """

    DEPLOY_LOG = Path("experiments/deploy_logs")

    def __init__(self, model_name: str, current_version: str):
        self.model_name      = model_name
        self.current_version = current_version
        self.DEPLOY_LOG.mkdir(parents=True, exist_ok=True)

    def rollback(self, reason: str, latency_summary: Optional[dict] = None):
        """
        Execute rollback. Called by DegradationMonitor.
        """
        logger.error(f"[DEPLOY] Initiating rollback for {self.model_name} — {reason}")

        previous_version = self._find_previous_version()

        if previous_version is None:
            logger.error(
                "[DEPLOY] No previous version found in registry. "
                "Cannot auto-rollback. Alerting and waiting for manual intervention."
            )
            self._alert(reason, rolled_back=False, latency_summary=latency_summary)
            return

        # Stop current container
        self._stop_container(f"frameworm-{self.model_name}-{self.current_version}")

        # Start previous version
        started = self._start_container(
            f"frameworm/{self.model_name}:{previous_version}",
            f"frameworm-{self.model_name}-{previous_version}",
        )

        if started:
            # Update registry
            self._update_registry_on_rollback(previous_version)
            logger.info(
                f"[DEPLOY] Rollback complete — "
                f"{self.current_version} → {previous_version}"
            )

        self._alert(reason, rolled_back=started,
                    previous_version=previous_version,
                    latency_summary=latency_summary)
        self._log_event(reason, previous_version, started, latency_summary)

    # ──────────────────────────────────────────────── registry lookup

    def _find_previous_version(self) -> Optional[str]:
        """Find the last production version before the current one."""
        try:
            from deploy.core.registry import ModelRegistry
            registry = ModelRegistry()
            history  = registry.get_production_history(self.model_name)
            for entry in reversed(history):
                if entry["version"] != self.current_version:
                    return entry["version"]
        except Exception as e:
            logger.warning(f"[DEPLOY] Registry lookup failed: {e}")
        return None

    # ──────────────────────────────────────────────── container ops

    def _stop_container(self, container_name: str):
        if not shutil.which("docker"):
            logger.warning("[DEPLOY] Docker not available — skipping container stop")
            return
        try:
            subprocess.run(
                ["docker", "stop", container_name],
                check=True, capture_output=True, timeout=30
            )
            logger.info(f"[DEPLOY] Stopped container: {container_name}")
        except subprocess.CalledProcessError:
            logger.warning(f"[DEPLOY] Could not stop {container_name} — may already be stopped")

    def _start_container(self, image: str, container_name: str) -> bool:
        if not shutil.which("docker"):
            logger.warning("[DEPLOY] Docker not available — cannot start rollback container")
            return False
        try:
            subprocess.run(
                ["docker", "run", "-d", "--name", container_name, image],
                check=True, capture_output=True, timeout=60
            )
            logger.info(f"[DEPLOY] Started rollback container: {container_name}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"[DEPLOY] Failed to start rollback container: {e}")
            return False

    # ──────────────────────────────────────────────── alert

    def _alert(
        self, reason: str,
        rolled_back: bool = False,
        previous_version: Optional[str] = None,
        latency_summary: Optional[dict] = None,
    ):
        """Fire Slack alert. Reuses FRAMEWORM Slack integration."""
        import os, urllib.request
        webhook = os.getenv("FRAMEWORM_SLACK_WEBHOOK")
        if not webhook:
            logger.warning("[DEPLOY] No Slack webhook configured — alert not sent")
            return

        status_emoji = "✅" if rolled_back else "🚨"
        msg_text = (
            f"{status_emoji} *FRAMEWORM DEPLOY — {'Rollback Complete' if rolled_back else 'Rollback Failed'}*\n"
            f"*Model:* `{self.model_name}`\n"
            f"*Reason:* {reason}\n"
            f"*From:* `{self.current_version}`"
        )
        if rolled_back and previous_version:
            msg_text += f"\n*To:* `{previous_version}`"
        if latency_summary:
            p95 = latency_summary.get("p95_ms", "—")
            msg_text += f"\n*p95 at rollback:* {p95:.0f}ms"

        data = json.dumps({"text": msg_text}).encode()
        try:
            req = urllib.request.Request(
                webhook, data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.warning(f"[DEPLOY] Slack alert failed: {e}")

    # ──────────────────────────────────────────────── logging

    def _log_event(self, reason, previous_version, success, latency_summary):
        log_file = self.DEPLOY_LOG / f"{self.model_name}_rollbacks.jsonl"
        entry = {
            "timestamp":        datetime.utcnow().isoformat(),
            "model_name":       self.model_name,
            "from_version":     self.current_version,
            "to_version":       previous_version,
            "reason":           reason,
            "success":          success,
            "latency_summary":  latency_summary,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def _update_registry_on_rollback(self, previous_version: str):
        try:
            from deploy.core.registry import ModelRegistry
            registry = ModelRegistry()
            registry.promote(self.model_name, previous_version, "production")
            registry.demote(self.model_name, self.current_version, "archived",
                            note="Auto-archived after rollback")
        except Exception as e:
            logger.warning(f"[DEPLOY] Registry update after rollback failed: {e}")

