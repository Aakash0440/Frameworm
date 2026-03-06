"""
FRAMEWORM DEPLOY — Latency tracker.
Tracks p50/p95/p99 inference latency per deployed model.
Persists to experiments/deploy_logs/<model_name>_latency.db

Two recording APIs:
    tracker.record(elapsed_ms)               ← direct insert (used by tests + monitor)
    tracker.end_request(start_time, success) ← used by FastAPI middleware
"""

import sqlite3
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np

logger = logging.getLogger("frameworm.deploy")

LOGS_DIR = Path("experiments/deploy_logs")


@dataclass
class LatencySnapshot:
    model_name:   str
    p50_ms:       float
    p95_ms:       float
    p99_ms:       float
    mean_ms:      float
    max_ms:       float
    min_ms:       float
    n_requests:   int
    error_rate:   float
    window_start: str
    window_end:   str

    def breaches_threshold(self, p95_warn_ms: float, p95_crit_ms: float) -> str:
        if self.p95_ms >= p95_crit_ms:
            return "critical"
        if self.p95_ms >= p95_warn_ms:
            return "warn"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "model_name":   self.model_name,
            "p50_ms":       round(self.p50_ms, 2),
            "p95_ms":       round(self.p95_ms, 2),
            "p99_ms":       round(self.p99_ms, 2),
            "mean_ms":      round(self.mean_ms, 2),
            "max_ms":       round(self.max_ms, 2),
            "min_ms":       round(self.min_ms, 2),
            "n_requests":   self.n_requests,
            "error_rate":   round(self.error_rate, 4),
            "window_start": self.window_start,
            "window_end":   self.window_end,
        }

    def print_summary(self):
        colours = {"ok": "\033[92m", "warn": "\033[93m", "critical": "\033[91m"}
        reset = "\033[0m"
        cfg = _load_config()
        status = self.breaches_threshold(
            cfg.get("latency_p95_warn_ms", 200),
            cfg.get("latency_p95_crit_ms", 500),
        )
        c = colours.get(status, "")
        print(f"\n[DEPLOY] Latency — {self.model_name}  {c}[{status.upper()}]{reset}")
        print(f"  p50={self.p50_ms:.1f}ms  p95={self.p95_ms:.1f}ms  "
              f"p99={self.p99_ms:.1f}ms  mean={self.mean_ms:.1f}ms")
        print(f"  requests={self.n_requests}  "
              f"error_rate={self.error_rate*100:.2f}%\n")


def _load_config() -> dict:
    try:
        import yaml
        with open("configs/deploy_config.yaml") as f:
            return yaml.safe_load(f).get("deploy", {})
    except Exception:
        return {}


class LatencyTracker:
    WINDOW_SIZE = 1000

    def __init__(self, model_name: str, logs_dir: Optional[str] = None,
                 persist_every: int = 50):
        self.model_name    = model_name
        self.logs_dir      = Path(logs_dir) if logs_dir else LOGS_DIR
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.persist_every = persist_every

        self._lock          = threading.Lock()
        self._timings_ms    = deque(maxlen=self.WINDOW_SIZE)
        self._errors        = deque(maxlen=self.WINDOW_SIZE)
        self._request_count = 0
        self._window_start  = datetime.utcnow().isoformat()

        self._db_path = self.logs_dir / f"{model_name}_latency.db"
        self._init_db()

    # ─── recording APIs ───────────────────────────────────────────────────────

    def record(self, elapsed_ms: float, success: bool = True):
        """
        Directly record a latency value in milliseconds.
        Used by tests, the middleware, and DegradationMonitor.
        """
        with self._lock:
            self._timings_ms.append(elapsed_ms)
            self._errors.append(not success)
            self._request_count += 1
            if self._request_count % self.persist_every == 0:
                # Use _snapshot_unlocked to avoid re-entrant lock deadlock
                snap = self._snapshot_unlocked()
                if snap is not None:
                    self._write_snapshot_to_db(snap)

    def start_request(self) -> float:
        """Call at request start. Returns start timestamp."""
        return time.perf_counter()

    def end_request(self, start_time: float, success: bool = True):
        """
        Call at request end. Computes elapsed ms and delegates to record().
        Deadlock-safe: does NOT call snapshot() while holding the lock.
        """
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        self.record(elapsed_ms, success=success)

    # ─── stats ────────────────────────────────────────────────────────────────

    def snapshot(self) -> Optional[LatencySnapshot]:
        """Compute p50/p95/p99 from rolling window. Acquires lock."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self) -> Optional[LatencySnapshot]:
        """Same as snapshot() but assumes lock is already held by caller."""
        timings = list(self._timings_ms)
        errors  = list(self._errors)
        if not timings:
            return None
        arr = np.array(timings)
        return LatencySnapshot(
            model_name=self.model_name,
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            mean_ms=float(np.mean(arr)),
            max_ms=float(np.max(arr)),
            min_ms=float(np.min(arr)),
            n_requests=len(timings),
            error_rate=sum(errors) / len(errors) if errors else 0.0,
            window_start=self._window_start,
            window_end=datetime.utcnow().isoformat(),
        )

    def history(self, limit: int = 100) -> List[dict]:
        with self._conn() as conn:
            cur = conn.execute(
                "SELECT * FROM latency_snapshots ORDER BY window_end DESC LIMIT ?",
                (limit,),
            )
            cols = [d[0] for d in cur.description]
            return [dict(zip(cols, row)) for row in cur.fetchall()]

    def reset_window(self):
        with self._lock:
            self._timings_ms.clear()
            self._errors.clear()
            self._window_start = datetime.utcnow().isoformat()

    def close(self):
        snap = self.snapshot()
        if snap is not None:
            self._write_snapshot_to_db(snap)
        try:
            with self._conn() as conn:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except Exception:
            pass

    # ─── persistence ──────────────────────────────────────────────────────────

    def _write_snapshot_to_db(self, snap: LatencySnapshot):
        d    = snap.to_dict()
        cols = list(d.keys())
        vals = list(d.values())
        try:
            with self._conn() as conn:
                conn.execute(
                    f"INSERT INTO latency_snapshots ({','.join(cols)}) "
                    f"VALUES ({','.join('?' * len(vals))})",
                    vals,
                )
        except Exception as e:
            logger.warning(f"[DEPLOY] Failed to persist latency snapshot: {e}")

    def _init_db(self):
        with self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS latency_snapshots (
                    model_name   TEXT,
                    p50_ms       REAL,
                    p95_ms       REAL,
                    p99_ms       REAL,
                    mean_ms      REAL,
                    max_ms       REAL,
                    min_ms       REAL,
                    n_requests   INTEGER,
                    error_rate   REAL,
                    window_start TEXT,
                    window_end   TEXT
                )
            """)

    def _conn(self):
        conn = sqlite3.connect(str(self._db_path))
        conn.execute("PRAGMA journal_mode=DELETE")
        return conn


# ─── global singleton registry ───────────────────────────────────────────────

_trackers: Dict[str, LatencyTracker] = {}
_trackers_lock = threading.Lock()


def get_tracker(model_name: str) -> LatencyTracker:
    with _trackers_lock:
        if model_name not in _trackers:
            _trackers[model_name] = LatencyTracker(model_name)
        return _trackers[model_name]