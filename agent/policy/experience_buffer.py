"""
ExperienceBuffer — offline RL dataset.

Every agent decision is a training example:
    state:      observation vector at time of decision
    action:     what the agent did (as int index)
    reward:     how good the outcome was (from counterfactual delta)
    next_state: observation vector after intervention
    done:       whether episode ended (rollback or pause)

Persists to experiments/experiments.db via your existing
integrations/database.py so data accumulates across sessions.

State vector (16 features):
    [loss_ema, loss_delta, loss_z_score,
     grad_norm, grad_norm_var, grad_norm_z,
     lr_log, plateau_score, divergence_score, oscillation_score,
     anomaly_type_onehot (6 dims)]

Action space (6 actions — matches ActionType enum):
    0: WATCH
    1: ADJUST_LR
    2: ROLLBACK
    3: SWAP_SCHEDULER
    4: PAUSE
    5: ALERT
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from agent.classifier.anomaly_types import FAILURE_TYPES, AnomalyType
from agent.observer.signal_extractor import SignalSnapshot
from agent.react.action_parser import ActionType

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────

# State vector dimension
STATE_DIM = 16

# Action indices — must match ActionType enum order
ACTION_INDEX: Dict[ActionType, int] = {
    ActionType.WATCH: 0,
    ActionType.ADJUST_LR: 1,
    ActionType.ROLLBACK: 2,
    ActionType.SWAP_SCHEDULER: 3,
    ActionType.PAUSE: 4,
    ActionType.ALERT: 5,
}
INDEX_ACTION = {v: k for k, v in ACTION_INDEX.items()}
N_ACTIONS = len(ACTION_INDEX)  # 6

# Anomaly type one-hot indices
ANOMALY_INDEX: Dict[AnomalyType, int] = {atype: i for i, atype in enumerate(FAILURE_TYPES)}


# ── Transition dataclass ─────────────────────────────────────────


@dataclass
class Transition:
    """
    One (s, a, r, s', done) transition for offline RL training.
    """

    state: np.ndarray  # (STATE_DIM,) observation vector
    action: int  # action index (0–5)
    reward: float  # outcome quality
    next_state: np.ndarray  # (STATE_DIM,) post-intervention obs
    done: bool  # episode ended?

    # Metadata (not used for training — for logging and debugging)
    step: int = 0
    anomaly_type: str = ""
    action_name: str = ""
    loss_delta: float = 0.0  # from counterfactual (Part 4)
    resolved: bool = False
    run_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self):
        self.state = np.asarray(self.state, dtype=np.float32)
        self.next_state = np.asarray(self.next_state, dtype=np.float32)
        assert self.state.shape == (
            STATE_DIM,
        ), f"State dim mismatch: {self.state.shape} vs ({STATE_DIM},)"
        assert 0 <= self.action < N_ACTIONS, f"Invalid action index: {self.action}"


# ── State encoder ────────────────────────────────────────────────


def encode_state(
    signals: SignalSnapshot,
    anomaly_type: AnomalyType,
) -> np.ndarray:
    """
    Encode a SignalSnapshot + AnomalyType into a flat state vector.

    Returns: (STATE_DIM,) float32 array

    Layout:
        [0]  loss_ema
        [1]  loss_delta
        [2]  loss_z_score
        [3]  grad_norm (current)
        [4]  grad_norm_var
        [5]  grad_norm_z_score
        [6]  lr_log (log10 of current LR)
        [7]  plateau_score
        [8]  divergence_score
        [9]  oscillation_score
        [10–15] anomaly_type one-hot (6 dims = N_FAILURE_TYPES)
    """
    lr = max(signals.lr_current, 1e-10)
    lr_log = float(np.log10(lr))

    # Base features
    base = np.array(
        [
            signals.loss_ema,
            signals.loss_delta,
            signals.loss_z_score,
            signals.grad_norm_current,
            signals.grad_norm_var,
            signals.grad_norm_z_score,
            lr_log,
            signals.plateau_score,
            signals.divergence_score,
            signals.oscillation_score,
        ],
        dtype=np.float32,
    )

    # Anomaly type one-hot (6 dims)
    one_hot = np.zeros(len(FAILURE_TYPES), dtype=np.float32)
    if anomaly_type in ANOMALY_INDEX:
        one_hot[ANOMALY_INDEX[anomaly_type]] = 1.0

    state = np.concatenate([base, one_hot])
    assert state.shape == (STATE_DIM,), f"State shape error: {state.shape}"
    return state


def compute_reward(
    resolved: bool,
    loss_delta: float,
    fid_delta: Optional[float],
    action_type: ActionType,
    is_fallback: bool,
) -> float:
    """
    Compute scalar reward for a transition.

    Reward components:
        +2.0  resolved = True (intervention worked)
        +1.0  loss_delta < -0.05 (meaningful improvement)
        +0.5  fid improved
        -0.5  unresolved
        -1.0  ROLLBACK used (costly action)
        -0.3  LLM fallback was used (LLM should be replaced over time)
        -2.0  PAUSE triggered (worst case — human needed)

    Range: roughly [-3, +3.5]
    """
    reward = 0.0

    if resolved:
        reward += 2.0
    else:
        reward -= 0.5

    if loss_delta < -0.05:
        reward += 1.0
    elif loss_delta > 0.1:
        reward -= 0.5

    if fid_delta is not None and fid_delta < -1.0:
        reward += 0.5

    if action_type == ActionType.ROLLBACK:
        reward -= 1.0
    elif action_type == ActionType.PAUSE:
        reward -= 2.0

    if is_fallback:
        reward -= 0.3

    return float(reward)


# ── ExperienceBuffer ─────────────────────────────────────────────


class ExperienceBuffer:
    """
    Stores and persists agent transitions for offline RL training.

    Two backends:
        IN_MEMORY:  Fast, lost between sessions.
        SQLITE:     Persistent via your experiments/experiments.db.

    Args:
        db_path:        Path to SQLite DB. Uses your existing
                        experiments/experiments.db if it exists.
        max_memory:     Max transitions in memory cache.
        table_name:     SQLite table name (created if not exists).
    """

    TABLE_DDL = """
        CREATE TABLE IF NOT EXISTS agent_transitions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            state           BLOB NOT NULL,
            action          INTEGER NOT NULL,
            reward          REAL NOT NULL,
            next_state      BLOB NOT NULL,
            done            INTEGER NOT NULL,
            step            INTEGER,
            anomaly_type    TEXT,
            action_name     TEXT,
            loss_delta      REAL,
            resolved        INTEGER,
            run_id          TEXT,
            timestamp       REAL
        )
    """

    def __init__(
        self,
        db_path: Path = Path("experiments/experiments.db"),
        max_memory: int = 100_000,
        table_name: str = "agent_transitions",
    ) -> None:
        self.db_path = db_path
        self.max_memory = max_memory
        self.table_name = table_name
        self._memory: List[Transition] = []
        self._db_conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize SQLite table if DB is accessible."""
        try:
            self._db_conn = sqlite3.connect(str(self.db_path))
            self._db_conn.execute(self.TABLE_DDL)
            self._db_conn.commit()
            logger.info(f"ExperienceBuffer: using SQLite at {self.db_path}")
        except Exception as exc:
            logger.warning(
                f"ExperienceBuffer: SQLite unavailable ({exc}). "
                "Using in-memory only — data will not persist."
            )
            self._db_conn = None

    # ── Write ─────────────────────────────────────────────────────

    def add(self, transition: Transition) -> None:
        """Add one transition to buffer."""
        # Memory cache
        if len(self._memory) >= self.max_memory:
            self._memory.pop(0)  # evict oldest
        self._memory.append(transition)

        # Persist to SQLite
        if self._db_conn is not None:
            try:
                self._db_conn.execute(
                    f"""INSERT INTO {self.table_name}
                        (state, action, reward, next_state, done,
                         step, anomaly_type, action_name,
                         loss_delta, resolved, run_id, timestamp)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        transition.state.tobytes(),
                        transition.action,
                        transition.reward,
                        transition.next_state.tobytes(),
                        int(transition.done),
                        transition.step,
                        transition.anomaly_type,
                        transition.action_name,
                        transition.loss_delta,
                        int(transition.resolved),
                        transition.run_id,
                        transition.timestamp,
                    ),
                )
                self._db_conn.commit()
            except Exception as exc:
                logger.debug(f"ExperienceBuffer SQLite write failed: {exc}")

    def add_from_decision(
        self,
        signals: SignalSnapshot,
        anomaly_type: AnomalyType,
        action_type: ActionType,
        next_signals: Optional[SignalSnapshot],
        resolved: bool,
        loss_delta: float,
        fid_delta: Optional[float] = None,
        is_fallback: bool = False,
        step: int = 0,
        run_id: str = "",
    ) -> Transition:
        """
        Convenience method — build and add a Transition from
        agent decision components. Call this from agent._tick()
        after verification completes.
        """
        state = encode_state(signals, anomaly_type)

        # If next_signals not available, use current state shifted
        if next_signals is not None:
            next_state = encode_state(next_signals, AnomalyType.HEALTHY)
        else:
            next_state = state.copy()
            next_state[0] += loss_delta  # approximate: loss changed by delta

        reward = compute_reward(
            resolved=resolved,
            loss_delta=loss_delta,
            fid_delta=fid_delta,
            action_type=action_type,
            is_fallback=is_fallback,
        )

        transition = Transition(
            state=state,
            action=ACTION_INDEX.get(action_type, 0),
            reward=reward,
            next_state=next_state,
            done=(action_type in (ActionType.PAUSE, ActionType.ROLLBACK)),
            step=step,
            anomaly_type=anomaly_type.name,
            action_name=action_type.name,
            loss_delta=loss_delta,
            resolved=resolved,
            run_id=run_id,
        )

        self.add(transition)
        return transition

    # ── Read ──────────────────────────────────────────────────────

    def sample(self, batch_size: int) -> Optional[List[Transition]]:
        """
        Sample a random mini-batch from the buffer.
        Returns None if fewer than batch_size transitions available.
        """
        data = self._get_all()
        if len(data) < batch_size:
            return None
        indices = np.random.choice(len(data), batch_size, replace=False)
        return [data[i] for i in indices]

    def load_from_db(self) -> int:
        """
        Load all transitions from SQLite into memory.
        Call this at the start of a new session to resume
        from previous runs.

        Returns number of transitions loaded.
        """
        if self._db_conn is None:
            return 0
        try:
            cursor = self._db_conn.execute(
                f"SELECT state, action, reward, next_state, done, "
                f"step, anomaly_type, action_name, loss_delta, "
                f"resolved, run_id, timestamp "
                f"FROM {self.table_name} "
                f"ORDER BY id DESC LIMIT {self.max_memory}"
            )
            rows = cursor.fetchall()
            loaded = []
            for row in rows:
                state = np.frombuffer(row[0], dtype=np.float32).copy()
                next_state = np.frombuffer(row[3], dtype=np.float32).copy()
                if state.shape != (STATE_DIM,) or next_state.shape != (STATE_DIM,):
                    continue
                loaded.append(
                    Transition(
                        state=state,
                        action=int(row[1]),
                        reward=float(row[2]),
                        next_state=next_state,
                        done=bool(row[4]),
                        step=int(row[5] or 0),
                        anomaly_type=str(row[6] or ""),
                        action_name=str(row[7] or ""),
                        loss_delta=float(row[8] or 0.0),
                        resolved=bool(row[9]),
                        run_id=str(row[10] or ""),
                        timestamp=float(row[11] or 0.0),
                    )
                )
            self._memory = loaded
            logger.info(f"ExperienceBuffer: loaded {len(loaded)} " "transitions from DB.")
            return len(loaded)
        except Exception as exc:
            logger.warning(f"ExperienceBuffer load failed: {exc}")
            return 0

    def _get_all(self) -> List[Transition]:
        return list(self._memory)

    def to_arrays(self) -> Optional[Tuple[np.ndarray, ...]]:
        """
        Convert buffer to numpy arrays for batch training.
        Returns (states, actions, rewards, next_states, dones)
        or None if buffer is empty.
        """
        data = self._get_all()
        if not data:
            return None
        states = np.stack([t.state for t in data])
        actions = np.array([t.action for t in data], dtype=np.int64)
        rewards = np.array([t.reward for t in data], dtype=np.float32)
        next_states = np.stack([t.next_state for t in data])
        dones = np.array([t.done for t in data], dtype=np.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self._memory)

    @property
    def is_ready(self) -> bool:
        """True once we have enough data to start training (min 100)."""
        return len(self) >= 100

    def stats(self) -> dict:
        """Summary statistics of the buffer."""
        data = self._get_all()
        if not data:
            return {"size": 0}
        rewards = [t.reward for t in data]
        return {
            "size": len(data),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(np.mean([t.resolved for t in data])),
            "action_counts": {
                name: sum(1 for t in data if t.action_name == name)
                for name in set(t.action_name for t in data)
            },
        }
