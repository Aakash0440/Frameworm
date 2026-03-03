"""
CQL Policy — Conservative Q-Learning for offline RL.

Why CQL over standard Q-learning:
    In offline RL you cannot interact with the environment to
    collect new data. Standard Q-learning overestimates Q-values
    for out-of-distribution (OOD) actions that weren't in the
    dataset — causing the policy to choose actions it has never
    tried and has no evidence for.

    CQL adds a conservative penalty: minimize Q-values for
    actions NOT in the data, maximize Q-values for actions
    that ARE in the data. This keeps the policy from going
    off-distribution even without online exploration.

    This is critical for our use case: we don't want the agent
    to try ROLLBACK on every mild plateau just because
    Q-learning overestimates its Q-value.

Architecture:
    Q-network:  3-layer MLP
                input:  STATE_DIM (16)
                hidden: 128 → 64
                output: N_ACTIONS (6)

    Loss:       standard Bellman TD loss
                + CQL conservative penalty (alpha * gap)

    Training:   Adam optimizer
                target network updated every N steps
                early stopping on validation TD loss
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from agent.policy.experience_buffer import (
    ExperienceBuffer,
    N_ACTIONS,
    STATE_DIM,
    ACTION_INDEX,
    INDEX_ACTION,
)
from agent.react.action_parser import ActionType
from agent.classifier.anomaly_types import AnomalyType

logger = logging.getLogger(__name__)


@dataclass
class CQLConfig:
    """Hyperparameters for CQL training."""
    # Network
    hidden_size_1: int = 128
    hidden_size_2: int = 64
    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.99             # discount factor
    max_epochs: int = 100
    patience: int = 10
    target_update_freq: int = 50    # steps between target network updates
    # CQL-specific
    cql_alpha: float = 1.0          # conservative penalty weight
                                    # higher = more conservative
    # Checkpointing
    save_dir: str = "experiments/policy"
    # Evaluation
    eval_every_n_epochs: int = 5


class CQLPolicy:
    """
    Conservative Q-Learning policy for intervention action selection.

    After training, replaces the LLM for seen anomaly types.
    Falls back to LLM for anomaly types with fewer than
    min_samples transitions in the buffer.

    Args:
        config:         CQLConfig with hyperparameters.
        min_samples:    Min transitions per anomaly type before
                        trusting policy over LLM.
    """

    def __init__(
        self,
        config: Optional[CQLConfig] = None,
        min_samples: int = 20,
    ) -> None:
        self.config = config or CQLConfig()
        self.min_samples = min_samples
        self._q_network = None
        self._target_network = None
        self._optimizer = None
        self._device = None
        self._is_trained = False
        self._n_updates = 0
        # Track how many samples we have per anomaly type
        self._samples_per_type: dict = {}

    # ── Action selection ──────────────────────────────────────────

    def select_action(
        self,
        state: np.ndarray,
        anomaly_type: AnomalyType,
        epsilon: float = 0.0,
    ) -> Tuple[ActionType, float]:
        """
        Select best action for a given state.

        Returns (ActionType, confidence_score).
        confidence_score = max Q-value (higher = more certain).

        Falls back to (None, 0.0) if policy not confident enough,
        signaling the agent to use the LLM instead.
        """
        if not self.should_use_policy(anomaly_type):
            return None, 0.0

        if not self._is_trained or self._q_network is None:
            return None, 0.0

        # Epsilon-greedy for exploration (only during evaluation)
        if epsilon > 0 and np.random.random() < epsilon:
            action_idx = np.random.randint(N_ACTIONS)
            return INDEX_ACTION[action_idx], 0.0

        try:
            import torch
            self._q_network.eval()
            state_t = torch.tensor(
                state, dtype=torch.float32
            ).unsqueeze(0).to(self._device)

            with torch.no_grad():
                q_values = self._q_network(state_t).squeeze(0)
                best_action_idx = int(q_values.argmax().item())
                confidence = float(q_values.max().item())

            return INDEX_ACTION.get(best_action_idx, ActionType.WATCH), confidence

        except Exception as exc:
            logger.warning(f"CQLPolicy.select_action failed: {exc}")
            return None, 0.0

    def should_use_policy(self, anomaly_type: AnomalyType) -> bool:
        """
        True if policy has enough experience with this anomaly type
        to trust it over the LLM.
        """
        if not self._is_trained:
            return False
        n = self._samples_per_type.get(anomaly_type.name, 0)
        return n >= self.min_samples

    def update_sample_counts(self, buffer: ExperienceBuffer) -> None:
        """Update per-anomaly-type sample counts from buffer."""
        data = buffer._get_all()
        counts: dict = {}
        for t in data:
            counts[t.anomaly_type] = counts.get(t.anomaly_type, 0) + 1
        self._samples_per_type = counts

    # ── Training ──────────────────────────────────────────────────

    def _build_network(self):
        """Build Q-network and target network."""
        try:
            import torch.nn as nn
        except ImportError:
            raise ImportError("PyTorch required. pip install torch")

        cfg = self.config

        class _QNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(STATE_DIM, cfg.hidden_size_1),
                    nn.LayerNorm(cfg.hidden_size_1),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_size_1, cfg.hidden_size_2),
                    nn.LayerNorm(cfg.hidden_size_2),
                    nn.ReLU(),
                    nn.Linear(cfg.hidden_size_2, N_ACTIONS),
                )

            def forward(self, x):
                return self.net(x)

        return _QNetwork()

    def save(self, path: Optional[Path] = None) -> Path:
        """Save policy weights to disk."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for saving")

        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = path or (save_dir / "best_cql_policy.pt")

        torch.save({
            "q_network": self._q_network.state_dict()
                         if self._q_network else {},
            "config": asdict(self.config),
            "is_trained": self._is_trained,
            "n_updates": self._n_updates,
            "samples_per_type": self._samples_per_type,
        }, path)
        logger.info(f"CQLPolicy saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path) -> "CQLPolicy":
        """Load a saved policy."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch required for loading")

        checkpoint = torch.load(path, map_location="cpu")
        config = CQLConfig(**checkpoint["config"])
        policy = cls(config)
        policy._device = policy._get_device()
        policy._q_network = policy._build_network()
        policy._q_network.load_state_dict(checkpoint["q_network"])
        policy._q_network.to(policy._device)
        policy._q_network.eval()
        policy._is_trained = checkpoint.get("is_trained", True)
        policy._n_updates = checkpoint.get("n_updates", 0)
        policy._samples_per_type = checkpoint.get("samples_per_type", {})
        logger.info(f"CQLPolicy loaded from {path}")
        return policy

    def _get_device(self):
        try:
            import torch
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except ImportError:
            return None

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    @classmethod
    def from_checkpoint(
        cls,
        path: Path = Path("experiments/policy/best_cql_policy.pt"),
    ) -> "CQLPolicy":
        if path.exists():
            return cls.load(path)
        logger.warning(
            f"No CQL policy at {path}. "
            "Policy will always defer to LLM until trained."
        )
        return cls()


# ── Training function ─────────────────────────────────────────────

def train_cql_policy(
    policy: CQLPolicy,
    buffer: ExperienceBuffer,
    verbose: bool = True,
) -> dict:
    """
    Train the CQL policy on the experience buffer.

    CQL Loss = TD Loss + alpha * CQL_Penalty

    TD Loss (standard Bellman):
        L_TD = E[(Q(s,a) - (r + gamma * max_a' Q_target(s',a')))^2]

    CQL Penalty (conservative regularization):
        L_CQL = E[log sum_a exp(Q(s,a))] - E[Q(s, a_data)]
        This pushes Q-values DOWN for actions not in the data
        and UP for actions that were actually taken.

    Args:
        policy:     CQLPolicy instance.
        buffer:     ExperienceBuffer with transition data.
        verbose:    Print epoch losses.

    Returns:
        Training history dict.

    Usage:
        buffer = ExperienceBuffer()
        buffer.load_from_db()  # load past sessions
        policy = CQLPolicy()
        history = train_cql_policy(policy, buffer)
        # policy is now ready — use in agent loop
    """
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        raise ImportError("PyTorch required for CQL training")

    if not buffer.is_ready:
        logger.warning(
            f"ExperienceBuffer only has {len(buffer)} transitions "
            f"(need 100). Collect more data first."
        )
        return {"td_loss": [], "cql_loss": [], "total_loss": []}

    cfg = policy.config

    # Build networks
    policy._device = policy._get_device()
    policy._q_network = policy._build_network().to(policy._device)
    policy._target_network = copy.deepcopy(policy._q_network)
    policy._target_network.eval()

    policy._optimizer = torch.optim.Adam(
        policy._q_network.parameters(),
        lr=cfg.learning_rate,
        weight_decay=1e-4,
    )

    n_params = sum(p.numel() for p in policy._q_network.parameters())
    if verbose:
        logger.info(
            f"CQLPolicy: training on {len(buffer)} transitions, "
            f"{n_params} parameters, device={policy._device}"
        )

    # Convert buffer to tensors
    arrays = buffer.to_arrays()
    if arrays is None:
        return {}

    states, actions, rewards, next_states, dones = arrays
    n = len(states)
    val_size = max(1, int(n * 0.15))
    train_size = n - val_size

    # Shuffle and split
    idx = np.random.permutation(n)
    train_idx = idx[:train_size]
    val_idx = idx[train_size:]

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(policy._device)

    # Training tensors
    s_train = to_tensor(states[train_idx])
    a_train = torch.tensor(actions[train_idx], dtype=torch.long).to(policy._device)
    r_train = to_tensor(rewards[train_idx])
    ns_train = to_tensor(next_states[train_idx])
    d_train = to_tensor(dones[train_idx])

    # Val tensors
    s_val = to_tensor(states[val_idx])
    a_val = torch.tensor(actions[val_idx], dtype=torch.long).to(policy._device)
    r_val = to_tensor(rewards[val_idx])
    ns_val = to_tensor(next_states[val_idx])
    d_val = to_tensor(dones[val_idx])

    history = {"td_loss": [], "cql_loss": [], "total_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    patience_counter = 0
    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_path = save_dir / "best_cql_policy.pt"

    for epoch in range(cfg.max_epochs):
        policy._q_network.train()

        # Mini-batch training
        perm = torch.randperm(train_size).to(policy._device)
        epoch_td, epoch_cql, epoch_total = [], [], []

        for start in range(0, train_size, cfg.batch_size):
            batch_idx = perm[start:start + cfg.batch_size]
            if len(batch_idx) == 0:
                continue

            s_b = s_train[batch_idx]
            a_b = a_train[batch_idx]
            r_b = r_train[batch_idx]
            ns_b = ns_train[batch_idx]
            d_b = d_train[batch_idx]

            # ── TD Loss ──────────────────────────────────────────
            q_values = policy._q_network(s_b)           # (batch, N_ACTIONS)
            q_taken = q_values.gather(1, a_b.unsqueeze(1)).squeeze(1)

            with torch.no_grad():
                next_q = policy._target_network(ns_b)
                next_v = next_q.max(dim=1)[0]
                target = r_b + cfg.gamma * next_v * (1 - d_b)

            td_loss = F.mse_loss(q_taken, target)

            # ── CQL Penalty ───────────────────────────────────────
            # log sum_a exp(Q(s,a)) — logsumexp over all actions
            logsumexp = torch.logsumexp(q_values, dim=1).mean()
            # E[Q(s, a_data)] — Q-value of the action actually taken
            q_data = q_taken.mean()
            # CQL penalty: penalize OOD actions (logsumexp - q_data)
            cql_loss = logsumexp - q_data

            total_loss = td_loss + cfg.cql_alpha * cql_loss

            policy._optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                policy._q_network.parameters(), 1.0
            )
            policy._optimizer.step()

            policy._n_updates += 1
            if policy._n_updates % cfg.target_update_freq == 0:
                policy._target_network.load_state_dict(
                    policy._q_network.state_dict()
                )

            epoch_td.append(td_loss.item())
            epoch_cql.append(cql_loss.item())
            epoch_total.append(total_loss.item())

        # ── Validation ────────────────────────────────────────────
        policy._q_network.eval()
        with torch.no_grad():
            val_q = policy._q_network(s_val)
            val_taken = val_q.gather(1, a_val.unsqueeze(1)).squeeze(1)
            val_next = policy._target_network(ns_val).max(dim=1)[0]
            val_target = r_val + cfg.gamma * val_next * (1 - d_val)
            val_td = F.mse_loss(val_taken, val_target).item()

        mean_td = float(np.mean(epoch_td)) if epoch_td else 0.0
        mean_cql = float(np.mean(epoch_cql)) if epoch_cql else 0.0
        mean_total = float(np.mean(epoch_total)) if epoch_total else 0.0

        history["td_loss"].append(mean_td)
        history["cql_loss"].append(mean_cql)
        history["total_loss"].append(mean_total)
        history["val_loss"].append(val_td)

        if verbose and epoch % 10 == 0:
            logger.info(
                f"CQL Epoch {epoch+1:3d} | "
                f"TD={mean_td:.4f} | "
                f"CQL={mean_cql:.4f} | "
                f"total={mean_total:.4f} | "
                f"val={val_td:.4f}"
            )

        # Early stopping
        if val_td < best_val_loss:
            best_val_loss = val_td
            patience_counter = 0
            policy.save(best_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg.patience:
                logger.info(
                    f"CQL early stopping at epoch {epoch+1}"
                )
                break

    # Load best
    if best_path.exists():
        checkpoint = torch.load(best_path, map_location=policy._device)
        policy._q_network.load_state_dict(checkpoint["q_network"])

    policy._is_trained = True
    policy.update_sample_counts(buffer)

    logger.info(
        f"CQL training complete. Best val_loss={best_val_loss:.4f}, "
        f"updates={policy._n_updates}"
    )
    return history
