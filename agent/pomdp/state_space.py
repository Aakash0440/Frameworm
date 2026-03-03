"""
Formal POMDP state/observation/action/transition/reward definitions.

This is the theoretical contribution of the paper.
Each class is both:
    - A formal mathematical definition (for the paper)
    - A Python implementation that drives the belief updater

Put the POMDPSpec docstring in your paper's Section 3 (Problem Formulation).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from agent.policy.experience_buffer import STATE_DIM, N_ACTIONS
from agent.react.action_parser import ActionType
from agent.classifier.anomaly_types import AnomalyType


# ── State Space ───────────────────────────────────────────────────

@dataclass
class StateSpace:
    """
    S — True training state (partially unobservable).

    The true state captures everything that determines how
    training will proceed. Most of it is unobservable:

    Observable components (O ⊆ S):
        loss            scalar ∈ R+
        grad_norm       scalar ∈ R+
        lr              scalar ∈ R+
        layer_grads     vector ∈ R^L (L = num layers)

    Unobservable components (S \ O):
        weight_tensors  all model parameters ∈ R^P (billions of dims)
        optimizer_state Adam momentum m_t, v_t ∈ R^P
        data_dist       distribution of upcoming batches
        hardware_state  GPU memory pressure, thermal throttling

    Why this matters:
        Two runs can have identical loss curves but completely
        different weight configurations → different futures.
        The agent must reason under this partial observability.
        This is why POMDP is the right framework.

    Continuous state space dimension:
        |S| = P + 2P + d_data + d_hw ≈ O(10^8) for typical models
        Observable proxy dimension: |O| = 16 (our STATE_DIM)
    """
    # Observed dimension (proxy for true state)
    obs_dim: int = STATE_DIM      # 16
    # True state is infinite-dimensional (weights + optimizer)
    # We approximate it via the observation + belief updater

    @property
    def description(self) -> str:
        return (
            "Continuous partially-observable state space. "
            f"Observable proxy: R^{self.obs_dim}. "
            "True state includes model weights, optimizer momentum, "
            "data distribution (unobservable)."
        )


# ── Observation Space ─────────────────────────────────────────────

@dataclass
class ObservationSpace:
    """
    O — Observation space (what the agent can measure).

    O is a noisy, lossy projection of S via the training
    instrumentation (W&B, TensorBoard, local metrics file).

    Observation vector layout (16 dims — same as STATE_DIM):
        [0]  loss_ema          — smoothed loss
        [1]  loss_delta        — rate of change
        [2]  loss_z_score      — deviation from baseline
        [3]  grad_norm         — current gradient norm
        [4]  grad_norm_var     — gradient instability
        [5]  grad_norm_z       — gradient norm deviation
        [6]  lr_log            — log10(learning rate)
        [7]  plateau_score     — how stuck training is
        [8]  divergence_score  — fraction of steps getting worse
        [9]  oscillation_score — loss bouncing indicator
        [10-15] anomaly_type_onehot (6 dims)

    Observation noise model:
        o_t = f(s_t) + ε_t,  ε_t ~ N(0, Σ_obs)
        where Σ_obs is diagonal with known (estimated) variances
    """
    dim: int = STATE_DIM

    # Estimated observation noise std per feature
    # (calibrated from first 100 steps of each run)
    noise_std: np.ndarray = field(
        default_factory=lambda: np.array([
            0.02,   # loss_ema
            0.05,   # loss_delta
            0.3,    # loss_z_score
            0.1,    # grad_norm
            0.05,   # grad_norm_var
            0.3,    # grad_norm_z
            0.01,   # lr_log
            0.02,   # plateau_score
            0.05,   # divergence_score
            0.02,   # oscillation_score
            0.0,    # one-hot dims (no noise)
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ], dtype=np.float32)
    )


# ── Action Space ──────────────────────────────────────────────────

@dataclass
class ActionSpace:
    """
    A — Discrete action space (6 actions).

    Actions are interventions on the training process.
    Some are reversible (ADJUST_LR), some irreversible (ROLLBACK).

    Formal definition:
        A = {WATCH, ADJUST_LR, ROLLBACK, SWAP_SCHEDULER, PAUSE, ALERT}
        |A| = 6

    Action costs (used in reward model):
        WATCH:          0.0   (free — just observe)
        ADJUST_LR:      0.1   (small — reversible)
        SWAP_SCHEDULER: 0.2   (small — reversible)
        ALERT:          0.3   (human time cost)
        ROLLBACK:       1.0   (expensive — compute wasted)
        PAUSE:          2.0   (most expensive — human blocked)
    """
    n_actions: int = N_ACTIONS

    action_costs: Dict[ActionType, float] = field(default_factory=lambda: {
        ActionType.WATCH: 0.0,
        ActionType.ADJUST_LR: 0.1,
        ActionType.SWAP_SCHEDULER: 0.2,
        ActionType.ALERT: 0.3,
        ActionType.ROLLBACK: 1.0,
        ActionType.PAUSE: 2.0,
    })

    action_reversibility: Dict[ActionType, bool] = field(
        default_factory=lambda: {
            ActionType.WATCH: True,
            ActionType.ADJUST_LR: True,
            ActionType.SWAP_SCHEDULER: True,
            ActionType.ALERT: True,
            ActionType.ROLLBACK: False,
            ActionType.PAUSE: False,
        }
    )


# ── Transition Model ──────────────────────────────────────────────

@dataclass
class TransitionModel:
    """
    T(s' | s, a) — How training state evolves after an action.

    True transition model is unknown — we cannot compute
    P(s' | s, a) analytically for neural network training.

    We approximate it in two ways:
        1. GradForecaster (Part 3) — models T over observations O
           using LSTM trained on historical data
        2. CQL policy (this part) — learns T implicitly via
           Q-value backup: Q(s,a) = E[r + γ max_a' Q(s',a')]

    For the paper, we state this explicitly:
        "We do not assume access to the true transition model.
         Instead, we learn an observation-space approximation
         T̂(o' | o, a) from offline data..."
    """
    is_known: bool = False
    approximation_method: str = "offline_RL_and_LSTM_forecaster"

    @property
    def description(self) -> str:
        return (
            "Unknown true transition model. "
            "Approximated via: (1) LSTM gradient forecaster "
            "learning T̂(o'|o) from historical runs, "
            "(2) CQL Q-learning implicitly learning "
            "T̂(o'|o,a) via Bellman backup."
        )


# ── Reward Model ──────────────────────────────────────────────────

@dataclass
class RewardModel:
    """
    R(s, a) — Reward function.

    Formally:
        R(s, a) = w_recovery * I[resolved]
                - w_loss_delta * Δloss
                - w_fid_delta * ΔFID
                - cost(a)

    Where:
        I[resolved]  = 1 if post-intervention verification passed
        Δloss        = post_loss - pre_loss (negative = good)
        ΔFID         = post_FID - pre_FID (negative = good)
        cost(a)      = action cost from ActionSpace

    The reward is observable: we compute it from the
    counterfactual twin runner (Part 4) after each intervention.

    The counterfactual twin gives us Δloss and ΔFID causally —
    not just correlatively — which is the key improvement over
    naive reward shaping.
    """
    w_recovery: float = 2.0
    w_loss_delta: float = 1.0
    w_fid_delta: float = 0.5
    gamma: float = 0.99             # discount factor

    @property
    def description(self) -> str:
        return (
            f"R(s,a) = {self.w_recovery}*I[resolved] "
            f"- {self.w_loss_delta}*Δloss "
            f"- {self.w_fid_delta}*ΔFID "
            f"- cost(a). "
            f"Reward computed from counterfactual twin runner. "
            f"γ = {self.gamma}."
        )


# ── Full POMDP Spec ───────────────────────────────────────────────

class POMDPSpec:
    """
    Complete POMDP specification for the training monitoring problem.

    This is the formal theoretical contribution.
    Write the docstring of this class into your paper's
    Section 3 (Problem Formulation).

    POMDP = (S, O, A, T, R, γ, b₀)

    Key insight: standard training monitors assume full observability
    (MDP). We formalize the problem as a POMDP because:
        1. The true training state (weights + optimizer) is not
           directly observable — only metric proxies are.
        2. Two identical observation sequences can correspond to
           completely different true states with different futures.
        3. Optimal intervention requires maintaining a BELIEF over
           the true state, updated by each new observation.

    This framing:
        - Explains why LLM-only approaches work at all
          (they reason over the observation history = belief proxy)
        - Explains why the CQL policy can improve over time
          (more data = better belief approximation)
        - Provides a principled basis for the forecaster
          (predicting future observations = T̂ approximation)
    """

    def __init__(self) -> None:
        self.S = StateSpace()
        self.O = ObservationSpace()
        self.A = ActionSpace()
        self.T = TransitionModel()
        self.R = RewardModel()
        self.gamma = self.R.gamma

    def to_paper_text(self) -> str:
        """
        Generate LaTeX-ready problem formulation text.
        Paste into your paper's Section 3.
        """
        return """
\\section{Problem Formulation}

We formalize autonomous training monitoring as a
Partially Observable Markov Decision Process (POMDP),
defined by the tuple $(\\mathcal{S}, \\mathcal{O},
\\mathcal{A}, T, R, \\gamma, b_0)$.

\\textbf{State space} $\\mathcal{S}$:
The true training state $s_t \\in \\mathcal{S}$ encompasses
all information determining future training dynamics:
model weights $\\theta \\in \\mathbb{R}^P$, optimizer
momentum $m_t, v_t \\in \\mathbb{R}^P$, and data
distribution $\\mathcal{D}$. The state is partially
observable — we cannot directly access $\\theta$ or the
optimizer's internal state during monitoring.

\\textbf{Observation space} $\\mathcal{O}$:
At each step $t$, the agent observes $o_t \\in \\mathbb{R}^{16}$,
a noisy projection of the true state via training
instrumentation: $o_t = f(s_t) + \\varepsilon_t$, where
$\\varepsilon_t \\sim \\mathcal{N}(0, \\Sigma_{obs})$.
The observation includes loss statistics, gradient norms,
learning rate, and anomaly classification signals.

\\textbf{Action space} $\\mathcal{A}$:
The agent selects from six discrete interventions:
$\\mathcal{A} = \\{$WATCH, ADJUST\\_LR, ROLLBACK,
SWAP\\_SCHEDULER, PAUSE, ALERT$\\}$,
with associated costs $c(a) \\in [0, 2.0]$ reflecting
reversibility and human overhead.

\\textbf{Transition model} $T(s' | s, a)$:
The true transition model is unknown. We approximate it via
(1) an LSTM gradient forecaster learning $\\hat{T}(o'|o)$
from offline data, and (2) a Conservative Q-Learning policy
implicitly approximating $\\hat{T}(o'|o,a)$ via Bellman backup.

\\textbf{Reward function} $R(s, a)$:
$R(s, a) = 2.0 \\cdot \\mathbb{1}[\\text{resolved}]
- \\Delta\\text{loss} - 0.5 \\cdot \\Delta\\text{FID} - c(a)$,
where $\\Delta\\text{loss}$ and $\\Delta\\text{FID}$ are
computed causally via counterfactual twin runs.

\\textbf{Discount factor} $\\gamma = 0.99$:
Future training quality is weighted almost equivalently
to immediate recovery, reflecting the long-horizon nature
of model training.

\\textbf{Initial belief} $b_0$:
Uniform distribution over healthy training dynamics,
updated by the Bayesian belief updater at each observation.
"""

    def to_json(self) -> str:
        """Serialize spec for logging."""
        return json.dumps({
            "S": self.S.description,
            "O": {"dim": self.O.dim, "noise_std": self.O.noise_std.tolist()},
            "A": {
                "n_actions": self.A.n_actions,
                "costs": {k.name: v for k, v in self.A.action_costs.items()},
            },
            "T": self.T.description,
            "R": self.R.description,
            "gamma": self.gamma,
        }, indent=2)
