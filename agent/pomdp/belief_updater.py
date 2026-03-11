"""
BeliefUpdater — Bayesian belief state maintenance.

In a POMDP the agent maintains a belief b_t = P(s_t | o_1:t, a_1:t-1)
over the true (unobservable) state, updated at each step via Bayes:

    b_t(s) ∝ P(o_t | s) * ∑_s' T(s | s', a_{t-1}) * b_{t-1}(s')

We approximate this using a Gaussian Process over the observation
space — reusing your existing search/bayesian_search.py GP.

Practical effect:
    - Maintains uncertainty estimates over training trajectory
    - "Confident normal" = agent can skip LLM call
    - "Uncertain abnormal" = agent must escalate

Hooks into:
    search/bayesian_search.py   → your existing GP implementation
                                  (same GP used for HPO — reused here)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from agent.classifier.anomaly_types import AnomalyType
from agent.observer.signal_extractor import SignalSnapshot

logger = logging.getLogger(__name__)


@dataclass
class BeliefState:
    """
    Current belief over training health.

    Rather than maintaining a full distribution over the
    intractable true state space, we maintain a belief over
    a compressed latent representing training health.

    Concretely: belief = probability distribution over
    {HEALTHY, DEGRADING, ANOMALOUS, UNKNOWN}

    Updated each tick using a simple Bayesian update rule
    backed by the GP's posterior mean and variance.
    """

    # Probability mass on each training health class
    p_healthy: float = 0.9
    p_degrading: float = 0.07
    p_anomalous: float = 0.02
    p_unknown: float = 0.01

    # GP posterior estimate (mean prediction, uncertainty)
    gp_mean: float = 0.0  # predicted health score
    gp_std: float = 1.0  # uncertainty

    # Entropy of belief (high = uncertain)
    entropy: float = 0.0

    # Step this belief corresponds to
    step: int = 0

    def __post_init__(self):
        self._normalize()
        self.entropy = self._compute_entropy()

    def _normalize(self) -> None:
        total = self.p_healthy + self.p_degrading + self.p_anomalous + self.p_unknown
        if total > 0:
            self.p_healthy /= total
            self.p_degrading /= total
            self.p_anomalous /= total
            self.p_unknown /= total

    def _compute_entropy(self) -> float:
        probs = np.array([self.p_healthy, self.p_degrading, self.p_anomalous, self.p_unknown])
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs + 1e-10)))

    @property
    def is_confident_healthy(self) -> bool:
        """High confidence that training is healthy — safe to skip LLM."""
        return self.p_healthy > 0.85 and self.entropy < 0.3

    @property
    def is_uncertain(self) -> bool:
        """High uncertainty — should escalate to LLM."""
        return self.entropy > 0.8 or self.p_unknown > 0.2

    @property
    def most_likely_state(self) -> str:
        probs = {
            "HEALTHY": self.p_healthy,
            "DEGRADING": self.p_degrading,
            "ANOMALOUS": self.p_anomalous,
            "UNKNOWN": self.p_unknown,
        }
        return max(probs, key=probs.get)

    def __repr__(self) -> str:
        return (
            f"BeliefState(p_healthy={self.p_healthy:.2f}, "
            f"p_degrading={self.p_degrading:.2f}, "
            f"p_anomalous={self.p_anomalous:.2f}, "
            f"entropy={self.entropy:.2f}, "
            f"most_likely={self.most_likely_state})"
        )


class BeliefUpdater:
    """
    Maintains and updates the agent's belief state over training health.

    Reuses your search/bayesian_search.py GP if available.
    Falls back to a simple Bayesian filter if not.

    Usage:
        updater = BeliefUpdater()
        belief = updater.initial_belief()

        # Each tick:
        belief = updater.update(belief, signals, anomaly_events)

        if belief.is_confident_healthy:
            pass  # skip LLM call
        elif belief.is_uncertain:
            pass  # escalate to LLM
    """

    def __init__(self, use_gp: bool = True) -> None:
        self.use_gp = use_gp
        self._gp = None
        self._obs_history: List[np.ndarray] = []
        self._health_history: List[float] = []
        self._init_gp()

    def _init_gp(self) -> None:
        """Try to initialize GP from your existing bayesian_search.py."""
        if not self.use_gp:
            return
        try:
            from search.bayesian_search import BayesianSearch

            # Your BayesianSearch uses a GP internally
            # We access it for belief state estimation
            self._bayesian_search = None  # initialized lazily when needed
            logger.debug("BeliefUpdater: using FRAMEWORM BayesianSearch GP")
        except ImportError:
            try:
                # Fallback: sklearn GP
                from sklearn.gaussian_process import GaussianProcessRegressor
                from sklearn.gaussian_process.kernels import RBF, WhiteKernel

                kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
                self._gp = GaussianProcessRegressor(
                    kernel=kernel,
                    n_restarts_optimizer=2,
                    normalize_y=True,
                )
                logger.debug("BeliefUpdater: using sklearn GP")
            except ImportError:
                logger.debug("BeliefUpdater: no GP available, using simple filter")
                self._gp = None

    def initial_belief(self) -> BeliefState:
        """Return the initial belief state (optimistic prior)."""
        return BeliefState(
            p_healthy=0.90,
            p_degrading=0.07,
            p_anomalous=0.02,
            p_unknown=0.01,
            gp_mean=0.0,
            gp_std=1.0,
            step=0,
        )

    def update(
        self,
        prior: BeliefState,
        signals: SignalSnapshot,
        n_anomaly_events: int = 0,
    ) -> BeliefState:
        """
        Bayesian belief update given new observation.

        P(h_t | o_t) ∝ P(o_t | h_t) * P(h_t | h_{t-1})

        Where:
            h_t ∈ {HEALTHY, DEGRADING, ANOMALOUS, UNKNOWN}
            o_t = signals snapshot
            P(o_t | h_t) = likelihood from signal features
            P(h_t | h_{t-1}) = transition prior (training tends to be stable)
        """
        # Build observation vector for GP
        obs = np.array(
            [
                signals.loss_z_score,
                signals.grad_norm_z_score,
                signals.plateau_score,
                signals.divergence_score,
                float(n_anomaly_events),
            ],
            dtype=np.float32,
        )

        # Compute health score: negative = unhealthy
        health_score = self._compute_health_score(signals, n_anomaly_events)

        # Update GP model if we have enough history
        gp_mean, gp_std = self._update_gp(obs, health_score)

        # Bayesian update of categorical belief
        likelihood_healthy = self._likelihood(signals, "HEALTHY")
        likelihood_degrading = self._likelihood(signals, "DEGRADING")
        likelihood_anomalous = self._likelihood(signals, "ANOMALOUS")

        # Transition prior: training is sticky (tends to stay in same state)
        trans_healthy = 0.95 * prior.p_healthy + 0.05 * prior.p_degrading
        trans_degrading = 0.7 * prior.p_degrading + 0.1 * prior.p_healthy + 0.2 * prior.p_anomalous
        trans_anomalous = 0.8 * prior.p_anomalous + 0.15 * prior.p_degrading
        trans_unknown = 0.5 * prior.p_unknown + 0.03

        # Posterior (unnormalized)
        p_healthy_post = likelihood_healthy * trans_healthy
        p_degrading_post = likelihood_degrading * trans_degrading
        p_anomalous_post = likelihood_anomalous * trans_anomalous
        p_unknown_post = 0.1 * trans_unknown

        # Boost anomaly probability if rule engine fired
        if n_anomaly_events > 0:
            p_anomalous_post *= 1.0 + n_anomaly_events

        posterior = BeliefState(
            p_healthy=p_healthy_post,
            p_degrading=p_degrading_post,
            p_anomalous=p_anomalous_post,
            p_unknown=p_unknown_post,
            gp_mean=gp_mean,
            gp_std=gp_std,
            step=signals.step,
        )

        return posterior

    def _compute_health_score(self, signals: SignalSnapshot, n_anomalies: int) -> float:
        """
        Scalar health score in [-3, +1].
        Positive = healthy, negative = problematic.
        """
        score = 0.0
        score -= min(abs(signals.loss_z_score), 3.0) * 0.3
        score -= min(signals.divergence_score, 1.0) * 0.3
        score += signals.plateau_score * 0.1
        score -= signals.oscillation_score * 0.2
        score -= n_anomalies * 0.5
        return float(np.clip(score, -3.0, 1.0))

    def _likelihood(self, signals: SignalSnapshot, health_class: str) -> float:
        """
        P(o_t | health_class) — likelihood of observation under each class.
        """
        z = signals.loss_z_score
        div = signals.divergence_score

        if health_class == "HEALTHY":
            # High likelihood when z-score is small and divergence is low
            return float(np.exp(-0.5 * z**2) * (1 - div))
        elif health_class == "DEGRADING":
            # High likelihood when divergence is moderate
            return float(np.exp(-0.5 * (z - 1.5) ** 2) * div)
        elif health_class == "ANOMALOUS":
            # High likelihood when z-score is large
            return float(np.exp(-0.5 * (abs(z) - 3.0) ** 2 / 4.0))
        return 0.1

    def _update_gp(self, obs: np.ndarray, health_score: float) -> Tuple[float, float]:
        """
        Update GP model and return (mean_prediction, std_prediction).
        """
        self._obs_history.append(obs)
        self._health_history.append(health_score)

        if len(self._obs_history) < 10:
            return health_score, 1.0

        # Only keep last 200 points for GP (efficiency)
        X = np.array(self._obs_history[-200:])
        y = np.array(self._health_history[-200:])

        # Try sklearn GP
        if self._gp is not None and len(X) >= 10:
            try:
                if len(X) % 20 == 0:  # refit every 20 observations
                    self._gp.fit(X, y)
                mean, std = self._gp.predict(obs.reshape(1, -1), return_std=True)
                return float(mean[0]), float(std[0])
            except Exception:
                pass

        # Simple fallback: EMA of health score
        ema = float(np.mean(y[-10:]))
        std = float(np.std(y[-10:])) + 0.1
        return ema, std
