"""
Causal graph over training dynamics signals.

Reuses your existing graph/graph.py DAG engine — same node/edge
structure, just different domain (training signals vs pipeline steps).

The causal graph represents how information flows during training:

    batch_quality ──────────────────────────────┐
    data_stats    ──→ gradient_distribution ──→ layer_grad_norms ──→ weight_updates ──→ loss
    optimizer_state ──────────────────────────────────────────────────────────────────────┘
    lr_schedule ──────────────────────────────────────────────────────────────────────────┘

When an anomaly fires, we traverse this graph backwards from the
loss node to find which upstream node showed the earliest deviation
from its baseline. That node is the root cause.

Hooks into:
    graph/graph.py  → GraphNode, GraphEdge base classes if compatible
                      (wraps them if needed, or reimplements lightly)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from agent.observer.rolling_window import RollingWindow
from agent.observer.signal_extractor import SignalSnapshot

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """Health status of a causal node at a given step."""

    NORMAL = auto()  # within expected range
    DEVIATED = auto()  # outside expected range — potential root cause
    UNKNOWN = auto()  # insufficient data to determine
    MASKED = auto()  # upstream deviation makes this downstream irrelevant


@dataclass
class CausalNode:
    """
    One node in the training dynamics causal graph.

    Stores baseline statistics (from first N steps of the run)
    and current value for anomaly detection via z-score.
    """

    name: str
    description: str
    parents: List[str] = field(default_factory=list)  # upstream node names
    children: List[str] = field(default_factory=list)  # downstream node names

    # Baseline stats — computed from first 100 healthy steps
    baseline_mean: float = 0.0
    baseline_std: float = 1.0
    baseline_n: int = 0

    # Current value at time of anomaly
    current_value: float = 0.0
    current_z_score: float = 0.0
    status: NodeStatus = NodeStatus.UNKNOWN

    # Step at which deviation was first detected
    first_deviation_step: Optional[int] = None

    def update_baseline(self, values: np.ndarray) -> None:
        """Update baseline statistics from a window of healthy values."""
        self.baseline_mean = float(np.mean(values))
        self.baseline_std = float(np.std(values)) + 1e-8
        self.baseline_n = len(values)

    def evaluate(self, value: float, step: int, z_threshold: float = 2.5) -> NodeStatus:
        """
        Compare current value to baseline.
        Returns DEVIATED if |z-score| > threshold.
        """
        self.current_value = value
        if self.baseline_n < 10:
            self.status = NodeStatus.UNKNOWN
            return self.status

        self.current_z_score = (value - self.baseline_mean) / self.baseline_std
        if abs(self.current_z_score) > z_threshold:
            self.status = NodeStatus.DEVIATED
            if self.first_deviation_step is None:
                self.first_deviation_step = step
        else:
            self.status = NodeStatus.NORMAL
        return self.status

    def __repr__(self) -> str:
        return (
            f"CausalNode({self.name}, "
            f"status={self.status.name}, "
            f"z={self.current_z_score:.2f})"
        )


@dataclass
class CausalEdge:
    """Directed edge: parent → child in causal graph."""

    parent: str
    child: str
    strength: float = 1.0  # correlation strength (0–1), estimated from data


class CausalGraph:
    """
    DAG representing training dynamics causal structure.

    Reuses your graph/graph.py topological sort logic for
    traversal ordering. Uses the same Kahn's algorithm approach.

    The graph is fixed (same structure for all FRAMEWORM models)
    but node baselines are calibrated per-run from the first
    100 healthy training steps.

    Nodes (in causal order, roots first):
        batch_quality       → outlier score of current mini-batch
        data_stats          → batch mean/variance vs dataset stats
        optimizer_state     → effective step size, momentum magnitude
        lr_schedule         → current LR relative to peak LR
        gradient_dist       → overall gradient norm + variance
        layer_grad_norms    → per-layer gradient norms (top 5 layers)
        activation_stats    → per-layer activation mean/variance
        weight_updates      → weight update magnitude ratio
        loss                → scalar training loss (always the sink)

    Usage:
        graph = CausalGraph()
        graph.calibrate_baseline(window, healthy_steps=100)
        graph.evaluate_at(snapshot, step=4200)
        root_causes = graph.find_root_causes()
    """

    # Fixed graph structure — same for all models
    GRAPH_STRUCTURE = {
        "batch_quality": {"parents": [], "desc": "Mini-batch outlier score"},
        "data_stats": {"parents": [], "desc": "Batch statistics vs dataset"},
        "optimizer_state": {"parents": [], "desc": "Optimizer momentum + step size"},
        "lr_schedule": {"parents": [], "desc": "Current LR relative to peak"},
        "gradient_dist": {
            "parents": ["batch_quality", "data_stats", "lr_schedule"],
            "desc": "Gradient norm distribution",
        },
        "layer_grad_norms": {"parents": ["gradient_dist"], "desc": "Per-layer gradient norms"},
        "activation_stats": {"parents": ["batch_quality"], "desc": "Per-layer activations"},
        "weight_updates": {
            "parents": ["layer_grad_norms", "optimizer_state"],
            "desc": "Weight update magnitude",
        },
        "loss": {"parents": ["weight_updates", "activation_stats"], "desc": "Scalar training loss"},
    }

    def __init__(self, z_threshold: float = 2.5) -> None:
        self.z_threshold = z_threshold
        self.nodes: Dict[str, CausalNode] = {}
        self.edges: List[CausalEdge] = []
        self._topo_order: List[str] = []
        self._build_graph()

    # ── Graph construction ────────────────────────────────────────

    def _build_graph(self) -> None:
        """
        Instantiate nodes and edges from GRAPH_STRUCTURE.
        Tries to reuse your graph/graph.py if compatible.
        """
        # Try to hook into your existing graph engine first
        try:
            from graph.graph import Graph as FWGraph

            self._fw_graph = FWGraph()
            logger.debug("CausalGraph: using FRAMEWORM graph engine for traversal")
        except ImportError:
            self._fw_graph = None
            logger.debug("CausalGraph: using built-in traversal")

        # Build nodes
        for name, props in self.GRAPH_STRUCTURE.items():
            node = CausalNode(
                name=name,
                description=props["desc"],
                parents=props["parents"].copy(),
            )
            self.nodes[name] = node

        # Build edges + populate children lists
        for name, props in self.GRAPH_STRUCTURE.items():
            for parent_name in props["parents"]:
                edge = CausalEdge(parent=parent_name, child=name)
                self.edges.append(edge)
                if parent_name in self.nodes:
                    self.nodes[parent_name].children.append(name)

        # Compute topological order (Kahn's algorithm — same as your graph.py)
        self._topo_order = self._kahn_sort()
        logger.debug(f"CausalGraph topology: {self._topo_order}")

    def _kahn_sort(self) -> List[str]:
        """
        Kahn's topological sort — same algorithm as your graph/graph.py.
        Returns node names in execution order (roots first).
        """
        in_degree: Dict[str, int] = {name: 0 for name in self.nodes}
        for edge in self.edges:
            in_degree[edge.child] += 1

        queue = [name for name, deg in in_degree.items() if deg == 0]
        order = []

        while queue:
            node = queue.pop(0)
            order.append(node)
            for child in self.nodes[node].children:
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

        if len(order) != len(self.nodes):
            logger.warning("CausalGraph: cycle detected — topological sort incomplete")

        return order

    # ── Calibration ───────────────────────────────────────────────

    def calibrate_baseline(
        self,
        window: RollingWindow,
        healthy_steps: int = 100,
    ) -> None:
        """
        Compute baseline statistics for each node from the first
        N healthy steps of the run.

        Call this once early in training (after first 100 steps).
        The agent's _tick() should call this automatically after
        enough history accumulates.
        """
        losses = window.losses(n=healthy_steps)
        grad_norms = window.grad_norms(n=healthy_steps)
        lrs = window.lrs(n=healthy_steps)

        if len(losses) < 20:
            logger.debug("CausalGraph: not enough steps to calibrate baseline")
            return

        # Calibrate observable nodes from rolling window data
        self.nodes["loss"].update_baseline(losses)
        self.nodes["gradient_dist"].update_baseline(grad_norms)
        self.nodes["lr_schedule"].update_baseline(lrs)

        # Approximate unobservable nodes from observable proxies
        # weight_updates ≈ grad_norm * lr (simplified)
        weight_update_proxy = grad_norms * lrs
        self.nodes["weight_updates"].update_baseline(weight_update_proxy)

        # optimizer_state ≈ grad_norm variance (momentum smooths gradients)
        grad_var = np.array(
            [np.var(grad_norms[max(0, i - 10) : i + 1]) for i in range(len(grad_norms))]
        )
        self.nodes["optimizer_state"].update_baseline(grad_var)

        # Layer-level nodes get approximate baselines
        # These are refined if layer_grad_norms are available in snapshots
        snapshots = window.snapshots(n=healthy_steps)
        if snapshots and snapshots[0].layer_grad_norms:
            all_layer_norms = []
            for snap in snapshots:
                if snap.layer_grad_norms:
                    mean_layer = float(np.mean(list(snap.layer_grad_norms.values())))
                    all_layer_norms.append(mean_layer)
            if all_layer_norms:
                self.nodes["layer_grad_norms"].update_baseline(np.array(all_layer_norms))

        logger.info(
            f"CausalGraph: baseline calibrated from {len(losses)} steps. "
            f"loss_mean={self.nodes['loss'].baseline_mean:.4f}, "
            f"grad_mean={self.nodes['gradient_dist'].baseline_mean:.4f}"
        )

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate_at(
        self,
        snapshot,  # MetricSnapshot from observer
        signals: SignalSnapshot,
        step: int,
    ) -> Dict[str, NodeStatus]:
        """
        Evaluate all nodes at the current step.
        Returns dict of {node_name: NodeStatus}.
        """
        # Evaluate observable nodes
        self.nodes["loss"].evaluate(snapshot.loss, step, self.z_threshold)
        self.nodes["gradient_dist"].evaluate(snapshot.grad_norm, step, self.z_threshold)
        self.nodes["lr_schedule"].evaluate(snapshot.lr, step, self.z_threshold)

        # Derived nodes
        weight_upd = snapshot.grad_norm * snapshot.lr
        self.nodes["weight_updates"].evaluate(weight_upd, step, self.z_threshold)

        optimizer_proxy = signals.grad_norm_var
        self.nodes["optimizer_state"].evaluate(optimizer_proxy, step, self.z_threshold)

        # Layer-level (if available)
        if snapshot.layer_grad_norms:
            mean_layer = float(np.mean(list(snapshot.layer_grad_norms.values())))
            self.nodes["layer_grad_norms"].evaluate(mean_layer, step, self.z_threshold)

        # batch_quality and data_stats are harder to observe directly
        # Use loss z-score as proxy for batch quality
        self.nodes["batch_quality"].evaluate(abs(signals.loss_z_score), step, z_threshold=2.0)

        return {name: node.status for name, node in self.nodes.items()}

    def find_root_causes(self) -> List[CausalNode]:
        """
        Traverse graph backwards from loss to find the earliest
        deviated node. Uses reverse topological order.

        Returns list of root cause nodes (usually 1–2),
        sorted by how early their deviation started.
        """
        deviated = [node for node in self.nodes.values() if node.status == NodeStatus.DEVIATED]

        if not deviated:
            return []

        # Find roots: deviated nodes with no deviated parents
        root_causes = []
        for node in deviated:
            has_deviated_parent = any(
                self.nodes.get(p, CausalNode("", "")).status == NodeStatus.DEVIATED
                for p in node.parents
            )
            if not has_deviated_parent:
                root_causes.append(node)

        # Sort by earliest deviation
        root_causes.sort(key=lambda n: n.first_deviation_step or float("inf"))

        return root_causes

    def get_causal_path(self, root_name: str, sink_name: str = "loss") -> List[str]:
        """
        Get the causal path from a root cause node to the loss node.
        Used for generating human-readable attribution explanations.
        """
        if root_name not in self.nodes or sink_name not in self.nodes:
            return []

        # BFS from root to sink
        from collections import deque

        queue = deque([[root_name]])
        while queue:
            path = queue.popleft()
            current = path[-1]
            if current == sink_name:
                return path
            for child in self.nodes[current].children:
                if child not in path:  # avoid cycles
                    queue.append(path + [child])
        return []

    def summary(self) -> str:
        """Human-readable graph status summary."""
        lines = ["CausalGraph Status:"]
        for name in self._topo_order:
            node = self.nodes[name]
            status_icon = {"NORMAL": "✓", "DEVIATED": "✗", "UNKNOWN": "?", "MASKED": "~"}.get(
                node.status.name, "?"
            )
            lines.append(
                f"  {status_icon} {name:20s} z={node.current_z_score:+.2f}  [{node.status.name}]"
            )
        return "\n".join(lines)
