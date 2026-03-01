"""
Thread-safe priority queue for anomaly events.

When multiple anomalies fire simultaneously (e.g. gradient explosion
AND loss spike), this ensures the most urgent is handled first.

Also deduplicates: if the same anomaly type is already queued,
the new event only replaces it if higher severity.
"""

from __future__ import annotations

import heapq
import threading
from typing import List, Optional

from agent.classifier.anomaly_types import AnomalyEvent, AnomalyType


class AnomalyPriorityQueue:
    """
    Min-heap where lower anomaly priority number = dequeued first.

    Usage:
        q = AnomalyPriorityQueue()
        q.push_all(events)          # from rule engine
        event = q.pop()             # highest priority first
        q.clear()                   # after rollback
    """

    def __init__(self, max_size: int = 20) -> None:
        self.max_size = max_size
        self._heap: List[AnomalyEvent] = []
        self._lock = threading.Lock()
        self._seen_types: set = set()

    def push(self, event: AnomalyEvent) -> None:
        """
        Add an anomaly event. If same type is already queued,
        only replaces if new event has higher severity.
        """
        with self._lock:
            if event.anomaly_type in self._seen_types:
                # Replace if higher severity
                for i, existing in enumerate(self._heap):
                    if existing.anomaly_type == event.anomaly_type:
                        if event.severity.value > existing.severity.value:
                            self._heap[i] = event
                            heapq.heapify(self._heap)
                return

            if len(self._heap) >= self.max_size:
                return  # silently drop if full

            heapq.heappush(self._heap, event)
            self._seen_types.add(event.anomaly_type)

    def push_all(self, events: List[AnomalyEvent]) -> None:
        """Push a list of events (from rule engine output)."""
        for event in events:
            self.push(event)

    def pop(self) -> Optional[AnomalyEvent]:
        """
        Remove and return the highest-priority anomaly event.
        Returns None if queue is empty.
        """
        with self._lock:
            if not self._heap:
                return None
            event = heapq.heappop(self._heap)
            self._seen_types.discard(event.anomaly_type)
            return event

    def peek(self) -> Optional[AnomalyEvent]:
        """Return highest-priority event without removing it."""
        with self._lock:
            return self._heap[0] if self._heap else None

    def clear(self) -> None:
        """Empty the queue — call after a rollback."""
        with self._lock:
            self._heap.clear()
            self._seen_types.clear()

    def has_type(self, anomaly_type: AnomalyType) -> bool:
        """Check if a specific anomaly type is currently queued."""
        with self._lock:
            return anomaly_type in self._seen_types

    def __len__(self) -> int:
        with self._lock:
            return len(self._heap)

    def is_empty(self) -> bool:
        return len(self) == 0

    def all_events(self) -> List[AnomalyEvent]:
        """Snapshot of all queued events (does not modify queue)."""
        with self._lock:
            return sorted(self._heap.copy())
