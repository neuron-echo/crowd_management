"""Trend tracking from rolling CrowdState observations.

Maintains a per-gate sliding window of congestion scores and derives
INCREASING / STABLE / DECREASING trend by comparing recent vs older averages.

No dependencies on torch, cv2, or any vision model code.
"""

from __future__ import annotations

from collections import defaultdict, deque

from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import Trend


class TrendTracker:
    """Per-gate trend tracker using a rolling window of congestion scores.

    The window is split into two halves:
        older_half = first ceil(N/2) observations
        recent_half = last floor(N/2) observations

    If avg(recent) - avg(older) > threshold → INCREASING
    If avg(older) - avg(recent) > threshold → DECREASING
    Otherwise → STABLE
    """

    def __init__(self, config: DecisionConfig | None = None) -> None:
        if config is None:
            config = DecisionConfig()
        self._window_size = config.trend_window_size
        self._threshold = config.trend_change_threshold
        self._history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=self._window_size))

    def update(self, gate_id: str, congestion_score: float) -> Trend:
        """Record a new congestion_score observation and return the current trend.

        Args:
            gate_id: Gate identifier.
            congestion_score: Latest congestion score in [0, 1].

        Returns:
            Computed trend direction.
        """
        window = self._history[gate_id]
        window.append(congestion_score)

        if len(window) < 2:
            return Trend.STABLE

        scores = list(window)
        mid = len(scores) // 2
        if mid == 0:
            mid = 1

        older = scores[:mid]
        recent = scores[mid:]

        avg_older = sum(older) / len(older)
        avg_recent = sum(recent) / len(recent)

        diff = avg_recent - avg_older

        if diff > self._threshold:
            return Trend.INCREASING
        elif diff < -self._threshold:
            return Trend.DECREASING
        else:
            return Trend.STABLE

    def get_trend(self, gate_id: str) -> Trend:
        """Return the last computed trend for a gate without adding a new observation."""
        window = self._history.get(gate_id)
        if window is None or len(window) < 2:
            return Trend.STABLE

        scores = list(window)
        mid = len(scores) // 2
        if mid == 0:
            mid = 1

        avg_older = sum(scores[:mid]) / len(scores[:mid])
        avg_recent = sum(scores[mid:]) / len(scores[mid:])

        diff = avg_recent - avg_older
        if diff > self._threshold:
            return Trend.INCREASING
        elif diff < -self._threshold:
            return Trend.DECREASING
        else:
            return Trend.STABLE

    def reset(self, gate_id: str | None = None) -> None:
        """Reset trend history for a specific gate or all gates."""
        if gate_id is not None:
            self._history.pop(gate_id, None)
        else:
            self._history.clear()
