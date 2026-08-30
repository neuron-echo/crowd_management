"""Rule-based Decision Engine for crowd management.

Input:  dict[gate_id, CrowdState] + VenueTopology
Output: list[DecisionResult]

This engine has ZERO dependencies on:
- torch / torchvision
- cv2 / OpenCV
- CSRNet / RAFT
- FramePacket / FrameBuffer / FrameProducer

It only knows WHAT is happening at each gate and HOW gates are connected.
"""

from __future__ import annotations

import time

from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import (
    ActionType,
    CandidateEvaluation,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    Trend,
)
from crowd_management.decision.routing import rank_candidates
from crowd_management.venue.topology import VenueTopology


# Congestion level ordering for comparison
_LEVEL_ORDER = {
    CongestionLevel.LOW: 0,
    CongestionLevel.MODERATE: 1,
    CongestionLevel.HIGH: 2,
    CongestionLevel.CRITICAL: 3,
}


def _level_at_least(level: CongestionLevel, threshold_name: str) -> bool:
    """Check if a congestion level meets or exceeds a named threshold."""
    threshold_map = {
        "LOW": CongestionLevel.LOW,
        "MODERATE": CongestionLevel.MODERATE,
        "HIGH": CongestionLevel.HIGH,
        "CRITICAL": CongestionLevel.CRITICAL,
    }
    threshold = threshold_map.get(threshold_name.upper(), CongestionLevel.HIGH)
    return _LEVEL_ORDER[level] >= _LEVEL_ORDER[threshold]


class DecisionEngine:
    """Pure rule-based crowd management Decision Engine.

    For each gate with congestion above the configured threshold:
    1. Evaluate all reachable candidate gates
    2. Select the best viable candidate based on routing score
    3. Apply hysteresis to prevent decision oscillation
    4. Produce an explainable DecisionResult

    Safety rules:
    - No diversion if no viable destination exists
    - No diversion if all connected gates are congested
    - No diversion if observation data is stale
    - No diversion if confidence is too low
    - No diversion if source gate is not actually congested
    """

    def __init__(self, venue: VenueTopology, config: DecisionConfig | None = None) -> None:
        self._venue = venue
        self._config = config or DecisionConfig()

        # Hysteresis state: gate_id -> (target_gate_id, decision_timestamp)
        self._active_diversions: dict[str, tuple[str, float]] = {}

    @property
    def venue(self) -> VenueTopology:
        return self._venue

    @property
    def config(self) -> DecisionConfig:
        return self._config

    def evaluate(self, states: dict[str, CrowdState]) -> list[DecisionResult]:
        """Evaluate all gates and produce decisions.

        Args:
            states: Mapping of gate_id -> CrowdState for all observed gates.

        Returns:
            List of DecisionResult objects, one per gate that was evaluated.
        """
        results: list[DecisionResult] = []
        now = time.time()

        for gate_id in self._venue.gate_ids:
            state = states.get(gate_id)
            if state is None:
                continue

            result = self._evaluate_gate(gate_id, state, states, now)
            results.append(result)

        return results

    def _evaluate_gate(
        self,
        gate_id: str,
        state: CrowdState,
        all_states: dict[str, CrowdState],
        now: float,
    ) -> DecisionResult:
        """Evaluate a single gate and produce a DecisionResult."""

        # --- Safety check: stale observation ---
        if state.result_age_s > self._config.max_observation_age_s:
            return self._make_result(
                gate_id, ActionType.MONITOR, state.congestion_level,
                confidence=state.confidence * 0.5,
                reason_codes=("STALE_OBSERVATION",),
                reason=f"Observation for {gate_id} is stale ({state.result_age_s:.1f}s old). "
                       f"Not safe to issue diversion. Monitoring only.",
            )

        # --- Safety check: low confidence ---
        if state.confidence < self._config.min_confidence:
            return self._make_result(
                gate_id, ActionType.MONITOR, state.congestion_level,
                confidence=state.confidence,
                reason_codes=("LOW_CONFIDENCE",),
                reason=f"Observation confidence for {gate_id} is {state.confidence:.2f}, "
                       f"below minimum {self._config.min_confidence:.2f}. Monitoring only.",
            )

        # --- Check if congestion warrants action ---
        if not _level_at_least(state.congestion_level, self._config.divert_above_level):
            action = ActionType.MONITOR if state.congestion_level == CongestionLevel.MODERATE else ActionType.NO_ACTION
            level_label = state.congestion_level.value
            return self._make_result(
                gate_id, action, state.congestion_level,
                confidence=state.confidence,
                reason_codes=("BELOW_THRESHOLD",),
                reason=f"{gate_id} congestion is {level_label} (score={state.congestion_score:.2f}). "
                       f"No diversion needed.",
            )

        # --- Gate is congested: evaluate routing candidates ---
        candidates = rank_candidates(gate_id, all_states, self._venue, self._config)
        viable = [c for c in candidates if c.is_viable]

        if not viable:
            return self._make_result(
                gate_id, ActionType.MONITOR, state.congestion_level,
                confidence=state.confidence,
                reason_codes=("NO_VIABLE_TARGET",),
                reason=f"{gate_id} is {state.congestion_level.value} but no viable diversion target exists. "
                       f"All connected gates are congested or have insufficient capacity.",
                candidates=tuple(candidates),
            )

        # --- Select best candidate, applying hysteresis ---
        best = viable[0]
        target_id = best.gate_id

        # Hysteresis: if we already have an active diversion for this gate,
        # only switch if the new candidate beats the old one by switching_margin
        active = self._active_diversions.get(gate_id)
        if active is not None:
            prev_target, prev_time = active
            elapsed = now - prev_time

            if elapsed < self._config.hysteresis.cooldown_seconds:
                # Still in cooldown: keep the previous target if it's still viable
                prev_eval = next((c for c in viable if c.gate_id == prev_target), None)
                if prev_eval is not None:
                    margin = best.routing_score - prev_eval.routing_score
                    if margin < self._config.hysteresis.switching_margin:
                        target_id = prev_target
                        best = prev_eval

        # Update hysteresis state
        self._active_diversions[gate_id] = (target_id, now)

        target_state = all_states.get(target_id)
        trend_label = state.trend.value.lower()

        reason = (
            f"{gate_id} has {state.congestion_level.value} congestion "
            f"(score={state.congestion_score:.2f}) and {trend_label} density. "
            f"{target_id} is connected, {best.distance_m:.0f}m away, "
            f"has {best.congestion_level.value} congestion "
            f"(score={best.congestion_score:.2f}), "
            f"and capacity utilization of {best.capacity_utilization:.0%}."
        )

        return self._make_result(
            gate_id, ActionType.DIVERT, state.congestion_level,
            target_gate_id=target_id,
            confidence=state.confidence,
            reason_codes=("CONGESTED", "DIVERT_RECOMMENDED"),
            reason=reason,
            candidates=tuple(candidates),
        )

    def _make_result(
        self,
        source_gate_id: str,
        action: ActionType,
        severity: CongestionLevel,
        target_gate_id: str | None = None,
        confidence: float = 1.0,
        reason_codes: tuple[str, ...] = (),
        reason: str = "",
        candidates: tuple[CandidateEvaluation, ...] = (),
    ) -> DecisionResult:
        return DecisionResult(
            decision_id=DecisionResult.generate_id(),
            timestamp=time.time(),
            source_gate_id=source_gate_id,
            action=action,
            target_gate_id=target_gate_id,
            severity=severity,
            confidence=confidence,
            reason_codes=reason_codes,
            human_readable_reason=reason,
            candidate_evaluations=candidates,
        )

    def reset_hysteresis(self, gate_id: str | None = None) -> None:
        """Reset hysteresis state for a gate or all gates."""
        if gate_id is not None:
            self._active_diversions.pop(gate_id, None)
        else:
            self._active_diversions.clear()
