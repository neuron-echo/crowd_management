"""Routing candidate scorer for diversion decisions.

Scores candidate gates using a weighted combination of:
    - Low congestion (inverted congestion_score)
    - Available capacity (inverted capacity_utilization)
    - Short distance (inverted normalized distance)
    - Favorable trend bonus

All weights are configurable via DecisionConfig.routing_weights.

No dependencies on torch, cv2, or any vision model code.
"""

from __future__ import annotations

from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import (
    CandidateEvaluation,
    CongestionLevel,
    CrowdState,
    Trend,
)
from crowd_management.venue.topology import VenueTopology


def _trend_bonus(trend: Trend) -> float:
    """Convert trend to a routing bonus value.

    DECREASING (crowd is clearing) → 1.0 bonus
    STABLE → 0.5 bonus
    INCREASING (crowd is growing) → 0.0 bonus
    """
    if trend == Trend.DECREASING:
        return 1.0
    elif trend == Trend.STABLE:
        return 0.5
    else:
        return 0.0


def evaluate_candidate(
    source_gate_id: str,
    candidate_state: CrowdState,
    venue: VenueTopology,
    config: DecisionConfig,
    max_venue_distance: float,
) -> CandidateEvaluation:
    """Evaluate a single candidate gate for diversion routing.

    Args:
        source_gate_id: The congested gate requesting diversion.
        candidate_state: CrowdState of the candidate gate.
        venue: Venue topology for distance computation.
        config: Decision configuration with routing weights and viability thresholds.
        max_venue_distance: Maximum distance in the venue (for normalization).

    Returns:
        CandidateEvaluation with score and viability determination.
    """
    cand_id = candidate_state.gate_id
    distance = venue.shortest_distance(source_gate_id, cand_id)
    gate_capacity = venue.get_gate_capacity(cand_id)

    capacity_utilization = candidate_state.density_score
    remaining_ratio = 1.0 - capacity_utilization

    # Check viability
    rejection_reason: str | None = None

    if candidate_state.congestion_score > config.max_candidate_congestion:
        rejection_reason = (
            f"Congestion {candidate_state.congestion_score:.2f} exceeds "
            f"max {config.max_candidate_congestion:.2f}"
        )
    elif remaining_ratio < config.min_capacity_remaining_ratio:
        rejection_reason = (
            f"Remaining capacity ratio {remaining_ratio:.2f} below "
            f"min {config.min_capacity_remaining_ratio:.2f}"
        )
    elif distance == float("inf"):
        rejection_reason = "Gate is unreachable"

    is_viable = rejection_reason is None

    # Compute routing score (even for non-viable candidates, for transparency)
    w = config.routing_weights
    norm_distance = min(1.0, distance / max_venue_distance) if max_venue_distance > 0 else 0.5

    score = (
        w.w_congestion * (1.0 - candidate_state.congestion_score)
        + w.w_capacity * remaining_ratio
        + w.w_distance * (1.0 - norm_distance)
        + w.w_trend * _trend_bonus(candidate_state.trend)
    )

    weight_sum = w.w_congestion + w.w_capacity + w.w_distance + w.w_trend
    if weight_sum > 0:
        score /= weight_sum

    return CandidateEvaluation(
        gate_id=cand_id,
        distance_m=distance,
        congestion_level=candidate_state.congestion_level,
        congestion_score=candidate_state.congestion_score,
        capacity_utilization=capacity_utilization,
        trend=candidate_state.trend,
        routing_score=score,
        is_viable=is_viable,
        rejection_reason=rejection_reason,
    )


def rank_candidates(
    source_gate_id: str,
    all_states: dict[str, CrowdState],
    venue: VenueTopology,
    config: DecisionConfig,
) -> list[CandidateEvaluation]:
    """Evaluate and rank all candidate gates for diversion from source_gate_id.

    Returns candidates sorted by routing_score (highest first).
    Non-viable candidates are included but marked as not viable.
    """
    max_dist = venue.max_distance()
    reachable = venue.reachable_gates(source_gate_id)

    evaluations: list[CandidateEvaluation] = []
    for gid in reachable:
        if gid not in all_states:
            # No observation for this gate — cannot route there safely
            evaluations.append(CandidateEvaluation(
                gate_id=gid,
                distance_m=venue.shortest_distance(source_gate_id, gid),
                congestion_level=CongestionLevel.LOW,
                congestion_score=0.0,
                capacity_utilization=0.0,
                trend=Trend.STABLE,
                routing_score=0.0,
                is_viable=False,
                rejection_reason="No crowd observation available for this gate",
            ))
            continue

        candidate_state = all_states[gid]
        evaluation = evaluate_candidate(
            source_gate_id, candidate_state, venue, config, max_dist
        )
        evaluations.append(evaluation)

    # Sort: viable first, then by routing_score descending
    evaluations.sort(key=lambda e: (-int(e.is_viable), -e.routing_score))
    return evaluations
