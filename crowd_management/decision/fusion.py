"""Feature fusion: build model-agnostic CrowdState from raw observations.

This module computes congestion_score and congestion_level from raw density,
velocity, and capacity information. It does NOT import torch, cv2, CSRNet,
RAFT, or any model-specific types.

The congestion scoring formula is:

    congestion_score = w_density  * density_score
                     + w_velocity * inverse_velocity_score
                     + w_capacity * capacity_utilization

    where:
        density_score          = clamp(crowd_count / gate_capacity, 0, 1)
        inverse_velocity_score = 1 - clamp(velocity / max_expected_velocity, 0, 1)
        capacity_utilization   = density_score  (same as above)

    This formula is a CONFIGURABLE HEURISTIC and is NOT scientifically validated.
    All weights are exposed in DecisionConfig.congestion_weights.

Congestion level is derived from congestion_score using configurable thresholds:
    LOW:      score < 0.4
    MODERATE: 0.4 <= score < 0.65
    HIGH:     0.65 <= score < 0.8
    CRITICAL: score >= 0.8
"""

from __future__ import annotations

import time

from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import CongestionLevel, CrowdState, Trend


def classify_congestion(score: float, config: DecisionConfig) -> CongestionLevel:
    """Map a congestion_score to a CongestionLevel using configured thresholds."""
    thresholds = config.congestion_thresholds
    if score >= thresholds.critical_threshold:
        return CongestionLevel.CRITICAL
    elif score >= thresholds.high_threshold:
        return CongestionLevel.HIGH
    elif score >= thresholds.moderate_threshold:
        return CongestionLevel.MODERATE
    else:
        return CongestionLevel.LOW


def compute_congestion_score(
    density_score: float,
    velocity: float,
    capacity_utilization: float,
    config: DecisionConfig,
    has_motion: bool = True,
) -> float:
    """Compute a composite congestion score from observations.

    Args:
        density_score: Normalized density in [0, 1].
        velocity: Crowd movement velocity magnitude (≥ 0).
        capacity_utilization: crowd_count / capacity, clamped to [0, 1].
        config: Decision engine configuration with weights.
        has_motion: Whether valid current motion observation is available.

    Returns:
        Congestion score in [0, 1].
    """
    w = config.congestion_weights

    if not has_motion:
        # When motion is unavailable/stale, compute score purely from density and capacity utilization
        # without assuming either 0 velocity (artificial congestion) or max velocity (artificial freedom).
        weight_sum = w.w_density + w.w_capacity
        if weight_sum > 0:
            raw_score = (
                w.w_density * min(1.0, max(0.0, density_score))
                + w.w_capacity * min(1.0, max(0.0, capacity_utilization))
            ) / weight_sum
        else:
            raw_score = density_score
        return min(1.0, max(0.0, raw_score))

    # Higher velocity = less congestion (crowd is flowing freely)
    inv_velocity = 1.0 - min(1.0, max(0.0, velocity / w.max_expected_velocity))

    raw_score = (
        w.w_density * min(1.0, max(0.0, density_score))
        + w.w_velocity * inv_velocity
        + w.w_capacity * min(1.0, max(0.0, capacity_utilization))
    )

    weight_sum = w.w_density + w.w_velocity + w.w_capacity
    if weight_sum > 0:
        raw_score /= weight_sum

    return min(1.0, max(0.0, raw_score))


def build_crowd_state(
    gate_id: str,
    gate_capacity: int,
    crowd_count: float = 0.0,
    velocity: float = 0.0,
    direction: float = 0.0,
    flow_consistency: float = 0.0,
    confidence: float = 1.0,
    source_frame_id: int | None = None,
    result_age_s: float = 0.0,
    has_density: bool = True,
    has_motion: bool = True,
    motion_status: str = "OK",
    raft_pair_provenance: str | None = None,
    raft_result_age_s: float = 0.0,
    is_motion_fresh: bool = True,
    trend: Trend = Trend.STABLE,
    config: DecisionConfig | None = None,
    timestamp: float | None = None,
) -> CrowdState:
    """Build a complete CrowdState from raw observation values.

    This function computes density_score, capacity_utilization, congestion_score,
    and congestion_level from the raw inputs.

    Args:
        gate_id: Gate identifier.
        gate_capacity: Maximum capacity of this gate.
        crowd_count: Estimated crowd count.
        velocity: Crowd movement velocity magnitude.
        direction: Dominant movement direction in degrees.
        flow_consistency: Flow coherence in [0, 1].
        confidence: Overall observation confidence in [0, 1].
        source_frame_id: Optional source frame identifier.
        result_age_s: Age of the observation in seconds.
        has_density: Whether density data is available.
        has_motion: Whether motion data is available.
        motion_status: "OK", "STALE", or "UNAVAILABLE".
        raft_pair_provenance: String descriptor of frame pair (e.g. "82->83").
        raft_result_age_s: Age of the RAFT optical flow result in seconds.
        is_motion_fresh: Whether the motion observation is within freshness threshold.
        trend: Crowd density trend.
        config: Decision configuration (uses defaults if None).
        timestamp: Observation timestamp (uses current time if None).

    Returns:
        Complete CrowdState with computed congestion metrics.
    """
    if config is None:
        config = DecisionConfig()

    if timestamp is None:
        timestamp = time.time()

    # Compute density_score (normalized capacity utilization)
    density_score = min(1.0, max(0.0, crowd_count / gate_capacity)) if gate_capacity > 0 else 0.0
    capacity_utilization = density_score

    # If motion is not available or stale, zero out velocity/direction so they cannot be misinterpreted as valid motion
    actual_motion_valid = has_motion and is_motion_fresh
    actual_vel = velocity if actual_motion_valid else 0.0
    actual_dir = direction if actual_motion_valid else 0.0
    actual_flow_con = flow_consistency if actual_motion_valid else 0.0

    # Compute congestion score
    congestion_score = compute_congestion_score(
        density_score=density_score,
        velocity=actual_vel,
        capacity_utilization=capacity_utilization,
        config=config,
        has_motion=actual_motion_valid,
    )

    # Degrade confidence if observations are partial or stale
    effective_confidence = confidence
    if not has_density:
        effective_confidence *= 0.5
    if not actual_motion_valid:
        effective_confidence *= 0.7

    congestion_level = classify_congestion(congestion_score, config)

    return CrowdState(
        gate_id=gate_id,
        timestamp=timestamp,
        crowd_count=crowd_count,
        density_score=density_score,
        velocity=actual_vel,
        direction=actual_dir,
        flow_consistency=actual_flow_con,
        congestion_score=congestion_score,
        congestion_level=congestion_level,
        trend=trend,
        confidence=effective_confidence,
        source_frame_id=source_frame_id,
        result_age_s=result_age_s,
        has_density=has_density,
        has_motion=actual_motion_valid,
        motion_status=motion_status,
        raft_pair_provenance=raft_pair_provenance,
        raft_result_age_s=raft_result_age_s,
        is_motion_fresh=is_motion_fresh,
    )
