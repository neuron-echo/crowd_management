"""Configuration for the Decision Engine.

All thresholds, weights, and tuning parameters are centralized here.
No dependencies on torch, cv2, or any vision model code.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CongestionThresholds(BaseModel):
    """Thresholds for mapping congestion_score to CongestionLevel.

    LOW:      score < moderate_threshold
    MODERATE: moderate_threshold <= score < high_threshold
    HIGH:     high_threshold <= score < critical_threshold
    CRITICAL: score >= critical_threshold
    """

    moderate_threshold: float = Field(default=0.4, ge=0, le=1)
    high_threshold: float = Field(default=0.65, ge=0, le=1)
    critical_threshold: float = Field(default=0.8, ge=0, le=1)


class CongestionWeights(BaseModel):
    """Weights for computing congestion_score from observation features.

    congestion_score = w_density * density_score
                     + w_velocity * inverse_velocity_score
                     + w_capacity * capacity_utilization

    where inverse_velocity_score = 1.0 - clamp(velocity / max_expected_velocity, 0, 1)

    NOTE: This formula is a configurable heuristic, NOT scientifically validated.
    It is designed to be transparent and tunable for demonstration purposes.
    """

    w_density: float = Field(default=0.5, ge=0, le=1)
    w_velocity: float = Field(default=0.2, ge=0, le=1)
    w_capacity: float = Field(default=0.3, ge=0, le=1)
    max_expected_velocity: float = Field(
        default=15.0, gt=0,
        description="Velocity at or above which inverse_velocity_score = 0 (fast = not congested)",
    )


class RoutingWeights(BaseModel):
    """Weights for scoring candidate diversion targets.

    routing_score = w_congestion * (1 - candidate_congestion_score)
                  + w_capacity  * (1 - capacity_utilization)
                  + w_distance  * (1 - normalized_distance)
                  + w_trend     * trend_bonus

    Higher score = better candidate.
    """

    w_congestion: float = Field(default=0.35, ge=0, le=1)
    w_capacity: float = Field(default=0.30, ge=0, le=1)
    w_distance: float = Field(default=0.20, ge=0, le=1)
    w_trend: float = Field(default=0.15, ge=0, le=1)


class HysteresisConfig(BaseModel):
    """Decision stability configuration to prevent oscillation."""

    cooldown_seconds: float = Field(
        default=10.0, ge=0,
        description="Minimum time before the same source gate can change its diversion target",
    )
    switching_margin: float = Field(
        default=0.15, ge=0, le=1,
        description="A new candidate must beat the current target's score by this margin to trigger a switch",
    )


class DecisionConfig(BaseModel):
    """Complete configuration for the Decision Engine."""

    congestion_thresholds: CongestionThresholds = Field(default_factory=CongestionThresholds)
    congestion_weights: CongestionWeights = Field(default_factory=CongestionWeights)
    routing_weights: RoutingWeights = Field(default_factory=RoutingWeights)
    hysteresis: HysteresisConfig = Field(default_factory=HysteresisConfig)

    # When to trigger diversion evaluation
    divert_above_level: str = Field(
        default="HIGH",
        description="Minimum congestion level that triggers diversion evaluation (HIGH or CRITICAL)",
    )

    # Observation freshness
    max_observation_age_s: float = Field(
        default=5.0, gt=0,
        description="Maximum age of a CrowdState observation before it is considered stale",
    )
    min_confidence: float = Field(
        default=0.3, ge=0, le=1,
        description="Minimum observation confidence required for decision-making",
    )

    # Candidate viability
    max_candidate_congestion: float = Field(
        default=0.75, ge=0, le=1,
        description="Maximum congestion_score for a gate to be considered as a diversion target",
    )
    min_capacity_remaining_ratio: float = Field(
        default=0.15, ge=0, le=1,
        description="Minimum (1 - capacity_utilization) for a candidate gate to be viable",
    )

    # Trend history
    trend_window_size: int = Field(
        default=5, ge=2,
        description="Number of recent observations for trend calculation",
    )
    trend_change_threshold: float = Field(
        default=0.05, ge=0,
        description="Minimum score difference between recent and older observations to detect a trend",
    )
