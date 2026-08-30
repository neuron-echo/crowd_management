"""Model-agnostic data contracts for the Decision Engine.

These contracts define the interface between model inference (Phase 2) and
the Decision Engine (Phase 3). They contain NO imports from torch, cv2,
CSRNet, RAFT, or any vision model code.

The Decision Engine consumes CrowdState and produces DecisionResult.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ===================================================================
# Enumerations
# ===================================================================


class CongestionLevel(str, Enum):
    """Discrete congestion classification for a gate."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class Trend(str, Enum):
    """Crowd density trend direction."""

    DECREASING = "DECREASING"
    STABLE = "STABLE"
    INCREASING = "INCREASING"


class ActionType(str, Enum):
    """Decision action types."""

    NO_ACTION = "NO_ACTION"
    MONITOR = "MONITOR"
    DIVERT = "DIVERT"
    RESTRICT_INFLOW = "RESTRICT_INFLOW"
    HOLD = "HOLD"


# ===================================================================
# CrowdState — model-agnostic observation at a gate
# ===================================================================


@dataclass(frozen=True)
class CrowdState:
    """Model-independent observation of crowd conditions at a single gate.

    This is the ONLY input contract consumed by the Decision Engine.
    It does NOT reference CSRNet, RAFT, torch, cv2, or any model-specific types.

    Attributes:
        gate_id: Gate identifier (from venue config, NOT camera_id).
        timestamp: Observation timestamp (wall-clock seconds since epoch).
        crowd_count: Estimated crowd count (from any density model, or manual input).
        density_score: Normalized density in [0, 1] (0 = empty, 1 = at/over capacity).
        velocity: Crowd movement velocity (model-agnostic magnitude, ≥ 0).
        direction: Dominant movement direction in degrees [0, 360).
        flow_consistency: Flow coherence in [0, 1] (1 = perfectly uniform movement).
        congestion_score: Composite congestion score in [0, 1].
        congestion_level: Discrete congestion classification.
        trend: Crowd density trend direction.
        confidence: Overall observation confidence in [0, 1].
        source_frame_id: Optional identifier of the source frame.
        result_age_s: Age of the underlying model result in seconds.
        has_density: True if density/count data is present (not estimated/defaulted).
        has_motion: True if motion/flow data is present (not estimated/defaulted).
    """

    gate_id: str
    timestamp: float

    crowd_count: float = 0.0
    density_score: float = 0.0

    velocity: float = 0.0
    direction: float = 0.0
    flow_consistency: float = 0.0

    congestion_score: float = 0.0
    congestion_level: CongestionLevel = CongestionLevel.LOW
    trend: Trend = Trend.STABLE

    confidence: float = 1.0

    source_frame_id: int | None = None
    result_age_s: float = 0.0

    has_density: bool = True
    has_motion: bool = True
    motion_status: str = "OK"  # "OK", "STALE", "UNAVAILABLE"
    raft_pair_provenance: str | None = None
    raft_result_age_s: float = 0.0
    is_motion_fresh: bool = True

    def __post_init__(self) -> None:
        if not self.gate_id:
            raise ValueError("gate_id must not be empty")
        if not (0.0 <= self.density_score <= 1.0):
            raise ValueError(f"density_score must be in [0, 1], got {self.density_score}")
        if not (0.0 <= self.congestion_score <= 1.0):
            raise ValueError(f"congestion_score must be in [0, 1], got {self.congestion_score}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.velocity < 0:
            raise ValueError(f"velocity must be >= 0, got {self.velocity}")


# ===================================================================
# CandidateEvaluation — routing candidate assessment
# ===================================================================


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation of a candidate gate for diversion routing.

    Provides full transparency into why a gate was/wasn't selected.
    """

    gate_id: str
    distance_m: float
    congestion_level: CongestionLevel
    congestion_score: float
    capacity_utilization: float
    trend: Trend
    routing_score: float
    is_viable: bool
    rejection_reason: str | None = None


# ===================================================================
# DecisionResult — output of the Decision Engine
# ===================================================================


@dataclass(frozen=True)
class DecisionResult:
    """Immutable result of a decision evaluation for a single source gate.

    Contains the action, target, reasoning, and full candidate evaluations
    for audit and explainability.
    """

    decision_id: str
    timestamp: float
    source_gate_id: str
    action: ActionType
    target_gate_id: str | None = None
    severity: CongestionLevel = CongestionLevel.LOW
    confidence: float = 1.0
    reason_codes: tuple[str, ...] = ()
    human_readable_reason: str = ""
    candidate_evaluations: tuple[CandidateEvaluation, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_gate_id:
            raise ValueError("source_gate_id must not be empty")
        if self.action == ActionType.DIVERT and self.target_gate_id is None:
            raise ValueError("DIVERT action requires a target_gate_id")

    @staticmethod
    def generate_id() -> str:
        """Generate a unique decision identifier."""
        return f"DEC-{uuid.uuid4().hex[:8].upper()}"
