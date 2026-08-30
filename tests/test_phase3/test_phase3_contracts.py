"""Tests for Phase 3 model-agnostic contracts: CrowdState, DecisionResult, enums."""

from __future__ import annotations

import pytest

from crowd_management.decision.contracts import (
    ActionType,
    CandidateEvaluation,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    Trend,
)


class TestCrowdState:

    def test_valid_crowd_state(self):
        s = CrowdState(
            gate_id="G1", timestamp=1000.0,
            crowd_count=250.0, density_score=0.5,
            velocity=3.0, direction=90.0, flow_consistency=0.8,
            congestion_score=0.65, congestion_level=CongestionLevel.HIGH,
            trend=Trend.INCREASING, confidence=0.9,
        )
        assert s.gate_id == "G1"
        assert s.congestion_level == CongestionLevel.HIGH
        assert s.trend == Trend.INCREASING

    def test_invalid_density_score(self):
        with pytest.raises(ValueError, match="density_score"):
            CrowdState(gate_id="G1", timestamp=1000.0, density_score=1.5)

    def test_invalid_congestion_score(self):
        with pytest.raises(ValueError, match="congestion_score"):
            CrowdState(gate_id="G1", timestamp=1000.0, congestion_score=-0.1)

    def test_invalid_confidence(self):
        with pytest.raises(ValueError, match="confidence"):
            CrowdState(gate_id="G1", timestamp=1000.0, confidence=1.5)

    def test_invalid_velocity(self):
        with pytest.raises(ValueError, match="velocity"):
            CrowdState(gate_id="G1", timestamp=1000.0, velocity=-1.0)

    def test_empty_gate_id(self):
        with pytest.raises(ValueError, match="gate_id"):
            CrowdState(gate_id="", timestamp=1000.0)

    def test_missing_density(self):
        s = CrowdState(gate_id="G1", timestamp=1000.0, has_density=False)
        assert s.has_density is False

    def test_missing_motion(self):
        s = CrowdState(gate_id="G1", timestamp=1000.0, has_motion=False)
        assert s.has_motion is False


class TestDecisionResult:

    def test_valid_no_action(self):
        r = DecisionResult(
            decision_id="DEC-1", timestamp=1000.0,
            source_gate_id="G1", action=ActionType.NO_ACTION,
        )
        assert r.action == ActionType.NO_ACTION
        assert r.target_gate_id is None

    def test_valid_divert(self):
        r = DecisionResult(
            decision_id="DEC-2", timestamp=1000.0,
            source_gate_id="G1", action=ActionType.DIVERT,
            target_gate_id="G2",
        )
        assert r.action == ActionType.DIVERT
        assert r.target_gate_id == "G2"

    def test_divert_requires_target(self):
        with pytest.raises(ValueError, match="target_gate_id"):
            DecisionResult(
                decision_id="DEC-3", timestamp=1000.0,
                source_gate_id="G1", action=ActionType.DIVERT,
            )

    def test_generate_id(self):
        id1 = DecisionResult.generate_id()
        id2 = DecisionResult.generate_id()
        assert id1.startswith("DEC-")
        assert id1 != id2


class TestEnums:

    def test_congestion_levels(self):
        assert CongestionLevel.LOW.value == "LOW"
        assert CongestionLevel.CRITICAL.value == "CRITICAL"

    def test_trend_values(self):
        assert Trend.INCREASING.value == "INCREASING"
        assert Trend.STABLE.value == "STABLE"
        assert Trend.DECREASING.value == "DECREASING"

    def test_action_types(self):
        assert ActionType.DIVERT.value == "DIVERT"
        assert ActionType.NO_ACTION.value == "NO_ACTION"
        assert ActionType.MONITOR.value == "MONITOR"


class TestCandidateEvaluation:

    def test_viable_candidate(self):
        c = CandidateEvaluation(
            gate_id="G2", distance_m=25.0,
            congestion_level=CongestionLevel.LOW, congestion_score=0.2,
            capacity_utilization=0.3, trend=Trend.STABLE,
            routing_score=0.87, is_viable=True,
        )
        assert c.is_viable is True
        assert c.rejection_reason is None

    def test_rejected_candidate(self):
        c = CandidateEvaluation(
            gate_id="G3", distance_m=50.0,
            congestion_level=CongestionLevel.CRITICAL, congestion_score=0.95,
            capacity_utilization=0.9, trend=Trend.INCREASING,
            routing_score=0.1, is_viable=False,
            rejection_reason="Congestion too high",
        )
        assert c.is_viable is False
        assert c.rejection_reason == "Congestion too high"
