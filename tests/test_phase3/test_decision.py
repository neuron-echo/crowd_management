"""Comprehensive tests for the Decision Engine, fusion, routing, and trend.

Tests cover:
1. Feature fusion and congestion scoring
2. Missing/stale observation handling
3. Trend tracker
4. Multi-gate evaluation
5. Candidate routing and ranking
6. Capacity constraints
7. Distance weighting
8. Congestion weighting
9. No-action conditions
10. Decision stability (hysteresis)
11. Multi-camera/gate isolation
12. All 6 acceptance scenarios
"""

from __future__ import annotations

import time

import pytest

from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import (
    ActionType,
    CongestionLevel,
    CrowdState,
    Trend,
)
from crowd_management.decision.engine import DecisionEngine
from crowd_management.decision.fusion import (
    build_crowd_state,
    classify_congestion,
    compute_congestion_score,
)
from crowd_management.decision.routing import evaluate_candidate, rank_candidates
from crowd_management.decision.trend import TrendTracker
from crowd_management.venue.models import GateConfig, GeometryConfig, VenueConfig
from crowd_management.venue.topology import VenueTopology


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def config() -> DecisionConfig:
    return DecisionConfig()


@pytest.fixture
def venue_config() -> VenueConfig:
    return VenueConfig(
        id="test", name="Test Venue",
        geometry=GeometryConfig(radius_m=100.0),
        gates=[
            GateConfig(gate_id="G1", capacity=500, position_index=0, camera_id="cam_01"),
            GateConfig(gate_id="G2", capacity=600, position_index=1, camera_id="cam_02"),
            GateConfig(gate_id="G3", capacity=450, position_index=2, camera_id="cam_03"),
            GateConfig(gate_id="G4", capacity=700, position_index=3, camera_id="cam_04"),
        ],
    )


@pytest.fixture
def venue(venue_config: VenueConfig) -> VenueTopology:
    return VenueTopology(venue_config)


@pytest.fixture
def engine(venue: VenueTopology, config: DecisionConfig) -> DecisionEngine:
    return DecisionEngine(venue, config)


def _state(gate_id: str, cap: int, count: float, vel: float = 3.0,
            trend: Trend = Trend.STABLE, config: DecisionConfig | None = None,
            confidence: float = 1.0, result_age_s: float = 0.0) -> CrowdState:
    return build_crowd_state(
        gate_id=gate_id, gate_capacity=cap, crowd_count=count,
        velocity=vel, direction=90.0, flow_consistency=0.8,
        confidence=confidence, trend=trend,
        config=config or DecisionConfig(),
        result_age_s=result_age_s, timestamp=time.time(),
    )


# ===================================================================
# Test: Feature Fusion & Congestion Scoring
# ===================================================================

class TestFusion:

    def test_empty_gate_low_congestion(self, config: DecisionConfig):
        state = build_crowd_state("G1", 500, crowd_count=0, velocity=10.0, config=config)
        assert state.congestion_score < 0.3
        assert state.congestion_level == CongestionLevel.LOW

    def test_full_gate_high_congestion(self, config: DecisionConfig):
        state = build_crowd_state("G1", 500, crowd_count=480, velocity=0.5, config=config)
        assert state.congestion_score > 0.7
        assert state.congestion_level in (CongestionLevel.HIGH, CongestionLevel.CRITICAL)

    def test_density_score_normalized(self, config: DecisionConfig):
        state = build_crowd_state("G1", 500, crowd_count=250, config=config)
        assert abs(state.density_score - 0.5) < 0.01

    def test_density_score_capped_at_1(self, config: DecisionConfig):
        state = build_crowd_state("G1", 500, crowd_count=600, config=config)
        assert state.density_score == 1.0

    def test_velocity_reduces_congestion(self, config: DecisionConfig):
        slow = compute_congestion_score(0.5, velocity=0.5, capacity_utilization=0.5, config=config)
        fast = compute_congestion_score(0.5, velocity=10.0, capacity_utilization=0.5, config=config)
        assert slow > fast

    def test_classify_congestion_levels(self, config: DecisionConfig):
        assert classify_congestion(0.1, config) == CongestionLevel.LOW
        assert classify_congestion(0.45, config) == CongestionLevel.MODERATE
        assert classify_congestion(0.7, config) == CongestionLevel.HIGH
        assert classify_congestion(0.9, config) == CongestionLevel.CRITICAL

    def test_missing_density_degrades_confidence(self, config: DecisionConfig):
        state = build_crowd_state("G1", 500, has_density=False, confidence=1.0, config=config)
        assert state.confidence < 1.0

    def test_missing_motion_degrades_confidence(self, config: DecisionConfig):
        state = build_crowd_state("G1", 500, has_motion=False, confidence=1.0, config=config)
        assert state.confidence < 1.0


# ===================================================================
# Test: Trend Tracker
# ===================================================================

class TestTrendTracker:

    def test_single_observation_stable(self, config: DecisionConfig):
        tracker = TrendTracker(config)
        trend = tracker.update("G1", 0.5)
        assert trend == Trend.STABLE

    def test_increasing_trend(self, config: DecisionConfig):
        tracker = TrendTracker(config)
        for val in [0.3, 0.35, 0.4, 0.5, 0.6]:
            trend = tracker.update("G1", val)
        assert trend == Trend.INCREASING

    def test_decreasing_trend(self, config: DecisionConfig):
        tracker = TrendTracker(config)
        for val in [0.7, 0.65, 0.6, 0.5, 0.4]:
            trend = tracker.update("G1", val)
        assert trend == Trend.DECREASING

    def test_stable_trend(self, config: DecisionConfig):
        tracker = TrendTracker(config)
        for val in [0.5, 0.5, 0.5, 0.5, 0.5]:
            trend = tracker.update("G1", val)
        assert trend == Trend.STABLE

    def test_multi_gate_isolation(self, config: DecisionConfig):
        tracker = TrendTracker(config)
        for val in [0.3, 0.35, 0.4, 0.5, 0.6]:
            tracker.update("G1", val)
        for val in [0.7, 0.65, 0.6, 0.5, 0.4]:
            tracker.update("G2", val)
        assert tracker.get_trend("G1") == Trend.INCREASING
        assert tracker.get_trend("G2") == Trend.DECREASING

    def test_reset(self, config: DecisionConfig):
        tracker = TrendTracker(config)
        for val in [0.3, 0.5, 0.7]:
            tracker.update("G1", val)
        tracker.reset("G1")
        assert tracker.get_trend("G1") == Trend.STABLE


# ===================================================================
# Test: Routing & Candidate Selection
# ===================================================================

class TestRouting:

    def test_low_congestion_candidate_scores_high(self, venue: VenueTopology, config: DecisionConfig):
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 100, vel=5.0, trend=Trend.STABLE, config=config),
        }
        candidates = rank_candidates("G1", states, venue, config)
        g2_eval = next(c for c in candidates if c.gate_id == "G2")
        assert g2_eval.is_viable
        assert g2_eval.routing_score > 0.5

    def test_congested_candidate_rejected(self, venue: VenueTopology, config: DecisionConfig):
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, config=config),
            "G2": _state("G2", 600, 580, vel=0.5, config=config),
        }
        candidates = rank_candidates("G1", states, venue, config)
        g2_eval = next(c for c in candidates if c.gate_id == "G2")
        assert not g2_eval.is_viable

    def test_no_observation_candidate_rejected(self, venue: VenueTopology, config: DecisionConfig):
        states = {"G1": _state("G1", 500, 480, vel=0.5, config=config)}
        candidates = rank_candidates("G1", states, venue, config)
        for c in candidates:
            assert not c.is_viable

    def test_nearer_gate_preferred_over_farther(self, venue: VenueTopology, config: DecisionConfig):
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, config=config),
            "G2": _state("G2", 600, 100, vel=5.0, config=config),
            "G3": _state("G3", 450, 100, vel=5.0, config=config),
        }
        candidates = rank_candidates("G1", states, venue, config)
        viable = [c for c in candidates if c.is_viable]
        assert len(viable) >= 2
        # G2 is directly adjacent, G3 is 2 hops away
        # Distance should influence ranking
        g2 = next(c for c in viable if c.gate_id == "G2")
        g3 = next(c for c in viable if c.gate_id == "G3")
        assert g2.distance_m < g3.distance_m

    def test_capacity_constrained_candidate_rejected(self, venue: VenueTopology):
        config = DecisionConfig(max_candidate_congestion=0.99, min_capacity_remaining_ratio=0.20)
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, config=config),
            "G2": _state("G2", 600, 500, vel=10.0, config=config),  # 83.3% utilized -> remaining 16.7% < 20%
        }
        candidates = rank_candidates("G1", states, venue, config)
        g2_eval = next(c for c in candidates if c.gate_id == "G2")
        assert not g2_eval.is_viable
        assert "capacity" in g2_eval.rejection_reason.lower()

    def test_decreasing_trend_bonus(self, venue: VenueTopology, config: DecisionConfig):
        s_decreasing = _state("G2", 600, 200, vel=5.0, trend=Trend.DECREASING, config=config)
        s_increasing = _state("G3", 450, 200, vel=5.0, trend=Trend.INCREASING, config=config)
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, config=config),
            "G2": s_decreasing,
            "G3": s_increasing,
        }
        candidates = rank_candidates("G1", states, venue, config)
        g2 = next(c for c in candidates if c.gate_id == "G2")
        g3 = next(c for c in candidates if c.gate_id == "G3")
        # DECREASING trend should give a better score than INCREASING
        assert g2.routing_score > g3.routing_score


# ===================================================================
# Test: Decision Engine — Core Scenarios
# ===================================================================

class TestDecisionEngine:

    def test_scenario1_divert_g1_to_g2(self, engine: DecisionEngine, config: DecisionConfig):
        """G1 critical, G2 low → DIVERT G1 → nearby viable gate."""
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 100, vel=5.0, trend=Trend.STABLE, config=config),
            "G3": _state("G3", 450, 200, vel=4.0, config=config),
            "G4": _state("G4", 700, 100, vel=6.0, trend=Trend.DECREASING, config=config),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action == ActionType.DIVERT
        assert g1_result.target_gate_id is not None

    def test_scenario2_skip_congested_neighbor(self, engine: DecisionEngine, config: DecisionConfig):
        """G1 critical, G2 high, G3 low → skip G2, divert elsewhere."""
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 560, vel=0.5, trend=Trend.INCREASING, config=config),
            "G3": _state("G3", 450, 90, vel=7.0, trend=Trend.STABLE, config=config),
            "G4": _state("G4", 700, 100, vel=5.0, trend=Trend.DECREASING, config=config),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action == ActionType.DIVERT
        # Should NOT divert to G2 (congested)
        assert g1_result.target_gate_id != "G2"

    def test_scenario3_all_congested_no_viable(self, engine: DecisionEngine, config: DecisionConfig):
        """All gates critical → MONITOR, no diversion."""
        states = {
            "G1": _state("G1", 500, 490, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 580, vel=0.5, trend=Trend.INCREASING, config=config),
            "G3": _state("G3", 450, 440, vel=0.3, trend=Trend.INCREASING, config=config),
            "G4": _state("G4", 700, 690, vel=0.4, trend=Trend.INCREASING, config=config),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action in (ActionType.MONITOR, ActionType.NO_ACTION)
        assert g1_result.target_gate_id is None

    def test_scenario4_moderate_no_action(self, engine: DecisionEngine, config: DecisionConfig):
        """G1 moderate → NO_ACTION or MONITOR."""
        states = {
            "G1": _state("G1", 500, 220, vel=4.0, config=config),
            "G2": _state("G2", 600, 180, vel=5.0, config=config),
            "G3": _state("G3", 450, 100, vel=6.0, config=config),
            "G4": _state("G4", 700, 150, vel=5.5, config=config),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action in (ActionType.NO_ACTION, ActionType.MONITOR)

    def test_scenario5_skip_full_nearest(self, engine: DecisionEngine, config: DecisionConfig):
        """G1 critical, nearest gate full, second-nearest viable."""
        states = {
            "G1": _state("G1", 500, 490, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 590, vel=0.8, trend=Trend.INCREASING, config=config),
            "G3": _state("G3", 450, 100, vel=6.0, trend=Trend.DECREASING, config=config),
            "G4": _state("G4", 700, 680, vel=1.0, config=config),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action == ActionType.DIVERT
        assert g1_result.target_gate_id == "G3"

    def test_scenario6_stale_observation_no_divert(self, engine: DecisionEngine, config: DecisionConfig):
        """Stale observation → MONITOR, no unsafe diversion."""
        states = {
            "G1": _state("G1", 500, 490, vel=0.5, trend=Trend.INCREASING,
                         config=config, result_age_s=10.0),
            "G2": _state("G2", 600, 100, vel=5.0, config=config),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action == ActionType.MONITOR
        assert "STALE" in " ".join(g1_result.reason_codes)


# ===================================================================
# Test: Decision Stability (Hysteresis)
# ===================================================================

class TestHysteresis:

    def test_no_rapid_switching(self, venue: VenueTopology):
        """Same source gate should not oscillate between targets within cooldown."""
        config = DecisionConfig()
        config.hysteresis.cooldown_seconds = 60.0  # Long cooldown
        engine = DecisionEngine(venue, config)

        # First evaluation: G1 critical, should divert
        states1 = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 100, vel=5.0, config=config),
            "G3": _state("G3", 450, 120, vel=5.0, config=config),
            "G4": _state("G4", 700, 100, vel=5.0, config=config),
        }
        results1 = engine.evaluate(states1)
        g1_r1 = next(r for r in results1 if r.source_gate_id == "G1")
        target1 = g1_r1.target_gate_id

        # Second evaluation: slightly different scores, within margin
        states2 = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 110, vel=5.0, config=config),
            "G3": _state("G3", 450, 95, vel=5.2, config=config),
            "G4": _state("G4", 700, 105, vel=4.8, config=config),
        }
        results2 = engine.evaluate(states2)
        g1_r2 = next(r for r in results2 if r.source_gate_id == "G1")
        target2 = g1_r2.target_gate_id

        # Should keep same target during cooldown (if previous is still viable)
        assert target1 == target2

    def test_reset_hysteresis(self, engine: DecisionEngine):
        engine.reset_hysteresis()
        # Should not raise


# ===================================================================
# Test: Low Confidence Handling
# ===================================================================

class TestLowConfidence:

    def test_low_confidence_monitor(self, engine: DecisionEngine, config: DecisionConfig):
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, config=config, confidence=0.1),
        }
        results = engine.evaluate(states)
        g1_result = next(r for r in results if r.source_gate_id == "G1")
        assert g1_result.action == ActionType.MONITOR
        assert "LOW_CONFIDENCE" in g1_result.reason_codes


# ===================================================================
# Test: Multi-Gate Simultaneous Evaluation
# ===================================================================

class TestMultiGate:

    def test_evaluates_all_gates(self, engine: DecisionEngine, config: DecisionConfig):
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 100, vel=5.0, config=config),
            "G3": _state("G3", 450, 430, vel=0.5, trend=Trend.INCREASING, config=config),
            "G4": _state("G4", 700, 100, vel=5.0, config=config),
        }
        results = engine.evaluate(states)
        assert len(results) == 4
        gate_ids_evaluated = {r.source_gate_id for r in results}
        assert gate_ids_evaluated == {"G1", "G2", "G3", "G4"}

    def test_multiple_diversions_possible(self, engine: DecisionEngine, config: DecisionConfig):
        """Two congested gates can independently divert."""
        states = {
            "G1": _state("G1", 500, 480, vel=0.5, trend=Trend.INCREASING, config=config),
            "G2": _state("G2", 600, 100, vel=5.0, config=config),
            "G3": _state("G3", 450, 430, vel=0.5, trend=Trend.INCREASING, config=config),
            "G4": _state("G4", 700, 100, vel=5.0, config=config),
        }
        results = engine.evaluate(states)
        divert_results = [r for r in results if r.action == ActionType.DIVERT]
        assert len(divert_results) >= 2
