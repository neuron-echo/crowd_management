"""Standalone Phase 3 Decision Engine Demo.

This script demonstrates the Decision Engine WITHOUT requiring:
- torch / torchvision
- cv2 / OpenCV
- CSRNet / RAFT
- video ingestion
- GPU

It loads a venue configuration, creates synthetic CrowdState objects,
and runs the Decision Engine to produce explainable DecisionResults.

Usage:
    python examples/run_decision_demo.py --venue config/venue/circular_venue.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import (
    ActionType,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    Trend,
)
from crowd_management.decision.engine import DecisionEngine
from crowd_management.decision.fusion import build_crowd_state
from crowd_management.decision.trend import TrendTracker
from crowd_management.venue.loader import load_venue_topology
from crowd_management.venue.topology import VenueTopology


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 Decision Engine Standalone Demo")
    parser.add_argument(
        "--venue", type=str, default="config/venue/circular_venue.yaml",
        help="Path to venue YAML configuration",
    )
    return parser.parse_args()


def print_header(venue: VenueTopology) -> None:
    summary = venue.summary()
    print("=" * 70)
    print("PHASE 3 -- DECISION ENGINE STANDALONE DEMO")
    print("=" * 70)
    print(f"Venue         : {summary['venue_name']} ({summary['venue_id']})")
    print(f"Geometry      : {summary['geometry_type']}, radius={summary['radius_m']}m")
    print(f"Gates         : {summary['gate_count']} ({', '.join(summary['gate_ids'])})")
    print(f"Connections   : {summary['connection_count']}")
    print()
    print("NOTE: This demo uses SYNTHETIC CrowdState data.")
    print("      No torch, cv2, CSRNet, RAFT, or video files are used.")
    print("=" * 70)


def print_gate_states(states: dict[str, CrowdState], venue: VenueTopology) -> None:
    print()
    print("--- GATE STATUS ---")
    for gid in venue.gate_ids:
        s = states.get(gid)
        if s is None:
            print(f"  {gid}: (no observation)")
            continue
        cap = venue.get_gate_capacity(gid)
        print(
            f"  {gid}: "
            f"Congestion={s.congestion_level.value:8s} "
            f"Score={s.congestion_score:.2f}  "
            f"Count={s.crowd_count:.0f}/{cap}  "
            f"Density={s.density_score:.2f}  "
            f"Velocity={s.velocity:.1f}  "
            f"Trend={s.trend.value:10s}  "
            f"Confidence={s.confidence:.2f}"
        )


def print_decision(result: DecisionResult, venue: VenueTopology) -> None:
    print()
    print(f"  Source Gate    : {result.source_gate_id}")
    print(f"  Action        : {result.action.value}")
    if result.target_gate_id:
        print(f"  Target Gate   : {result.target_gate_id}")
    print(f"  Severity      : {result.severity.value}")
    print(f"  Confidence    : {result.confidence:.2f}")
    print(f"  Reason Codes  : {', '.join(result.reason_codes)}")
    print(f"  Reason        : {result.human_readable_reason}")

    if result.candidate_evaluations:
        print()
        print("  Candidate Evaluations:")
        for c in result.candidate_evaluations:
            viable_str = "VIABLE" if c.is_viable else f"REJECTED ({c.rejection_reason})"
            print(
                f"    {c.gate_id}: "
                f"Distance={c.distance_m:.0f}m  "
                f"Congestion={c.congestion_level.value}  "
                f"Score={c.routing_score:.3f}  "
                f"CapUtil={c.capacity_utilization:.0%}  "
                f"Trend={c.trend.value}  "
                f"-> {viable_str}"
            )


def make_state(
    gate_id: str, cap: int, count: float, velocity: float = 3.0,
    trend: Trend = Trend.STABLE, config: DecisionConfig | None = None,
    confidence: float = 1.0, result_age_s: float = 0.0,
    has_density: bool = True, has_motion: bool = True,
) -> CrowdState:
    """Create a synthetic CrowdState for demonstration."""
    return build_crowd_state(
        gate_id=gate_id,
        gate_capacity=cap,
        crowd_count=count,
        velocity=velocity,
        direction=90.0,
        flow_consistency=0.8,
        confidence=confidence,
        source_frame_id=1,
        result_age_s=result_age_s,
        has_density=has_density,
        has_motion=has_motion,
        trend=trend,
        config=config,
        timestamp=time.time(),
    )


def run_scenario(
    name: str,
    states: dict[str, CrowdState],
    engine: DecisionEngine,
    venue: VenueTopology,
) -> None:
    print()
    print("=" * 70)
    print(f"SCENARIO: {name}")
    print("=" * 70)

    print_gate_states(states, venue)

    results = engine.evaluate(states)

    print()
    print("--- DECISIONS ---")
    for r in results:
        if r.action != ActionType.NO_ACTION:
            print_decision(r, venue)

    # If all NO_ACTION, still print the summary
    if all(r.action == ActionType.NO_ACTION for r in results):
        print()
        print("  All gates are within acceptable congestion levels.")
        print("  Action: NO_ACTION for all gates.")


def main() -> None:
    args = parse_args()
    venue = load_venue_topology(args.venue)
    config = DecisionConfig()
    engine = DecisionEngine(venue, config)

    print_header(venue)

    gate_ids = venue.gate_ids
    capacities = {gid: venue.get_gate_capacity(gid) for gid in gate_ids}

    # Check venue size and run appropriate scenarios
    if len(gate_ids) == 2:
        g1, g2 = gate_ids[0], gate_ids[1]

        # Scenario 1: G1 critical, G2 low -> DIVERT G1 -> G2
        engine.reset_hysteresis()
        s1 = {
            g1: make_state(g1, capacities[g1], capacities[g1] * 0.95, velocity=1.0, trend=Trend.INCREASING, config=config),
            g2: make_state(g2, capacities[g2], capacities[g2] * 0.25, velocity=5.0, trend=Trend.STABLE, config=config),
        }
        run_scenario(f"1 -- {g1} critical, {g2} low -> expect DIVERT {g1} -> {g2}", s1, engine, venue)

        # Scenario 2: Both G1 and G2 critical -> NO_ACTION / MONITOR
        engine.reset_hysteresis()
        s2 = {
            g1: make_state(g1, capacities[g1], capacities[g1] * 0.95, velocity=0.5, trend=Trend.INCREASING, config=config),
            g2: make_state(g2, capacities[g2], capacities[g2] * 0.95, velocity=0.6, trend=Trend.INCREASING, config=config),
        }
        run_scenario(f"2 -- Both {g1} and {g2} critical -> expect NO viable diversion", s2, engine, venue)

        # Scenario 3: G1 moderate, G2 low -> NO_ACTION
        engine.reset_hysteresis()
        s3 = {
            g1: make_state(g1, capacities[g1], capacities[g1] * 0.40, velocity=4.0, trend=Trend.STABLE, config=config),
            g2: make_state(g2, capacities[g2], capacities[g2] * 0.20, velocity=5.0, trend=Trend.STABLE, config=config),
        }
        run_scenario(f"3 -- {g1} moderate -> expect NO_ACTION", s3, engine, venue)

        # Scenario 4: Stale G1 observation -> no unsafe diversion (MONITOR)
        engine.reset_hysteresis()
        s4 = {
            g1: make_state(g1, capacities[g1], capacities[g1] * 0.95, velocity=0.5,
                           trend=Trend.INCREASING, config=config, result_age_s=10.0),
            g2: make_state(g2, capacities[g2], capacities[g2] * 0.20, velocity=5.0,
                           trend=Trend.STABLE, config=config),
        }
        run_scenario(f"4 -- Stale {g1} observation -> expect MONITOR, no diversion", s4, engine, venue)

    else:
        # Multi-gate venue scenarios (N >= 3)
        # SCENARIO 1: G1 critical, G2 low -> DIVERT G1 -> G2
        engine.reset_hysteresis()
        s1 = {
            gate_ids[0]: make_state(gate_ids[0], capacities[gate_ids[0]], 450, velocity=1.0, trend=Trend.INCREASING, config=config),
            gate_ids[1]: make_state(gate_ids[1], capacities[gate_ids[1]], 150, velocity=5.0, trend=Trend.STABLE, config=config),
            gate_ids[2]: make_state(gate_ids[2], capacities[gate_ids[2]], 200, velocity=4.0, trend=Trend.STABLE, config=config),
            gate_ids[3]: make_state(gate_ids[3], capacities[gate_ids[3]], 100, velocity=6.0, trend=Trend.DECREASING, config=config),
        }
        run_scenario("1 -- G1 critical, G2 low -> expect DIVERT G1 -> best nearby", s1, engine, venue)

        # SCENARIO 2: G1 critical, G2 high, G3 low -> DIVERT G1 -> G3
        engine.reset_hysteresis()
        s2 = {
            gate_ids[0]: make_state(gate_ids[0], capacities[gate_ids[0]], 480, velocity=0.5, trend=Trend.INCREASING, config=config),
            gate_ids[1]: make_state(gate_ids[1], capacities[gate_ids[1]], 500, velocity=1.5, trend=Trend.INCREASING, config=config),
            gate_ids[2]: make_state(gate_ids[2], capacities[gate_ids[2]], 90, velocity=7.0, trend=Trend.STABLE, config=config),
            gate_ids[3]: make_state(gate_ids[3], capacities[gate_ids[3]], 120, velocity=5.0, trend=Trend.DECREASING, config=config),
        }
        run_scenario("2 -- G1 critical, G2 high, G3 low -> expect DIVERT G1 -> G3 or G4", s2, engine, venue)

        # SCENARIO 3: G1 critical, ALL nearby critical -> NO_ACTION / MONITOR
        engine.reset_hysteresis()
        s3 = {
            gate_ids[0]: make_state(gate_ids[0], capacities[gate_ids[0]], 475, velocity=0.5, trend=Trend.INCREASING, config=config),
            gate_ids[1]: make_state(gate_ids[1], capacities[gate_ids[1]], 550, velocity=0.8, trend=Trend.INCREASING, config=config),
            gate_ids[2]: make_state(gate_ids[2], capacities[gate_ids[2]], 430, velocity=0.3, trend=Trend.INCREASING, config=config),
            gate_ids[3]: make_state(gate_ids[3], capacities[gate_ids[3]], 670, velocity=0.4, trend=Trend.INCREASING, config=config),
        }
        run_scenario("3 -- All gates critical -> expect NO viable diversion", s3, engine, venue)

        # SCENARIO 4: G1 moderate -> NO_ACTION / MONITOR
        engine.reset_hysteresis()
        s4 = {
            gate_ids[0]: make_state(gate_ids[0], capacities[gate_ids[0]], 220, velocity=4.0, trend=Trend.STABLE, config=config),
            gate_ids[1]: make_state(gate_ids[1], capacities[gate_ids[1]], 180, velocity=5.0, trend=Trend.STABLE, config=config),
            gate_ids[2]: make_state(gate_ids[2], capacities[gate_ids[2]], 100, velocity=6.0, trend=Trend.STABLE, config=config),
            gate_ids[3]: make_state(gate_ids[3], capacities[gate_ids[3]], 150, velocity=5.5, trend=Trend.STABLE, config=config),
        }
        run_scenario("4 -- G1 moderate -> expect NO_ACTION or MONITOR", s4, engine, venue)

        # SCENARIO 5: G1 critical, nearest full, second-nearest has capacity
        engine.reset_hysteresis()
        s5 = {
            gate_ids[0]: make_state(gate_ids[0], capacities[gate_ids[0]], 490, velocity=0.5, trend=Trend.INCREASING, config=config),
            gate_ids[1]: make_state(gate_ids[1], capacities[gate_ids[1]], 590, velocity=0.8, trend=Trend.INCREASING, config=config),
            gate_ids[2]: make_state(gate_ids[2], capacities[gate_ids[2]], 100, velocity=6.0, trend=Trend.DECREASING, config=config),
            gate_ids[3]: make_state(gate_ids[3], capacities[gate_ids[3]], 650, velocity=1.0, trend=Trend.STABLE, config=config),
        }
        run_scenario("5 -- G1 critical, nearest gate full, second-nearest viable", s5, engine, venue)

        # SCENARIO 6: Stale observations -> no unsafe diversion
        engine.reset_hysteresis()
        s6 = {
            gate_ids[0]: make_state(gate_ids[0], capacities[gate_ids[0]], 490, velocity=0.5,
                                    trend=Trend.INCREASING, config=config, result_age_s=10.0),
            gate_ids[1]: make_state(gate_ids[1], capacities[gate_ids[1]], 100, velocity=5.0,
                                    trend=Trend.STABLE, config=config),
            gate_ids[2]: make_state(gate_ids[2], capacities[gate_ids[2]], 100, velocity=5.0,
                                    trend=Trend.STABLE, config=config),
            gate_ids[3]: make_state(gate_ids[3], capacities[gate_ids[3]], 100, velocity=5.0,
                                    trend=Trend.STABLE, config=config),
        }
        run_scenario("6 -- Stale G1 observation -> expect MONITOR, no diversion", s6, engine, venue)

    print()
    print("=" * 70)
    print("PHASE 3 DEMO COMPLETE")
    print()
    print("This demo ran without importing:")
    print("  - torch / torchvision")
    print("  - cv2 / OpenCV")
    print("  - CSRNet / RAFT")
    print("  - FramePacket / FrameBuffer / FrameProducer")
    print("  - Any video files or GPU inference")
    print()
    print("Simulation is NOT implemented in this phase.")
    print("=" * 70)


if __name__ == "__main__":
    main()
