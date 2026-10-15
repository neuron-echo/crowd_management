"""Rule-Based Advisory Engine for Event Organisers.

Consumes the latest CameraState snapshots and produces structured,
actionable advisories — richer than the basic DIVERT/MONITOR decision.

Rules implemented (all purely rule-based, no ML):
    1. CRUSH_RISK          — High count + near-zero velocity
    2. BOTTLENECK_FORMING  — Rapid count surge (trend INCREASING + high score)
    3. GATE_OVERLOAD       — Crowd count exceeds gate capacity
    4. IMBALANCED_LOAD     — One gate much busier than others
    5. SLOW_FLOW_WARNING   — Very low velocity at moderate+ congestion
    6. DIVERT_IN_PROGRESS  — Active DIVERT decision confirmation
    7. ALL_GATES_FULL      — No viable diversion target exists
    8. SYSTEM_HEALTHY      — All gates within normal parameters
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

# ── Advisory severity levels ──────────────────────────────────────────────────
SEVERITY_EMERGENCY = "EMERGENCY"   # 🔴 Immediate action required
SEVERITY_ALERT     = "ALERT"       # 🟠 Action needed within minutes
SEVERITY_WARNING   = "WARNING"     # 🟡 Monitor closely
SEVERITY_INFO      = "INFO"        # 🟢 Informational


@dataclass
class Advisory:
    rule_id: str
    severity: str
    gate_id: str
    title: str
    description: str
    action: str           # What the organiser should physically do
    target_gate: str | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "rule_id":    self.rule_id,
            "severity":   self.severity,
            "gate_id":    self.gate_id,
            "title":      self.title,
            "description": self.description,
            "action":     self.action,
            "target_gate": self.target_gate,
            "timestamp":  self.timestamp,
        }


def generate_advisories(cameras: dict[str, Any]) -> list[Advisory]:
    """
    Generate rule-based organisational advisories from camera snapshot data.

    Args:
        cameras: dict of camera_id -> camera state dict (from DashboardState.snapshot())

    Returns:
        list of Advisory objects, sorted by severity (EMERGENCY first).
    """
    advisories: list[Advisory] = []
    cam_list = list(cameras.values())
    if not cam_list:
        return []

    # Pre-compute totals for cross-gate rules
    total_crowd = sum(c.get("crowd_count", 0) for c in cam_list)
    active_gates = [c for c in cam_list if c.get("crowd_count", 0) > 0]
    avg_count = total_crowd / len(active_gates) if active_gates else 0

    seen_gates: set[str] = set()  # avoid duplicate advisories per gate

    for cam in cam_list:
        gate_id         = cam.get("gate_id", "?")
        count           = cam.get("crowd_count", 0.0)
        capacity        = cam.get("gate_capacity", 500)
        velocity        = cam.get("velocity", 0.0)
        congestion_score= cam.get("congestion_score", 0.0)
        congestion_level= cam.get("congestion_level", "LOW")
        trend           = cam.get("trend", "STABLE")
        action          = cam.get("action", "NO_ACTION")
        target_gate     = cam.get("target_gate")
        motion_status   = cam.get("motion_status", "OK")
        flow_consistency= cam.get("flow_consistency", 1.0)

        util = count / capacity if capacity > 0 else 0

        # ── RULE 1: CRUSH RISK ─────────────────────────────────────────────
        # High count but crowd is barely moving = stampede/crush risk
        if (congestion_level in ("HIGH", "CRITICAL")
                and velocity < 5.0
                and motion_status != "UNAVAILABLE"
                and count > 20):
            advisories.append(Advisory(
                rule_id="CRUSH_RISK",
                severity=SEVERITY_EMERGENCY,
                gate_id=gate_id,
                title=f"[CRUSH RISK] High density and near-zero motion at {gate_id}",
                description=(
                    f"{gate_id} has {count:.0f} people but crowd velocity is critically low "
                    f"({velocity:.1f} px/s). People are packed and unable to move freely. "
                    f"Congestion score: {congestion_score:.0%}."
                ),
                action=(
                    f"IMMEDIATE: Deploy ground staff to {gate_id}. "
                    f"Stop all incoming entries. Create physical space by opening barriers. "
                    f"Redirect all incoming visitors to nearest alternative gate."
                ),
                target_gate=target_gate,
            ))

        # ── RULE 2: RAPID SURGE ────────────────────────────────────────────
        elif (trend == "INCREASING"
              and congestion_score > 0.65
              and congestion_level in ("HIGH", "CRITICAL")):
            advisories.append(Advisory(
                rule_id="BOTTLENECK_FORMING",
                severity=SEVERITY_ALERT,
                gate_id=gate_id,
                title=f"[SURGE WARNING] Rapid crowd buildup at {gate_id}",
                description=(
                    f"Crowd at {gate_id} is growing rapidly. Count: {count:.0f}/{capacity}, "
                    f"congestion score {congestion_score:.0%} and still INCREASING. "
                    f"Bottleneck will form within minutes if unchecked."
                ),
                action=(
                    f"Alert staff at {gate_id} to prepare for diversion. "
                    f"Pre-position barriers for flow redirection. "
                    f"Announce alternative entry points on PA system."
                ),
                target_gate=target_gate,
            ))

        # ── RULE 3: GATE CAPACITY EXCEEDED ────────────────────────────────
        elif util >= 1.0:
            advisories.append(Advisory(
                rule_id="GATE_OVERLOAD",
                severity=SEVERITY_EMERGENCY,
                gate_id=gate_id,
                title=f"[GATE OVERLOAD] Capacity exceeded at {gate_id}",
                description=(
                    f"{gate_id} has {count:.0f} people - capacity is {capacity}. "
                    f"Gate is at {util:.0%} utilization. Immediate action required."
                ),
                action=(
                    f"CLOSE {gate_id} to new entries immediately. "
                    f"Station staff to physically block incoming queue. "
                    f"Redirect all visitors to alternative gates."
                ),
                target_gate=target_gate,
            ))

        # ── RULE 4: DIVERT IN PROGRESS (confirmation + instructions) ──────
        elif action == "DIVERT" and target_gate:
            advisories.append(Advisory(
                rule_id="DIVERT_IN_PROGRESS",
                severity=SEVERITY_ALERT,
                gate_id=gate_id,
                title=f"[ACTIVE DIVERSION] Redirecting {gate_id} -> {target_gate}",
                description=(
                    f"AI system is diverting crowd from {gate_id} (score: {congestion_score:.0%}) "
                    f"to {target_gate}. Count at source: {count:.0f}/{capacity}."
                ),
                action=(
                    f"1. Station staff at the {gate_id} -> {target_gate} pathway. "
                    f"2. Place directional signage pointing to {target_gate}. "
                    f"3. Make PA announcement: 'Please proceed to {target_gate} - {gate_id} is temporarily at capacity.'"
                ),
                target_gate=target_gate,
            ))

        # ── RULE 5: SLOW FLOW WARNING ─────────────────────────────────────
        elif (congestion_level in ("MODERATE", "HIGH")
              and velocity < 10.0
              and count > 15
              and motion_status != "UNAVAILABLE"):
            advisories.append(Advisory(
                rule_id="SLOW_FLOW_WARNING",
                severity=SEVERITY_WARNING,
                gate_id=gate_id,
                title=f"[SLOW FLOW] Reduced pedestrian speed at {gate_id}",
                description=(
                    f"Crowd at {gate_id} is moving slowly (velocity: {velocity:.1f} px/s) "
                    f"with {count:.0f} people. Flow consistency: {flow_consistency:.0%}. "
                    f"If velocity drops further, crush risk will increase."
                ),
                action=(
                    f"Monitor {gate_id} closely. "
                    f"If velocity drops below 5 px/s, initiate diversion. "
                    f"Consider opening an additional lane at {gate_id} to improve throughput."
                ),
            ))

        # ── RULE 6: MONITOR with INCREASING trend ─────────────────────────
        elif action == "MONITOR" and trend == "INCREASING":
            advisories.append(Advisory(
                rule_id="MONITOR_SURGE",
                severity=SEVERITY_WARNING,
                gate_id=gate_id,
                title=f"[WATCHLIST] Inflow growing at {gate_id}",
                description=(
                    f"{gate_id} is under monitoring. Count: {count:.0f}/{capacity} "
                    f"({util:.0%} capacity) and still increasing. "
                    f"Congestion score: {congestion_score:.0%}."
                ),
                action=(
                    f"Alert staff at {gate_id} to be ready for action. "
                    f"Check if nearby gates have spare capacity for overflow. "
                    f"No immediate action required but prepare for escalation."
                ),
            ))

    # ── RULE 7: IMBALANCED LOAD (cross-gate) ──────────────────────────────
    for cam in cam_list:
        count = cam.get("crowd_count", 0.0)
        gate_id = cam.get("gate_id", "?")
        if avg_count > 0 and count > avg_count * 2.5 and count > 30:
            # This gate has more than 2.5x the average crowd
            lightest = min(cam_list, key=lambda c: c.get("crowd_count", 9999))
            light_gate = lightest.get("gate_id", "?")
            light_count = lightest.get("crowd_count", 0)
            # Only add if not already covered by a higher severity rule
            existing_rules = {a.rule_id for a in advisories if a.gate_id == gate_id}
            if "CRUSH_RISK" not in existing_rules and "GATE_OVERLOAD" not in existing_rules:
                advisories.append(Advisory(
                    rule_id="IMBALANCED_LOAD",
                    severity=SEVERITY_WARNING,
                    gate_id=gate_id,
                    title=f"[LOAD IMBALANCE] Unequal distribution: {gate_id} vs {light_gate}",
                    description=(
                        f"{gate_id} has {count:.0f} people while {light_gate} has only "
                        f"{light_count:.0f}. The venue load is unevenly distributed."
                    ),
                    action=(
                        f"Proactively direct a portion of visitors arriving at {gate_id} "
                        f"to {light_gate} before congestion worsens. "
                        f"Place directional staff and signs near the {gate_id} approach path."
                    ),
                    target_gate=light_gate,
                ))

    # ── RULE 8: ALL CLEAR ─────────────────────────────────────────────────
    if not advisories:
        advisories.append(Advisory(
            rule_id="ALL_CLEAR",
            severity=SEVERITY_INFO,
            gate_id="ALL",
            title="[ALL CLEAR] All gates operating within safe limits",
            description=(
                f"All {len(cam_list)} monitored gates are within safe operating parameters. "
                f"Total crowd across venue: {total_crowd:.0f} people."
            ),
            action="No action required. Continue routine monitoring.",
        ))

    # Sort: EMERGENCY → ALERT → WARNING → INFO
    ORDER = {SEVERITY_EMERGENCY: 0, SEVERITY_ALERT: 1, SEVERITY_WARNING: 2, SEVERITY_INFO: 3}
    advisories.sort(key=lambda a: ORDER.get(a.severity, 99))

    return advisories
