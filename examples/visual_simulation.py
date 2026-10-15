"""
Crowd Safety Operations Center — Interactive 2D Venue Simulation
=================================================================
Demo Mode for Presentations & Academic Evaluation:
- No video dependency: controlled interactively via keyboard
- Keys [1], [2], [3] inject people directly into Gate 1, Gate 2, or Gate 3
- Real DecisionEngine evaluates gate capacities and triggers diversion orders
- Outside-the-Venue Perimeter Routing: Diverted pedestrians walk strictly along
  the outer ring concourse (OUTSIDE the circular venue), NEVER cutting through the event!
- Professional, standard C2 (Command & Control) aesthetic

Controls:
  [1] Inject crowd burst into Gate 1 (North)
  [2] Inject crowd burst into Gate 2 (East)
  [3] Inject crowd burst into Gate 3 (West)
  [F] Toggle Speed (1.0x Normal / 2.0x Fast / 3.5x Turbo)
  [C] Clear / Reset Venue crowd
  [SPACE] Pause / Resume simulation
  [Q] Exit
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

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
from crowd_management.venue.loader import load_venue_topology

# ─── Dimensions & Framerate ───────────────────────────────────────────────────
WIDTH = 1340
HEIGHT = 800
FPS = 60

# ─── Professional Theme Palette (Executive Dark Slate & ISO Standards) ────────
C_BG = (26, 22, 18)            # Deep charcoal background #12161a
C_PANEL_BG = (32, 27, 22)      # Sidebar background
C_PANEL_BORDER = (55, 48, 40)  # Subtle structural dividing lines
C_CARD_BG = (40, 34, 28)       # Metric card surface
C_GRID = (34, 29, 24)          # Floor coordinate grid
C_VENUE_FILL = (30, 26, 22)    # Restricted event interior
C_VENUE_BORDER = (75, 65, 55)  # Venue circular boundary
C_PATHWAY = (38, 33, 27)       # Outer paved concourse ring
C_PATH_LINE = (50, 44, 36)     # Concourse boundary guide lines

# ISO / Civil Safety Colors (BGR)
C_NORMAL = (80, 190, 80)       # Forest Green (Optimal / Normal)
C_ELEVATED = (20, 160, 240)    # Amber / Orange (Elevated)
C_CRITICAL = (50, 50, 230)     # Crimson Red (Capacity Limit / Divert)

# Pedestrian Dot Colors
C_DOT_NORMAL = (220, 215, 210) # Clean silver-white regular pedestrians
C_DOT_DIVERTED = (15, 140, 245)# Traffic Warning Amber for diverted pedestrians

# Typography
C_TEXT_TITLE = (245, 245, 245)
C_TEXT_MAIN = (210, 215, 220)
C_TEXT_MUTED = (130, 135, 145)
C_TEXT_DIM = (90, 95, 105)

# ─── Venue Coordinates & Geometry ─────────────────────────────────────────────
# Circular Venue Center & Radii
VENUE_CENTER = (840, 420)
VENUE_RADIUS = 165             # Event perimeter (RESTRICTED AREA)
CONCOURSE_RADIUS = 225         # Outer evacuation & diversion ring (OUTSIDE VENUE)

# Gates on the Event Perimeter Circle (120 degrees apart)
GATES = {
    "G1": {
        "name": "Gate North (Main)",
        "pos": (840, 255),               # Entrance portal on circle
        "spawn": (840, 60),              # Outside spawn point
        "cap": 40,                       # Capacity threshold for demo
        "angle_deg": -90,
    },
    "G2": {
        "name": "Gate East (Concourse)",
        "pos": (983, 503),               # 30 deg on circle
        "spawn": (1240, 503),            # Outside spawn point
        "cap": 60,
        "angle_deg": 30,
    },
    "G3": {
        "name": "Gate West (Concourse)",
        "pos": (697, 503),               # 150 deg on circle
        "spawn": (440, 503),             # Outside spawn point
        "cap": 55,
        "angle_deg": 150,
    },
}

# Outer Waypoints along the Ring Concourse (strictly R = 225 > 165 outside circle)
def _ring_point(angle_deg: float) -> tuple[int, int]:
    rad = math.radians(angle_deg)
    return (
        int(VENUE_CENTER[0] + CONCOURSE_RADIUS * math.cos(rad)),
        int(VENUE_CENTER[1] + CONCOURSE_RADIUS * math.sin(rad)),
    )

WAYPOINTS = {
    "W_G1": _ring_point(-90),          # (840, 195)
    "W_G1_G2": _ring_point(-30),       # (1035, 307)
    "W_G2": _ring_point(30),           # (1035, 532)
    "W_G2_G3": _ring_point(90),        # (840, 645)
    "W_G3": _ring_point(150),          # (645, 532)
    "W_G3_G1": _ring_point(210),       # (645, 307)
}


def get_outer_diversion_route(src: str, tgt: str) -> list[tuple[int, int]]:
    """Returns route strictly along the outer ring concourse outside the event circle."""
    if src == "G1" and tgt == "G2":
        return [WAYPOINTS["W_G1"], WAYPOINTS["W_G1_G2"], WAYPOINTS["W_G2"], GATES["G2"]["pos"]]
    elif src == "G1" and tgt == "G3":
        return [WAYPOINTS["W_G1"], WAYPOINTS["W_G3_G1"], WAYPOINTS["W_G3"], GATES["G3"]["pos"]]
    elif src == "G2" and tgt == "G1":
        return [WAYPOINTS["W_G2"], WAYPOINTS["W_G1_G2"], WAYPOINTS["W_G1"], GATES["G1"]["pos"]]
    elif src == "G2" and tgt == "G3":
        return [WAYPOINTS["W_G2"], WAYPOINTS["W_G2_G3"], WAYPOINTS["W_G3"], GATES["G3"]["pos"]]
    elif src == "G3" and tgt == "G1":
        return [WAYPOINTS["W_G3"], WAYPOINTS["W_G3_G1"], WAYPOINTS["W_G1"], GATES["G1"]["pos"]]
    elif src == "G3" and tgt == "G2":
        return [WAYPOINTS["W_G3"], WAYPOINTS["W_G2_G3"], WAYPOINTS["W_G2"], GATES["G2"]["pos"]]
    return [GATES[tgt]["pos"]]


# ─── Pedestrian Agent ─────────────────────────────────────────────────────────
class Pedestrian:
    __slots__ = ("x", "y", "target_id", "is_diverted", "state", "speed", "wait_ticks", "waypoints")

    def __init__(self, x: float, y: float, target_id: str):
        self.x = float(x)
        self.y = float(y)
        self.target_id = target_id
        self.is_diverted = False
        self.state = "walking"  # "walking" | "at_gate"
        self.speed = random.uniform(2.8, 4.4)
        self.wait_ticks = 0
        # Default route to assigned gate
        self.waypoints: list[tuple[int, int]] = [GATES[target_id]["pos"]]

    def divert_to(self, new_target: str, src_gate: str):
        self.target_id = new_target
        self.is_diverted = True
        self.waypoints = get_outer_diversion_route(src_gate, new_target)

    def update(self, sim_speed: float):
        effective_speed = self.speed * sim_speed
        if self.state == "walking":
            if not self.waypoints:
                self.state = "at_gate"
                return

            tx, ty = self.waypoints[0]
            # Add slight random jitter around waypoint for natural crowd spread
            dx = tx - self.x
            dy = ty - self.y
            dist = math.hypot(dx, dy)

            # Reached current waypoint
            if dist < 18:
                self.waypoints.pop(0)
                if not self.waypoints:
                    self.state = "at_gate"
            else:
                self.x += (dx / dist) * effective_speed
                self.y += (dy / dist) * effective_speed

        elif self.state == "at_gate":
            self.wait_ticks += 1


# ─── Drawing Utilities ────────────────────────────────────────────────────────
def draw_card(img: np.ndarray, x: int, y: int, w: int, h: int, bg: tuple, border: tuple | None = None):
    cv2.rectangle(img, (x, y), (x + w, y + h), bg, -1)
    if border:
        cv2.rectangle(img, (x, y), (x + w, y + h), border, 1)


def draw_progress_bar(img: np.ndarray, x: int, y: int, w: int, h: int, pct: float, fill_color: tuple):
    pct = max(0.0, min(1.0, pct))
    cv2.rectangle(img, (x, y), (x + w, y + h), (46, 40, 32), -1)
    fill_w = int(w * pct)
    if fill_w > 0:
        cv2.rectangle(img, (x, y), (x + fill_w, y + h), fill_color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (65, 58, 48), 1)


def draw_dashed_arc(img: np.ndarray, center: tuple[int, int], radius: int, start_deg: int, end_deg: int, color: tuple, thickness: int = 2):
    step = 5
    d = start_deg
    while d < end_deg:
        p1_rad = math.radians(d)
        p2_rad = math.radians(min(d + 3, end_deg))
        x1 = int(center[0] + radius * math.cos(p1_rad))
        y1 = int(center[1] + radius * math.sin(p1_rad))
        x2 = int(center[0] + radius * math.cos(p2_rad))
        y2 = int(center[1] + radius * math.sin(p2_rad))
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)
        d += step


# ─── Main Simulation Loop ─────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Professional Crowd Management 2D Simulation")
    parser.add_argument("--venue", default="config/venue/circular_venue_live_scaled.yaml")
    args = parser.parse_args()

    print("=" * 70)
    print("CROWD MANAGEMENT DEMO — TOPOLOGY & DECISION ENGINE")
    print("=" * 70)
    print("Controls:")
    print("  [1] Add crowd to Gate 1 (North)")
    print("  [2] Add crowd to Gate 2 (East)")
    print("  [3] Add crowd to Gate 3 (West)")
    print("  [F] Toggle Simulation Speed")
    print("  [C] Clear crowd")
    print("  [SPACE] Pause / Resume")
    print("  [Q] Exit")
    print("=" * 70)

    # Initialize Decision Engine with 3 Gates Topology
    venue_path = Path(args.venue)
    if not venue_path.exists():
        venue_path = Path("config/venue/circular_venue.yaml")
    venue = load_venue_topology(str(venue_path))
    decision_config = DecisionConfig()
    engine = DecisionEngine(venue, decision_config)

    # Simulation State
    agents: list[Pedestrian] = []
    active_diversions: dict[str, str] = {}
    last_decisions: dict[str, DecisionResult] = {}
    gate_counts: dict[str, int] = {gid: 0 for gid in GATES}

    sim_speed = 1.0
    speed_label = "1.0x (Normal)"
    is_paused = False
    tick = 0

    # Initial baseline crowd
    for gid, ginfo in GATES.items():
        initial = 12 if gid != "G1" else 25
        for _ in range(initial):
            rx = ginfo["pos"][0] + random.randint(-18, 18)
            ry = ginfo["pos"][1] + random.randint(-18, 18)
            p = Pedestrian(rx, ry, gid)
            p.state = "at_gate"
            p.waypoints.clear()
            agents.append(p)

    def spawn_batch(gid: str, count: int = 10):
        """Spawns an incoming batch of people approaching the designated gate."""
        ginfo = GATES[gid]
        sx, sy = ginfo["spawn"]
        for _ in range(count):
            if gid == "G1":
                px = sx + random.randint(-35, 35)
                py = sy + random.randint(-20, 20)
            else:
                px = sx + random.randint(-20, 20)
                py = sy + random.randint(-35, 35)

            p = Pedestrian(px, py, gid)
            # If the gate is already under active diversion, divert incoming person along outer ring!
            if gid in active_diversions:
                p.divert_to(active_diversions[gid], gid)
            agents.append(p)

    window_name = "Crowd Safety Operations Center — 2D Topology Dispatch"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WIDTH, HEIGHT)

    while True:
        loop_start = time.time()
        frame = np.full((HEIGHT, WIDTH, 3), C_BG, dtype=np.uint8)

        # -------------------------------------------------------------
        # 1. Update Pedestrians
        # -------------------------------------------------------------
        if not is_paused:
            gate_counts = {gid: 0 for gid in GATES}
            surviving: list[Pedestrian] = []

            for p in agents:
                # If approaching gate is newly diverted, route along outer perimeter!
                if p.state == "walking" and (not p.is_diverted) and (p.target_id in active_diversions):
                    # Ensure person is outside the gate before rerouting
                    dist_to_gate = math.hypot(p.x - GATES[p.target_id]["pos"][0], p.y - GATES[p.target_id]["pos"][1])
                    if dist_to_gate > 20:
                        p.divert_to(active_diversions[p.target_id], p.target_id)

                p.update(sim_speed)

                if p.state == "at_gate":
                    gate_counts[p.target_id] = gate_counts.get(p.target_id, 0) + 1
                    # People enter the stadium over time (turnstile throughput)
                    if p.wait_ticks > random.randint(100, 220):
                        continue  # Enters venue (leaves gate queue)

                surviving.append(p)

            agents = surviving

        # -------------------------------------------------------------
        # 2. Decision Engine Evaluation (Every 6 frames)
        # -------------------------------------------------------------
        if tick % 6 == 0 and not is_paused:
            crowd_states: dict[str, CrowdState] = {}
            for gid, ginfo in GATES.items():
                cap = ginfo["cap"]
                cnt = float(gate_counts.get(gid, 0))
                util = cnt / cap

                if util >= 0.85:
                    level = CongestionLevel.CRITICAL
                    score = min(1.0, 0.85 + (util - 0.85) * 0.8)
                    trend = Trend.INCREASING
                elif util >= 0.65:
                    level = CongestionLevel.HIGH
                    score = 0.70
                    trend = Trend.INCREASING
                elif util >= 0.35:
                    level = CongestionLevel.MODERATE
                    score = 0.45
                    trend = Trend.STABLE
                else:
                    level = CongestionLevel.LOW
                    score = max(0.1, util * 0.4)
                    trend = Trend.DECREASING if cnt < 8 else Trend.STABLE

                crowd_states[gid] = build_crowd_state(
                    gate_id=gid,
                    gate_capacity=cap,
                    crowd_count=cnt,
                    velocity=2.5 if level != CongestionLevel.CRITICAL else 0.8,
                    direction=90.0,
                    flow_consistency=0.85,
                    confidence=0.98,
                    source_frame_id=tick,
                    result_age_s=0.1,
                    has_density=True,
                    has_motion=True,
                    trend=trend,
                    config=decision_config,
                    timestamp=time.time(),
                )

            # Evaluate with DecisionEngine
            decisions = engine.evaluate(crowd_states)
            active_diversions.clear()
            for dec in decisions:
                last_decisions[dec.source_gate_id] = dec
                if dec.action == ActionType.DIVERT and dec.target_gate_id:
                    active_diversions[dec.source_gate_id] = dec.target_gate_id

        # -------------------------------------------------------------
        # 3. RENDER — Standard Command Center Visuals
        # -------------------------------------------------------------

        # Background Coordinates Grid
        for gx in range(350, WIDTH, 50):
            cv2.line(frame, (gx, 0), (gx, HEIGHT), C_GRID, 1)
        for gy in range(0, HEIGHT, 50):
            cv2.line(frame, (350, gy), (WIDTH, gy), C_GRID, 1)

        # ── OUTER RING CONCOURSE (Outside the Circle!) ────────────────
        # Paved ring corridor between radius 165 and 235
        cv2.circle(frame, VENUE_CENTER, CONCOURSE_RADIUS + 15, C_PATHWAY, -1)
        cv2.circle(frame, VENUE_CENTER, CONCOURSE_RADIUS + 15, C_PATH_LINE, 1)
        cv2.circle(frame, VENUE_CENTER, VENUE_RADIUS, C_BG, -1)  # Clear inner ring gap

        # Outer Approach Corridors (Straight paths leading to gates from outside)
        # North approach (G1)
        cv2.rectangle(frame, (805, 50), (875, 195), C_PATHWAY, -1)
        cv2.rectangle(frame, (805, 50), (875, 195), C_PATH_LINE, 1)
        # East approach (G2)
        cv2.rectangle(frame, (1035, 470), (1260, 540), C_PATHWAY, -1)
        cv2.rectangle(frame, (1035, 470), (1260, 540), C_PATH_LINE, 1)
        # West approach (G3)
        cv2.rectangle(frame, (420, 470), (645, 540), C_PATHWAY, -1)
        cv2.rectangle(frame, (420, 470), (645, 540), C_PATH_LINE, 1)

        # ── EVENT CIRCULAR VENUE (Restricted Interior) ────────────────
        cv2.circle(frame, VENUE_CENTER, VENUE_RADIUS, C_VENUE_FILL, -1)
        cv2.circle(frame, VENUE_CENTER, VENUE_RADIUS, C_VENUE_BORDER, 2)
        cv2.circle(frame, VENUE_CENTER, VENUE_RADIUS - 12, (45, 40, 34), 1)

        # Interior Technical Text
        cv2.putText(frame, "RESTRICTED EVENT VENUE", (VENUE_CENTER[0] - 110, VENUE_CENTER[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TEXT_TITLE, 1, cv2.LINE_AA)
        cv2.putText(frame, "INTERNAL ARENA (RADIUS 100m)", (VENUE_CENTER[0] - 105, VENUE_CENTER[1] + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_DIM, 1, cv2.LINE_AA)

        # Outer Concourse Technical Label
        cv2.putText(frame, "<-- OUTER PERIMETER BYPASS CONCOURSE (OUTSIDE EVENT) -->",
                    (VENUE_CENTER[0] - 220, VENUE_CENTER[1] + 195),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (85, 95, 110), 1, cv2.LINE_AA)

        # Active Diversion Outer Path Highlight
        for src, tgt in active_diversions.items():
            route = get_outer_diversion_route(src, tgt)
            for i in range(len(route) - 1):
                p1 = route[i]
                p2 = route[i + 1]
                cv2.line(frame, p1, p2, C_DOT_DIVERTED, 2)
                # Draw small directional indicator
                mx = (p1[0] + p2[0]) // 2
                my = (p1[1] + p2[1]) // 2
                cv2.circle(frame, (mx, my), 3, C_DOT_DIVERTED, -1)

        # Draw Pedestrian Dots
        for p in agents:
            col = C_DOT_DIVERTED if p.is_diverted else C_DOT_NORMAL
            cv2.circle(frame, (int(p.x), int(p.y)), 3, col, -1, cv2.LINE_AA)

        # Draw Gates (Portals on Venue Boundary)
        for gid, ginfo in GATES.items():
            gx, gy = ginfo["pos"]
            cap = ginfo["cap"]
            cnt = gate_counts.get(gid, 0)
            util = cnt / cap
            is_div = gid in active_diversions

            # Status color
            if is_div or util >= 0.85:
                stat_col = C_CRITICAL
            elif util >= 0.65:
                stat_col = C_ELEVATED
            else:
                stat_col = C_NORMAL

            # Physical Gate Barrier on boundary
            cv2.rectangle(frame, (gx - 28, gy - 12), (gx + 28, gy + 12), C_CARD_BG, -1)
            cv2.rectangle(frame, (gx - 28, gy - 12), (gx + 28, gy + 12), stat_col, 2)
            cv2.putText(frame, gid, (gx - 11, gy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_TEXT_TITLE, 2, cv2.LINE_AA)

            # Capacity Tag (Placed on outside side of the gate)
            tag_y = gy - 40 if gid == "G1" else gy + 22
            draw_card(frame, gx - 60, tag_y, 120, 34, C_CARD_BG, C_PANEL_BORDER)
            cv2.putText(frame, f"{cnt}/{cap} ({int(util * 100)}%)", (gx - 54, tag_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_TEXT_MAIN, 1, cv2.LINE_AA)
            draw_progress_bar(frame, gx - 54, tag_y + 19, 108, 6, util, stat_col)

        # -------------------------------------------------------------
        # 4. SIDEBAR — Incident Control & Telemetry Panel
        # -------------------------------------------------------------
        draw_card(frame, 0, 0, 350, HEIGHT, C_PANEL_BG, C_PANEL_BORDER)

        cv2.putText(frame, "CROWD CONTROL CONSOLE", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.62, C_TEXT_TITLE, 2, cv2.LINE_AA)
        cv2.putText(frame, "RULE-BASED INCIDENT DISPATCH", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.line(frame, (20, 64), (330, 64), C_PANEL_BORDER, 1)

        # Overview Metrics
        y_pos = 82
        draw_card(frame, 20, y_pos, 145, 54, C_CARD_BG, C_PANEL_BORDER)
        cv2.putText(frame, "TOTAL PEDESTRIANS", (28, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{len(agents)}", (28, y_pos + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.72, C_TEXT_TITLE, 2, cv2.LINE_AA)

        draw_card(frame, 175, y_pos, 155, 54, C_CARD_BG, C_PANEL_BORDER)
        cv2.putText(frame, "ACTIVE DIVERSIONS", (183, y_pos + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, C_TEXT_MUTED, 1, cv2.LINE_AA)
        div_col = C_CRITICAL if active_diversions else C_NORMAL
        cv2.putText(frame, f"{len(active_diversions)} ACTIVE", (183, y_pos + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, div_col, 2, cv2.LINE_AA)

        # Gate Cards
        y_pos += 74
        cv2.putText(frame, "OBSERVED GATE CAPACITIES", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TEXT_TITLE, 1, cv2.LINE_AA)
        y_pos += 14

        for gid in ["G1", "G2", "G3"]:
            ginfo = GATES[gid]
            cnt = gate_counts.get(gid, 0)
            cap = ginfo["cap"]
            util = cnt / cap
            is_div = gid in active_diversions

            card_h = 70
            draw_card(frame, 20, y_pos, 310, card_h, C_CARD_BG, C_PANEL_BORDER)
            col = C_CRITICAL if is_div or util >= 0.85 else (C_ELEVATED if util >= 0.65 else C_NORMAL)
            cv2.rectangle(frame, (20, y_pos), (24, y_pos + card_h), col, -1)

            cv2.putText(frame, f"{gid} - {ginfo['name']}", (32, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, C_TEXT_TITLE, 1, cv2.LINE_AA)
            status_lbl = "DIVERTING" if is_div else ("ELEVATED" if util >= 0.65 else "OPTIMAL")
            cv2.putText(frame, status_lbl, (240, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)

            cv2.putText(frame, f"Occupancy: {cnt} / {cap} pax", (32, y_pos + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{int(util * 100)}%", (280, y_pos + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MAIN, 1, cv2.LINE_AA)
            draw_progress_bar(frame, 32, y_pos + 48, 275, 7, util, col)
            y_pos += card_h + 10

        # Decision Engine Section
        y_pos += 8
        cv2.putText(frame, "INCIDENT DISPATCH DECISIONS", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_TEXT_TITLE, 1, cv2.LINE_AA)
        y_pos += 14

        if active_diversions:
            src_div = list(active_diversions.keys())[0]
            tgt_div = active_diversions[src_div]
            dec = last_decisions.get(src_div)

            draw_card(frame, 20, y_pos, 310, 115, (26, 26, 46), C_CRITICAL)
            cv2.putText(frame, f"ACTION: DIVERT {src_div} -> {tgt_div}", (32, y_pos + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_CRITICAL, 2, cv2.LINE_AA)
            cv2.putText(frame, "Routing: Outer Ring Bypass Concourse", (32, y_pos + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_TITLE, 1, cv2.LINE_AA)
            cv2.putText(frame, f"{src_div} capacity exceeded safety threshold.", (32, y_pos + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Incoming crowd redirected outside venue to {tgt_div}.", (32, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_TEXT_MUTED, 1, cv2.LINE_AA)
            conf = dec.confidence if dec else 1.0
            cv2.putText(frame, f"Engine Rule Confidence: {conf:.2f}", (32, y_pos + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.35, C_NORMAL, 1, cv2.LINE_AA)
            y_pos += 128
        else:
            draw_card(frame, 20, y_pos, 310, 60, C_CARD_BG, C_PANEL_BORDER)
            cv2.putText(frame, "ALL GATES NORMAL — MONITORING", (32, y_pos + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.44, C_NORMAL, 1, cv2.LINE_AA)
            cv2.putText(frame, "Crowd within safe threshold.", (32, y_pos + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MUTED, 1, cv2.LINE_AA)
            y_pos += 72

        # Interactive Controls Box
        draw_card(frame, 20, HEIGHT - 150, 310, 130, C_CARD_BG, C_PANEL_BORDER)
        cv2.putText(frame, "INTERACTIVE DEMO CONTROLS", (32, HEIGHT - 128), cv2.FONT_HERSHEY_SIMPLEX, 0.40, C_TEXT_TITLE, 1, cv2.LINE_AA)
        cv2.putText(frame, "[1] / [2] / [3] Add crowd to Gate 1, 2, or 3", (32, HEIGHT - 108), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (20, 180, 240), 1, cv2.LINE_AA)
        cv2.putText(frame, f"[F] Sim Speed: {speed_label}", (32, HEIGHT - 88), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(frame, "[C] Clear All Crowd    [SPACE] Pause", (32, HEIGHT - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(frame, "[Q] Quit", (32, HEIGHT - 48), cv2.FONT_HERSHEY_SIMPLEX, 0.36, C_TEXT_MUTED, 1, cv2.LINE_AA)

        # Top Header Status
        cv2.putText(frame, "TOPOLOGY: CIRCULAR VENUE (3 GATES)", (370, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_TEXT_TITLE, 1, cv2.LINE_AA)
        fps_act = int(1.0 / max(0.001, time.time() - loop_start))
        status_line = f"ENGINE: ACTIVE | SPEED: {speed_label} | FPS: {fps_act}"
        if is_paused:
            status_line += " [PAUSED]"
        cv2.putText(frame, status_line, (370, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, C_NORMAL if not is_paused else C_ELEVATED, 1, cv2.LINE_AA)

        cv2.imshow(window_name, frame)

        # Keyboard Input Handling
        key = cv2.waitKey(max(1, int(1000 / FPS))) & 0xFF
        if key in (ord('q'), ord('Q'), 27):
            print("\nExiting simulation...")
            break
        elif key in (ord('1'), ord('!')):
            print("[USER] Adding crowd burst to Gate 1 (North)")
            spawn_batch("G1", 8)
        elif key in (ord('2'), ord('@')):
            print("[USER] Adding crowd burst to Gate 2 (East)")
            spawn_batch("G2", 8)
        elif key in (ord('3'), ord('#')):
            print("[USER] Adding crowd burst to Gate 3 (West)")
            spawn_batch("G3", 8)
        elif key in (ord('f'), ord('F')):
            if sim_speed == 1.0:
                sim_speed = 2.0
                speed_label = "2.0x (Fast)"
            elif sim_speed == 2.0:
                sim_speed = 3.5
                speed_label = "3.5x (Turbo)"
            else:
                sim_speed = 1.0
                speed_label = "1.0x (Normal)"
        elif key in (ord('c'), ord('C'), ord('r'), ord('R')):
            agents.clear()
            active_diversions.clear()
            print("[USER] Cleared venue crowd.")
        elif key == ord(' '):
            is_paused = not is_paused

        tick += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
