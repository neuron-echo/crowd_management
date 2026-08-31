"""Visualization utilities for Phase 2/3 dashboard rendering.

Panels:
  1. Real video frame + Decision badge
  2. CSRNet density heatmap overlay
  3. ByteTrack motion summary (person tracks + velocity stats)
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from crowd_management.models.contracts import ByteTrackResult, CSRNetResult

DEFAULT_STALE_THRESHOLD_S = 1.0


# ─── Low-level rendering helpers ─────────────────────────────────────────────

def render_density_heatmap(density_map: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    d_min, d_max = float(np.min(density_map)), float(np.max(density_map))
    norm = ((density_map - d_min) / (d_max - d_min)) if d_max > d_min else np.zeros_like(density_map)
    heatmap = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)
    return heatmap


def render_density_overlay(bgr_frame: np.ndarray, density_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    h, w = bgr_frame.shape[:2]
    heatmap = render_density_heatmap(density_map, target_size=(w, h))
    return cv2.addWeighted(bgr_frame, 1.0 - alpha, heatmap, alpha, 0)


def render_flow_hsv(u_flow: np.ndarray, v_flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(u_flow, v_flow, angleInDegrees=True)
    hsv = np.zeros((*u_flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = (angle / 2.0).astype(np.uint8)
    hsv[..., 1] = 255
    mag_max = float(np.max(magnitude))
    hsv[..., 2] = (np.clip(magnitude / (mag_max * 0.8), 0, 1) * 255).astype(np.uint8) if mag_max > 1e-3 else 0
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render_quiver_arrows(
    bgr_frame: np.ndarray, u_flow: np.ndarray, v_flow: np.ndarray,
    step: int = 32, scale: float = 1.5, min_mag: float = 0.5,
) -> np.ndarray:
    canvas = bgr_frame.copy()
    fh, fw = canvas.shape[:2]
    if u_flow.shape[:2] != (fh, fw):
        u_flow = cv2.resize(u_flow, (fw, fh), interpolation=cv2.INTER_LINEAR)
        v_flow = cv2.resize(v_flow, (fw, fh), interpolation=cv2.INTER_LINEAR)
    for y in range(step // 2, fh, step):
        for x in range(step // 2, fw, step):
            dx, dy = u_flow[y, x] * scale, v_flow[y, x] * scale
            if np.sqrt(dx**2 + dy**2) >= min_mag:
                cv2.arrowedLine(canvas, (x, y), (int(x + dx), int(y + dy)), (0, 255, 255), 1, tipLength=0.35)
                cv2.circle(canvas, (x, y), 2, (0, 0, 255), -1)
    return canvas


def compute_result_age(result_video_timestamp: float, current_video_timestamp: float) -> float:
    return max(0.0, current_video_timestamp - result_video_timestamp)


def is_result_fresh(result_age_s: float, stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S) -> bool:
    return result_age_s <= stale_threshold_s


# ─── Phase 3 dashboard ───────────────────────────────────────────────────────

def create_phase3_dashboard_panel(
    original_frame: np.ndarray,
    cam_id: str,
    gate_id: str,
    csrnet_res: CSRNetResult | None,
    raft_res: ByteTrackResult | None,          # kept as raft_res for call-site compat
    crowd_state: Any | None = None,
    decision_res: Any | None = None,
    flow_uv: tuple[np.ndarray, np.ndarray] | None = None,
    telemetry: dict[str, Any] | None = None,
    target_width: int = 1440,
    stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S,
    is_eof: bool = False,
    is_new_csrnet: bool = True,
    is_new_raft: bool = True,
    sync_result: Any | None = None,
) -> np.ndarray:
    """3-panel Phase 3 dashboard:

    [Real Video + Decision Badge] | [CSRNet Density Overlay] | [ByteTrack Motion Panel]
    + Header bar + Bottom telemetry console.
    """
    telemetry = telemetry or {}
    bt_res: ByteTrackResult | None = raft_res   # alias for clarity

    frame_id   = csrnet_res.frame_id        if csrnet_res else (bt_res.frame_id if bt_res else 0)
    source_idx = csrnet_res.source_frame_index if csrnet_res else 0
    video_time = csrnet_res.video_timestamp if csrnet_res else (bt_res.video_timestamp if bt_res else 0.0)

    panel_w = target_width // 3
    panel_h = int(panel_w * 0.75)

    # Freshness
    csrnet_age   = compute_result_age(csrnet_res.video_timestamp, video_time) if csrnet_res else 0.0
    csrnet_fresh = is_result_fresh(csrnet_age, stale_threshold_s)

    # Sync status from sync_result or fallback
    if sync_result is not None:
        bt_res       = getattr(sync_result, "bytetrack_result", bt_res)
        bt_fresh     = getattr(sync_result, "is_valid_motion", False)
        sync_status  = getattr(sync_result.status, "value", str(sync_result.status)) if hasattr(sync_result, "status") else "NEW"
        bt_age_frames = getattr(sync_result, "age_frames", 0)
    else:
        bt_fresh     = True if is_new_raft else False
        sync_status  = "NEW" if is_new_raft else "LAST_KNOWN"
        bt_age_frames = 0

    # ── Panel 1: Real Video + Decision Badge ──────────────────────────────────
    panel_orig = cv2.resize(original_frame, (panel_w, panel_h))
    cv2.putText(panel_orig, f"1. REAL VIDEO ({gate_id})", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(panel_orig, f"Frame {frame_id:04d} | Src {source_idx} | t={video_time:.2f}s",
                (12, panel_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    if decision_res is not None:
        act = str(decision_res.action.value if hasattr(decision_res.action, "value") else decision_res.action)
        if act == "DIVERT" and decision_res.target_gate_id:
            badge_text  = f"DECISION: DIVERT -> {decision_res.target_gate_id}"
            badge_color = (0, 165, 255)
        elif act == "MONITOR":
            badge_text  = f"DECISION: MONITOR ({gate_id})"
            badge_color = (0, 255, 255)
        else:
            badge_text  = f"DECISION: {act}"
            badge_color = (0, 255, 0)
        cv2.rectangle(panel_orig, (8, panel_h - 32), (panel_w - 8, panel_h - 8), (0, 0, 0), -1)
        cv2.putText(panel_orig, badge_text, (14, panel_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 2, cv2.LINE_AA)
    elif is_eof:
        cv2.rectangle(panel_orig, (8, panel_h - 32), (panel_w - 8, panel_h - 8), (0, 0, 0), -1)
        cv2.putText(panel_orig, "STATUS: STREAM EOF", (14, panel_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 2, cv2.LINE_AA)

    # ── Panel 2: CSRNet Density Heatmap ───────────────────────────────────────
    if csrnet_res is not None:
        panel_density = render_density_overlay(original_frame, csrnet_res.density_map, alpha=0.45)
        panel_density = cv2.resize(panel_density, (panel_w, panel_h))
        status_tag = "[NEW]" if is_new_csrnet else "[LAST KNOWN]"
        cv2.putText(panel_density,
                    f"2. DENSITY (Count: {csrnet_res.crowd_count:.1f}) {status_tag}", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                    (0, 255, 0) if is_new_csrnet else (0, 200, 255), 2)
        age_color = (0, 255, 0) if csrnet_fresh else (0, 0, 255)
        cv2.putText(panel_density,
                    f"Age: {csrnet_age:.2f}s | Latency: {csrnet_res.inference_time_ms:.0f}ms",
                    (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, age_color, 1, cv2.LINE_AA)
    else:
        panel_density = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel_density, "2. DENSITY (WAITING)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 2)

    # ── Panel 3: ByteTrack Motion Panel ───────────────────────────────────────
    panel_motion = original_frame.copy() if original_frame is not None else np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    panel_motion = cv2.resize(panel_motion, (panel_w, panel_h))

    if bt_res is not None:
        if sync_status == "NEW":
            title_color = (0, 255, 0)
            title_tag   = "[NEW]"
        elif sync_status == "LAST_KNOWN":
            title_color = (0, 220, 255)
            title_tag   = "[LAST KNOWN]"
        elif sync_status == "STALE":
            title_color = (0, 0, 255)
            title_tag   = "[STALE - EXCLUDED]"
        else:
            title_color = (128, 128, 128)
            title_tag   = "[UNAVAILABLE]"

        cv2.putText(panel_motion,
                    f"3. BYTETRACK (Med: {bt_res.median_velocity:.1f} px/s) {title_tag}",
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, title_color, 2)
        cv2.putText(panel_motion,
                    f"Tracks: {bt_res.track_count} | Dir: {bt_res.dominant_direction_deg:.0f}deg | Cons: {bt_res.flow_consistency:.2f}",
                    (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        bt_color = (0, 255, 0) if bt_fresh else (0, 0, 255)
        cv2.putText(panel_motion,
                    f"Gap: {bt_age_frames}f | Conf: {bt_res.motion_confidence:.2f} | Latency: {bt_res.inference_time_ms:.0f}ms",
                    (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, bt_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel_motion, "3. BYTETRACK MOTION (PRIMING)", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2)

    # ── Combine panels ────────────────────────────────────────────────────────
    panel_row = np.hstack([panel_orig, panel_density, panel_motion])

    # ── Header bar ────────────────────────────────────────────────────────────
    header_h = 55
    header = np.full((header_h, panel_row.shape[1], 3), (20, 22, 28), dtype=np.uint8)
    cv2.putText(header,
                f"PHASE 3 LIVE | CAM: {cam_id} -> GATE: {gate_id} | Frame: {frame_id:04d} | Src: {source_idx} | t={video_time:.2f}s",
                (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)

    if is_eof:
        status_text, status_color = "STREAM: EOF", (0, 140, 255)
    elif bt_res is not None and csrnet_res is not None:
        if sync_status == "NEW":
            status_text, status_color = "MOTION: FRESH [NEW]", (0, 255, 0)
        elif sync_status == "LAST_KNOWN":
            status_text, status_color = f"MOTION: OK [LAST KNOWN gap={bt_age_frames}f]", (0, 220, 255)
        elif sync_status == "STALE":
            status_text, status_color = f"MOTION: STALE (gap={bt_age_frames}f) EXCLUDED", (0, 0, 255)
        else:
            status_text, status_color = "MOTION: UNAVAILABLE", (128, 128, 128)
    else:
        status_text, status_color = "INITIALIZING...", (0, 200, 255)

    cv2.putText(header, status_text, (panel_row.shape[1] - 470, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)
    cv2.putText(header,
                f"Device: {telemetry.get('device', 'cpu').upper()} | Target FPS: {telemetry.get('ingestion_fps', 5.0):.0f}",
                (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 128), 1, cv2.LINE_AA)
    cv2.putText(header,
                f"Queues: CSR={telemetry.get('csrnet_queue', 0)} BT={telemetry.get('raft_queue', 0)} | Stale Discarded: {telemetry.get('raft_stale_discarded', 0)}",
                (panel_row.shape[1] - 560, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # ── Bottom telemetry console ──────────────────────────────────────────────
    footer_h = 175
    footer = np.full((footer_h, panel_row.shape[1], 3), (12, 14, 18), dtype=np.uint8)

    # Col 1: Model observations
    if bt_res:
        bt_vel_str  = f"ByteTrack Med Vel  : {bt_res.median_velocity:.2f} px/s ({sync_status})"
        bt_dir_str  = f"ByteTrack Direction: {bt_res.dominant_direction_deg:.1f} deg (Cons: {bt_res.flow_consistency:.2f})"
        bt_trk_str  = f"ByteTrack Tracks   : {bt_res.track_count} | Active: {bt_res.active_motion_ratio:.2f}"
    else:
        bt_vel_str = bt_dir_str = bt_trk_str = "ByteTrack: (priming)"

    col1 = [
        "--- MODEL OBSERVATIONS ---",
        f"CSRNet Count   : {csrnet_res.crowd_count:.1f} people ({'NEW' if is_new_csrnet else 'LAST KNOWN'})" if csrnet_res else "CSRNet Count   : (waiting)",
        f"CSRNet Density : {csrnet_res.density_map_width}x{csrnet_res.density_map_height} ({csrnet_res.inference_time_ms:.0f}ms)" if csrnet_res else "CSRNet Density : (waiting)",
        bt_vel_str,
        bt_dir_str,
        bt_trk_str,
    ]

    # Col 2: CrowdState
    col2 = ["--- CROWDSTATE (ADAPTER) ---"]
    if crowd_state is not None:
        motion_tag = f"{crowd_state.motion_status} (gap={bt_age_frames}f)" if crowd_state.is_motion_fresh else f"{crowd_state.motion_status} (EXCLUDED)"
        col2.extend([
            f"Gate ID        : {crowd_state.gate_id}",
            f"Congestion     : {crowd_state.congestion_level.value} (score: {crowd_state.congestion_score:.2f})",
            f"Density Score  : {crowd_state.density_score:.2f} | Vel: {crowd_state.velocity:.1f} px/s",
            f"Motion Status  : {motion_tag}",
            f"Trend / Conf   : {crowd_state.trend.value} | Conf: {crowd_state.confidence:.2f}",
        ])
    else:
        col2.extend(["CrowdState     : (fusing...)"] * 5)

    # Col 3: Decision Engine
    col3 = ["--- DECISION ENGINE ---"]
    if decision_res is not None:
        act_str = str(decision_res.action.value if hasattr(decision_res.action, "value") else decision_res.action)
        tgt_str = f" -> {decision_res.target_gate_id}" if decision_res.target_gate_id else ""
        col3.extend([
            f"Action         : {act_str}{tgt_str}",
            f"Severity       : {decision_res.severity.value if hasattr(decision_res.severity, 'value') else decision_res.severity}",
            f"Confidence     : {decision_res.confidence:.2f}",
            f"Reason Code    : {', '.join(decision_res.reason_codes[:2])}",
            f"Reason         : {decision_res.human_readable_reason[:50]}",
        ])
    else:
        col3.extend(["Decision       : (evaluating...)"] * 5)

    y_spacing = 25
    for i, line in enumerate(col1):
        cv2.putText(footer, line, (20, 22 + i * y_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if i == 0 else (220, 220, 220), 1, cv2.LINE_AA)
    for i, line in enumerate(col2):
        cv2.putText(footer, line, (panel_w + 20, 22 + i * y_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if i == 0 else (220, 220, 220), 1, cv2.LINE_AA)
    for i, line in enumerate(col3):
        cv2.putText(footer, line, (panel_w * 2 + 20, 22 + i * y_spacing),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255) if i == 0 else (220, 220, 220), 1, cv2.LINE_AA)

    return np.vstack([header, panel_row, footer])
