"""Comprehensive visualization utilities for Phase 2 CSRNet density and RAFT optical flow outputs.

Includes result-age tracking, SYNC freshness indicators, pair provenance display,
and robust velocity statistics (median, P90, P95).
"""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np

from crowd_management.models.contracts import CSRNetResult, RAFTResult

# Default freshness threshold in seconds
DEFAULT_STALE_THRESHOLD_S = 1.0


def render_density_heatmap(density_map: np.ndarray, target_size: tuple[int, int] | None = None) -> np.ndarray:
    """Convert 2D density map into a colorized JET heatmap.

    Args:
        density_map: 2D numpy array of float values
        target_size: Optional (width, height) to resize heatmap

    Returns:
        BGR heatmap image (H, W, 3) uint8
    """
    d_min, d_max = float(np.min(density_map)), float(np.max(density_map))
    if d_max > d_min:
        norm = (density_map - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(density_map)

    norm_uint8 = (norm * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(norm_uint8, cv2.COLORMAP_JET)

    if target_size is not None:
        heatmap = cv2.resize(heatmap, target_size, interpolation=cv2.INTER_CUBIC)

    return heatmap


def render_density_overlay(
    bgr_frame: np.ndarray,
    density_map: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend density heatmap onto the original BGR frame."""
    h, w = bgr_frame.shape[:2]
    heatmap = render_density_heatmap(density_map, target_size=(w, h))
    return cv2.addWeighted(bgr_frame, 1.0 - alpha, heatmap, alpha, 0)


def render_flow_hsv(u_flow: np.ndarray, v_flow: np.ndarray) -> np.ndarray:
    """Render optical flow (u, v) as an HSV color-wheel BGR image."""
    magnitude, angle = cv2.cartToPolar(u_flow, v_flow, angleInDegrees=True)
    hsv = np.zeros((u_flow.shape[0], u_flow.shape[1], 3), dtype=np.uint8)

    # Hue corresponds to direction angle (0-179 in OpenCV HSV)
    hsv[..., 0] = (angle / 2.0).astype(np.uint8)

    # Saturation is fixed high
    hsv[..., 1] = 255

    # Value (brightness) corresponds to normalized velocity magnitude
    mag_max = float(np.max(magnitude))
    if mag_max > 1e-3:
        val_norm = np.clip(magnitude / (mag_max * 0.8), 0.0, 1.0) * 255.0
    else:
        val_norm = np.zeros_like(magnitude)
    hsv[..., 2] = val_norm.astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def render_quiver_arrows(
    bgr_frame: np.ndarray,
    u_flow: np.ndarray,
    v_flow: np.ndarray,
    step: int = 32,
    scale: float = 1.5,
    min_mag: float = 0.5,
) -> np.ndarray:
    """Overlay direction vector arrows on top of a frame."""
    canvas = bgr_frame.copy()
    fh, fw = canvas.shape[:2]

    # Resize flow if dimensions differ
    if u_flow.shape[:2] != (fh, fw):
        u_scaled = cv2.resize(u_flow, (fw, fh), interpolation=cv2.INTER_LINEAR)
        v_scaled = cv2.resize(v_flow, (fw, fh), interpolation=cv2.INTER_LINEAR)
    else:
        u_scaled, v_scaled = u_flow, v_flow

    for y in range(step // 2, fh, step):
        for x in range(step // 2, fw, step):
            dx = u_scaled[y, x] * scale
            dy = v_scaled[y, x] * scale
            mag = np.sqrt(dx**2 + dy**2)
            if mag >= min_mag:
                pt1 = (x, y)
                pt2 = (int(round(x + dx)), int(round(y + dy)))
                cv2.arrowedLine(canvas, pt1, pt2, (0, 255, 255), 1, tipLength=0.35)
                cv2.circle(canvas, pt1, 2, (0, 0, 255), -1)

    return canvas


def compute_result_age(result_video_timestamp: float, current_video_timestamp: float) -> float:
    """Compute result age as difference between current stream time and result time.

    Returns age in seconds. A positive value means the result is older than the current frame.
    """
    return max(0.0, current_video_timestamp - result_video_timestamp)


def is_result_fresh(result_age_s: float, stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S) -> bool:
    """Return True if result age is within acceptable freshness threshold."""
    return result_age_s <= stale_threshold_s


def create_live_dashboard_panel(
    original_frame: np.ndarray,
    csrnet_res: CSRNetResult | None,
    raft_res: RAFTResult | None,
    flow_uv: tuple[np.ndarray, np.ndarray] | None = None,
    telemetry: dict[str, Any] | None = None,
    target_width: int = 1440,
    stale_threshold_s: float = DEFAULT_STALE_THRESHOLD_S,
) -> np.ndarray:
    """Create a 3-panel live visualization dashboard:

    [Original Frame] | [CSRNet Density Overlay] | [RAFT Motion / Quiver]
    with top status bar showing sync status and bottom metrics console
    with result age, pair provenance, and robust velocity statistics.
    """
    telemetry = telemetry or {}
    cam_id = csrnet_res.camera_id if csrnet_res else (raft_res.camera_id if raft_res else "cam_??")
    frame_id = csrnet_res.frame_id if csrnet_res else (raft_res.current_frame_id if raft_res else 0)
    source_idx = csrnet_res.source_frame_index if csrnet_res else 0
    video_time = csrnet_res.video_timestamp if csrnet_res else (raft_res.current_timestamp if raft_res else 0.0)

    panel_w = target_width // 3
    panel_h = int(panel_w * 0.75)  # 4:3 aspect ratio per panel

    # --- Result Age Calculations ---
    csrnet_age = 0.0
    raft_age = 0.0
    csrnet_fresh = True
    raft_fresh = True

    if csrnet_res is not None:
        csrnet_age = compute_result_age(csrnet_res.video_timestamp, video_time)
        csrnet_fresh = is_result_fresh(csrnet_age, stale_threshold_s)

    if raft_res is not None:
        raft_age = compute_result_age(raft_res.current_timestamp, video_time)
        raft_fresh = is_result_fresh(raft_age, stale_threshold_s)

    overall_sync = csrnet_fresh and raft_fresh

    # --- 1. Left panel: Original Frame ---
    panel_orig = cv2.resize(original_frame, (panel_w, panel_h))
    cv2.putText(panel_orig, "1. ORIGINAL FRAME", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    # Show frame provenance
    cv2.putText(panel_orig, f"Frame {frame_id:04d} | Src {source_idx} | t={video_time:.2f}s",
                (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # --- 2. Center panel: CSRNet Density Heatmap Overlay ---
    if csrnet_res is not None:
        panel_density = render_density_overlay(original_frame, csrnet_res.density_map, alpha=0.45)
        panel_density = cv2.resize(panel_density, (panel_w, panel_h))
        cv2.putText(
            panel_density,
            f"2. DENSITY (Count: {csrnet_res.crowd_count:.1f})",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0), 2,
        )
        age_color = (0, 255, 0) if csrnet_fresh else (0, 0, 255)
        cv2.putText(panel_density, f"Age: {csrnet_age:.2f}s | {csrnet_res.inference_time_ms:.0f}ms",
                    (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, age_color, 1, cv2.LINE_AA)
    else:
        panel_density = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel_density, "2. DENSITY (WAITING)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 2)

    # --- 3. Right panel: RAFT Optical Flow Motion / Quiver ---
    if flow_uv is not None:
        u, v = flow_uv
        panel_motion = render_flow_hsv(u, v)
        panel_motion = render_quiver_arrows(panel_motion, u, v, step=28, scale=2.0)
        panel_motion = cv2.resize(panel_motion, (panel_w, panel_h))
    elif raft_res is not None and original_frame is not None:
        panel_motion = original_frame.copy()
        panel_motion = cv2.resize(panel_motion, (panel_w, panel_h))
    else:
        panel_motion = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    if raft_res is not None:
        title_text = f"3. RAFT (Med: {raft_res.median_velocity:.1f} P95: {raft_res.p95_velocity:.1f} px/f)"
        cv2.putText(panel_motion, title_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 2)
        # Pair provenance
        pair_text = f"Pair: {raft_res.previous_frame_id}->{raft_res.current_frame_id} dt={raft_res.delta_time:.3f}s"
        cv2.putText(panel_motion, pair_text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        age_color = (0, 255, 0) if raft_fresh else (0, 0, 255)
        age_text = f"Age: {raft_age:.2f}s | {raft_res.inference_time_ms:.0f}ms"
        cv2.putText(panel_motion, age_text, (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, age_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel_motion, "3. RAFT MOTION (PRIMING)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (128, 128, 128), 2)

    # --- Combine 3 panels horizontally ---
    panel_row = np.hstack([panel_orig, panel_density, panel_motion])

    # --- Top Header Bar (height=55) ---
    header_h = 55
    header = np.zeros((header_h, panel_row.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 22, 28)

    # Main title
    cv2.putText(
        header,
        f"PHASE 2 LIVE | CAM: {cam_id} | Frame: {frame_id:04d} | Src: {source_idx} | t={video_time:.2f}s",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # SYNC status indicator
    if raft_res is not None and csrnet_res is not None:
        if overall_sync:
            sync_text = "SYNC: OK"
            sync_color = (0, 255, 0)
        else:
            worst_age = max(csrnet_age, raft_age)
            sync_text = f"SYNC: STALE ({worst_age:.2f}s)"
            sync_color = (0, 0, 255)
    else:
        sync_text = "SYNC: PRIMING"
        sync_color = (0, 200, 255)

    cv2.putText(header, sync_text, (panel_row.shape[1] - 300, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, sync_color, 2, cv2.LINE_AA)

    # Device & FPS info
    device_str = f"Device: {telemetry.get('device', 'cpu').upper()} | Ingestion: {telemetry.get('ingestion_fps', 5.0):.0f} FPS"
    cv2.putText(header, device_str, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 128), 1, cv2.LINE_AA)

    queue_str = f"Queues: CSR={telemetry.get('csrnet_queue', 0)} RAFT={telemetry.get('raft_queue', 0)} | Stale Discarded: {telemetry.get('raft_stale_discarded', 0)}"
    cv2.putText(header, queue_str, (panel_row.shape[1] - 520, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    # --- Bottom Telemetry Console (height=160) ---
    footer_h = 160
    footer = np.zeros((footer_h, panel_row.shape[1], 3), dtype=np.uint8)
    footer[:] = (12, 14, 18)

    # Column 1: CSRNet details
    col1 = []
    if csrnet_res is not None:
        col1 = [
            f"CSRNet Crowd Count : {csrnet_res.crowd_count:.1f} people",
            f"Density Map Dim    : {csrnet_res.density_map_width}x{csrnet_res.density_map_height}",
            f"CSRNet Latency     : {csrnet_res.inference_time_ms:.0f} ms",
            f"CSRNet Result Age  : {csrnet_age:.3f} s",
            f"Checkpoint         : {csrnet_res.checkpoint_identifier}",
        ]
    else:
        col1 = ["CSRNet: (waiting for first result)"]

    # Column 2: RAFT robust velocity + provenance
    col2 = []
    if raft_res is not None:
        col2 = [
            f"RAFT Pair          : {raft_res.previous_frame_id} -> {raft_res.current_frame_id}",
            f"Src Indices        : {raft_res.previous_source_frame_index} -> {raft_res.current_source_frame_index}",
            f"Video Time         : {raft_res.previous_timestamp:.2f}s -> {raft_res.current_timestamp:.2f}s (dt={raft_res.delta_time:.3f}s)",
            f"Median Velocity    : {raft_res.median_velocity:.2f} px/f (inference res)",
            f"P95 Velocity       : {raft_res.p95_velocity:.2f} px/f | Max: {raft_res.max_velocity:.1f} px/f",
        ]
    else:
        col2 = ["RAFT: (priming — waiting for frame pair)"]

    # Column 3: RAFT direction + age + system
    col3 = []
    if raft_res is not None:
        col3 = [
            f"Direction          : {raft_res.dominant_direction_deg:.1f} deg",
            f"Flow Consistency   : {raft_res.flow_consistency:.3f}",
            f"Active Motion      : {raft_res.active_motion_ratio:.2f} | Conf: {raft_res.motion_confidence:.2f}",
            f"RAFT Latency       : {raft_res.inference_time_ms:.0f} ms",
            f"RAFT Result Age    : {raft_age:.3f} s",
        ]
    else:
        col3 = [
            f"Queue CSRNet       : {telemetry.get('csrnet_queue', 0)}",
            f"Queue RAFT         : {telemetry.get('raft_queue', 0)}",
        ]

    y_spacing = 24
    for i, line in enumerate(col1):
        cv2.putText(footer, line, (20, 24 + i * y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    for i, line in enumerate(col2):
        cv2.putText(footer, line, (panel_w + 20, 24 + i * y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)
    for i, line in enumerate(col3):
        cv2.putText(footer, line, (panel_w * 2 + 20, 24 + i * y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

    # Stack full dashboard
    dashboard = np.vstack([header, panel_row, footer])
    return dashboard


def create_phase3_dashboard_panel(
    original_frame: np.ndarray,
    cam_id: str,
    gate_id: str,
    csrnet_res: CSRNetResult | None,
    raft_res: RAFTResult | None,
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
    """Create a 3-panel live visualization dashboard for Phase 3:

    [Real Video + Decision Badge] | [CSRNet Density Overlay] | [RAFT Motion / Quiver]
    with top status bar (temporal freshness + alignment + new/last-known status) and 3-column bottom console:
    [Model Observations] | [CrowdState Fusion] | [Decision Engine Output].
    """
    telemetry = telemetry or {}
    frame_id = csrnet_res.frame_id if csrnet_res else (raft_res.current_frame_id if raft_res else 0)
    source_idx = csrnet_res.source_frame_index if csrnet_res else 0
    video_time = csrnet_res.video_timestamp if csrnet_res else (raft_res.current_timestamp if raft_res else 0.0)

    panel_w = target_width // 3
    panel_h = int(panel_w * 0.75)  # 4:3 aspect ratio per panel

    # --- Freshness and Alignment Calculations ---
    csrnet_age = 0.0
    csrnet_fresh = True

    if csrnet_res is not None:
        csrnet_age = compute_result_age(csrnet_res.video_timestamp, video_time)
        csrnet_fresh = is_result_fresh(csrnet_age, stale_threshold_s)

    if sync_result is not None:
        raft_res = sync_result.raft_result
        raft_age = sync_result.age_s
        raft_fresh = sync_result.is_valid_motion
        sync_status_val = sync_result.status.value if hasattr(sync_result.status, "value") else str(sync_result.status)
    else:
        raft_age = compute_result_age(raft_res.current_timestamp, video_time) if raft_res else 0.0
        raft_fresh = is_result_fresh(raft_age, stale_threshold_s) if raft_res else False
        sync_status_val = "NEW" if is_new_raft else "LAST_KNOWN"

    # Frame alignment check
    is_aligned = False
    if csrnet_res is not None and raft_res is not None:
        is_aligned = abs(csrnet_res.frame_id - raft_res.current_frame_id) <= 1

    # --- 1. Left panel: Real Video Frame ---
    panel_orig = cv2.resize(original_frame, (panel_w, panel_h))
    cv2.putText(panel_orig, f"1. REAL VIDEO ({gate_id})", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cv2.putText(panel_orig, f"Frame {frame_id:04d} | Src {source_idx} | t={video_time:.2f}s",
                (12, panel_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # Decision Badge on video frame
    if decision_res is not None:
        act = str(decision_res.action.value if hasattr(decision_res.action, "value") else decision_res.action)
        if act == "DIVERT" and decision_res.target_gate_id:
            badge_text = f"DECISION: DIVERT -> {decision_res.target_gate_id}"
            badge_color = (0, 165, 255)  # Orange
        elif act == "MONITOR":
            badge_text = f"DECISION: MONITOR ({gate_id})"
            badge_color = (0, 255, 255)  # Yellow
        else:
            badge_text = f"DECISION: {act}"
            badge_color = (0, 255, 0)    # Green

        cv2.rectangle(panel_orig, (8, panel_h - 32), (panel_w - 8, panel_h - 8), (0, 0, 0), -1)
        cv2.putText(panel_orig, badge_text, (14, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, badge_color, 2, cv2.LINE_AA)
    elif is_eof:
        cv2.rectangle(panel_orig, (8, panel_h - 32), (panel_w - 8, panel_h - 8), (0, 0, 0), -1)
        cv2.putText(panel_orig, "STATUS: STREAM EOF", (14, panel_h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 140, 255), 2, cv2.LINE_AA)

    # --- 2. Center panel: CSRNet Density Heatmap Overlay ---
    if csrnet_res is not None:
        panel_density = render_density_overlay(original_frame, csrnet_res.density_map, alpha=0.45)
        panel_density = cv2.resize(panel_density, (panel_w, panel_h))
        status_tag = "[NEW]" if is_new_csrnet else "[LAST KNOWN]"
        cv2.putText(
            panel_density,
            f"2. DENSITY (Count: {csrnet_res.crowd_count:.1f}) {status_tag}",
            (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 255, 0) if is_new_csrnet else (0, 200, 255), 2,
        )
        age_color = (0, 255, 0) if csrnet_fresh else (0, 0, 255)
        cv2.putText(panel_density, f"Age: {csrnet_age:.2f}s | Latency: {csrnet_res.inference_time_ms:.0f}ms",
                    (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, age_color, 1, cv2.LINE_AA)
    else:
        panel_density = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
        cv2.putText(panel_density, "2. DENSITY (WAITING)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (128, 128, 128), 2)

    # --- 3. Right panel: RAFT Optical Flow Motion / Quiver ---
    if flow_uv is not None and raft_fresh:
        u, v = flow_uv
        panel_motion = render_flow_hsv(u, v)
        panel_motion = render_quiver_arrows(panel_motion, u, v, step=28, scale=2.0)
        panel_motion = cv2.resize(panel_motion, (panel_w, panel_h))
    elif raft_res is not None and original_frame is not None:
        panel_motion = original_frame.copy()
        panel_motion = cv2.resize(panel_motion, (panel_w, panel_h))
    else:
        panel_motion = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

    if raft_res is not None:
        if sync_status_val == "NEW":
            raft_tag = "[NEW]"
            title_text = f"3. RAFT (Med: {raft_res.median_velocity:.1f} P95: {raft_res.p95_velocity:.1f}) {raft_tag}"
            title_color = (0, 255, 0)
        elif sync_status_val == "LAST_KNOWN":
            raft_tag = "[LAST KNOWN]"
            title_text = f"3. RAFT (Med: {raft_res.median_velocity:.1f} P95: {raft_res.p95_velocity:.1f}) {raft_tag}"
            title_color = (0, 220, 255)
        elif sync_status_val == "STALE":
            title_text = f"3. RAFT [STALE (>{stale_threshold_s:.1f}s) - EXCLUDED]"
            title_color = (0, 0, 255)
        else:
            title_text = "3. RAFT [UNAVAILABLE]"
            title_color = (128, 128, 128)

        cv2.putText(panel_motion, title_text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, title_color, 2)
        pair_text = f"Pair: {raft_res.previous_frame_id}->{raft_res.current_frame_id} dt={raft_res.delta_time:.3f}s"
        cv2.putText(panel_motion, pair_text, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        age_color = (0, 255, 0) if raft_fresh else (0, 0, 255)
        age_text = f"Age: {raft_age:.2f}s | Latency: {raft_res.inference_time_ms:.0f}ms"
        cv2.putText(panel_motion, age_text, (12, panel_h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, age_color, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel_motion, "3. RAFT MOTION (UNAVAILABLE)", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (128, 128, 128), 2)

    # --- Combine 3 panels horizontally ---
    panel_row = np.hstack([panel_orig, panel_density, panel_motion])

    # --- Top Header Bar (height=55) ---
    header_h = 55
    header = np.zeros((header_h, panel_row.shape[1], 3), dtype=np.uint8)
    header[:] = (20, 22, 28)

    cv2.putText(
        header,
        f"PHASE 3 LIVE | CAM: {cam_id} -> GATE: {gate_id} | Frame: {frame_id:04d} | Src: {source_idx} | t={video_time:.2f}s",
        (15, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
    )

    # Status indicators (Freshness + Alignment + EOF)
    if is_eof:
        status_text = "STREAM: EOF"
        status_color = (0, 140, 255)
    elif raft_res is not None and csrnet_res is not None:
        if sync_status_val == "NEW":
            fresh_str = "FRESH: OK (NEW)"
        elif sync_status_val == "LAST_KNOWN":
            fresh_str = f"FRESH: OK (LAST KNOWN {raft_age:.1f}s)"
        elif sync_status_val == "STALE":
            fresh_str = f"FRESH: STALE ({raft_age:.1f}s)"
        else:
            fresh_str = "MOTION: UNAVAILABLE"

        align_str = "ALIGN: OK" if is_aligned else ("ALIGN: HISTORICAL" if raft_res.current_frame_id <= csrnet_res.frame_id else "ALIGN: MISMATCH")
        status_text = f"{fresh_str} | {align_str}"
        status_color = (0, 255, 0) if (raft_fresh and is_aligned) else (0, 200, 255) if raft_fresh else (0, 0, 255)
    else:
        status_text = "INITIALIZING..."
        status_color = (0, 200, 255)

    cv2.putText(header, status_text, (panel_row.shape[1] - 470, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, status_color, 2, cv2.LINE_AA)

    device_str = f"Device: {telemetry.get('device', 'cuda').upper()} | Target FPS: {telemetry.get('ingestion_fps', 5.0):.0f}"
    cv2.putText(header, device_str, (15, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 128), 1, cv2.LINE_AA)

    queue_str = f"Queues: CSR={telemetry.get('csrnet_queue', 0)} RAFT={telemetry.get('raft_queue', 0)} | RAFT Stale Discarded: {telemetry.get('raft_stale_discarded', 0)}"
    cv2.putText(header, queue_str, (panel_row.shape[1] - 560, 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1, cv2.LINE_AA)

    # --- Bottom Telemetry Console (height=175) ---
    footer_h = 175
    footer = np.zeros((footer_h, panel_row.shape[1], 3), dtype=np.uint8)
    footer[:] = (12, 14, 18)

    # Column 1: Model Observations
    if raft_res:
        raft_prov_tag = f"({sync_status_val})"
        raft_med_str = f"RAFT Median Vel: {raft_res.median_velocity:.2f} px/f (P95: {raft_res.p95_velocity:.1f}) {raft_prov_tag}"
        raft_dir_str = f"RAFT Direction : {raft_res.dominant_direction_deg:.1f} deg (Con: {raft_res.flow_consistency:.2f})"
        raft_pair_str = f"RAFT Pair      : {raft_res.previous_frame_id}->{raft_res.current_frame_id} (dt={raft_res.delta_time:.3f}s)"
    else:
        raft_med_str = "RAFT Median Vel: (unavailable)"
        raft_dir_str = "RAFT Direction : (unavailable)"
        raft_pair_str = "RAFT Pair      : (none)"

    col1 = [
        "--- MODEL OBSERVATIONS ---",
        f"CSRNet Count   : {csrnet_res.crowd_count:.1f} people ({'NEW' if is_new_csrnet else 'LAST KNOWN'})" if csrnet_res else "CSRNet Count   : (waiting)",
        f"CSRNet Density : {csrnet_res.density_map_width}x{csrnet_res.density_map_height} ({csrnet_res.inference_time_ms:.0f}ms)" if csrnet_res else "CSRNet Density : (waiting)",
        raft_med_str,
        raft_dir_str,
        raft_pair_str,
    ]

    # Column 2: CrowdState Fusion
    col2 = ["--- CROWDSTATE (ADAPTER) ---"]
    if crowd_state is not None:
        motion_tag = f"{crowd_state.motion_status} ({crowd_state.raft_result_age_s:.1f}s)" if crowd_state.is_motion_fresh else f"{crowd_state.motion_status} (EXCLUDED)"
        col2.extend([
            f"Gate ID        : {crowd_state.gate_id}",
            f"Congestion     : {crowd_state.congestion_level.value} (score: {crowd_state.congestion_score:.2f})",
            f"Density Score  : {crowd_state.density_score:.2f} | Vel: {crowd_state.velocity:.1f} px/f",
            f"Motion Status  : {motion_tag}",
            f"Trend / Conf   : {crowd_state.trend.value} | Conf: {crowd_state.confidence:.2f}",
        ])
    else:
        col2.extend(["CrowdState     : (fusing...)"] * 5)

    # Column 3: Decision Engine Output
    col3 = ["--- DECISION ENGINE ---"]
    if decision_res is not None:
        act_str = str(decision_res.action.value if hasattr(decision_res.action, "value") else decision_res.action)
        tgt_str = f" -> {decision_res.target_gate_id}" if decision_res.target_gate_id else ""
        col3.extend([
            f"Action         : {act_str}{tgt_str}",
            f"Severity       : {decision_res.severity.value if hasattr(decision_res.severity, 'value') else decision_res.severity}",
            f"Confidence     : {decision_res.confidence:.2f}",
            f"Reason Code    : {', '.join(decision_res.reason_codes[:2])}",
            f"Reason         : {decision_res.human_readable_reason[:48]}...",
        ])
    else:
        col3.extend(["Decision       : (evaluating...)"] * 5)

    y_spacing = 25
    for i, line in enumerate(col1):
        color = (0, 255, 255) if i == 0 else (220, 220, 220)
        cv2.putText(footer, line, (20, 22 + i * y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    for i, line in enumerate(col2):
        color = (0, 255, 255) if i == 0 else (220, 220, 220)
        cv2.putText(footer, line, (panel_w + 20, 22 + i * y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    for i, line in enumerate(col3):
        color = (0, 255, 255) if i == 0 else (220, 220, 220)
        cv2.putText(footer, line, (panel_w * 2 + 20, 22 + i * y_spacing), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    dashboard = np.vstack([header, panel_row, footer])
    return dashboard

