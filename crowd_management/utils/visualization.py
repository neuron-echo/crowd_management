"""Visualization utilities for Phase 2 CSRNet density and RAFT optical flow outputs."""

from __future__ import annotations

import cv2
import numpy as np

from crowd_management.models.contracts import CSRNetResult, RAFTResult


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
    alpha: float = 0.5,
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


def draw_phase2_hud(
    bgr_frame: np.ndarray,
    csrnet_res: CSRNetResult | None = None,
    raft_res: RAFTResult | None = None,
) -> np.ndarray:
    """Draw comprehensive diagnostic HUD overlay with CSRNet and RAFT metrics."""
    out = bgr_frame.copy()
    box_w = 460
    box_h = 240

    overlay = out.copy()
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (15, 15, 20), -1)
    cv2.addWeighted(overlay, 0.8, out, 0.2, 0, out)
    cv2.rectangle(out, (15, 15), (15 + box_w, 15 + box_h), (0, 200, 255), 2)

    lines = []
    if csrnet_res is not None:
        lines += [
            f"[CSRNet] Cam: {csrnet_res.camera_id} | Frame: {csrnet_res.frame_id}",
            f"  Predicted Crowd Count : {csrnet_res.crowd_count:.1f} people",
            f"  Density Map Dim       : {csrnet_res.density_map_width}x{csrnet_res.density_map_height}",
            f"  Inference Latency     : {csrnet_res.inference_time_ms:.1f} ms ({csrnet_res.device})",
        ]
    if raft_res is not None:
        lines += [
            f"[RAFT] Motion Pair: {raft_res.previous_frame_id} -> {raft_res.current_frame_id}",
            f"  Avg Velocity : {raft_res.average_velocity:.2f} px/frame",
            f"  Dominant Dir : {raft_res.dominant_direction_deg:.1f} deg",
            f"  Consistency  : {raft_res.flow_consistency:.2f} | Confidence: {raft_res.motion_confidence:.2f}",
            f"  Inference    : {raft_res.inference_time_ms:.1f} ms",
        ]

    y = 42
    for i, line in enumerate(lines):
        color = (0, 255, 128) if line.startswith("[") else (240, 240, 240)
        cv2.putText(out, line, (25, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 24

    return out
