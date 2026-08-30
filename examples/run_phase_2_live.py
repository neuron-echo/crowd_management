"""Live Multi-Camera Phase 2 Visualization Runner with Freshness-Aware RAFT Synchronization.

Features:
- Real-time CSRNet + RAFT inference on streaming HAJJv2 video
- Result-age tracking with SYNC: OK / SYNC: STALE indicators
- RAFT pair provenance (prev→curr frame IDs, source indices, timestamps)
- Robust velocity display (median, P95) instead of mean
- --debug-raft mode to save suspicious frame pairs to disk
- VRAM monitoring on CUDA
- Comprehensive final statistics with freshness percentiles
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd_management.config.loader import load_ingestion_config
from crowd_management.models.contracts import CSRNetResult, RAFTResult
from crowd_management.models.manager import ModelManager
from crowd_management.pipeline.fanout_pipeline import FanOutPipeline
from crowd_management.utils.visualization import (
    create_live_dashboard_panel,
    compute_result_age,
    is_result_fresh,
    render_flow_hsv,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Phase2Live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Live Phase 2 Fan-Out Pipeline with Synchronized Telemetry Dashboard")
    parser.add_argument(
        "--config", type=str, default="config/ingestion/hajj_ingestion.yaml",
        help="Path to ingestion YAML config file",
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs/phase_2/live",
        help="Output directory for saved live snapshots and latest dashboard images",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0.0,
        help="Maximum duration in seconds to run (0 for indefinite until Q/Ctrl+C)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable interactive cv2.imshow window (headless mode, saves to disk)",
    )
    parser.add_argument(
        "--stale-threshold", type=float, default=1.0,
        help="Maximum acceptable result age in seconds before marking SYNC: STALE",
    )
    parser.add_argument(
        "--debug-raft", action="store_true",
        help="Save RAFT debug artifacts (frame pairs, flow visualization, metadata JSON) for every result",
    )
    parser.add_argument(
        "--debug-raft-dir", type=str, default="outputs/phase_2/raft_debug",
        help="Directory for RAFT debug artifacts",
    )
    return parser.parse_args()


def print_banner(config, args) -> None:
    print("=" * 90)
    print("PHASE 2 — REAL-TIME MULTI-CAMERA FAN-OUT CROWD ANALYTICS PIPELINE")
    print(f"Dataset         : {config.dataset.name} ({config.dataset.root})")
    print(f"Cameras         : {[c.camera_id for c in config.cameras if c.enabled]}")
    print(f"Stale Threshold : {args.stale_threshold:.2f}s")
    print(f"Debug RAFT      : {args.debug_raft}")
    print("Controls: Press 'Q' or 'ESC' in dashboard window, or Ctrl+C in terminal to stop.")
    print("=" * 90)


def save_raft_debug(
    raft_res: RAFTResult,
    prev_frame: np.ndarray | None,
    curr_frame: np.ndarray | None,
    flow_uv: tuple[np.ndarray, np.ndarray] | None,
    debug_dir: Path,
    raft_age: float,
) -> None:
    """Save RAFT frame pair, flow visualization, and metadata JSON for investigation."""
    prefix = f"{raft_res.camera_id}_pair_{raft_res.previous_frame_id:04d}_{raft_res.current_frame_id:04d}"

    if prev_frame is not None:
        cv2.imwrite(str(debug_dir / f"{prefix}_prev.jpg"), prev_frame)
    if curr_frame is not None:
        cv2.imwrite(str(debug_dir / f"{prefix}_curr.jpg"), curr_frame)

    if flow_uv is not None:
        u, v = flow_uv
        flow_viz = render_flow_hsv(u, v)
        flow_viz = cv2.resize(flow_viz, (512, 384))
        cv2.imwrite(str(debug_dir / f"{prefix}_flow.jpg"), flow_viz)

    meta = {
        "camera_id": raft_res.camera_id,
        "previous_frame_id": raft_res.previous_frame_id,
        "current_frame_id": raft_res.current_frame_id,
        "previous_source_frame_index": raft_res.previous_source_frame_index,
        "current_source_frame_index": raft_res.current_source_frame_index,
        "previous_video_timestamp": raft_res.previous_timestamp,
        "current_video_timestamp": raft_res.current_timestamp,
        "delta_time_s": raft_res.delta_time,
        "mean_velocity_pxf": raft_res.mean_velocity,
        "median_velocity_pxf": raft_res.median_velocity,
        "p90_velocity_pxf": raft_res.p90_velocity,
        "p95_velocity_pxf": raft_res.p95_velocity,
        "max_velocity_pxf": raft_res.max_velocity,
        "velocity_variance": raft_res.velocity_variance,
        "dominant_direction_deg": raft_res.dominant_direction_deg,
        "flow_consistency": raft_res.flow_consistency,
        "active_motion_ratio": raft_res.active_motion_ratio,
        "motion_confidence": raft_res.motion_confidence,
        "inference_time_ms": raft_res.inference_time_ms,
        "result_age_s": raft_age,
        "device": raft_res.device,
        "model_name": raft_res.model_name,
        "flow_resolution": f"{raft_res.flow_width}x{raft_res.flow_height}",
        "velocity_unit": "px/frame at inference resolution",
    }
    with open(debug_dir / f"{prefix}.json", "w") as f:
        json.dump(meta, f, indent=2)


def get_vram_mb() -> float:
    """Get current GPU memory allocated in MB. Returns 0.0 if CUDA unavailable."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 * 1024)
    except Exception:
        pass
    return 0.0


def main() -> None:
    args = parse_args()
    config = load_ingestion_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    debug_dir = Path(args.debug_raft_dir)
    if args.debug_raft:
        debug_dir.mkdir(parents=True, exist_ok=True)

    print_banner(config, args)

    # Initialize shared model manager
    model_mgr = ModelManager(models_config=config.models)

    # Per-camera state tracking for synchronization
    latest_csrnet: dict[str, CSRNetResult] = {}
    latest_raft: dict[str, RAFTResult] = {}
    latest_raw_frames: dict[str, np.ndarray] = {}
    camera_flow_uv: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    # Statistics collectors
    raft_result_ages: list[float] = []
    csrnet_result_ages: list[float] = []
    raft_velocities_median: list[float] = []
    raft_velocities_p95: list[float] = []
    sync_ok_count = 0
    sync_stale_count = 0
    total_display_cycles = 0

    start_time = time.monotonic()
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nStopping pipeline gracefully...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    def on_csrnet_result(res: CSRNetResult) -> None:
        latest_csrnet[res.camera_id] = res
        if res.frame is not None:
            latest_raw_frames[res.camera_id] = res.frame

    def on_raft_result(res: RAFTResult) -> None:
        latest_raft[res.camera_id] = res
        # Unpack flow summary for quiver rendering
        if res.flow_summary is not None:
            u_flow = res.flow_summary[..., 0]
            v_flow = res.flow_summary[..., 1]
            camera_flow_uv[res.camera_id] = (u_flow, v_flow)

        # Collect velocity statistics
        raft_velocities_median.append(res.median_velocity)
        raft_velocities_p95.append(res.p95_velocity)

        # Debug: save every result if --debug-raft
        if args.debug_raft:
            curr_video_time = res.current_timestamp
            age = 0.0  # At callback time, age is ~0 relative to RAFT's own pair
            # Use latest raw frame if available (for visual context)
            raw = latest_raw_frames.get(res.camera_id)
            save_raft_debug(res, raw, raw, camera_flow_uv.get(res.camera_id), debug_dir, age)

    pipeline = FanOutPipeline(
        config=config,
        model_manager=model_mgr,
        csrnet_callback=on_csrnet_result,
        raft_callback=on_raft_result,
        queue_capacity=config.ingestion.buffer_size,
    )

    print("\nStarting multi-camera ingestion and inference fan-out workers...")
    pipeline.start()

    window_name = "Phase 2 Crowd Analytics Live Dashboard"
    can_display = not args.no_display

    saved_snapshots = 0
    last_save_time = 0.0
    last_log_time = 0.0

    try:
        while running:
            now = time.monotonic()
            if args.max_seconds > 0 and (now - start_time) >= args.max_seconds:
                print(f"\nReached max runtime ({args.max_seconds:.1f}s), stopping...")
                break

            stats = pipeline.stats()

            # Render dashboard for active cameras
            active_camera_ids = pipeline.producer.camera_ids
            dashboard_panels = []

            for cam_id in active_camera_ids:
                csr = latest_csrnet.get(cam_id)
                raft = latest_raft.get(cam_id)
                raw_frame = latest_raw_frames.get(cam_id)

                if raw_frame is None and csr is not None and csr.frame is not None:
                    raw_frame = csr.frame
                if raw_frame is None:
                    raw_frame = np.zeros((540, 720, 3), dtype=np.uint8)

                flow_uv = camera_flow_uv.get(cam_id)

                # Compute result ages
                current_video_time = csr.video_timestamp if csr else 0.0

                csrnet_age = 0.0
                raft_age = 0.0
                if csr is not None:
                    csrnet_age = compute_result_age(csr.video_timestamp, current_video_time)
                if raft is not None:
                    raft_age = compute_result_age(raft.current_timestamp, current_video_time)
                    raft_result_ages.append(raft_age)
                if csr is not None:
                    csrnet_result_ages.append(csrnet_age)

                # Track sync status
                total_display_cycles += 1
                if raft is not None and csr is not None:
                    if is_result_fresh(raft_age, args.stale_threshold):
                        sync_ok_count += 1
                    else:
                        sync_stale_count += 1

                cam_telemetry = {
                    "ingestion_fps": config.ingestion.target_fps,
                    "csrnet_fps": (1000.0 / stats["csrnet"]["avg_latency_ms"]) if stats["csrnet"]["avg_latency_ms"] > 0 else 0.0,
                    "raft_fps": (1000.0 / stats["raft"]["avg_latency_ms"]) if stats["raft"]["avg_latency_ms"] > 0 else 0.0,
                    "device": stats["device"],
                    "csrnet_queue": stats["csrnet"]["queue_size"],
                    "raft_queue": stats["raft"]["queue_size"],
                    "raft_stale_discarded": pipeline.raft_worker.stale_frames_discarded,
                }

                dash_panel = create_live_dashboard_panel(
                    original_frame=raw_frame,
                    csrnet_res=csr,
                    raft_res=raft,
                    flow_uv=flow_uv,
                    telemetry=cam_telemetry,
                    target_width=1350,
                    stale_threshold_s=args.stale_threshold,
                )
                dashboard_panels.append(dash_panel)

            # Terminal log every 1.5 seconds
            if (now - last_log_time) >= 1.5 and active_camera_ids:
                last_log_time = now
                for cam_id in active_camera_ids:
                    csr = latest_csrnet.get(cam_id)
                    raft = latest_raft.get(cam_id)
                    current_vt = csr.video_timestamp if csr else 0.0

                    csr_str = f"Count: {csr.crowd_count:5.1f} ({csr.inference_time_ms:4.0f}ms)" if csr else "CSR: (waiting)"
                    if raft:
                        r_age = compute_result_age(raft.current_timestamp, current_vt)
                        sync = "OK" if is_result_fresh(r_age, args.stale_threshold) else f"STALE({r_age:.2f}s)"
                        raft_str = (
                            f"RAFT[{raft.previous_frame_id:04d}->{raft.current_frame_id:04d}] "
                            f"Med={raft.median_velocity:5.1f} P95={raft.p95_velocity:5.1f} px/f "
                            f"Dir={raft.dominant_direction_deg:5.1f}° Con={raft.flow_consistency:.2f} "
                            f"({raft.inference_time_ms:4.0f}ms) SYNC:{sync}"
                        )
                    else:
                        raft_str = "RAFT: (priming)"

                    frame_str = f"Frame {csr.frame_id:04d} (t={csr.video_timestamp:5.2f}s)" if csr else "Frame ????"
                    q_str = f"Q[CSR:{stats['csrnet']['queue_size']:02d} RAFT:{stats['raft']['queue_size']:02d}]"
                    vram = get_vram_mb()
                    vram_str = f"VRAM:{vram:.0f}MB" if vram > 0 else ""

                    print(f"[{cam_id}] {frame_str} | {csr_str} | {raft_str} | {q_str} {vram_str}")

            if dashboard_panels:
                full_dashboard = np.vstack(dashboard_panels)

                # Save latest dashboard image
                latest_path = out_dir / "live_dashboard_latest.jpg"
                cv2.imwrite(str(latest_path), full_dashboard)

                # Periodically save snapshot frame (every 3 seconds)
                if (now - last_save_time) >= 3.0:
                    last_save_time = now
                    saved_snapshots += 1
                    snap_path = out_dir / f"live_snapshot_{saved_snapshots:04d}.jpg"
                    cv2.imwrite(str(snap_path), full_dashboard)

                # Display in interactive window if available
                if can_display:
                    try:
                        cv2.imshow(window_name, full_dashboard)
                        key = cv2.waitKey(20) & 0xFF
                        if key in [ord("q"), ord("Q"), 27]:  # 'q' or ESC
                            print("\nUser requested exit (Q/ESC pressed).")
                            break
                    except Exception as e:
                        can_display = False
                        logger.info("GUI window not available (%s); continuing in background/file-save mode.", e)

            time.sleep(0.05)

    finally:
        pipeline.stop()
        if can_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # ===================================================================
    # FINAL STATISTICS REPORT
    # ===================================================================
    total_time = time.monotonic() - start_time
    final_stats = pipeline.stats()

    print("\n" + "=" * 90)
    print("PHASE 2 LIVE EXECUTION — FINAL STATISTICS REPORT")
    print("=" * 90)
    print(f"Total Runtime              : {total_time:.2f} s")
    print(f"Execution Device           : {final_stats['device'].upper()}")
    print(f"GPU VRAM (final)           : {get_vram_mb():.1f} MB")
    print()

    # Per-camera ingestion stats
    print("--- INGESTION ---")
    for cam_id, cam_stats in final_stats["cameras"].items():
        print(f"  [{cam_id}] Produced: {cam_stats['produced']} | Consumed: {cam_stats['consumed']} | "
              f"Dropped: {cam_stats['dropped']} | Buffer: {cam_stats['buffer_size']}")
    print(f"  Distributor Routed       : {final_stats['distributor']['packets_routed']} packets")

    # CSRNet stats
    print()
    print("--- CSRNET ---")
    print(f"  Frames Processed         : {final_stats['csrnet']['frames_processed']}")
    print(f"  Avg Latency              : {final_stats['csrnet']['avg_latency_ms']:.1f} ms")
    csr_fps = (1000.0 / final_stats['csrnet']['avg_latency_ms']) if final_stats['csrnet']['avg_latency_ms'] > 0 else 0
    print(f"  Effective FPS            : {csr_fps:.2f}")

    # RAFT stats
    print()
    print("--- RAFT (Freshness-Aware) ---")
    print(f"  Pairs Processed          : {final_stats['raft']['pairs_processed']}")
    print(f"  Avg Latency              : {final_stats['raft']['avg_latency_ms']:.1f} ms")
    raft_fps = (1000.0 / final_stats['raft']['avg_latency_ms']) if final_stats['raft']['avg_latency_ms'] > 0 else 0
    print(f"  Effective FPS            : {raft_fps:.2f}")
    print(f"  Stale Frames Discarded   : {pipeline.raft_worker.stale_frames_discarded}")

    # Result age statistics
    print()
    print("--- RESULT FRESHNESS ---")
    if raft_result_ages:
        ages = np.array(raft_result_ages)
        print(f"  RAFT Result Age (median) : {np.median(ages):.3f} s")
        print(f"  RAFT Result Age (P95)    : {np.percentile(ages, 95):.3f} s")
        print(f"  RAFT Result Age (max)    : {np.max(ages):.3f} s")
    else:
        print("  RAFT Result Age          : N/A (no pairs processed)")

    if total_display_cycles > 0:
        total_with_results = sync_ok_count + sync_stale_count
        if total_with_results > 0:
            print(f"  Synchronized (OK)        : {sync_ok_count}/{total_with_results} ({100*sync_ok_count/total_with_results:.1f}%)")
            print(f"  Stale Results            : {sync_stale_count}/{total_with_results} ({100*sync_stale_count/total_with_results:.1f}%)")

    # RAFT velocity distribution
    print()
    print("--- RAFT VELOCITY DISTRIBUTION ---")
    if raft_velocities_median:
        med_arr = np.array(raft_velocities_median)
        p95_arr = np.array(raft_velocities_p95)
        print(f"  Median Velocity (all)    : mean={np.mean(med_arr):.2f}, median={np.median(med_arr):.2f}, max={np.max(med_arr):.2f} px/f")
        print(f"  P95 Velocity (all)       : mean={np.mean(p95_arr):.2f}, median={np.median(p95_arr):.2f}, max={np.max(p95_arr):.2f} px/f")
    else:
        print("  No RAFT velocity data collected.")

    print()
    print(f"Live Artifacts Saved       : {out_dir}")
    if args.debug_raft:
        print(f"RAFT Debug Artifacts       : {debug_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
