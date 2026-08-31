"""Live Multi-Camera Phase 3 Integration Runner: End-to-End Real Vision -> CrowdState -> Decision Engine.

NO SYNTHETIC CROWDSTATE DATA IS USED.
Every decision originates from the real CSRNet crowd count and YOLOv8+ByteTrack
crowd motion computed on streaming HAJJv2 video frames.

Data Flow:
    Real Video Stream (Phase 1)
        |
        v
    FramePacket -> FrameDistributor
        |                    |
        v                    v
    CSRNetWorker        ByteTrackWorker (Asynchronous)
    (CSRNetResult)      (ByteTrackResult)
        |                    |
        +---------+----------+
                  |
                  v
          TemporalSynchronizer (causality check: bytetrack.frame_id <= csr.frame_id)
                  |
                  v
            Phase2Adapter (fuse with staleness check)
                  |
                  v
              CrowdState (model-agnostic, stale/future motion excluded)
                  |
                  v
            DecisionEngine (evaluate active observed gates)
                  |
                  v
            DecisionResult
                  |
                  v
          Live Dashboard & Provenance Telemetry

Usage:
    .venv\\Scripts\\python examples/run_phase_3_live.py ^
        --config config/ingestion/hajj_ingestion.yaml ^
        --venue config/venue/circular_venue_live_scaled.yaml ^
        --debug
"""

from __future__ import annotations

import argparse
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

from crowd_management.adapters.phase2_adapter import Phase2Adapter
from crowd_management.adapters.synchronizer import MotionFreshness, SyncResult, TemporalSynchronizer
from crowd_management.config.loader import load_ingestion_config
from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.contracts import (
    ActionType,
    CongestionLevel,
    CrowdState,
    DecisionResult,
    Trend,
)
from crowd_management.decision.engine import DecisionEngine
from crowd_management.decision.trend import TrendTracker
from crowd_management.models.contracts import ByteTrackResult, CSRNetResult
from crowd_management.models.manager import ModelManager
from crowd_management.pipeline.fanout_pipeline import FanOutPipeline
from crowd_management.utils.visualization import (
    compute_result_age,
    create_phase3_dashboard_panel,
    is_result_fresh,
    render_flow_hsv,
)
from crowd_management.venue.loader import load_venue_topology
from crowd_management.venue.topology import VenueTopology

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("Phase3Live")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 Real-Time End-to-End Crowd Management Live Integration Runner"
    )
    parser.add_argument(
        "--config", type=str, default="config/ingestion/hajj_ingestion.yaml",
        help="Path to ingestion YAML config file",
    )
    parser.add_argument(
        "--venue", type=str, default="config/venue/circular_venue_live_scaled.yaml",
        help="Path to venue YAML config file",
    )
    parser.add_argument(
        "--out-dir", type=str, default="outputs/phase_3/live",
        help="Output directory for saved live snapshots and dashboard images",
    )
    parser.add_argument(
        "--max-seconds", type=float, default=0.0,
        help="Maximum duration in seconds to run (0 for indefinite until EOF or Q/Ctrl+C)",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="Disable interactive cv2.imshow window (headless mode, saves snapshots to disk)",
    )
    parser.add_argument(
        "--stale-frames", type=int, default=10,
        help="Maximum frame-id gap before ByteTrack motion is marked STALE and excluded",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose provenance debug logging for every frame evaluation",
    )
    return parser.parse_args()


def validate_camera_gate_mapping(config: Any, venue: VenueTopology) -> list[str]:
    """Validate 1-to-1 mapping between active enabled cameras and observed venue gates."""
    active_cameras = [c.camera_id for c in config.cameras if c.enabled]
    if not active_cameras:
        raise ValueError("No enabled cameras found in ingestion configuration.")

    for cam_id in active_cameras:
        gate_id = venue.gate_for_camera(cam_id)
        if gate_id is None:
            raise ValueError(f"Active camera '{cam_id}' has no corresponding gate in venue '{venue.venue_id}'")

    for gid in venue.gate_ids:
        cam_id = venue.camera_for_gate(gid)
        if cam_id is None or cam_id not in active_cameras:
            raise ValueError(
                f"Gate '{gid}' in venue '{venue.venue_id}' maps to camera '{cam_id}' "
                f"which is not an active enabled camera in ingestion config."
            )

    return active_cameras


def print_startup_banner(
    config: Any,
    venue: VenueTopology,
    active_cameras: list[str],
    args: argparse.Namespace,
) -> None:
    print("=" * 90)
    print("PHASE 3 -- REAL-TIME END-TO-END CROWD MANAGEMENT LIVE INTEGRATION RUNNER")
    print("=" * 90)
    print("LIVE MODE:")
    print("  REAL VIDEO + REAL CSRNET + YOLOV8+BYTETRACK + TEMPORAL SYNCHRONIZER + DECISION ENGINE")
    print("  NO SYNTHETIC CROWDSTATE | STRICT TEMPORAL CAUSALITY (bytetrack.frame_id <= csr.frame_id)")
    print()
    print(f"Dataset         : {config.dataset.name} ({config.dataset.root})")
    print(f"Venue           : {venue.venue_name} ({venue.venue_id})")
    print(f"Geometry        : {venue._config.geometry.type}, radius={venue._config.geometry.radius_m}m")
    print(f"Total Gates     : {venue.gate_count} (Circular adjacency established)")
    print()
    print("ACTIVE CAMERAS:")
    for cam_id in active_cameras:
        gid = venue.gate_for_camera(cam_id)
        cap = venue.get_gate_capacity(gid)
        print(f"  {cam_id} -> {gid} (Capacity: {cap})")
    print()
    print("ACTIVE OBSERVED GATES:")
    for gid in venue.gate_ids:
        cam = venue.camera_for_gate(gid)
        cap = venue.get_gate_capacity(gid)
        print(f"  {gid} (Camera: {cam}, Capacity: {cap})")
    print()
    bt_cfg = config.models.bytetrack
    print(f"ByteTrack Model : {bt_cfg.model_path} (conf={bt_cfg.confidence_threshold})")
    print(f"Stale Threshold : {args.stale_frames} frames")
    print(f"Debug Mode      : {args.debug}")
    print("Controls: Press 'Q' or 'ESC' in dashboard window, or Ctrl+C in terminal to stop.")
    print("=" * 90)


def get_vram_mb() -> float:
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
    venue = load_venue_topology(args.venue)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate camera <-> gate architecture
    active_cameras = validate_camera_gate_mapping(config, venue)
    print_startup_banner(config, venue, active_cameras, args)

    # Initialize Decision & Synchronization components
    decision_config = DecisionConfig(max_observation_age_s=float(args.stale_frames) * 0.5)
    trend_tracker = TrendTracker(decision_config)
    synchronizer = TemporalSynchronizer(
        max_history_per_camera=100,
        stale_frame_gap=args.stale_frames,
    )
    adapter = Phase2Adapter(decision_config)
    decision_engine = DecisionEngine(venue, decision_config)

    # Initialize shared model manager for Phase 2 vision models
    model_mgr = ModelManager(models_config=config.models)

    # Per-camera & Per-gate state tracking
    latest_csrnet: dict[str, CSRNetResult] = {}
    latest_bytetrack: dict[str, ByteTrackResult] = {}
    latest_sync_results: dict[str, SyncResult] = {}
    latest_raw_frames: dict[str, np.ndarray] = {}
    latest_crowd_states: dict[str, CrowdState] = {}
    latest_decisions: dict[str, DecisionResult] = {}
    camera_eof: dict[str, bool] = {cam_id: False for cam_id in active_cameras}

    last_rendered_csr_id: dict[str, int] = {cam_id: -1 for cam_id in active_cameras}
    last_logged_csr_id: dict[str, int] = {cam_id: -1 for cam_id in active_cameras}

    # Telemetry statistics
    decisions_count_by_action: dict[str, int] = {"DIVERT": 0, "MONITOR": 0, "NO_ACTION": 0, "HOLD": 0}
    total_evaluations = 0

    start_time = time.monotonic()
    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\nStopping pipeline gracefully...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)

    def on_csrnet_result(res: CSRNetResult) -> None:
        nonlocal total_evaluations
        cam_id = res.camera_id
        latest_csrnet[cam_id] = res
        if res.frame is not None:
            latest_raw_frames[cam_id] = res.frame

        gate_id = venue.gate_for_camera(cam_id)
        if gate_id is None:
            return

        gate_cap = venue.get_gate_capacity(gate_id)

        # Synchronize with the latest valid ByteTrack result (frame_id <= csr.frame_id)
        sync_res = synchronizer.synchronize(res, stale_frame_gap=args.stale_frames)
        latest_sync_results[cam_id] = sync_res

        # Get current trend from tracker
        trend = trend_tracker.get_trend(gate_id)

        # Fuse observations into model-agnostic CrowdState
        crowd_state = adapter.fuse(
            csrnet_result=res,
            bytetrack_result=sync_res.bytetrack_result,
            gate_id=gate_id,
            gate_capacity=gate_cap,
            trend=trend,
            stale_frame_gap=args.stale_frames,
            sync_result=sync_res,
        )
        latest_crowd_states[gate_id] = crowd_state

        # Update trend tracker with real congestion score
        trend_tracker.update(gate_id, crowd_state.congestion_score)

        # Evaluate through Decision Engine
        decisions = decision_engine.evaluate(latest_crowd_states)
        for dec in decisions:
            latest_decisions[dec.source_gate_id] = dec
            act_name = dec.action.value if hasattr(dec.action, "value") else str(dec.action)
            decisions_count_by_action[act_name] = decisions_count_by_action.get(act_name, 0) + 1

        total_evaluations += 1

        # Debug provenance logging
        if args.debug:
            dec = latest_decisions.get(gate_id)
            act_str = f"{dec.action.value} -> {dec.target_gate_id}" if (dec and dec.target_gate_id) else (str(dec.action.value) if dec else "None")
            conf_str = f"{dec.confidence:.2f}" if dec else "0.00"
            sev_str = str(dec.severity.value) if dec else "N/A"

            print(f"[DEBUG PROVENANCE] {cam_id} -> {gate_id}")
            print(f"  CSRNet observation : frame={res.frame_id}, count={res.crowd_count:.1f}, video_time={res.video_timestamp:.2f}s, lat={res.inference_time_ms:.0f}ms [NEW]")
            if sync_res.bytetrack_result:
                bt = sync_res.bytetrack_result
                print(f"  ByteTrack synced   : {sync_res.pair_provenance}, tracks={bt.track_count}, gap={sync_res.age_frames}f, status={sync_res.status.value}, med_vel={bt.median_velocity:.1f} px/s")
            else:
                print(f"  ByteTrack synced   : status=UNAVAILABLE, reason={sync_res.reason}")
            print(f"  CrowdState fused   : gate={gate_id}, count={crowd_state.crowd_count:.1f}, density={crowd_state.density_score:.2f}, vel={crowd_state.velocity:.1f} px/s (motion={crowd_state.motion_status}), cong={crowd_state.congestion_level.value} (score={crowd_state.congestion_score:.2f}), trend={crowd_state.trend.value}")
            print(f"  Decision emitted   : {act_str} | Severity: {sev_str} | Conf: {conf_str}")

    def on_bytetrack_result(res: ByteTrackResult) -> None:
        cam_id = res.camera_id
        synchronizer.add_bytetrack_result(res)
        latest_bytetrack[cam_id] = res

    pipeline = FanOutPipeline(
        config=config,
        model_manager=model_mgr,
        csrnet_callback=on_csrnet_result,
        bytetrack_callback=on_bytetrack_result,
        queue_capacity=config.ingestion.buffer_size,
    )

    print("\nStarting multi-camera ingestion and inference fan-out workers...")
    pipeline.start()

    window_name = "Phase 3 Live End-to-End Crowd Management Dashboard"
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
            active_camera_ids = pipeline.producer.camera_ids

            # Check EOF status for each camera
            all_eof = True
            for cam_id in active_camera_ids:
                cam_stats = stats["cameras"].get(cam_id, {})
                is_comp = cam_stats.get("is_complete", False)
                buf_sz = cam_stats.get("buffer_size", 0)
                csr_q = stats["csrnet"]["queue_size"]
                bt_q = stats["bytetrack"]["queue_size"]

                if is_comp and buf_sz == 0 and csr_q == 0 and bt_q == 0:
                    if not camera_eof.get(cam_id, False):
                        camera_eof[cam_id] = True
                        gate_id = venue.gate_for_camera(cam_id) or "G?"
                        print(f"\n[{cam_id} -> {gate_id}] STREAM EOF -- video completed & buffer drained.")
                else:
                    all_eof = False

            if all_eof and active_camera_ids:
                print("\n" + "=" * 90)
                print("ALL CAMERAS FINISHED (EOF)")
                print("PIPELINE DRAINED")
                print("LIVE DEMO COMPLETE")
                print("=" * 90)
                break

            # Render dashboard panels
            dashboard_panels = []
            for cam_id in active_camera_ids:
                gate_id = venue.gate_for_camera(cam_id) or "G?"
                csr = latest_csrnet.get(cam_id)
                sync_res = latest_sync_results.get(cam_id)
                bt = sync_res.bytetrack_result if sync_res else latest_bytetrack.get(cam_id)
                raw_frame = latest_raw_frames.get(cam_id)
                crowd_st = latest_crowd_states.get(gate_id)
                dec_res = latest_decisions.get(gate_id)
                is_eof = camera_eof.get(cam_id, False)

                if raw_frame is None and csr is not None and csr.frame is not None:
                    raw_frame = csr.frame
                if raw_frame is None:
                    raw_frame = np.zeros((540, 720, 3), dtype=np.uint8)

                is_new_csr = (csr is not None and csr.frame_id != last_rendered_csr_id[cam_id])
                if is_new_csr and csr:
                    last_rendered_csr_id[cam_id] = csr.frame_id

                is_new_bt = (sync_res is not None and sync_res.status == MotionFreshness.NEW)

                cam_telemetry = {
                    "ingestion_fps": config.ingestion.target_fps,
                    "device": stats["device"],
                    "csrnet_queue": stats["csrnet"]["queue_size"],
                    "raft_queue": stats["bytetrack"]["queue_size"],  # key kept for dashboard compat
                    "raft_stale_discarded": pipeline.bytetrack_worker.stale_frames_discarded,
                }

                try:
                    panel = create_phase3_dashboard_panel(
                        original_frame=raw_frame,
                        cam_id=cam_id,
                        gate_id=gate_id,
                        csrnet_res=csr,
                        raft_res=None,          # dashboard compat: no flow HSV needed
                        crowd_state=crowd_st,
                        decision_res=dec_res,
                        flow_uv=None,
                        telemetry=cam_telemetry,
                        target_width=1350,
                        stale_threshold_s=1.0,
                        is_eof=is_eof,
                        is_new_csrnet=is_new_csr,
                        is_new_raft=is_new_bt,
                        sync_result=None,
                    )
                    dashboard_panels.append(panel)
                except Exception:
                    pass

            # Periodic terminal telemetry (every 1.5s)
            if (now - last_log_time) >= 1.5 and active_camera_ids:
                last_log_time = now
                print("-" * 80)
                for cam_id in active_camera_ids:
                    gate_id = venue.gate_for_camera(cam_id) or "G?"
                    csr = latest_csrnet.get(cam_id)
                    sync_res = latest_sync_results.get(cam_id)
                    c_state = latest_crowd_states.get(gate_id)
                    dec = latest_decisions.get(gate_id)

                    is_new_csr_log = (csr is not None and csr.frame_id != last_logged_csr_id[cam_id])
                    if is_new_csr_log and csr:
                        last_logged_csr_id[cam_id] = csr.frame_id
                    csr_status_str = "[NEW]" if is_new_csr_log else "[LAST KNOWN]"

                    frame_str = f"Frame {csr.frame_id:04d} (t={csr.video_timestamp:5.2f}s)" if csr else "Frame ????"
                    csr_str = f"Count={csr.crowd_count:4.1f} ({csr.inference_time_ms:3.0f}ms) {csr_status_str}" if csr else "CSR:(waiting)"

                    if sync_res and sync_res.bytetrack_result:
                        bt = sync_res.bytetrack_result
                        status_tag = sync_res.status.value
                        bt_str = f"ByteTrack[frame {bt.frame_id:04d}] tracks={bt.track_count} Med={bt.median_velocity:.1f}px/s dir={bt.dominant_direction_deg:.0f}° ({status_tag}, gap={sync_res.age_frames}f)"
                    elif sync_res:
                        bt_str = f"ByteTrack:({sync_res.status.value} - {sync_res.reason})"
                    else:
                        bt_str = "ByteTrack:(priming)"

                    if c_state:
                        motion_str = c_state.motion_status
                        state_str = f"Congestion={c_state.congestion_level.value:8s} Score={c_state.congestion_score:.2f} Motion={motion_str} Trend={c_state.trend.value}"
                    else:
                        state_str = "State:(none)"

                    if dec:
                        tgt = f" -> {dec.target_gate_id}" if dec.target_gate_id else ""
                        dec_str = f"DECISION: {dec.action.value}{tgt} (Sev: {dec.severity.value}, Conf: {dec.confidence:.2f})"
                    else:
                        dec_str = "DECISION: (evaluating...)"

                    print(f"[{cam_id} -> {gate_id}] {frame_str} | {csr_str}")
                    print(f"               {bt_str}")
                    print(f"               {state_str} | {dec_str}")
                print("-" * 80)

            # Display and save dashboard
            if dashboard_panels:
                full_dashboard = np.vstack(dashboard_panels)
                latest_path = out_dir / "live_dashboard_latest.jpg"
                cv2.imwrite(str(latest_path), full_dashboard)

                if (now - last_save_time) >= 3.0:
                    last_save_time = now
                    saved_snapshots += 1
                    snap_path = out_dir / f"live_snapshot_{saved_snapshots:04d}.jpg"
                    cv2.imwrite(str(snap_path), full_dashboard)

                if can_display:
                    try:
                        cv2.imshow(window_name, full_dashboard)
                        key = cv2.waitKey(20) & 0xFF
                        if key in [ord("q"), ord("Q"), 27]:
                            print("\nUser requested exit (Q/ESC pressed).")
                            break
                    except Exception as e:
                        can_display = False
                        logger.info("GUI display window not available (%s); continuing in file-save mode.", e)

            time.sleep(0.04)

    finally:
        pipeline.stop()
        if can_display:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

    # Final statistics report
    total_time = time.monotonic() - start_time
    final_stats = pipeline.stats()
    sync_stats = synchronizer.stats

    print("\n" + "=" * 90)
    print("PHASE 3 LIVE EXECUTION -- FINAL STATISTICS REPORT")
    print("=" * 90)
    print(f"Total Runtime              : {total_time:.2f} s")
    print(f"Execution Device           : {final_stats['device'].upper()}")
    print(f"GPU VRAM (final)           : {get_vram_mb():.1f} MB")
    print()

    print("--- INGESTION ---")
    for cam_id, cam_stats in final_stats["cameras"].items():
        gate_id = venue.gate_for_camera(cam_id) or "G?"
        print(f"  [{cam_id} -> {gate_id}] Produced: {cam_stats['produced']} | Consumed: {cam_stats['consumed']} | "
              f"Dropped: {cam_stats['dropped']} | Buffer: {cam_stats['buffer_size']}")
    print(f"  Distributor Routed       : {final_stats['distributor']['packets_routed']} packets")

    print()
    print("--- CSRNET INFERENCE ---")
    print(f"  Frames Processed         : {final_stats['csrnet']['frames_processed']}")
    print(f"  Avg Latency              : {final_stats['csrnet']['avg_latency_ms']:.1f} ms")
    csr_fps = (1000.0 / final_stats['csrnet']['avg_latency_ms']) if final_stats['csrnet']['avg_latency_ms'] > 0 else 0
    print(f"  Effective FPS            : {csr_fps:.2f}")

    print()
    print("--- YOLOV8 + BYTETRACK INFERENCE ---")
    print(f"  Frames Processed         : {final_stats['bytetrack']['frames_processed']}")
    print(f"  Avg Latency              : {final_stats['bytetrack']['avg_latency_ms']:.1f} ms")
    bt_fps = (1000.0 / final_stats['bytetrack']['avg_latency_ms']) if final_stats['bytetrack']['avg_latency_ms'] > 0 else 0
    print(f"  Effective FPS            : {bt_fps:.2f}")
    print(f"  Stale Frames Discarded   : {pipeline.bytetrack_worker.stale_frames_discarded}")

    print()
    print("--- TEMPORAL SYNCHRONIZATION (CAUSALITY-ENFORCED) ---")
    total_sync = sync_stats["total_sync_queries"]
    print(f"  Total Synchronization Queries : {total_sync}")
    print(f"  NEW Observations (Fresh)      : {sync_stats['new_count']}")
    print(f"  LAST KNOWN Observations       : {sync_stats['last_known_count']}")
    print(f"  STALE Observations (Excluded) : {sync_stats['stale_count']}")
    print(f"  UNAVAILABLE Observations      : {sync_stats['unavailable_count']}")
    if total_sync > 0:
        fresh_total = sync_stats['new_count'] + sync_stats['last_known_count']
        print(f"  Temporally Valid Motion (OK)  : {fresh_total}/{total_sync} ({100*fresh_total/total_sync:.1f}%)")

    print()
    print("--- DECISION ENGINE SUMMARY ---")
    print(f"  Total Evaluations             : {total_evaluations}")
    for act_name, count in decisions_count_by_action.items():
        print(f"  Action: {act_name:<21s} : {count}")

    print()
    print(f"Live Artifacts Saved            : {out_dir}")
    print("=" * 90)


if __name__ == "__main__":
    main()
