"""Run the live multi-camera Phase 2 Fan-Out pipeline on real HAJJv2 video streams."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd_management.config.loader import load_ingestion_config
from crowd_management.models.contracts import CSRNetResult, RAFTResult
from crowd_management.models.manager import ModelManager
from crowd_management.pipeline.fanout_pipeline import FanOutPipeline
from crowd_management.utils.visualization import draw_phase2_hud, render_density_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase 2 Fan-Out CSRNet + RAFT pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/ingestion/hajj_ingestion.yaml",
        help="Path to ingestion YAML config",
    )
    parser.add_argument("--max-frames", type=int, default=15, help="Maximum frames to process per camera")
    parser.add_argument("--max-seconds", type=float, default=20.0, help="Maximum pipeline run duration")
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/phase_2/demo_stream",
        help="Directory to save periodic visualization frames",
    )
    return parser.parse_args()


def print_header(config) -> None:
    print("=" * 75)
    print("PHASE 2 — MULTI-CAMERA FAN-OUT PIPELINE (CSRNET + RAFT)")
    print(f"Dataset : {config.dataset.name}")
    print("=" * 75)


def run_pipeline_demo(config_path: str, max_frames: int, max_seconds: float, out_dir: str) -> None:
    config = load_ingestion_config(config_path)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print_header(config)

    # Latest results per camera for synchronized HUD drawing
    latest_csrnet: dict[str, CSRNetResult] = {}
    latest_raft: dict[str, RAFTResult] = {}
    saved_frames: dict[str, int] = {cam.camera_id: 0 for cam in config.cameras if cam.enabled}
    processed_counts: dict[str, int] = {cam.camera_id: 0 for cam in config.cameras if cam.enabled}

    def on_csrnet_result(res: CSRNetResult) -> None:
        latest_csrnet[res.camera_id] = res
        processed_counts[res.camera_id] = processed_counts.get(res.camera_id, 0) + 1
        raft_res = latest_raft.get(res.camera_id)

        print(
            f"[{res.camera_id}] Frame {res.frame_id:03d} (t={res.video_timestamp:.2f}s) | "
            f"CSRNet Count: {res.crowd_count:5.1f} ({res.inference_time_ms:.0f}ms) | "
            + (
                f"RAFT: {raft_res.average_velocity:.2f} px/f, {raft_res.dominant_direction_deg:.0f}° ({raft_res.inference_time_ms:.0f}ms)"
                if raft_res
                else "RAFT: (priming...)"
            )
        )

        # Save periodic demo visualizations
        if res.frame_id in [1, 5, 10, 15] and saved_frames[res.camera_id] < res.frame_id:
            saved_frames[res.camera_id] = res.frame_id
            base_frame = res.frame if res.frame is not None else np.zeros((res.input_height, res.input_width, 3), dtype=np.uint8)
            hud_frame = render_density_overlay(base_frame, res.density_map, alpha=0.35)
            hud_frame = draw_phase2_hud(hud_frame, res, raft_res)
            save_file = output_path / f"{res.camera_id}_frame_{res.frame_id:04d}_phase2.jpg"
            cv2.imwrite(str(save_file), hud_frame)
            print(f"  [SAVED DEMO OVERLAY] {save_file.name}")

    def on_raft_result(res: RAFTResult) -> None:
        latest_raft[res.camera_id] = res

    # Initialize shared model manager
    model_mgr = ModelManager(
        device="cpu",
        csrnet_resolution=(720, 540),
        raft_model_size="large",
        raft_resolution=(512, 384),
    )

    pipeline = FanOutPipeline(
        config=config,
        model_manager=model_mgr,
        csrnet_callback=on_csrnet_result,
        raft_callback=on_raft_result,
        queue_capacity=15,
    )

    print("\nStarting ingestion and inference workers...")
    pipeline.start()

    start_time = time.monotonic()
    deadline = start_time + max_seconds

    # Save periodic annotated frames
    while time.monotonic() < deadline:
        all_reached = all(count >= max_frames for count in processed_counts.values())
        if all_reached:
            break

        time.sleep(0.05)

    pipeline.stop()
    total_elapsed = time.monotonic() - start_time

    print("\n" + "=" * 75)
    print("PHASE 2 PIPELINE EXECUTION SUMMARY")
    print("=" * 75)
    stats = pipeline.stats()
    print(f"Total Elapsed Time : {total_elapsed:.2f} s")
    print(f"Inference Device   : {stats['device']}")
    print(f"CSRNet Processed   : {stats['csrnet']['frames_processed']} frames (Avg: {stats['csrnet']['avg_latency_ms']:.1f} ms)")
    print(f"RAFT Processed     : {stats['raft']['pairs_processed']} pairs (Avg: {stats['raft']['avg_latency_ms']:.1f} ms)")
    print(f"Distributor Routed : {stats['distributor']['packets_routed']} packets")
    print(f"Queue Status       : CSRNet Queue={stats['distributor']['csrnet_queue_size']}, RAFT Queue={stats['distributor']['raft_queue_size']}")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    run_pipeline_demo(args.config, args.max_frames, args.max_seconds, args.out_dir)
