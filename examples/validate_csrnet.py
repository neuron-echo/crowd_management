"""Standalone CSRNet Validation Utility on Real HAJJv2 Video Streams."""

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

from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.ingestion.video_source import VideoFileSource
from crowd_management.models.csrnet.infer import CSRNetInference
from crowd_management.utils.gt_loader import HajjGroundTruth
from crowd_management.utils.visualization import render_density_heatmap, render_density_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate CSRNet crowd density estimation on real Hajj videos")
    parser.add_argument(
        "--video",
        type=str,
        default="HAJJv2.Dataset/Testing/Videos/2.mp4",
        help="Path to real Hajj video file",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="HAJJv2.Dataset/Testing/Labels/HajjDataset_V2_Test.csv",
        help="Path to Hajj ground-truth annotation CSV",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/phase_2/csrnet_validation",
        help="Output directory for saved visualizations",
    )
    parser.add_argument("--num-frames", type=int, default=6, help="Number of sampled frames to evaluate")
    parser.add_argument("--stride", type=int, default=15, help="Sampling stride between frames")
    return parser.parse_args()


def run_validation(video_path: str, labels_path: str, out_dir: str, num_frames: int, stride: int) -> None:
    v_path = Path(video_path)
    l_path = Path(labels_path)
    o_dir = Path(out_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 2 — CSRNET REAL-HAJJ VALIDATION AUDIT")
    print(f"Video Source : {v_path}")
    print(f"Labels File  : {l_path}")
    print(f"Output Dir   : {o_dir}")
    print("=" * 75)

    gt = HajjGroundTruth(l_path) if l_path.exists() else None
    if gt and gt.total_annotated_frames > 0:
        print(f"Loaded ground-truth annotations: {gt.total_annotated_frames} annotated frames in dataset.")
    else:
        print("No ground-truth annotations found or empty dataset; running visual & sanity validation.")

    engine = CSRNetInference(
        device="cpu",
        inference_resolution=(720, 540),
    )
    print(f"CSRNet initialized on device: {engine.device} (Inference Resolution: 720x540)")

    source = VideoFileSource(v_path)
    source.open()
    print(f"Video opened: {source.resolution[0]}x{source.resolution[1]} @ {source.source_fps:.2f} FPS ({source.frame_count} total frames)")

    results = []
    evaluated_packets = []
    frame_idx = 0
    sampled_count = 0

    while sampled_count < num_frames:
        frame, s_idx = source.read_next()
        if frame is None or s_idx is None:
            break

        if s_idx % stride == 0:
            sampled_count += 1
            pkt = FramePacket(
                camera_id="cam_01",
                frame_id=sampled_count,
                source_frame_index=int(s_idx),
                video_timestamp=source.timestamp_for_index(int(s_idx)),
                wall_clock_timestamp=time.time(),
                frame=frame,
                source_fps=source.source_fps,
                stream_fps=source.source_fps / float(stride),
                width=int(source.resolution[0]),
                height=int(source.resolution[1]),
            )
            evaluated_packets.append(pkt)

            res = engine.estimate(pkt)
            results.append((pkt, res))

            # Render and save visuals
            heatmap = render_density_heatmap(res.density_map, target_size=(pkt.width, pkt.height))
            overlay = render_density_overlay(pkt.frame, res.density_map, alpha=0.45)

            # Draw HUD on overlay
            gt_count = gt.get_count_for_frame(int(s_idx) + 1) if gt else None
            gt_str = f" | GT Count: {gt_count}" if gt_count is not None else ""

            hud_text = f"Cam: {res.camera_id} | Frame {res.frame_id} (Src {res.source_frame_index}) | Count: {res.crowd_count:.1f}{gt_str}"
            cv2.putText(overlay, hud_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            # Save individual files
            base_name = f"frame_{res.frame_id:04d}_src_{res.source_frame_index:04d}"
            cv2.imwrite(str(o_dir / f"{base_name}_orig.jpg"), pkt.frame)
            cv2.imwrite(str(o_dir / f"{base_name}_heatmap.jpg"), heatmap)
            cv2.imwrite(str(o_dir / f"{base_name}_overlay.jpg"), overlay)

            print(f"  [SAVED] {base_name}: Predicted={res.crowd_count:.1f} people (Inference: {res.inference_time_ms:.1f} ms){gt_str}")

    source.close()

    # Sanity Checks
    print("\n" + "=" * 75)
    print("SANITY & INTEGRITY CHECKS:")
    print("=" * 75)

    # 1. Determinism
    det_res1 = engine.estimate(evaluated_packets[0])
    det_res2 = engine.estimate(evaluated_packets[0])
    det_pass = np.isclose(det_res1.crowd_count, det_res2.crowd_count, atol=1e-5)
    print(f"1. Determinism (Same frame -> same prediction): {'PASS' if det_pass else 'FAIL'}")

    # 2. Finite values
    all_finite = all(np.all(np.isfinite(r.density_map)) and np.isfinite(r.crowd_count) for _, r in results)
    print(f"2. Finite Values (No NaN / Inf): {'PASS' if all_finite else 'FAIL'}")

    # 3. Non-negative counts
    all_non_neg = all(r.crowd_count >= 0.0 for _, r in results)
    print(f"3. Non-Negative Crowd Counts: {'PASS' if all_non_neg else 'FAIL'}")

    # 4. Density Map Dimensions
    valid_dims = all(r.density_map.ndim == 2 and r.density_map_width > 0 and r.density_map_height > 0 for _, r in results)
    print(f"4. Valid Density Map Dimensions: {'PASS' if valid_dims else 'FAIL'} (Shape: {results[0][1].density_map_width}x{results[0][1].density_map_height})")

    # 5. Dynamic Response
    counts = [r.crowd_count for _, r in results]
    dynamic_response = len(set(np.round(counts, 2))) > 1
    print(f"5. Dynamic Response (Predictions respond across different video scenes): {'PASS' if dynamic_response else 'FAIL'} (Range: {min(counts):.1f} - {max(counts):.1f})")

    # Summary
    print("\n" + "=" * 75)
    print("CSRNET VALIDATION SUMMARY:")
    print("=" * 75)
    avg_latency = np.mean([r.inference_time_ms for _, r in results])
    print(f"Evaluated Frames        : {len(results)}")
    print(f"Average Inference Time  : {avg_latency:.1f} ms ({1000.0 / avg_latency:.2f} FPS)")
    print(f"Predicted Count Mean    : {np.mean(counts):.1f} ± {np.std(counts):.1f}")
    print(f"Visual Artifacts Saved  : {o_dir}")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    run_validation(args.video, args.labels, args.out_dir, args.num_frames, args.stride)
