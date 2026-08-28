"""Standalone RAFT Optical Flow Validation Utility on Real HAJJv2 Streams & Synthetic Translations."""

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
from crowd_management.models.raft.infer import RAFTInference
from crowd_management.utils.visualization import (
    render_density_heatmap,
    render_flow_hsv,
    render_quiver_arrows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate RAFT optical flow on real Hajj video streams")
    parser.add_argument(
        "--video",
        type=str,
        default="HAJJv2.Dataset/Testing/Videos/2.mp4",
        help="Path to real Hajj video file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/phase_2/raft_validation",
        help="Output directory for saved visualizations",
    )
    parser.add_argument("--num-pairs", type=int, default=4, help="Number of consecutive pairs to evaluate")
    parser.add_argument("--stride", type=int, default=5, help="Stride between sampled frames")
    return parser.parse_args()


def run_controlled_sanity_tests(engine: RAFTInference, out_dir: Path) -> dict[str, bool]:
    """Run controlled synthetic translation tests with known directional displacement."""
    print("\n" + "=" * 75)
    print("CONTROLLED MATHEMATICAL SANITY TESTS (SYNTHETIC BENCHMARK):")
    print("=" * 75)

    synth_dir = out_dir / "synthetic_controlled"
    synth_dir.mkdir(parents=True, exist_ok=True)

    # Textured background pattern
    x = np.linspace(0, 4 * np.pi, 256)
    y = np.linspace(0, 4 * np.pi, 256)
    xx, yy = np.meshgrid(x, y)
    pattern = (np.sin(xx) * np.cos(yy) * 127.0 + 128.0).astype(np.uint8)
    base_frame = cv2.cvtColor(pattern, cv2.COLOR_GRAY2BGR)

    shifts = {
        "East (+X shift)": (8, 0, 0.0),
        "South (+Y shift)": (0, 8, 90.0),
        "West (-X shift)": (-8, 0, 180.0),
        "North (-Y shift)": (0, -8, 270.0),
        "South-East (+X, +Y shift)": (8, 8, 45.0),
        "Static (0 shift)": (0, 0, None),
    }

    test_results = {}

    for name, (dx, dy, exp_angle) in shifts.items():
        M = np.float32([[1, 0, dx], [0, 1, dy]])
        shifted = cv2.warpAffine(base_frame, M, (256, 256))

        p1 = FramePacket("test_cam", 1, 0, 0.0, time.time(), base_frame, 25.0, 5.0, 256, 256)
        p2 = FramePacket("test_cam", 2, 5, 0.2, time.time(), shifted, 25.0, 5.0, 256, 256)

        res = engine.estimate_flow(p1, p2)

        if exp_angle is not None:
            diff = abs(res.dominant_direction_deg - exp_angle)
            if diff > 180.0:
                diff = 360.0 - diff
            passed = diff < 30.0 and res.average_velocity > 1.0 and res.flow_consistency > 0.6
            status = "PASS" if passed else "FAIL"
            print(f"  {name:28s} -> Measured: {res.dominant_direction_deg:5.1f}°, Expected: {exp_angle:5.1f}°, Consistency: {res.flow_consistency:.2f} [{status}]")
        else:
            passed = res.average_velocity < 1.0
            status = "PASS" if passed else "FAIL"
            print(f"  {name:28s} -> Measured Avg Velocity: {res.average_velocity:5.2f} px/frame [{status}]")

        test_results[name] = passed

    return test_results


def run_validation(video_path: str, out_dir: str, num_pairs: int, stride: int) -> None:
    v_path = Path(video_path)
    o_dir = Path(out_dir)
    o_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 2 — RAFT OPTICAL FLOW REAL-HAJJ VALIDATION")
    print(f"Video Source : {v_path}")
    print(f"Output Dir   : {o_dir}")
    print("=" * 75)

    engine = RAFTInference(
        device="cpu",
        model_size="large",
        inference_resolution=(512, 384),
    )
    print(f"RAFT initialized on device: {engine.device} (Inference Resolution: 512x384)")

    # 1. Run Controlled Sanity Tests
    sanity_results = run_controlled_sanity_tests(engine, o_dir)

    # 2. Run Real Hajj Video Evaluation
    print("\n" + "=" * 75)
    print("REAL HAJJ VIDEO OPTICAL FLOW INFERENCE:")
    print("=" * 75)

    source = VideoFileSource(v_path)
    source.open()
    print(f"Video opened: {source.resolution[0]}x{source.resolution[1]} @ {source.source_fps:.2f} FPS")

    sampled_packets = []
    while len(sampled_packets) < (num_pairs + 1):
        frame, s_idx = source.read_next()
        if frame is None or s_idx is None:
            break
        if s_idx % stride == 0:
            pkt = FramePacket(
                camera_id="cam_01",
                frame_id=len(sampled_packets) + 1,
                source_frame_index=int(s_idx),
                video_timestamp=source.timestamp_for_index(int(s_idx)),
                wall_clock_timestamp=time.time(),
                frame=frame,
                source_fps=source.source_fps,
                stream_fps=source.source_fps / float(stride),
                width=int(source.resolution[0]),
                height=int(source.resolution[1]),
            )
            sampled_packets.append(pkt)

    source.close()

    real_results = []
    for i in range(len(sampled_packets) - 1):
        p1 = sampled_packets[i]
        p2 = sampled_packets[i + 1]

        res = engine.estimate_flow(p1, p2)
        real_results.append(res)

        # Generate visualizations
        # We compute flow directly for detailed visual inspection
        import torch
        img1, (pw, ph), (orig_w, orig_h) = engine._preprocess_frame(p1.frame)
        img2, _, _ = engine._preprocess_frame(p2.frame)
        with torch.no_grad():
            with_torch = engine.model(img1, img2)[-1].squeeze(0).detach().cpu().numpy()
        u = with_torch[0] * (orig_w / float(pw))
        v = with_torch[1] * (orig_h / float(ph))

        flow_hsv = render_flow_hsv(u, v)
        flow_hsv_resized = cv2.resize(flow_hsv, (orig_w, orig_h))

        mag = np.sqrt(u**2 + v**2)
        vel_heatmap = render_density_heatmap(mag, target_size=(orig_w, orig_h))

        quiver_overlay = render_quiver_arrows(p2.frame, u, v, step=48, scale=2.5, min_mag=0.5)

        # Save artifacts
        pair_name = f"pair_{p1.frame_id:04d}_to_{p2.frame_id:04d}"
        cv2.imwrite(str(o_dir / f"{pair_name}_frame_t.jpg"), p1.frame)
        cv2.imwrite(str(o_dir / f"{pair_name}_frame_tplus1.jpg"), p2.frame)
        cv2.imwrite(str(o_dir / f"{pair_name}_flow_hsv.jpg"), flow_hsv_resized)
        cv2.imwrite(str(o_dir / f"{pair_name}_vel_heatmap.jpg"), vel_heatmap)
        cv2.imwrite(str(o_dir / f"{pair_name}_quiver.jpg"), quiver_overlay)

        print(
            f"  [PAIR {p1.frame_id}->{p2.frame_id}] Velocity: Avg={res.average_velocity:.2f} px/f, "
            f"Max={res.max_velocity:.2f} px/f | Dominant Dir={res.dominant_direction_deg:.1f}° | "
            f"Consistency={res.flow_consistency:.2f} | Latency={res.inference_time_ms:.1f} ms"
        )

    # Summary
    print("\n" + "=" * 75)
    print("RAFT VALIDATION SUMMARY:")
    print("=" * 75)
    avg_lat = np.mean([r.inference_time_ms for r in real_results])
    all_passed_sanity = all(sanity_results.values())
    print(f"Controlled Translation Tests: {'ALL PASS' if all_passed_sanity else 'SOME FAIL'}")
    print(f"Evaluated Real Hajj Pairs   : {len(real_results)}")
    print(f"Average Inference Latency   : {avg_lat:.1f} ms ({1000.0 / avg_lat:.2f} FPS on CPU)")
    print(f"Visual Artifacts Saved      : {o_dir}")
    print("=" * 75)


if __name__ == "__main__":
    args = parse_args()
    run_validation(args.video, args.out_dir, args.num_pairs, args.stride)
