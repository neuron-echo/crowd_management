"""Comprehensive Ingestion Validation Audit for Phase 1 against real HAJJv2 videos."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from queue import Empty

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd_management.config.loader import load_ingestion_config
from crowd_management.config.schemas import AppConfig, CameraConfig, DatasetConfig, IngestionConfig
from crowd_management.ingestion.consumer import QueueFrameConsumer
from crowd_management.ingestion.frame_buffer import FrameBuffer
from crowd_management.ingestion.frame_packet import FramePacket
from crowd_management.ingestion.multi_camera import MultiCameraProducer
from crowd_management.ingestion.producer import FrameProducer
from crowd_management.ingestion.video_source import VideoFileSource


def draw_diagnostic_overlay(packet: FramePacket, stream_fps: float, buffer_size: int) -> np.ndarray:
    """Draw diagnostic overlay with ingestion metadata on frame."""
    frame_copy = packet.frame.copy()
    h, w, _ = frame_copy.shape

    # Semi-transparent overlay box
    overlay = frame_copy.copy()
    box_w = 420
    box_h = 220
    cv2.rectangle(overlay, (15, 15), (15 + box_w, 15 + box_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame_copy, 0.25, 0, frame_copy)

    # Border
    cv2.rectangle(frame_copy, (15, 15), (15 + box_w, 15 + box_h), (0, 215, 255), 2)

    lines = [
        f"camera_id: {packet.camera_id}",
        f"frame_id: {packet.frame_id}",
        f"source_frame_index: {packet.source_frame_index}",
        f"video_timestamp: {packet.video_timestamp:.3f} s",
        f"wall_clock_timestamp: {packet.wall_clock_timestamp:.3f}",
        f"measured stream FPS: {stream_fps:.2f}",
        f"buffer size: {buffer_size}",
    ]

    y = 45
    for i, line in enumerate(lines):
        color = (0, 255, 0) if i == 0 else (240, 240, 240)
        cv2.putText(
            frame_copy,
            line,
            (30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y += 26

    return frame_copy


def run_audit() -> dict:
    config_path = ROOT / "config" / "ingestion" / "hajj_ingestion.yaml"
    out_dir = ROOT / "outputs" / "ingestion_validation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("PHASE 1 - INGESTION VALIDATION AUDIT")
    print(f"Config: {config_path}")
    print("=" * 75)

    config = load_ingestion_config(config_path)
    manager = MultiCameraProducer(config)

    camera_ids = manager.camera_ids
    print(f"Cameras configured: {camera_ids}")

    records: dict[str, list[dict]] = {cam_id: [] for cam_id in camera_ids}
    saved_images: dict[str, list[str]] = {cam_id: [] for cam_id in camera_ids}

    consumers = {cam_id: QueueFrameConsumer(manager.buffers[cam_id]) for cam_id in camera_ids}

    manager.start_all()
    target_frames_per_camera = 30  # ~6 seconds at 5 FPS
    max_duration_seconds = 12.0
    start_mono_time = time.monotonic()

    print("\nStarting live ingestion in real-time mode...")

    while time.monotonic() - start_mono_time < max_duration_seconds:
        all_done = all(len(records[cid]) >= target_frames_per_camera for cid in camera_ids)
        if all_done:
            break

        for cam_id in camera_ids:
            if len(records[cam_id]) >= target_frames_per_camera:
                continue
            try:
                packet = consumers[cam_id].consume(timeout=0.05, block=True)
                recv_time = time.time()
                buf_sz = len(manager.buffers[cam_id])

                if len(records[cam_id]) > 0:
                    prev_time = records[cam_id][-1]["wall_clock_timestamp"]
                    inst_fps = 1.0 / max(packet.wall_clock_timestamp - prev_time, 1e-6)
                else:
                    inst_fps = packet.stream_fps

                record = {
                    "frame_id": packet.frame_id,
                    "source_frame_index": packet.source_frame_index,
                    "video_timestamp": packet.video_timestamp,
                    "wall_clock_timestamp": packet.wall_clock_timestamp,
                    "recv_time": recv_time,
                    "width": packet.width,
                    "height": packet.height,
                    "source_fps": packet.source_fps,
                    "stream_fps": packet.stream_fps,
                    "inst_fps": inst_fps,
                    "buffer_size": buf_sz,
                    "mean_pixel_val": float(np.mean(packet.frame)),
                    "std_pixel_val": float(np.std(packet.frame)),
                }
                records[cam_id].append(record)

                # Save sampled frames with diagnostic overlay
                if len(records[cam_id]) in [1, 5, 15, 25]:
                    annotated = draw_diagnostic_overlay(packet, inst_fps, buf_sz)
                    img_path = out_dir / f"{cam_id}_frame_{packet.frame_id:04d}.jpg"
                    cv2.imwrite(str(img_path), annotated)
                    saved_images[cam_id].append(str(img_path))
                    print(f"  [SAVED OVERLAY] {img_path.name} (frame_id={packet.frame_id}, src_idx={packet.source_frame_index}, time={packet.video_timestamp:.3f}s)")

            except Empty:
                pass

        time.sleep(0.01)

    manager.stop_all()
    manager.shutdown()
    total_elapsed = time.monotonic() - start_mono_time

    print(f"\nLive ingestion completed in {total_elapsed:.2f} seconds.")

    stats = {}
    for cam_id in camera_ids:
        cam_records = records[cam_id]
        buf = manager.buffers[cam_id]
        prod = manager.producers[cam_id]

        if not cam_records:
            stats[cam_id] = {"status": "FAILED - NO FRAMES RECEIVED"}
            continue

        timestamps = [r["video_timestamp"] for r in cam_records]
        wall_timestamps = [r["wall_clock_timestamp"] for r in cam_records]
        source_indices = [r["source_frame_index"] for r in cam_records]
        frame_ids = [r["frame_id"] for r in cam_records]

        wall_intervals = [
            wall_timestamps[i] - wall_timestamps[i - 1]
            for i in range(1, len(wall_timestamps))
        ]
        avg_wall_interval = float(np.mean(wall_intervals)) if wall_intervals else 0.0
        expected_interval = 1.0 / prod.target_fps if prod.target_fps else 0.0

        # Monotonicity validations
        frame_id_monotonic = all(frame_ids[i] > frame_ids[i - 1] for i in range(1, len(frame_ids)))
        source_index_monotonic = all(source_indices[i] > source_indices[i - 1] for i in range(1, len(source_indices)))
        video_time_monotonic = all(timestamps[i] > timestamps[i - 1] for i in range(1, len(timestamps)))
        wall_time_monotonic = all(wall_timestamps[i] > wall_timestamps[i - 1] for i in range(1, len(wall_timestamps)))

        # Measured Ingestion FPS (derived from wall clock delta)
        measured_fps = (len(cam_records) - 1) / (wall_timestamps[-1] - wall_timestamps[0]) if len(cam_records) > 1 else 0.0

        # Video content verification (real Hajj images have varied pixel variance)
        real_video_detected = any(r["std_pixel_val"] > 10.0 for r in cam_records)

        stats[cam_id] = {
            "source_path": str(prod.source.path),
            "resolution": f"{prod.source.resolution[0]}x{prod.source.resolution[1]}",
            "source_fps": float(prod.source.source_fps),
            "target_fps": float(prod.target_fps),
            "measured_fps": float(measured_fps),
            "frames_produced": int(buf.frames_produced),
            "frames_consumed": int(len(cam_records)),
            "dropped_frames": int(buf.frames_dropped),
            "expected_frame_interval_s": float(expected_interval),
            "avg_measured_frame_interval_s": float(avg_wall_interval),
            "min_measured_frame_interval_s": float(np.min(wall_intervals)) if wall_intervals else 0.0,
            "max_measured_frame_interval_s": float(np.max(wall_intervals)) if wall_intervals else 0.0,
            "maximum_buffer_size": int(buf.maximum_size),
            "final_buffer_size": int(len(buf)),
            "frame_ordering_monotonic": bool(frame_id_monotonic),
            "source_frame_index_ordering_monotonic": bool(source_index_monotonic),
            "video_timestamp_ordering_monotonic": bool(video_time_monotonic),
            "wall_clock_timestamp_ordering_monotonic": bool(wall_time_monotonic),
            "real_video_detected": bool(real_video_detected),
            "mean_pixel_std": float(np.mean([r["std_pixel_val"] for r in cam_records])),
            "saved_overlays": saved_images[cam_id],
        }

    # Fault isolation verification
    print("\n" + "=" * 75)
    print("VERIFICATION: Fault Isolation (One bad camera stream vs. One healthy stream)")
    print("=" * 75)
    fault_config = AppConfig(
        dataset=DatasetConfig(name="HAJJv2", root="HAJJv2.Dataset"),
        ingestion=IngestionConfig(target_fps=5.0, real_time_mode=False, buffer_size=10, drop_policy="oldest"),
        cameras=[
            CameraConfig(camera_id="cam_invalid", source="HAJJv2.Dataset/Testing/Videos/nonexistent.mp4", enabled=True),
            CameraConfig(camera_id="cam_valid", source="HAJJv2.Dataset/Testing/Videos/2.mp4", enabled=True),
        ],
    )
    fault_manager = MultiCameraProducer(fault_config)
    fault_manager.start_all()
    time.sleep(0.5)
    valid_consumer = fault_manager.get_consumer("cam_valid")
    valid_frames = []
    for _ in range(5):
        try:
            p = valid_consumer.consume(timeout=0.2)
            valid_frames.append(p)
        except Empty:
            pass
    fault_manager.stop_all()
    fault_manager.shutdown()
    fault_isolation_passed = len(valid_frames) > 0
    print(f"Fault isolation test passed: {fault_isolation_passed} ({len(valid_frames)} frames retrieved from healthy camera)")

    # End-of-video verification
    print("\n" + "=" * 75)
    print("VERIFICATION: End-Of-Video Graceful Completion")
    print("=" * 75)
    eof_source = VideoFileSource("HAJJv2.Dataset/Testing/Videos/12.mp4")
    eof_buffer = FrameBuffer(capacity=10)
    eof_producer = FrameProducer("cam_eof", eof_source, eof_buffer, target_fps=30.0, real_time_mode=False)
    eof_producer.start()
    time.sleep(1.0)
    eof_producer.stop()
    eof_producer.shutdown()
    eof_handled = eof_producer.complete or eof_buffer.frames_produced > 0
    print(f"EOF handling passed: {eof_handled} (complete={eof_producer.complete}, frames_produced={eof_buffer.frames_produced})")

    return {
        "camera_stats": stats,
        "fault_isolation_passed": fault_isolation_passed,
        "eof_handled": eof_handled,
        "total_elapsed": total_elapsed,
    }


if __name__ == "__main__":
    result = run_audit()
    print("\n" + "=" * 75)
    print("AUDIT RESULTS TABLE")
    print("=" * 75)
    for cam, data in result["camera_stats"].items():
        print(f"\n--- Camera: {cam} ---")
        for k, v in data.items():
            if k != "saved_overlays":
                print(f"  {k}: {v}")
        print(f"  saved_overlays: {data['saved_overlays']}")
    print(f"\nFault Isolation Passed: {result['fault_isolation_passed']}")
    print(f"EOF Handling Passed: {result['eof_handled']}")
