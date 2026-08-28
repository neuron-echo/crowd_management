"""Run the ingestion subsystem against configured camera sources."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from queue import Empty

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd_management.config.loader import load_ingestion_config
from crowd_management.ingestion.consumer import QueueFrameConsumer
from crowd_management.ingestion.multi_camera import MultiCameraProducer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the HAJJ ingestion pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to the ingestion YAML config")
    parser.add_argument("--max-frames", type=int, default=20, help="Maximum frames to display per camera")
    parser.add_argument("--max-seconds", type=float, default=10.0, help="Maximum seconds to run the ingestion session")
    return parser.parse_args()


def print_header() -> None:
    print("============================================================")
    print("INGESTION STARTED")
    print("============================================================")


def print_camera_summary(camera_id: str, metadata: dict) -> None:
    src = Path(metadata["source"]).name
    print(f"\nCamera: {camera_id}")
    print(f"Source: {src}")
    print(f"Source FPS: {metadata['source_fps']:.2f}")
    print(f"Target FPS: {metadata['target_fps']:.2f}")
    print(f"Resolution: {metadata['resolution'][0]}x{metadata['resolution'][1]}")
    print(f"Buffer: {metadata['buffer_size']} | drop_policy={metadata['drop_policy']}")


def run_ingestion(config_path: str, max_frames: int, max_seconds: float) -> None:
    config = load_ingestion_config(config_path)
    manager = MultiCameraProducer(config)

    print_header()
    print(f"Dataset: {config.dataset.name}")
    metadata = manager.camera_metadata()
    for camera_id in manager.camera_ids:
        print_camera_summary(camera_id, metadata[camera_id])

    consumers = {camera_id: QueueFrameConsumer(manager.buffers[camera_id]) for camera_id in manager.camera_ids}
    manager.start_all()
    end_time = time.monotonic() + max_seconds
    counts = {camera_id: 0 for camera_id in manager.camera_ids}

    while time.monotonic() < end_time:
        any_packet = False
        for camera_id in manager.camera_ids:
            if counts[camera_id] >= max_frames:
                continue
            try:
                packet = consumers[camera_id].consume(timeout=0.05, block=False)
            except Empty:
                continue

            counts[camera_id] += 1
            any_packet = True
            buffer_size = len(manager.buffers[camera_id])
            print(
                f"{camera_id}\n"
                f"Frame ID: {packet.frame_id}\n"
                f"Source Frame: {packet.source_frame_index}\n"
                f"Video Time: {packet.video_timestamp:.3f} s\n"
                f"Wall Time: {packet.wall_clock_timestamp:.3f}\n"
                f"Buffer: {buffer_size}"
            )

        if not any_packet and all(producer.complete for producer in manager.producers.values()):
            break
        time.sleep(0.02)

    manager.stop_all()
    manager.shutdown()

    print("\n============================================================")
    print("INGESTION SUMMARY")
    print("============================================================")
    for camera_id in manager.camera_ids:
        producer = manager.producers[camera_id]
        buffer = manager.buffers[camera_id]
        print(f"Camera: {camera_id}")
        print(f"Source FPS: {producer.source.source_fps:.2f}")
        print(f"Target FPS: {producer.target_fps:.2f}")
        print(f"Frames produced: {buffer.frames_produced}")
        print(f"Frames consumed: {buffer.frames_consumed}")
        print(f"Frames dropped: {buffer.frames_dropped}")
        print(f"Buffer maximum: {buffer.maximum_size}")
        print()


if __name__ == "__main__":
    args = parse_args()
    run_ingestion(args.config, args.max_frames, args.max_seconds)
