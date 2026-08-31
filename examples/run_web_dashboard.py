"""Web dashboard entry-point — pipeline + FastAPI server in one process."""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from crowd_management.adapters.phase2_adapter import Phase2Adapter
from crowd_management.adapters.synchronizer import TemporalSynchronizer
from crowd_management.config.loader import load_ingestion_config
from crowd_management.decision.config import DecisionConfig
from crowd_management.decision.engine import DecisionEngine
from crowd_management.decision.trend import TrendTracker
from crowd_management.models.contracts import ByteTrackResult, CSRNetResult
from crowd_management.models.manager import ModelManager
from crowd_management.pipeline.fanout_pipeline import FanOutPipeline
from crowd_management.venue.loader import load_venue_topology
from crowd_management.web.server import app
from crowd_management.web.state import dashboard_state


def parse_args():
    p = argparse.ArgumentParser(description="Crowd Management Web Dashboard")
    p.add_argument("--config", default="config/ingestion/hajj_ingestion.yaml")
    p.add_argument("--venue",  default="config/venue/circular_venue_live_scaled.yaml")
    p.add_argument("--host",   default="0.0.0.0")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--stale-frames", type=int, default=10)
    return p.parse_args()


def run_pipeline(args):
    config = load_ingestion_config(args.config)
    venue  = load_venue_topology(args.venue)

    decision_config  = DecisionConfig()
    trend_tracker    = TrendTracker(decision_config)
    synchronizer     = TemporalSynchronizer(stale_frame_gap=args.stale_frames)
    adapter          = Phase2Adapter(decision_config)
    decision_engine  = DecisionEngine(venue, decision_config)
    model_mgr        = ModelManager(models_config=config.models)

    dashboard_state.device = model_mgr.device
    dashboard_state.pipeline_running = True

    # Initialise per-camera state
    for cam_cfg in config.cameras:
        if not cam_cfg.enabled:
            continue
        gate_id = venue.gate_for_camera(cam_cfg.camera_id) or "G?"
        cap     = venue.get_gate_capacity(gate_id) if gate_id != "G?" else 500
        dashboard_state.init_camera(cam_cfg.camera_id, gate_id, cap)

    latest_crowd_states = {}

    def on_csrnet(res: CSRNetResult):
        cam_id  = res.camera_id
        gate_id = venue.gate_for_camera(cam_id) or "G?"
        cap     = venue.get_gate_capacity(gate_id) if gate_id != "G?" else 500

        dashboard_state.update_csrnet(cam_id, res)

        sync_res    = synchronizer.synchronize(res, stale_frame_gap=args.stale_frames)
        trend       = trend_tracker.get_trend(gate_id)
        crowd_state = adapter.fuse(
            csrnet_result=res,
            bytetrack_result=sync_res.bytetrack_result,
            gate_id=gate_id,
            gate_capacity=cap,
            trend=trend,
            stale_frame_gap=args.stale_frames,
            sync_result=sync_res,
        )
        latest_crowd_states[gate_id] = crowd_state
        trend_tracker.update(gate_id, crowd_state.congestion_score)
        dashboard_state.update_crowd_state(cam_id, gate_id, crowd_state)

        decisions = decision_engine.evaluate(latest_crowd_states)
        for dec in decisions:
            src_cam = venue.camera_for_gate(dec.source_gate_id)
            if src_cam:
                dashboard_state.update_decision(src_cam, dec)

    def on_bytetrack(res: ByteTrackResult):
        synchronizer.add_bytetrack_result(res)
        dashboard_state.update_bytetrack(res.camera_id, res)

    pipeline = FanOutPipeline(
        config=config,
        model_manager=model_mgr,
        csrnet_callback=on_csrnet,
        bytetrack_callback=on_bytetrack,
        queue_capacity=config.ingestion.buffer_size,
    )
    pipeline.start()

    try:
        while dashboard_state.pipeline_running:
            time.sleep(1.0)
    finally:
        pipeline.stop()


def main():
    args = parse_args()

    # Start pipeline in background thread
    t = threading.Thread(target=run_pipeline, args=(args,), daemon=True)
    t.start()

    print(f"\n{'='*60}")
    print("  Crowd Management Web Dashboard")
    print(f"  Open: http://localhost:{args.port}")
    print(f"{'='*60}\n")

    def shutdown(sig, frame):
        dashboard_state.pipeline_running = False
        sys.exit(0)
    signal.signal(signal.SIGINT, shutdown)

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
