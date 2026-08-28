"""Unified Fan-Out processing pipeline orchestrator for Phase 2."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from queue import Empty
from typing import Any, Callable

from crowd_management.config.loader import load_ingestion_config
from crowd_management.config.schemas import AppConfig
from crowd_management.ingestion.multi_camera import MultiCameraProducer
from crowd_management.models.contracts import CSRNetResult, RAFTResult
from crowd_management.models.manager import ModelManager
from crowd_management.pipeline.csrnet_worker import CSRNetWorker
from crowd_management.pipeline.distributor import FrameDistributor
from crowd_management.pipeline.raft_worker import RAFTWorker

logger = logging.getLogger(__name__)


class FanOutPipeline:
    """Coordinates MultiCameraProducer ingestion, fan-out distribution,

    and parallel CSRNet + RAFT model inference workers.
    """

    def __init__(
        self,
        config: AppConfig | str | Path,
        model_manager: ModelManager | None = None,
        csrnet_callback: Callable[[CSRNetResult], None] | None = None,
        raft_callback: Callable[[RAFTResult], None] | None = None,
        csrnet_resolution: tuple[int, int] | None = (720, 540),
        raft_resolution: tuple[int, int] | None = (512, 384),
        queue_capacity: int = 15,
    ) -> None:
        if isinstance(config, (str, Path)):
            config = load_ingestion_config(config)
        self.config = config

        # Upstream multi-camera ingestion
        self.producer = MultiCameraProducer(self.config)

        # Vision models
        self.model_manager = model_manager or ModelManager(
            csrnet_resolution=csrnet_resolution,
            raft_resolution=raft_resolution,
        )

        # Fan-out distributor
        self.distributor = FrameDistributor(
            csrnet_buffer_size=queue_capacity,
            raft_buffer_size=queue_capacity,
            drop_policy=self.config.ingestion.drop_policy,
        )

        # Downstream workers
        self.csrnet_worker = CSRNetWorker(
            input_queue=self.distributor.csrnet_queue,
            engine=self.model_manager.csrnet,
            result_callback=csrnet_callback,
        )
        self.raft_worker = RAFTWorker(
            input_queue=self.distributor.raft_queue,
            engine=self.model_manager.raft,
            result_callback=raft_callback,
        )

        self._router_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self) -> None:
        """Start all producers, distribution router, and inference workers."""
        if self._is_running:
            return

        self._stop_event.clear()
        self._is_running = True

        # Start downstream workers first
        self.csrnet_worker.start()
        self.raft_worker.start()

        # Start distributor routing thread
        self._router_thread = threading.Thread(target=self._route_frames, name="FanOutRouter", daemon=True)
        self._router_thread.start()

        # Start upstream camera producers
        self.producer.start_all()
        logger.info("FanOutPipeline started with cameras: %s", self.producer.camera_ids)

    def stop(self) -> None:
        """Gracefully stop producers, router, and workers."""
        if not self._is_running:
            return

        self._stop_event.set()
        self.producer.stop_all()

        if self._router_thread is not None:
            self._router_thread.join(timeout=2.0)

        self.csrnet_worker.stop()
        self.raft_worker.stop()
        self.distributor.close()
        self.producer.shutdown()
        self._is_running = False
        logger.info("FanOutPipeline stopped successfully.")

    def _route_frames(self) -> None:
        """Continuous loop polling camera buffers and dispatching to distributor."""
        camera_ids = self.producer.camera_ids
        consumers = {cam_id: self.producer.get_consumer(cam_id) for cam_id in camera_ids}

        while not self._stop_event.is_set():
            any_dispatched = False
            for cam_id in camera_ids:
                consumer = consumers[cam_id]
                try:
                    packet = consumer.consume(timeout=0.02, block=False)
                    self.distributor.dispatch(packet)
                    any_dispatched = True
                except Empty:
                    pass

            if not any_dispatched:
                time.sleep(0.005)

    def stats(self) -> dict[str, Any]:
        """Aggregate real-time metrics across ingestion, distribution, and models."""
        camera_stats = {}
        for cam_id in self.producer.camera_ids:
            prod = self.producer.producers[cam_id]
            buf = self.producer.buffers[cam_id]
            camera_stats[cam_id] = {
                "produced": buf.frames_produced,
                "consumed": buf.frames_consumed,
                "dropped": buf.frames_dropped,
                "buffer_size": len(buf),
                "is_complete": prod.complete,
            }

        dist_stats = self.distributor.stats()

        return {
            "is_running": self._is_running,
            "device": self.model_manager.device,
            "cameras": camera_stats,
            "distributor": dist_stats,
            "csrnet": {
                "frames_processed": self.csrnet_worker.frames_processed,
                "avg_latency_ms": self.csrnet_worker.avg_inference_time_ms,
                "queue_size": len(self.distributor.csrnet_queue),
            },
            "raft": {
                "pairs_processed": self.raft_worker.pairs_processed,
                "avg_latency_ms": self.raft_worker.avg_inference_time_ms,
                "queue_size": len(self.distributor.raft_queue),
            },
        }

    def __enter__(self) -> "FanOutPipeline":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()
