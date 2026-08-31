"""Central environment-variable loader for the Capstone Decision Module.

All runtime-configurable values (paths, model settings, device, FPS, etc.)
are read from environment variables. A ``.env`` file in the project root is
loaded automatically via *python-dotenv* if it exists.

Usage anywhere in the codebase::

    from crowd_management.config.env import Env

    checkpoint = Env.csrnet_checkpoint()    # returns a Path
    device     = Env.device()               # "auto" | "cpu" | "cuda"
    fps        = Env.ingestion_target_fps() # float

All accessors return typed Python objects, never raw strings.

Priority (highest → lowest):
    1. Actual OS environment variable (e.g. set in CI / Docker)
    2. ``.env`` file in the project root
    3. Built-in default value
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()          # crowd_management/config/env.py
_PROJECT_ROOT = _HERE.parents[2]          # Capstone-Decision-Module/


def _load_dotenv() -> None:
    """Load .env from the project root if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
        env_file = _PROJECT_ROOT / ".env"
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
    except ImportError:
        pass  # python-dotenv is optional; OS env vars still work


_load_dotenv()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


def _get_bool(key: str, default: bool = True) -> bool:
    val = _get(key, "true" if default else "false").lower()
    return val in ("1", "true", "yes", "on")


def _get_int(key: str, default: int = 0) -> int:
    try:
        return int(_get(key, str(default)))
    except ValueError:
        return default


def _get_float(key: str, default: float = 0.0) -> float:
    try:
        return float(_get(key, str(default)))
    except ValueError:
        return default


def _resolve_path(raw: str, relative_to: Path = _PROJECT_ROOT) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else (relative_to / p)


def _get_resolution(key: str, default: str = "720x540") -> Tuple[int, int]:
    raw = _get(key, default)
    try:
        w, h = raw.lower().split("x")
        return int(w.strip()), int(h.strip())
    except (ValueError, AttributeError):
        w_str, h_str = default.split("x")
        return int(w_str), int(h_str)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class Env:
    """Typed accessors for all environment-variable configuration."""

    @staticmethod
    def dataset_root() -> Path:
        """Root directory of the HAJJv2 dataset. Env: ``DATASET_ROOT``."""
        return _resolve_path(_get("DATASET_ROOT", "HAJJv2.Dataset"))

    @staticmethod
    def camera_source(camera_id: str) -> Optional[Path]:
        """Path to the video file for *camera_id*. Env: ``CAMERA_01_SOURCE``, etc."""
        key = f"CAMERA_{camera_id.upper().replace('CAM_', '').replace('CAM', '')}_SOURCE"
        raw = _get(key) or _get(f"CAMERA_{camera_id.upper()}_SOURCE")
        return _resolve_path(raw) if raw else None

    @staticmethod
    def venue_config() -> Path:
        """Path to the venue YAML config. Env: ``VENUE_CONFIG``."""
        return _resolve_path(_get("VENUE_CONFIG", "config/venue/circular_venue.yaml"))

    @staticmethod
    def ingestion_target_fps() -> float:
        """Target ingestion FPS. Env: ``INGESTION_TARGET_FPS`` (default 5)."""
        return _get_float("INGESTION_TARGET_FPS", 5.0)

    @staticmethod
    def ingestion_real_time_mode() -> bool:
        """Throttle to real-time speed. Env: ``INGESTION_REAL_TIME_MODE``."""
        return _get_bool("INGESTION_REAL_TIME_MODE", True)

    @staticmethod
    def ingestion_buffer_size() -> int:
        """Frame buffer size per camera. Env: ``INGESTION_BUFFER_SIZE`` (default 30)."""
        return _get_int("INGESTION_BUFFER_SIZE", 30)

    @staticmethod
    def ingestion_drop_policy() -> Literal["oldest", "newest", "drop_latest"]:
        """Buffer drop policy. Env: ``INGESTION_DROP_POLICY`` (default 'oldest')."""
        val = _get("INGESTION_DROP_POLICY", "oldest").lower()
        return val if val in ("oldest", "newest", "drop_latest") else "oldest"  # type: ignore

    @staticmethod
    def csrnet_enabled() -> bool:
        """Enable CSRNet. Env: ``CSRNET_ENABLED`` (default true)."""
        return _get_bool("CSRNET_ENABLED", True)

    @staticmethod
    def csrnet_checkpoint() -> Path:
        """CSRNet checkpoint path. Env: ``CSRNET_CHECKPOINT``."""
        return _resolve_path(_get("CSRNET_CHECKPOINT", "checkpoints/csrnet_shanghaitech_part_a.pth"))

    @staticmethod
    def csrnet_resolution() -> Tuple[int, int]:
        """Inference resolution (W, H). Env: ``CSRNET_RESOLUTION`` e.g. '720x540'."""
        return _get_resolution("CSRNET_RESOLUTION", "720x540")

    @staticmethod
    def bytetrack_enabled() -> bool:
        """Enable ByteTrack. Env: ``BYTETRACK_ENABLED`` (default true)."""
        return _get_bool("BYTETRACK_ENABLED", True)

    @staticmethod
    def bytetrack_model() -> str:
        """YOLOv8 model variant. Env: ``BYTETRACK_MODEL`` (default 'yolov8l.pt')."""
        return _get("BYTETRACK_MODEL", "yolov8l.pt")

    @staticmethod
    def bytetrack_conf() -> float:
        """YOLO confidence threshold. Env: ``BYTETRACK_CONF`` (default 0.3)."""
        return max(0.0, min(1.0, _get_float("BYTETRACK_CONF", 0.3)))

    @staticmethod
    def bytetrack_iou() -> float:
        """YOLO NMS IoU threshold. Env: ``BYTETRACK_IOU`` (default 0.45)."""
        return max(0.0, min(1.0, _get_float("BYTETRACK_IOU", 0.45)))

    @staticmethod
    def bytetrack_motion_threshold_px() -> float:
        """Min pixel displacement to count as moving. Env: ``BYTETRACK_MOTION_THRESHOLD_PX``."""
        return max(0.0, _get_float("BYTETRACK_MOTION_THRESHOLD_PX", 2.0))

    @staticmethod
    def device() -> Literal["auto", "cpu", "cuda"]:
        """Inference device. Env: ``DEVICE`` (default 'auto')."""
        val = _get("DEVICE", "auto").lower()
        return val if val in ("auto", "cpu", "cuda") else "auto"  # type: ignore

    @staticmethod
    def summary() -> dict:
        """Return all resolved env values as a dict for logging/debugging."""
        return {
            "dataset_root": str(Env.dataset_root()),
            "venue_config": str(Env.venue_config()),
            "device": Env.device(),
            "ingestion": {
                "target_fps": Env.ingestion_target_fps(),
                "real_time_mode": Env.ingestion_real_time_mode(),
                "buffer_size": Env.ingestion_buffer_size(),
                "drop_policy": Env.ingestion_drop_policy(),
            },
            "csrnet": {
                "enabled": Env.csrnet_enabled(),
                "checkpoint": str(Env.csrnet_checkpoint()),
                "resolution": Env.csrnet_resolution(),
            },
            "bytetrack": {
                "enabled": Env.bytetrack_enabled(),
                "model": Env.bytetrack_model(),
                "conf": Env.bytetrack_conf(),
                "iou": Env.bytetrack_iou(),
                "motion_threshold_px": Env.bytetrack_motion_threshold_px(),
            },
        }
