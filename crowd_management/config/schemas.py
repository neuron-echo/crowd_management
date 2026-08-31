"""Configuration schemas for the ingestion pipeline."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetConfig(BaseModel):
    name: str = Field(..., min_length=1)
    root: str = Field(..., min_length=1)


class CameraConfig(BaseModel):
    camera_id: str = Field(..., min_length=1)
    name: str | None = None
    source_type: Literal["video"] = "video"
    source: str = Field(..., min_length=1)
    enabled: bool = True


class IngestionConfig(BaseModel):
    target_fps: float = Field(default=5.0, gt=0)
    real_time_mode: bool = True
    buffer_size: int = Field(default=30, gt=0)
    drop_policy: Literal["oldest", "newest", "drop_latest"] = "oldest"

    @model_validator(mode="after")
    def validate_target_fps(self):
        if self.target_fps <= 0:
            raise ValueError("target_fps must be > 0")
        return self


class CSRNetModelConfig(BaseModel):
    enabled: bool = True
    checkpoint_path: str = "checkpoints/csrnet_shanghaitech_part_a.pth"
    inference_resolution: tuple[int, int] | None = (720, 540)


class ByteTrackModelConfig(BaseModel):
    """Configuration for YOLOv8 + ByteTrack motion estimation model."""

    model_config = {"protected_namespaces": ()}

    enabled: bool = True
    model_path: str = Field(
        default="yolov8l.pt",
        description="YOLOv8 weights file. Downloaded automatically on first run if not present.",
    )
    tracker_config: str = Field(
        default="bytetrack.yaml",
        description="ByteTrack YAML config bundled with ultralytics.",
    )
    confidence_threshold: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Minimum detection confidence for a person to be tracked.",
    )
    motion_threshold_px: float = Field(
        default=2.0, gt=0,
        description="Minimum displacement (px/sec) for a track to count as 'moving'.",
    )


class ModelsConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    csrnet: CSRNetModelConfig = Field(default_factory=CSRNetModelConfig)
    bytetrack: ByteTrackModelConfig = Field(default_factory=ByteTrackModelConfig)


class AppConfig(BaseModel):
    dataset: DatasetConfig
    ingestion: IngestionConfig
    cameras: list[CameraConfig] = Field(default_factory=list)
    models: ModelsConfig = Field(default_factory=ModelsConfig)

    @model_validator(mode="after")
    def validate_cameras(self):
        enabled_cameras = [camera for camera in self.cameras if camera.enabled]
        if not enabled_cameras:
            raise ValueError("At least one enabled camera must be defined")
        return self
