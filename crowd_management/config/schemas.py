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


class AppConfig(BaseModel):
    dataset: DatasetConfig
    ingestion: IngestionConfig
    cameras: list[CameraConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_cameras(self):
        enabled_cameras = [camera for camera in self.cameras if camera.enabled]
        if not enabled_cameras:
            raise ValueError("At least one enabled camera must be defined")
        return self
