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


class RAFTModelConfig(BaseModel):
    model_config = {"protected_namespaces": ()}
    enabled: bool = True
    model_size: Literal["small", "large"] = "large"
    inference_resolution: tuple[int, int] | None = (512, 384)


class ModelsConfig(BaseModel):
    device: Literal["auto", "cpu", "cuda"] = "auto"
    csrnet: CSRNetModelConfig = Field(default_factory=CSRNetModelConfig)
    raft: RAFTModelConfig = Field(default_factory=RAFTModelConfig)


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

