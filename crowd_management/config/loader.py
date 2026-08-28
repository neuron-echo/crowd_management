"""Load and validate ingestion configuration files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from crowd_management.config.schemas import AppConfig
from crowd_management.utils.exceptions import ConfigurationError


def load_ingestion_config(config_path: str | Path) -> AppConfig:
    """Load and validate a YAML ingestion configuration.

    Relative paths for the dataset and camera sources are resolved relative to the
    YAML file directory when they are not absolute.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise ConfigurationError(f"Configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle) or {}

    if not isinstance(loaded, dict):
        raise ConfigurationError("Configuration root must be a mapping")

    resolved = dict(loaded)
    base_dir = config_file.parent
    dataset_root = resolved.get("dataset", {}).get("root")
    if isinstance(dataset_root, str) and not Path(dataset_root).is_absolute():
        if (base_dir / dataset_root).exists():
            resolved["dataset"]["root"] = str((base_dir / dataset_root).resolve())
        elif (Path.cwd() / dataset_root).exists():
            resolved["dataset"]["root"] = str((Path.cwd() / dataset_root).resolve())
        else:
            resolved["dataset"]["root"] = str((base_dir / dataset_root).resolve())

    cameras = resolved.get("cameras", [])
    for index, camera in enumerate(cameras):
        if not isinstance(camera, dict):
            continue
        source = camera.get("source")
        if isinstance(source, str) and not Path(source).is_absolute():
            if (base_dir / source).exists():
                resolved["cameras"][index]["source"] = str((base_dir / source).resolve())
            elif (Path.cwd() / source).exists():
                resolved["cameras"][index]["source"] = str((Path.cwd() / source).resolve())
            else:
                resolved["cameras"][index]["source"] = str((base_dir / source).resolve())

    try:
        return AppConfig.model_validate(resolved)
    except Exception as exc:  # pragma: no cover - surfaced to caller
        raise ConfigurationError(f"Invalid ingestion configuration: {exc}") from exc


def resolve_dataset_root(dataset_root: str | Path) -> Path:
    """Return the dataset root path as a Path object."""
    path = Path(dataset_root)
    return path.resolve() if path.is_absolute() else (Path.cwd() / path).resolve()
