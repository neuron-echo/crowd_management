"""Load venue configuration from YAML files.

No dependencies on torch, cv2, or any vision model code.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from crowd_management.venue.models import VenueConfig
from crowd_management.venue.topology import VenueTopology


def load_venue_config(config_path: str | Path) -> VenueConfig:
    """Load and validate a venue YAML configuration file.

    Args:
        config_path: Path to the venue YAML file.

    Returns:
        Validated VenueConfig instance.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the configuration is invalid.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Venue configuration file not found: {config_file}")

    with config_file.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if not isinstance(raw, dict):
        raise ValueError("Venue configuration root must be a mapping")

    # Support top-level 'venue:' key or flat structure
    venue_data = raw.get("venue", raw)

    try:
        return VenueConfig.model_validate(venue_data)
    except Exception as exc:
        raise ValueError(f"Invalid venue configuration: {exc}") from exc


def load_venue_topology(config_path: str | Path) -> VenueTopology:
    """Load a venue configuration and build the topology graph.

    Convenience function that combines loading + topology construction.
    """
    config = load_venue_config(config_path)
    return VenueTopology(config)
