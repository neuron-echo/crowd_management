"""Logging helpers for the ingestion subsystem."""

from __future__ import annotations

import logging


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure the root logger for the project."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )
    return logging.getLogger("crowd_management")
