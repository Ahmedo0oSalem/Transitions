"""Centralized filesystem paths for TRANSITIONS."""

from __future__ import annotations

import os
from pathlib import Path

from ..core.constants import (
    DEFAULT_COMPETITIONS_CSV,
    DEFAULT_DATA_DIR,
    DEFAULT_PLAYERS_CSV,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_EVENTS_DIR,
    DEFAULT_RAW_METADATA_DIR,
    DEFAULT_RAW_ROSTERS_DIR,
    DEFAULT_RAW_TRACKING_DIR,
    DEFAULT_RESOURCE_DIR,
)


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from this file's location to find the project root."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / marker).is_file():
            return parent
    return current.parents[1]


PROJECT_ROOT = _find_project_root()

_env_data_dir = os.environ.get("TRANSITIONS_DATA_DIR")
if _env_data_dir:
    DATA_DIR = Path(_env_data_dir).resolve()
else:
    DATA_DIR = PROJECT_ROOT / DEFAULT_DATA_DIR

RAW_METADATA_DIR = DATA_DIR / "meta_data"
RAW_TRACKING_DIR = DATA_DIR / "tracking_data"
RAW_ROSTERS_DIR = DATA_DIR / "rosters"
RAW_EVENTS_DIR = DATA_DIR / "event_data"
PROCESSED_DIR = PROJECT_ROOT / "Processed_Tracking"
RESOURCE_DIR = DATA_DIR / "resources"
EPV_GRID_PATH = RESOURCE_DIR / "EPV_grid.csv"
PLAYERS_CSV_PATH = DATA_DIR / "players.csv"
COMPETITIONS_CSV_PATH = DATA_DIR / "competitions.csv"


def find_match_file(base_dir: str | Path, match_id: str | int, suffix: str) -> Path | None:
    """Return a match file under ``base_dir``, including nested subfolders."""

    root = Path(base_dir)
    match_id = str(match_id)
    direct = root / f"{match_id}{suffix}"
    if direct.is_file():
        return direct
    for path in root.rglob(f"{match_id}{suffix}"):
        if path.is_file():
            return path
    return None


def match_dir(match_id: str | int, processed_dir: str | Path | None = None) -> Path:
    """Return the processed match directory for a match ID."""

    base = Path(processed_dir) if processed_dir is not None else PROCESSED_DIR
    return base / str(match_id)
