"""Schema validation helpers for match data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _get(d: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d:
            return d[k]
    return default


def validate_metadata(data: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize a metadata dict, raising on missing fields."""

    pitch = data.get("pitch", {})
    if not isinstance(pitch, dict) or "length" not in pitch or "width" not in pitch:
        raise ValueError("metadata must contain pitch.length and pitch.width")

    for field in ("id", "fps", "homeTeam", "awayTeam", "homeTeamStartLeft"):
        if field not in data:
            raise ValueError(f"metadata missing required field: {field}")

    return data


def validate_roster(data: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate a roster list, returning it unchanged if valid."""

    if not isinstance(data, list):
        raise TypeError("roster must be a list of player entries")
    return data


def validate_event(event: dict[str, Any]) -> dict[str, Any]:
    """Validate a single event record, returning it unchanged if valid."""

    for field in ("gameEventId", "period", "sequence"):
        if field not in event:
            raise ValueError(f"event missing required field: {field}")
    return event
