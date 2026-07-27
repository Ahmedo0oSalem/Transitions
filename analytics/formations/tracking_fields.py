"""Field-name aliasing primitives for tracking data parsing.

Extracted from goalkeeper.py into its own module so that multiple
consumers (detector.py, frame_reliability.py, etc.) can import these
without creating circular dependencies.
"""

from __future__ import annotations

from ...io.field_keys import (
    PLAYER_ID_KEYS,
    PLAYER_NUMBER_KEYS,
    PLAYER_X_KEYS,
    PLAYER_Y_KEYS,
)

REJECT_CONFIDENCE_VALUES = ["LOW"]
REJECT_VISIBILITY_VALUES: list[str] = []


def _get_first(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def extract_player_xy(player_dict):
    """Extract a player identifier and coordinates from a frame entry."""

    if REJECT_CONFIDENCE_VALUES and player_dict.get("confidence") in REJECT_CONFIDENCE_VALUES:
        return None
    if REJECT_VISIBILITY_VALUES and player_dict.get("visibility") in REJECT_VISIBILITY_VALUES:
        return None
    x = _get_first(player_dict, PLAYER_X_KEYS)
    y = _get_first(player_dict, PLAYER_Y_KEYS)
    pid = _get_first(player_dict, PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)
