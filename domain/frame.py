"""Frame domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .ball import Ball
from .player import PlayerPosition


@dataclass(slots=True)
class Frame:
    """Standardized tracking frame."""

    period: int
    timestamp: float
    home_players: list[PlayerPosition] = field(default_factory=list)
    away_players: list[PlayerPosition] = field(default_factory=list)
    ball: Ball | None = None
    metadata: dict[str, Any] = field(default_factory=dict)