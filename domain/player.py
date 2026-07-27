"""Player domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PlayerPosition:
    """Single player position in a frame."""

    player_id: str
    x: float
    y: float
    team: str | None = None
    jersey_number: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)