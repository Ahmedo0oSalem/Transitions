"""Match domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Match:
    """Standardized match container."""

    metadata: dict[str, Any]
    tracking: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    roster: list[dict[str, Any]] = field(default_factory=list)