"""Ball domain model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Ball:
    """Single ball state in a frame."""

    x: float
    y: float
    z: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)