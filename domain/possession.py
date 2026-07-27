"""Possession domain model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class PossessionSequence:
    """Possession interval representation."""

    team: str | None
    period: int
    start_sec: float
    end_sec: float
    duration: float
    start_idx: int | None = None
    end_idx: int | None = None