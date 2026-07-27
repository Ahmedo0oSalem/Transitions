"""EPV domain models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EPVPoint:
    """Single EPV observation."""

    period: int
    second_into_period: int
    mean_signed_epv: float


@dataclass(slots=True)
class DangerousAttackingSequence:
    """EPV-derived dangerous attacking sequence."""

    team: str
    period: int
    start_sec: float
    end_sec: float
    duration: float
    peak_epv: float
    is_das: bool
