"""Formation domain model."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class FormationWindow:
    """Detected formation over a time window."""

    match_id: str | int
    team: str
    period: int
    window_index: int
    window_start_sec: float
    window_end_sec: float
    formation: str
    orientation: str
    n_outfield_players: int
    n_frames: int
    avg_cost_per_player: float