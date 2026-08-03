"""Formation domain model."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

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
    confidence: float = 1.0
    # Per-frame spatial metrics (computed during detection)
    mean_compactness: Optional[float] = None
    std_compactness: Optional[float] = None
    mean_center_x: Optional[float] = None
    std_center_x: Optional[float] = None
    range_center_x: Optional[float] = None
    mean_center_y: Optional[float] = None
    std_center_y: Optional[float] = None
    range_center_y: Optional[float] = None
    mean_width: Optional[float] = None
    std_width: Optional[float] = None
    mean_depth: Optional[float] = None
    std_depth: Optional[float] = None
    net_centroid_displacement: Optional[float] = None
    mean_centroid_velocity: Optional[float] = None
    mean_template_displacement: Optional[float] = None

@dataclass(slots=True)
class FormationSegment:
    """Continuous formation segment with aggregated properties."""
    match_id: str | int
    team: str
    period: int
    formation: str
    variant: str  # formation without _flipped suffix
    family: str   # e.g., "back-4", "back-3"
    start_sec: float
    end_sec: float
    duration: float
    n_windows: int
    n_frames: int
    mean_confidence: float
    min_confidence: float
    # Temporal breakdown
    in_possession_sec: float
    out_of_possession_sec: float
    loose_ball_sec: float
    n_turnovers: int
    # Spatial shape metrics
    mean_compactness: float
    std_compactness: float
    mean_width: float
    std_width: float
    mean_depth: float
    std_depth: float
    mean_elongation: float  # depth/width ratio
    # Location metrics
    mean_center_x: float
    std_center_x: float
    range_center_x: float
    mean_center_y: float
    std_center_y: float
    range_center_y: float
    # Movement metrics
    net_centroid_displacement: float
    mean_centroid_velocity: float
    # Tactical execution
    mean_template_displacement: float