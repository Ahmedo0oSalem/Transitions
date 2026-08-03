"""Formation analysis artifacts."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import pandas as pd
from ..domain import FormationWindow, FormationSegment

@dataclass(slots=True)
class FormationResult:
    """Typed result for a formation detection run."""
    match_id: str | int
    windows: list[FormationWindow] = field(default_factory=list)
    segments: list[FormationSegment] = field(default_factory=list)

def formation_result_from_dataframe(match_id: str | int, formations_df: pd.DataFrame) -> FormationResult:
    """Convert a formation output dataframe into a typed artifact.
    
    Args:
        match_id: Match identifier.
        formations_df: Formation output dataframe.
        
    Returns:
        A typed formation artifact.
    """
    windows: list[FormationWindow] = []
    for row in formations_df.to_dict(orient="records"):
        windows.append(
            FormationWindow(
                match_id=row["matchId"],
                team=row["team"],
                period=int(row["period"]),
                window_index=int(row["windowIndex"]),
                window_start_sec=float(row["windowStartSec"]),
                window_end_sec=float(row["windowEndSec"]),
                formation=str(row["formation"]),
                orientation=str(row["orientation"]),
                n_outfield_players=int(row["nOutfieldPlayers"]),
                n_frames=int(row["nFrames"]),
                avg_cost_per_player=float(row["avgCostPerPlayer"]),
                confidence=float(row.get("confidence", 1.0)),
                mean_compactness=float(row["meanCompactness"]) if "meanCompactness" in row else None,
                std_compactness=float(row["stdCompactness"]) if "stdCompactness" in row else None,
                mean_center_x=float(row["meanCenterX"]) if "meanCenterX" in row else None,
                mean_center_y=float(row["meanCenterY"]) if "meanCenterY" in row else None,
                mean_width=float(row["meanWidth"]) if "meanWidth" in row else None,
                mean_depth=float(row["meanDepth"]) if "meanDepth" in row else None,
                net_centroid_displacement=float(row["netCentroidDisplacement"]) if "netCentroidDisplacement" in row else None,
                mean_centroid_velocity=float(row["meanCentroidVelocity"]) if "meanCentroidVelocity" in row else None,
            )
        )
    
    # Note: segments would be loaded from formation_segments.csv if needed
    # For now, we keep segments empty in this conversion function
    
    return FormationResult(match_id=match_id, windows=windows, segments=[])