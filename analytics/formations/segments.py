"""Continuous formation segment aggregation.
Groups raw, overlapping formation windows into continuous segments
and calculates exact temporal and spatial properties for each segment.
Now includes EPV and DAS integration."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ...core.logger import get_logger
from ...io.paths import match_dir
from .taxonomy import derive_hierarchy

logger = get_logger(__name__)

def load_possession_sequences(match_dir_path: Path) -> list[dict]:
    """Load possession sequences from events.json if available."""
    events_path = match_dir_path / "events.json"
    if not events_path.is_file():
        return []
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)
    metadata_path = match_dir_path / "metadata.json"
    if not metadata_path.is_file():
        return []
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    home_team_id = str(metadata.get("homeTeam", {}).get("id", ""))
    from ..possession.events import possession_sequences_from_events
    return possession_sequences_from_events(events, home_team_id)

def calculate_possession_overlap(segment_start: float, segment_end: float, 
                                 sequences: list[dict], team: str, period: int) -> dict:
    """Calculate exact in/out/loose ball seconds for a specific segment."""
    in_poss = 0.0
    out_poss = 0.0
    loose_ball = 0.0
    turnovers = 0
    team_seqs = [s for s in sequences if s["period"] == period]
    team_seqs.sort(key=lambda x: x["start_sec"])
    prev_owner = None
    for s in team_seqs:
        overlap_start = max(segment_start, s["start_sec"])
        overlap_end = min(segment_end, s["end_sec"])
        if overlap_start < overlap_end:
            duration = overlap_end - overlap_start
            if s["team"] == team:
                in_poss += duration
            elif s["team"] is None:
                loose_ball += duration
            else:
                out_poss += duration
            if prev_owner is not None and s["team"] != prev_owner and s["team"] is not None and prev_owner is not None:
                turnovers += 1
            if s["team"] is not None:
                prev_owner = s["team"]
    total_calc = in_poss + out_poss + loose_ball
    segment_duration = segment_end - segment_start
    if total_calc < segment_duration - 0.1:
        loose_ball += (segment_duration - total_calc)
    return {
        "in_possession_sec": round(in_poss, 2),
        "out_of_possession_sec": round(out_poss, 2),
        "loose_ball_sec": round(loose_ball, 2),
        "n_turnovers": turnovers
    }

def aggregate_spatial_metrics(windows_df: pd.DataFrame, segment_start: float, segment_end: float) -> dict:
    """Calculate duration-weighted average of spatial metrics for overlapping windows."""
    mask = (windows_df["windowStartSec"] < segment_end) & (windows_df["windowEndSec"] > segment_start)
    overlap_df = windows_df[mask].copy()
    if overlap_df.empty:
        return {}
    overlap_df["overlap_start"] = overlap_df["windowStartSec"].clip(lower=segment_start)
    overlap_df["overlap_end"] = overlap_df["windowEndSec"].clip(upper=segment_end)
    overlap_df["overlap_duration"] = overlap_df["overlap_end"] - overlap_df["overlap_start"]
    total_weight = overlap_df["overlap_duration"].sum()
    if total_weight <= 0:
        return {}
    def weighted_avg(col):
        if col not in overlap_df.columns: return np.nan
        return (overlap_df[col] * overlap_df["overlap_duration"]).sum() / total_weight
    def weighted_std(col):
        if col not in overlap_df.columns: return np.nan
        mean = weighted_avg(col)
        variance = ((overlap_df[col] - mean)**2 * overlap_df["overlap_duration"]).sum() / total_weight
        return np.sqrt(variance)
    return {
        "mean_compactness": round(weighted_avg("meanCompactness"), 3),
        "std_compactness": round(weighted_std("meanCompactness"), 3),
        "mean_width": round(weighted_avg("meanWidth"), 3),
        "std_width": round(weighted_std("meanWidth"), 3),
        "mean_depth": round(weighted_avg("meanDepth"), 3),
        "std_depth": round(weighted_std("meanDepth"), 3),
        "mean_center_x": round(weighted_avg("meanCenterX"), 2),
        "std_center_x": round(weighted_std("meanCenterX"), 2),
        "range_center_x": round(overlap_df["meanCenterX"].max() - overlap_df["meanCenterX"].min(), 2) if "meanCenterX" in overlap_df.columns else np.nan,
        "mean_center_y": round(weighted_avg("meanCenterY"), 2),
        "std_center_y": round(weighted_std("meanCenterY"), 2),
        "range_center_y": round(overlap_df["meanCenterY"].max() - overlap_df["meanCenterY"].min(), 2) if "meanCenterY" in overlap_df.columns else np.nan,
        "mean_elongation": round(weighted_avg("meanDepth") / weighted_avg("meanWidth"), 3) if weighted_avg("meanWidth") > 0 else np.nan,
        "net_centroid_displacement": round(overlap_df["netCentroidDisplacement"].mean(), 2) if "netCentroidDisplacement" in overlap_df.columns else np.nan,
        "mean_centroid_velocity": round(weighted_avg("meanCentroidVelocity"), 3) if "meanCentroidVelocity" in overlap_df.columns else np.nan,
        "mean_template_displacement": round(weighted_avg("avgCostPerPlayer"), 3),
        "mean_confidence": round(weighted_avg("confidence"), 4),
        "min_confidence": round(overlap_df["confidence"].min(), 4),
        "n_windows": len(overlap_df),
        "n_frames": int(overlap_df["nFrames"].sum())
    }

# --- NEW: EPV/DAS Aggregation Helpers ---

def _load_epv_data(match_dir_path: Path) -> pd.DataFrame | None:
    epv_path = match_dir_path / "epv_timeseries.csv"
    if epv_path.is_file():
        try:
            df = pd.read_csv(epv_path)
            logger.info("Loaded %s EPV rows.", len(df))
            return df
        except Exception as e:
            logger.warning("Failed to load EPV data: %s", e)
    return None

def _load_das_data(match_dir_path: Path) -> pd.DataFrame | None:
    das_path = match_dir_path / "das_sequences.csv"
    if das_path.is_file():
        try:
            df = pd.read_csv(das_path)
            logger.info("Loaded %s DAS sequences.", len(df))
            return df
        except Exception as e:
            logger.warning("Failed to load DAS data: %s", e)
    return None

def _aggregate_epv_for_segment(epv_df: pd.DataFrame, team: str, period: int, 
                               start_sec: float, end_sec: float) -> dict:
    """Aggregate EPV values within a segment's time window."""
    if epv_df is None or epv_df.empty:
        return {"cumulative_epv": 0.0, "mean_epv": 0.0, "epv_per_min": 0.0}
    
    # Filter by period and time range
    sub = epv_df[
        (epv_df["period"] == period) &
        (epv_df["secondIntoPeriod"] >= start_sec) &
        (epv_df["secondIntoPeriod"] < end_sec)
    ]
    
    if sub.empty:
        return {"cumulative_epv": 0.0, "mean_epv": 0.0, "epv_per_min": 0.0}
    
    epv_values = sub["meanSignedEPV"].values
    
    # For away team, flip the sign so positive always means "good for this team"
    if team == "away":
        epv_values = -epv_values
    
    duration_min = (end_sec - start_sec) / 60.0
    cum_epv = float(np.sum(epv_values))
    mean_epv = float(np.mean(epv_values))
    epv_per_min = cum_epv / duration_min if duration_min > 0 else 0.0
    
    return {
        "cumulative_epv": round(cum_epv, 4),
        "mean_epv": round(mean_epv, 4),
        "epv_per_min": round(epv_per_min, 4)
    }

def _aggregate_das_for_segment(das_df: pd.DataFrame, team: str, period: int,
                               start_sec: float, end_sec: float) -> dict:
    """Count DAS events that fall within a segment's time window."""
    if das_df is None or das_df.empty:
        return {"das_count": 0, "das_per_min": 0.0}
    
    # Filter by team, period, and time overlap
    sub = das_df[
        (das_df["team"] == team) &
        (das_df["period"] == period) &
        (das_df["isDAS"] == True) &
        (das_df["startSec"] < end_sec) &
        (das_df["endSec"] > start_sec)
    ]
    
    das_count = int(len(sub))
    duration_min = (end_sec - start_sec) / 60.0
    das_per_min = das_count / duration_min if duration_min > 0 else 0.0
    
    return {
        "das_count": das_count,
        "das_per_min": round(das_per_min, 4)
    }

# -----------------------------------------

def build_formation_segments(match_id: str | int, processed_dir: str) -> pd.DataFrame:
    """Main entrypoint: reads formations.csv and builds continuous segments."""
    match_dir_path = match_dir(match_id, processed_dir)
    formations_path = match_dir_path / "formations.csv"
    if not formations_path.is_file():
        logger.warning("[%s] formations.csv not found. Run detection first.", match_id)
        return pd.DataFrame()
    df = pd.read_csv(formations_path)
    if df.empty:
        return pd.DataFrame()
        
    # --- FIX: Avoid duplicate 'formation' column ---
    hierarchies = df["formation"].apply(derive_hierarchy).apply(pd.Series)
    if 'formation' in hierarchies.columns:
        hierarchies = hierarchies.drop(columns=['formation'])
    df = pd.concat([df, hierarchies], axis=1)
    # ----------------------------------------------
    
    possession_seqs = load_possession_sequences(match_dir_path)
    logger.info("[%s] loaded %s possession sequences.", match_id, len(possession_seqs))
    
    # Load EPV and DAS data once
    epv_df = _load_epv_data(match_dir_path)
    das_df = _load_das_data(match_dir_path)
    
    segments = []
    for (team, period), grp in df.groupby(["team", "period"], sort=False):
        grp = grp.sort_values("windowStartSec").reset_index(drop=True)
        current_formation = None
        seg_start = None
        seg_end = None
        for _, row in grp.iterrows():
            # Now row["formation"] is a clean string, not a Series
            formation = str(row["formation"]).strip()
            start = float(row["windowStartSec"])
            end = float(row["windowEndSec"])
            if formation == current_formation:
                seg_end = end
            else:
                if current_formation is not None:
                    seg_data = _finalize_segment(match_id, team, period, current_formation, seg_start, seg_end, df, possession_seqs, epv_df, das_df)
                    segments.append(seg_data)
                current_formation = formation
                seg_start = start
                seg_end = end
        if current_formation is not None:
            seg_data = _finalize_segment(match_id, team, period, current_formation, seg_start, seg_end, df, possession_seqs, epv_df, das_df)
            segments.append(seg_data)
    segments_df = pd.DataFrame(segments)
    out_path = match_dir_path / "formation_segments.csv"
    segments_df.to_csv(out_path, index=False)
    logger.info("[%s] wrote %s continuous segments -> %s", match_id, len(segments_df), out_path)
    return segments_df

def _finalize_segment(match_id, team, period, formation, start, end, df, possession_seqs, epv_df=None, das_df=None):
    """Helper to build a single segment dictionary."""
    segment_windows = df[(df["team"] == team) & (df["period"] == period) & 
                         (df["windowStartSec"] < end) & (df["windowEndSec"] > start)]
    hierarchy = derive_hierarchy(formation)
    duration = end - start
    poss_props = calculate_possession_overlap(start, end, possession_seqs, team, period)
    spatial_props = aggregate_spatial_metrics(segment_windows, start, end)
    
    # Aggregate EPV and DAS
    epv_props = _aggregate_epv_for_segment(epv_df, team, period, start, end)
    das_props = _aggregate_das_for_segment(das_df, team, period, start, end)
    
    return {
        "matchId": match_id,
        "team": team,
        "period": period,
        "formation": formation,
        "variant": hierarchy.get("variant", formation),
        "family": hierarchy.get("family", "other"),
        "start_sec": round(start, 2),
        "end_sec": round(end, 2),
        "duration": round(duration, 2),
        **poss_props,
        **spatial_props,
        **epv_props,
        **das_props
    }