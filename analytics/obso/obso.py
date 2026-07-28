"""Off-Ball Scoring Opportunity computation using EPV grid."""

import json
import bz2
import numpy as np
import pandas as pd
from pathlib import Path

from ...core.logger import get_logger
from ...io.paths import match_dir, EPV_GRID_PATH
from ...analytics.possession import epv_value, get_base_directions, attack_direction, load_epv_grid
from ...analytics.possession.tracking import stream_ball_and_owner, smooth_owner

logger = get_logger(__name__)

def compute_obso_for_match(match_id, processed_dir, epv_grid_path=None, radius=5.0, downsample=1):
    """
    Compute Off-Ball Scoring Opportunity (OBSO) for each frame.
    
    OBSO is the maximum EPV within a radius around the ball location,
    representing the best scoring opportunity available in the vicinity.
    
    Parameters
    ----------
    match_id : str/int
    processed_dir : str/Path
    epv_grid_path : str/Path, optional (default: EPV_GRID_PATH)
    radius : float, in metres (default 5.0)
    downsample : int, default 1
    
    Returns
    -------
    pd.DataFrame with columns: period, elapsed, team, obso
    """
    if epv_grid_path is None:
        epv_grid_path = EPV_GRID_PATH
    
    match_dir_path = match_dir(match_id, processed_dir)
    metadata_path = match_dir_path / "metadata.json"
    tracking_path = match_dir_path / "tracking.jsonl.bz2"
    
    if not metadata_path.is_file() or not tracking_path.is_file():
        raise FileNotFoundError(f"Processed files missing for {match_id} in {match_dir_path}")
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    
    # Load EPV grid
    epv_grid = load_epv_grid(epv_grid_path)
    home_dir_p1, away_dir_p1 = get_base_directions(metadata)
    
    # Get ball stream and possession owner
    periods, elapsed, ball_x, ball_y, owner = stream_ball_and_owner(
        tracking_path, pitch_length, pitch_width
    )
    # Use smoothed owner to reduce noise
    fps = metadata.get("fps", 25.0)
    smoothed = smooth_owner(owner, periods, fps)
    
    # Now compute OBSO for each frame
    records = []
    n_frames = len(periods)
    for i in range(0, n_frames, downsample):
        if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
            continue
        team = "home" if smoothed[i] == 1 else "away" if smoothed[i] == -1 else None
        if team is None:
            continue
        
        # Determine direction for this team/period
        direction = attack_direction(team, int(periods[i]), home_dir_p1, away_dir_p1)
        
        # Compute EPV at ball location
        ball_epv = epv_value(epv_grid, ball_x[i], ball_y[i], pitch_length, pitch_width, direction)
        
        # Sample points in a circle around the ball
        # For simplicity, we'll sample a grid of points within the radius
        # We can use a coarse grid (e.g., 1m spacing) to speed up
        step = 1.0  # metres
        x_range = np.arange(-radius, radius + step, step)
        y_range = np.arange(-radius, radius + step, step)
        max_epv = ball_epv
        for dx in x_range:
            for dy in y_range:
                if dx*dx + dy*dy > radius*radius:
                    continue
                x_test = ball_x[i] + dx
                y_test = ball_y[i] + dy
                if x_test < 0 or x_test > pitch_length or y_test < 0 or y_test > pitch_width:
                    continue
                epv_val = epv_value(epv_grid, x_test, y_test, pitch_length, pitch_width, direction)
                if epv_val > max_epv:
                    max_epv = epv_val
        
        records.append({
            "period": int(periods[i]),
            "elapsed": float(elapsed[i]),
            "team": team,
            "obso": max_epv,
        })
    
    df = pd.DataFrame(records)
    logger.info("Computed OBSO for %d frames (radius=%.1f m)", len(df), radius)
    return df


def aggregate_obso_by_window(obso_df, formations_df):
    """Average OBSO per formation window."""
    results = []
    for _, row in formations_df.iterrows():
        period = row["period"]
        start = row["windowStartSec"]
        end = row["windowEndSec"]
        mask = (obso_df["period"] == period) & (obso_df["elapsed"] >= start) & (obso_df["elapsed"] < end)
        sub = obso_df[mask]
        if len(sub) == 0:
            mean_obso = np.nan
        else:
            mean_obso = sub["obso"].mean()
        out = row.to_dict()
        out["mean_obso"] = mean_obso
        results.append(out)
    return pd.DataFrame(results)