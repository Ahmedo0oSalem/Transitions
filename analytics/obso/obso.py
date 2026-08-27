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


def compute_obso_for_match(match_id, processed_dir, epv_grid_path=None, radius=5.0, downsample=1, force_recompute=False):
    """
    Compute Off-Ball Scoring Opportunity (OBSO) for each frame.

    Reuses a cached CSV if it already exists unless a fresh recomputation is forced.

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
    cache_path = match_dir_path / "obso_frames.csv"
    if cache_path.is_file() and not force_recompute:
        logger.info("Loading cached OBSO data from %s", cache_path)
        return pd.read_csv(cache_path)

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

    # --- DIAGNOSTIC (safe, does not raise) ---
    unique_vals = np.unique(smoothed[~pd.isna(smoothed)]) if hasattr(smoothed, "__len__") else []
    logger.info("smoothed owner unique values: %s", unique_vals)
    # -------------------------------------------

    # Map raw owner-encoding to "home"/"away".
    # IMPORTANT: adjust HOME_VAL / AWAY_VAL below to match what the diagnostic
    # log line above actually prints for your data. Common encodings:
    #   {1: home, -1: away}   or   {1: home, 0: away}
    HOME_VAL = 1
    AWAY_VAL = 2 # <-- change to 0 if np.unique(smoothed) shows {0, 1} instead of {1, -1}

    # Now compute OBSO for each frame
    records = []
    n_frames = len(periods)
    for i in range(0, n_frames, downsample):
        if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
            continue
        if pd.isna(smoothed[i]):
            continue
        if smoothed[i] == HOME_VAL:
            team = "home"
        elif smoothed[i] == AWAY_VAL:
            team = "away"
        else:
            continue

        # Determine direction for this team/period
        direction = attack_direction(team, int(periods[i]), home_dir_p1, away_dir_p1)

        # Compute EPV at ball location
        ball_epv = epv_value(epv_grid, ball_x[i], ball_y[i], pitch_length, pitch_width, direction)

        # Sample points in a circle around the ball
        step = 1.0  # metres
        x_range = np.arange(-radius, radius + step, step)
        y_range = np.arange(-radius, radius + step, step)
        max_epv = ball_epv
        for dx in x_range:
            for dy in y_range:
                if dx * dx + dy * dy > radius * radius:
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
    if not df.empty:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("Saved OBSO cache to %s", cache_path)
        logger.info("OBSO team counts: %s", df["team"].value_counts().to_dict())
    logger.info("Computed OBSO for %d frames (radius=%.1f m)", len(df), radius)
    return df


def aggregate_obso_by_window(obso_df, formations_df):
    """
    Average OBSO per formation window, per team (long format).

    Returns one row per (window, team) with columns:
        windowStartSec, windowEndSec, formation, period, team, mean_obso
    """
    if formations_df.empty:
        return pd.DataFrame()

    windows = formations_df[['windowStartSec', 'windowEndSec', 'formation', 'period']].drop_duplicates()

    results = []
    for _, win in windows.iterrows():
        start = win['windowStartSec']
        end = win['windowEndSec']
        period = win['period']
        formation = win['formation']

        for team in ('home', 'away'):
            mask = (
                (obso_df['period'] == period) &
                (obso_df['elapsed'] >= start) &
                (obso_df['elapsed'] < end) &
                (obso_df['team'] == team)
            )
            sub = obso_df[mask]
            mean_val = sub['obso'].mean() if not sub.empty else np.nan

            results.append({
                'windowStartSec': start,
                'windowEndSec': end,
                'formation': formation,
                'period': period,
                'team': team,
                'mean_obso': mean_val,
            })

    return pd.DataFrame(results)