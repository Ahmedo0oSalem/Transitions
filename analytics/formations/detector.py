"""Public formation detection pipeline.
This module assembles the smaller formation helpers into the same
behaviour exposed by the legacy detect_formation.py script.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ...core.config import (
    FORMATION_MIN_FRAMES_PER_WINDOW,
    FORMATION_MIN_OUTFIELD_PLAYERS,
    FORMATION_STRIDE_SECONDS,
    FORMATION_WINDOW_SECONDS,
    PROCESSED_DIR as PACKAGE_PROCESSED_DIR,
)
from ...core.logger import get_logger
from ...io.paths import match_dir
from .goalkeeper import (
    PLAYER_ID_KEYS,
    PLAYER_X_KEYS,
    PLAYER_Y_KEYS,
    _get_first,
    resolve_goalkeepers,
)
from .matching import match_formation
from .templates import build_templates
from .windows import COORDS_ARE_CENTERED, accumulate_positions

logger = get_logger(__name__)

PROCESSED_DIR = str(PACKAGE_PROCESSED_DIR)

def get_orientation(team_key, period, home_team_start_left):
    """Return the template orientation for the team's attacking direction."""
    home_attacks_left_to_right = (
        home_team_start_left if period % 2 == 1 else not home_team_start_left
    )
    if team_key == "homePlayers":
        return "normal" if home_attacks_left_to_right else "flipped"
    return "flipped" if home_attacks_left_to_right else "normal"

def compute_spatial_metrics(players_dict, pitch_length, pitch_width):
    """
    Compute spatial metrics for a single window's player positions.
    """
    player_centroids = []
    for pid, triples in players_dict.items():
        arr = np.array(triples, dtype=float)
        w_sum = arr[:, 2].sum()
        if w_sum <= 0:
            continue
        wx = (arr[:, 0] * arr[:, 2]).sum() / w_sum
        wy = (arr[:, 1] * arr[:, 2]).sum() / w_sum
        player_centroids.append((wx, wy))
    
    if len(player_centroids) < 2:
        return {}
    
    player_arr = np.array(player_centroids)
    xs = player_arr[:, 0]
    ys = player_arr[:, 1]
    
    centroid_x = np.mean(xs)
    centroid_y = np.mean(ys)
    
    distances_to_centroid = np.sqrt((xs - centroid_x)**2 + (ys - centroid_y)**2)
    mean_compactness = float(np.mean(distances_to_centroid))
    std_compactness = float(np.std(distances_to_centroid))
    
    width = float(np.max(xs) - np.min(xs))
    depth = float(np.max(ys) - np.min(ys))
    
    return {
        "mean_compactness": mean_compactness,
        "std_compactness": std_compactness,
        "mean_center_x": centroid_x,
        "mean_center_y": centroid_y,
        "mean_width": width,
        "mean_depth": depth,
        "player_centroids": player_centroids,
    }

def process_match(match_id, processed_dir=PROCESSED_DIR, window_seconds=None, stride_seconds=None):
    """Run formation detection for a single match."""
    w_sec = window_seconds if window_seconds is not None else FORMATION_WINDOW_SECONDS
    s_sec = stride_seconds if stride_seconds is not None else FORMATION_STRIDE_SECONDS
    
    match_dir_path = match_dir(match_id, processed_dir)
    metadata_path = match_dir_path / "metadata.json"
    tracking_path = match_dir_path / "tracking.jsonl.bz2"
    
    if not metadata_path.is_file() or not tracking_path.is_file():
        logger.warning("[%s] processed metadata/tracking missing, skipping.", match_id)
        return
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    
    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    home_team_start_left = metadata["homeTeamStartLeft"]
    
    logger.info("[%s] building formation templates (%sx%s)...", match_id, pitch_length, pitch_width)
    templates = build_templates(pitch_length, pitch_width)
    
    logger.info("[%s] resolving goalkeepers...", match_id)
    goalkeepers = resolve_goalkeepers(tracking_path, metadata)
    logger.info("[%s] goalkeepers: %s", match_id, goalkeepers)
    
    events_path = match_dir_path / "events.json"
    events = None
    if events_path.is_file():
        with open(events_path, "r", encoding="utf-8") as f:
            events = json.load(f)
        logger.info("[%s] loaded %s events", match_id, len(events))
    
    weight_lookup = None
    if events is not None:
        from .frame_reliability import compute_frame_weights, MIN_WINDOW_CONFIDENCE as _MIN_CONF
        logger.info("[%s] computing per-frame reliability weights...", match_id)
        weight_lookup = compute_frame_weights(
            tracking_path, metadata, events=events, goalkeepers=goalkeepers,
        )
    else:
        _MIN_CONF = 0.0
    
    logger.info("[%s] accumulating positions into %ss windows (stride %ss)...", match_id, w_sec, s_sec)
    buckets = accumulate_positions(
        tracking_path,
        goalkeepers,
        pitch_length,
        pitch_width,
        stride_seconds=s_sec,
        window_seconds=w_sec,
        weight_lookup=weight_lookup,
    )
    
    rows = []
    for (team, period, window_index), players in sorted(buckets.items()):
        n_frames = sum(len(v) for v in players.values())
        if n_frames < FORMATION_MIN_FRAMES_PER_WINDOW:
            continue
        
        avg_xy = []
        window_weight_sum = 0.0
        for pid, triples in players.items():
            arr = np.array(triples, dtype=float)
            w_sum = arr[:, 2].sum()
            if w_sum <= 0:
                continue
            wx = (arr[:, 0] * arr[:, 2]).sum() / w_sum
            wy = (arr[:, 1] * arr[:, 2]).sum() / w_sum
            avg_xy.append((wx, wy))
            window_weight_sum += w_sum
        
        if not avg_xy:
            continue
        
        avg_xy = np.array(avg_xy)
        if avg_xy.shape[0] < FORMATION_MIN_OUTFIELD_PLAYERS:
            continue
        
        spatial_metrics = compute_spatial_metrics(players, pitch_length, pitch_width)
        
        orientation = get_orientation(team, period, home_team_start_left)
        formation, cost, _assigned_names = match_formation(avg_xy, templates, orientation)
        
        fit_quality = 1.0 / (1.0 + float(cost))
        confidence = window_weight_sum * fit_quality
        
        if events is not None and confidence < _MIN_CONF:
            logger.debug(
                "[%s] dropping window (team=%s, period=%s, idx=%s): "
                "confidence %.4f < %.4f",
                match_id, team, period, window_index, confidence, _MIN_CONF,
            )
            continue
        
        window_start = window_index * s_sec
        window_end = window_start + w_sec
        
        row = {
            "matchId": match_id,
            "team": "home" if team == "homePlayers" else "away",
            "period": period,
            "windowIndex": window_index,
            "windowStartSec": window_start,
            "windowEndSec": window_end,
            "nOutfieldPlayers": avg_xy.shape[0],
            "nFrames": n_frames,
            "formation": formation,
            "orientation": orientation,
            "avgCostPerPlayer": round(float(cost), 3),
            "confidence": round(float(confidence), 4),
        }
        
        if spatial_metrics:
            row["meanCompactness"] = round(spatial_metrics.get("mean_compactness", 0), 3)
            row["stdCompactness"] = round(spatial_metrics.get("std_compactness", 0), 3)
            row["meanCenterX"] = round(spatial_metrics.get("mean_center_x", 0), 2)
            row["meanCenterY"] = round(spatial_metrics.get("mean_center_y", 0), 2)
            row["meanWidth"] = round(spatial_metrics.get("mean_width", 0), 2)
            row["meanDepth"] = round(spatial_metrics.get("mean_depth", 0), 2)
            row["_player_centroids"] = spatial_metrics.get("player_centroids", [])
        
        rows.append(row)
    
    out_df = pd.DataFrame(rows).sort_values(["team", "period", "windowIndex"])
    
    if len(out_df) > 0:
        out_df = _calculate_displacement_and_velocity(out_df)
    
    cols_to_drop = [c for c in out_df.columns if c.startswith("_")]
    if cols_to_drop:
        out_df = out_df.drop(columns=cols_to_drop)
    
    out_path = match_dir_path / "formations.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("[%s] wrote %s rows -> %s", match_id, len(out_df), out_path)
    
    return out_df

def _calculate_displacement_and_velocity(df):
    """
    Calculate net centroid displacement and mean velocity between consecutive windows.
    """
    if "netCentroidDisplacement" not in df.columns:
        df["netCentroidDisplacement"] = 0.0
    if "meanCentroidVelocity" not in df.columns:
        df["meanCentroidVelocity"] = 0.0
        
    for (team, period), grp in df.groupby(["team", "period"]):
        grp = grp.sort_values("windowIndex")
        centroids = []
        timestamps = []
        indices = []
        
        for idx, row in grp.iterrows():
            if "_player_centroids" in row and row["_player_centroids"]:
                centroids_arr = np.array(row["_player_centroids"])
                centroid = centroids_arr.mean(axis=0)
                centroids.append(centroid)
                timestamps.append(row["windowStartSec"])
                indices.append(idx)
        
        if not centroids:
            continue
            
        displacements = [0.0]
        velocities = [0.0]
        for i in range(1, len(centroids)):
            prev_centroid = centroids[i-1]
            curr_centroid = centroids[i]
            displacement = np.linalg.norm(curr_centroid - prev_centroid)
            time_diff = timestamps[i] - timestamps[i-1]
            velocity = displacement / time_diff if time_diff > 0 else 0.0
            displacements.append(displacement)
            velocities.append(velocity)
        
        df.loc[indices, "netCentroidDisplacement"] = pd.Series(displacements, index=indices)
        df.loc[indices, "meanCentroidVelocity"] = pd.Series(velocities, index=indices)
    
    return df

def main():
    """CLI entry point compatible with the legacy script."""
    parser = argparse.ArgumentParser(description="Detect formations per time window from tracking data.")
    parser.add_argument("match_ids", nargs="*", help="Match IDs to process (folder names under Processed_Tracking). "
                                                       "If omitted, processes every match found.")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    parser.add_argument("--window-seconds", type=int, default=FORMATION_WINDOW_SECONDS,
                         help="Length of each detection window, in seconds.")
    parser.add_argument("--stride-seconds", type=int, default=None,
                         help="How far the window slides forward between readings, in seconds. "
                              "Defaults to --window-seconds (plain non-overlapping windows). "
                              "Set smaller than --window-seconds for an overlapping/sliding window "
                              "(e.g. --window-seconds 300 --stride-seconds 60 = a 5-minute window, "
                              "re-evaluated every minute).")
    args = parser.parse_args()
    
    w_sec = args.window_seconds
    s_sec = args.stride_seconds if args.stride_seconds is not None else w_sec
    
    if args.match_ids:
        match_ids = args.match_ids
    else:
        match_ids = [
            p.name for p in Path(args.processed_dir).iterdir()
            if p.is_dir()
        ]
    
    logger.info("Processing %s match(es)...", len(match_ids))
    for match_id in match_ids:
        process_match(match_id, processed_dir=args.processed_dir, window_seconds=w_sec, stride_seconds=s_sec)
    logger.info("Done!")

if __name__ == "__main__":
    main()