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

    logger.info("[%s] accumulating positions into %ss windows (stride %ss)...", match_id, w_sec, s_sec)
    buckets = accumulate_positions(
        tracking_path,
        goalkeepers,
        pitch_length,
        pitch_width,
        stride_seconds=s_sec,
        window_seconds=w_sec,
    )

    rows = []
    for (team, period, window_index), players in sorted(buckets.items()):
        n_frames = sum(len(v) for v in players.values())
        if n_frames < FORMATION_MIN_FRAMES_PER_WINDOW:
            continue

        avg_xy = []
        for pid, coords in players.items():
            arr = np.array(coords, dtype=float)
            avg_xy.append(arr.mean(axis=0))
        avg_xy = np.array(avg_xy)

        if avg_xy.shape[0] < FORMATION_MIN_OUTFIELD_PLAYERS:
            continue

        orientation = get_orientation(team, period, home_team_start_left)
        formation, cost, _assigned_names = match_formation(avg_xy, templates, orientation)

        window_start = window_index * s_sec
        window_end = window_start + w_sec

        rows.append({
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
        })

    out_df = pd.DataFrame(rows).sort_values(["team", "period", "windowIndex"])
    out_path = match_dir_path / "formations.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("[%s] wrote %s rows -> %s", match_id, len(out_df), out_path)
    return out_df


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
