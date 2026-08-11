"""EPV momentum helpers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ...io.paths import match_dir
from ..possession.tracking import attack_direction, epv_value, get_base_directions, load_epv_grid


def load_metadata(match_id, processed_dir):
	"""Load processed metadata and tracking file paths for a match."""

	match_dir_path = match_dir(match_id, processed_dir)
	metadata_path = match_dir_path / "metadata.json"
	tracking_path = match_dir_path / "tracking.jsonl.bz2"
	if not metadata_path.is_file() or not tracking_path.is_file():
		raise FileNotFoundError(f"Missing metadata.json / tracking.jsonl.bz2 under {match_dir_path}")
	with open(metadata_path, "r", encoding="utf-8") as f:
		metadata = json.load(f)
	return metadata, tracking_path, match_dir_path


def compute_frame_epv(periods, elapsed, ball_x, ball_y, owner_smoothed,
					   epv_grid, pitch_length, pitch_width, home_dir_p1, away_dir_p1):
	"""Attribute EPV to the current possession owner on each frame."""

	n = len(periods)
	signed_epv = np.zeros(n, dtype=np.float32)

	valid = ~np.isnan(ball_x) & ~np.isnan(ball_y) & (owner_smoothed != 0)
	idx = np.where(valid)[0]
	for i in idx:
		team = "home" if owner_smoothed[i] == 1 else "away"
		direction = attack_direction(team, int(periods[i]), home_dir_p1, away_dir_p1)
		val = epv_value(epv_grid, float(ball_x[i]), float(ball_y[i]), pitch_length, pitch_width, direction)
		signed_epv[i] = val if team == "home" else -val

	return signed_epv


def bucket_epv_by_second(periods, elapsed, signed_epv, valid=None):
    """Downsample per-frame signed EPV to one-second buckets.

    Averages only frames flagged as *valid* (ball tracked + owner assigned).
    Frames where the ball is missing or no owner exists contribute a forced
    zero in ``signed_epv`` and would bias the per-second mean downward, so
    they are excluded when *valid* is provided.  The number of valid frames
    per bucket is reported in the ``nValidFrames`` column.
    """
    if valid is None:
        valid = np.ones(len(signed_epv), dtype=bool)
    rows = []
    for p in np.unique(periods):
        mask = periods == p
        e = elapsed[mask]
        s = signed_epv[mask]
        v = valid[mask]
        bucket = np.floor(e).astype(int)
        for b in np.unique(bucket):
            m = bucket == b
            vm = m & v
            rows.append({
                "period": int(p), "secondIntoPeriod": int(b),
                "meanSignedEPV": float(np.mean(s[vm])) if vm.any() else 0.0,
                "nValidFrames": int(vm.sum()),
            })
    return pd.DataFrame(rows)
