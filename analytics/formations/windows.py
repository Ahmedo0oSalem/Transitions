"""Formation windowing helpers."""

from __future__ import annotations

import bz2
import json
import math
from collections import defaultdict
from typing import Any

from ...core.config import FORMATION_STRIDE_SECONDS, FORMATION_WINDOW_SECONDS
from ...core.logger import get_logger

from .tracking_fields import extract_player_xy


COORDS_ARE_CENTERED = True
logger = get_logger(__name__)


def get_window_indices(elapsed_seconds, stride_seconds, window_seconds):
	"""Return all sliding-window indices covering a timestamp."""

	k_max = int(elapsed_seconds // stride_seconds)
	k_min = max(0, math.ceil((elapsed_seconds - window_seconds) / stride_seconds))
	return range(k_min, k_max + 1)


def accumulate_positions(tracking_path, goalkeepers, pitch_length, pitch_width,
						  team_keys=("homePlayers", "awayPlayers"),
						  stride_seconds=FORMATION_STRIDE_SECONDS,
						  window_seconds=FORMATION_WINDOW_SECONDS,
						  weight_lookup=None):
	"""Bucket outfield player coordinates into sliding windows.

	Parameters
	----------
	weight_lookup : dict or None
		``{period: {elapsed: {"homePlayers": w, "awayPlayers": w}}}`` as
		returned by ``frame_reliability.compute_frame_weights()``.
		When *None*, every frame gets weight 1.0 (original behaviour).

	Returns
	-------
	dict
		``{(team, period, window_index): {player_id: [(x, y, w), ...]}}``
		where each coordinate tuple now includes the per-frame weight *w*.
	"""
	buckets: dict[Any, dict[Any, list]] = defaultdict(lambda: defaultdict(list))
	x_shift = pitch_length / 2 if COORDS_ARE_CENTERED else 0.0
	y_shift = pitch_width / 2 if COORDS_ARE_CENTERED else 0.0

	# Pre-fetch weight_lookup references at the period level for speed
	period_weights: dict[int | None, dict] = {}
	if weight_lookup is not None:
		for p, lookup in weight_lookup.items():
			period_weights[p] = lookup

	n_lines = 0
	try:
		with bz2.open(tracking_path, "rt") as f:
			for line in f:
				frame = json.loads(line)
				n_lines += 1
				period = frame.get("period")
				elapsed = frame.get("periodElapsedTime")
				if period is None or elapsed is None:
					continue

				# Resolve weights for this frame (per team)
				frame_w: dict[str, float] = {"homePlayers": 1.0, "awayPlayers": 1.0}
				if period in period_weights:
					ew = period_weights[period].get(elapsed)
					if ew is not None:
						frame_w.update(ew)

				window_indices = list(get_window_indices(elapsed, stride_seconds, window_seconds))

				for team in team_keys:
					w = frame_w.get(team, 1.0)
					gk_ids = goalkeepers.get(team) or set()
					for p in frame.get(team, []):
						parsed = extract_player_xy(p)
						if parsed is None:
							continue
						pid, x, y = parsed
						if str(pid) in gk_ids:
							continue
						xyw = (x + x_shift, y + y_shift, w)
						for k in window_indices:
							buckets[(team, period, k)][pid].append(xyw)
	except EOFError:
		logger.warning(
			"    !! WARNING: %s appears truncated/corrupted (bz2 stream ended early after %s frames). Continuing with the frames successfully read.",
			tracking_path,
			n_lines,
		)

	return buckets
