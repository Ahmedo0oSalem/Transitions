"""Formation windowing helpers."""

from __future__ import annotations

import bz2
import json
import math
from collections import defaultdict

from ...core.config import FORMATION_STRIDE_SECONDS, FORMATION_WINDOW_SECONDS
from ...core.logger import get_logger

from .goalkeeper import extract_player_xy


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
						  window_seconds=FORMATION_WINDOW_SECONDS):
	"""Bucket outfield player coordinates into sliding windows."""

	buckets = defaultdict(lambda: defaultdict(list))
	x_shift = pitch_length / 2 if COORDS_ARE_CENTERED else 0.0
	y_shift = pitch_width / 2 if COORDS_ARE_CENTERED else 0.0

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

				window_indices = list(get_window_indices(elapsed, stride_seconds, window_seconds))

				for team in team_keys:
					gk_ids = goalkeepers.get(team) or set()
					for p in frame.get(team, []):
						parsed = extract_player_xy(p)
						if parsed is None:
							continue
						pid, x, y = parsed
						if str(pid) in gk_ids:
							continue
						xy = (x + x_shift, y + y_shift)
						for k in window_indices:
							buckets[(team, period, k)][pid].append(xy)
	except EOFError:
		logger.warning(
			"    !! WARNING: %s appears truncated/corrupted (bz2 stream ended early after %s frames). Continuing with the frames successfully read.",
			tracking_path,
			n_lines,
		)

	return buckets
