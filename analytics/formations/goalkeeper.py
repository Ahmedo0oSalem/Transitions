"""Goalkeeper helpers for formation detection."""

from __future__ import annotations

import bz2
import json
from collections import defaultdict

from ...core.config import FORMATION_GK_MIN_FRAMES
from ...core.logger import get_logger

from .tracking_fields import (
    PLAYER_ID_KEYS,
    PLAYER_NUMBER_KEYS,
    PLAYER_X_KEYS,
    PLAYER_Y_KEYS,
    REJECT_CONFIDENCE_VALUES,
    REJECT_VISIBILITY_VALUES,
    _get_first,
    extract_player_xy,
)

GK_MIN_FRAMES = FORMATION_GK_MIN_FRAMES
logger = get_logger(__name__)


def goalkeepers_from_metadata(metadata):
	"""Read goalkeeper identifiers from metadata."""

	gk_meta = metadata.get("goalkeepers", {}) or {}
	result = {}
	for team_key, side in (("homePlayers", "home"), ("awayPlayers", "away")):
		entry = gk_meta.get(side)
		if not entry:
			result[team_key] = None
			continue
		ids = {str(v) for v in entry.values() if v is not None}
		result[team_key] = ids if ids else None
	return result


def identify_goalkeepers(tracking_path, team_keys=("homePlayers", "awayPlayers")):
	"""Infer goalkeepers from average per-frame displacement."""

	last_xy = {team: {} for team in team_keys}
	total_dist = {team: defaultdict(float) for team in team_keys}
	frame_count = {team: defaultdict(int) for team in team_keys}

	n_lines = 0
	try:
		with bz2.open(tracking_path, "rt") as f:
			for line in f:
				frame = json.loads(line)
				n_lines += 1
				for team in team_keys:
					for p in frame.get(team, []):
						parsed = extract_player_xy(p)
						if parsed is None:
							continue
						pid, x, y = parsed
						frame_count[team][pid] += 1
						if pid in last_xy[team]:
							lx, ly = last_xy[team][pid]
							total_dist[team][pid] += ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
						last_xy[team][pid] = (x, y)
	except EOFError:
		logger.warning(
			"    !! WARNING: %s appears truncated/corrupted (bz2 stream ended early after %s frames). Continuing with the frames successfully read -- re-check/re-generate this file if results look incomplete.",
			tracking_path,
			n_lines,
		)

	goalkeepers = {}
	for team in team_keys:
		candidates = {
			pid: total_dist[team][pid] / frame_count[team][pid]
			for pid in total_dist[team]
			if frame_count[team][pid] >= GK_MIN_FRAMES
		}
		if not candidates:
			if frame_count[team]:
				candidates = {
					pid: total_dist[team][pid] / frame_count[team][pid]
					for pid in total_dist[team]
				}
			else:
				goalkeepers[team] = None
				continue
		goalkeepers[team] = {str(min(candidates, key=candidates.get))}
	return goalkeepers


def resolve_goalkeepers(tracking_path, metadata):
	"""Use metadata goalkeepers first, then fall back to inference."""

	from_roster = goalkeepers_from_metadata(metadata)
	missing = [team for team, ids in from_roster.items() if ids is None]

	if not missing:
		return from_roster

	logger.warning(
		"    goalkeeper(s) missing from roster for: %s -- falling back to distance-based estimation for those.",
		missing,
	)
	from_distance = identify_goalkeepers(tracking_path, team_keys=tuple(missing))

	resolved = dict(from_roster)
	for team in missing:
		resolved[team] = from_distance.get(team)
	return resolved
