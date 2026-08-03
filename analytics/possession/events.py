"""Event-derived possession helpers."""

from __future__ import annotations

import json
from pathlib import Path

from ...core.logger import get_logger

logger = get_logger(__name__)


def load_events(match_dir):
	"""Load events.json if it exists."""

	path = Path(match_dir) / "events.json"
	if not path.is_file():
		return None
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def possession_sequences_from_events(events, home_team_id):
	"""Build possession sequences directly from event data."""

	home_team_id = str(home_team_id)

	otb_events = [ev for ev in events
				  if ev.get("gameEventType") == "OTB" and not ev.get("nonEvent")]

	groups = {}
	for ev in otb_events:
		period = ev.get("period")
		seq = ev.get("sequence")
		team_id = ev.get("teamId")
		start = ev.get("periodElapsedTimeEstimate")
		if period is None or seq is None or team_id is None or start is None:
			continue
		groups.setdefault((period, seq), []).append(ev)

	sequences = []
	for (period, seq), evs in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
		team_counts = {}
		for e in evs:
			t = str(e.get("teamId"))
			team_counts[t] = team_counts.get(t, 0) + 1
		if len(team_counts) > 1:
			logger.warning(
				"    !! WARNING: sequence %s (period %s) has rows from multiple teamIds %s -- using the majority team. Check possessionEventType 'IT' rows or a sequence-numbering edge case.",
				seq,
				period,
				list(team_counts),
			)
		majority_team_id = max(team_counts, key=team_counts.get)

		starts = [e["periodElapsedTimeEstimate"] for e in evs]
		ends = [e["periodElapsedTimeEstimate"] + (e.get("duration") or 0.0) for e in evs]
		start_sec = min(starts)
		end_sec = max(ends)

		sequences.append({
			"team": "home" if majority_team_id == home_team_id else "away",
			"period": period,
			"start_sec": start_sec,
			"end_sec": end_sec,
		})

	for s in sequences:
		s["duration"] = s["end_sec"] - s["start_sec"]

	return sequences
