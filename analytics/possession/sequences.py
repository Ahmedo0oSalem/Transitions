"""Possession sequence helpers."""

from __future__ import annotations

import numpy as np

from .tracking import MIN_SEQUENCE_SECONDS

_TEAM_OF = {0: None, 1: "home", 2: "away"}


def detect_possession_sequences(owner_smoothed, periods, elapsed, fps,
								 min_seconds=MIN_SEQUENCE_SECONDS):
	"""Run-length encode smoothed owner states into possession sequences."""

	sequences = []

	for p in np.unique(periods):
		idx = np.where(periods == p)[0]
		o = owner_smoothed[idx]
		e = elapsed[idx]

		change_points = np.where(np.diff(o) != 0)[0] + 1
		starts = np.concatenate(([0], change_points))
		ends = np.concatenate((change_points, [len(o)]))

		raw_seqs = []
		for s, en in zip(starts, ends):
			end_sec = float(e[en]) if en < len(e) else float(e[en - 1]) + 1.0 / fps
			raw_seqs.append({
				"team": _TEAM_OF[int(o[s])],
				"period": int(p),
				"start_idx": int(idx[s]),
				"end_idx": int(idx[en - 1]),
				"start_sec": float(e[s]),
				"end_sec": end_sec,
			})

		merged = []
		i = 0
		while i < len(raw_seqs):
			seg = raw_seqs[i]
			dur = seg["end_sec"] - seg["start_sec"]
			can_merge = (
				dur < min_seconds
				and merged
				and i + 1 < len(raw_seqs)
				and merged[-1]["team"] == raw_seqs[i + 1]["team"]
				and merged[-1]["team"] is not None
			)
			if can_merge:
				merged[-1]["end_sec"] = seg["end_sec"]
				merged[-1]["end_idx"] = seg["end_idx"]
				i += 1
				continue
			merged.append(dict(seg))
			i += 1

		for seg in merged:
			seg["duration"] = seg["end_sec"] - seg["start_sec"]

		coalesced = []
		for seg in merged:
			if coalesced and coalesced[-1]["team"] == seg["team"]:
				coalesced[-1]["end_sec"] = seg["end_sec"]
				coalesced[-1]["end_idx"] = seg["end_idx"]
				coalesced[-1]["duration"] = coalesced[-1]["end_sec"] - coalesced[-1]["start_sec"]
			else:
				coalesced.append(dict(seg))

		sequences.extend(coalesced)

	return sequences


def forward_fill_owner(owner_smoothed, periods, elapsed, max_gap_seconds=3.0):
	"""Carry possession forward through short no-owner gaps."""

	filled = owner_smoothed.copy()

	for p in np.unique(periods):
		idx = np.where(periods == p)[0]
		o = filled[idx]
		e = elapsed[idx]

		i = 0
		n = len(o)
		while i < n:
			if o[i] == 0:
				j = i
				while j < n and o[j] == 0:
					j += 1
				gap_duration = float(e[j - 1] - e[i]) if j > i else 0.0
				prev_team = o[i - 1] if i > 0 else 0
				if prev_team != 0 and gap_duration <= max_gap_seconds:
					o[i:j] = prev_team
				i = j
			else:
				i += 1

		filled[idx] = o

	return filled
