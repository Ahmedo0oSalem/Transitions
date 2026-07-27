"""Possession analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from ..domain import PossessionSequence


@dataclass(slots=True)
class PossessionResult:
	"""Typed result for possession inference."""

	match_id: str | int
	sequences: list[PossessionSequence] = field(default_factory=list)


def possession_result_from_sequences(match_id: str | int, sequences: list[dict[str, object]]) -> PossessionResult:
	"""Convert possession sequence dictionaries into a typed artifact."""

	items: list[PossessionSequence] = []
	for sequence in sequences:
		items.append(
			PossessionSequence(
				team=sequence.get("team") if sequence.get("team") in ("home", "away") else None,
				period=int(sequence["period"]),
				start_sec=float(sequence["start_sec"]),
				end_sec=float(sequence["end_sec"]),
				duration=float(sequence["duration"]),
				start_idx=int(sequence["start_idx"]) if sequence.get("start_idx") is not None else None,
				end_idx=int(sequence["end_idx"]) if sequence.get("end_idx") is not None else None,
			)
		)
	return PossessionResult(match_id=match_id, sequences=items)

