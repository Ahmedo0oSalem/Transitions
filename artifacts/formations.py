"""Formation analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..domain import FormationWindow


@dataclass(slots=True)
class FormationResult:
	"""Typed result for a formation detection run."""

	match_id: str | int
	windows: list[FormationWindow] = field(default_factory=list)


def formation_result_from_dataframe(match_id: str | int, formations_df: pd.DataFrame) -> FormationResult:
	"""Convert a formation output dataframe into a typed artifact.

	Args:
		match_id: Match identifier.
		formations_df: Formation output dataframe.

	Returns:
		A typed formation artifact.
	"""

	windows: list[FormationWindow] = []
	for row in formations_df.to_dict(orient="records"):
		windows.append(
			FormationWindow(
				match_id=row["matchId"],
				team=row["team"],
				period=int(row["period"]),
				window_index=int(row["windowIndex"]),
				window_start_sec=float(row["windowStartSec"]),
				window_end_sec=float(row["windowEndSec"]),
				formation=str(row["formation"]),
				orientation=str(row["orientation"]),
				n_outfield_players=int(row["nOutfieldPlayers"]),
				n_frames=int(row["nFrames"]),
				avg_cost_per_player=float(row["avgCostPerPlayer"]),
			)
		)
	return FormationResult(match_id=match_id, windows=windows)

