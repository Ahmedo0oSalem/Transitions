"""EPV analysis artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import pandas as pd

from ..domain import DangerousAttackingSequence, EPVPoint


@dataclass(slots=True)
class EPVResult:
	"""Typed result for EPV momentum analysis."""

	match_id: str | int
	points: list[EPVPoint] = field(default_factory=list)
	dangerous_sequences: list[DangerousAttackingSequence] = field(default_factory=list)


def epv_result_from_dataframes(match_id: str | int, epv_df: pd.DataFrame, das_df: pd.DataFrame) -> EPVResult:
	"""Convert EPV and DAS dataframes into a typed artifact."""

	points: list[EPVPoint] = []
	for row in epv_df.to_dict(orient="records"):
		points.append(
			EPVPoint(
				period=int(row["period"]),
				second_into_period=int(row["secondIntoPeriod"]),
				mean_signed_epv=float(row["meanSignedEPV"]),
			)
		)

	dangerous_sequences: list[DangerousAttackingSequence] = []
	for row in das_df.to_dict(orient="records"):
		dangerous_sequences.append(
			DangerousAttackingSequence(
				team=str(row["team"]),
				period=int(row["period"]),
				start_sec=float(row["startSec"]),
				end_sec=float(row["endSec"]),
				duration=float(row["duration"]),
				peak_epv=float(row["peakEPV"]),
				is_das=bool(row["isDAS"]),
			)
		)

	return EPVResult(match_id=match_id, points=points, dangerous_sequences=dangerous_sequences)

