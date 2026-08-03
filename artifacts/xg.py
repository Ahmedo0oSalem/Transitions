"""Future xG artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class XGResult:
	"""Typed result placeholder for future xG analysis."""

	match_id: str | int
	shots: list[dict] = field(default_factory=list)

