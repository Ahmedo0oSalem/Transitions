"""Future ratings artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class RatingsResult:
	"""Typed result placeholder for future player ratings."""

	match_id: str | int
	ratings: dict[str, float] = field(default_factory=dict)

