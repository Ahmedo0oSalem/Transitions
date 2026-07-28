"""Off-Ball Scoring Opportunity analytics package."""

from .obso import compute_obso_for_match, aggregate_obso_by_window

__all__ = ["compute_obso_for_match", "aggregate_obso_by_window"]