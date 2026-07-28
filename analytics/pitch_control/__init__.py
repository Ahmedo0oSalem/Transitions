"""Pitch control analytics package."""

from .control import compute_pitch_control_for_match, aggregate_pitch_control_by_window
from .artifact import PitchControlResult

__all__ = [
    "compute_pitch_control_for_match",
    "aggregate_pitch_control_by_window",
    "PitchControlResult",
]