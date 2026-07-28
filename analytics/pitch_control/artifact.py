"""Typed artifact for pitch control results."""

from dataclasses import dataclass, field
import pandas as pd


@dataclass
class PitchControlResult:
    match_id: str | int
    frame_control: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    window_control: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())


def pitch_control_result_from_frames(match_id, frame_df, window_df=None):
    return PitchControlResult(
        match_id=match_id,
        frame_control=frame_df,
        window_control=window_df if window_df is not None else pd.DataFrame(),
    )