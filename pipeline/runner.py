"""Pipeline helper functions built on package modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ..analytics.epv import das as epv_das
from ..analytics.formations import detector as formations
from ..artifacts import (
    EPVResult,
    FormationResult,
    epv_result_from_dataframes,
    formation_result_from_dataframe,
)

from ..analytics.obso import compute_obso_for_match, aggregate_obso_by_window

import matplotlib.pyplot as plt
from matplotlib.table import Table
import pandas as pd


from ..analytics.pitch_control import compute_pitch_control_for_match, aggregate_pitch_control_by_window
from ..analytics.pitch_control.artifact import PitchControlResult, pitch_control_result_from_frames

from ..io import paths as data_paths
from ..preprocessing import preprocess
from ..ui import timeline, viewer
from ..io.paths import match_dir

def preprocess_all_matches(match_id: str | None = None, raw_tracking_dir: str | None = None) -> None:
    """Run preprocessing over tracking files for one or all matches.

    Args:
        match_id: If given, process only this match. If None, process all.
        raw_tracking_dir: Override the raw tracking directory.
    """

    if raw_tracking_dir is not None:
        data_paths.RAW_TRACKING_DIR = Path(raw_tracking_dir)
        preprocess.RAW_TRACKING_DIR = str(raw_tracking_dir)
    preprocess.main(match_id=match_id)


def detect_formations(
    match_ids: Iterable[str] | None = None,
    processed_dir: str = formations.PROCESSED_DIR,
    window_seconds: int | None = None,
    stride_seconds: int | None = None,
) -> None:
    """Run the formation detector for one or more matches."""

    if match_ids is None:
        match_ids = [
            p.name for p in Path(processed_dir).iterdir()
            if p.is_dir()
        ]
    for match_id in match_ids:
        formations.process_match(match_id, processed_dir=processed_dir,
                                  window_seconds=window_seconds, stride_seconds=stride_seconds)


def detect_formations_artifact(
    match_id: str | int,
    processed_dir: str = formations.PROCESSED_DIR,
    window_seconds: int | None = None,
    stride_seconds: int | None = None,
) -> FormationResult:
    """Run formation detection and return a typed artifact."""

    formations_df = formations.process_match(
        str(match_id), processed_dir=processed_dir,
        window_seconds=window_seconds, stride_seconds=stride_seconds,
    )
    return formation_result_from_dataframe(match_id, formations_df)


def run_epv(match_id: str | int, processed_dir: str, epv_grid_path: str) -> tuple:
    """Run EPV/DAS analysis for a match and return the generated figures."""

    return epv_das.run_analysis(str(match_id), str(processed_dir), str(epv_grid_path))


def run_epv_artifact(
    match_id: str | int,
    processed_dir: str,
    epv_grid_path: str,
) -> EPVResult:
    """Run EPV/DAS analysis and return a typed artifact from the outputs."""

    epv_das.run_analysis(str(match_id), str(processed_dir), str(epv_grid_path))
    match_dir = Path(processed_dir) / str(match_id)
    import pandas as pd

    epv_df = pd.read_csv(match_dir / "epv_timeseries.csv")
    das_df = pd.read_csv(match_dir / "das_sequences.csv")
    return epv_result_from_dataframes(match_id, epv_df, das_df)


def run_timeline(match_id: str | int, processed_dir: str) -> object:
    """Render the formation timeline for a match."""

    return timeline.plot_formation_timeline(str(match_id), processed_dir=str(processed_dir))


def run_viewer(match_id: str | int, processed_dir: str, speed: float = 1.0) -> object:
    """Launch the interactive viewer for a match."""

    return viewer.run_app(str(match_id), str(processed_dir), speed)

# In Transitions/pipeline/runner.py

def compute_pitch_control_artifact(
    match_id: str | int,
    processed_dir: str,
    downsample: int = 1,
) -> PitchControlResult:
    """Compute pitch control and merge with formations."""
    frame_df = compute_pitch_control_for_match(match_id, processed_dir, downsample=downsample)
    
    # Load formations
    match_dir_path = match_dir(match_id, processed_dir)
    formations_path = match_dir_path / "formations.csv"
    if formations_path.is_file():
        formations_df = pd.read_csv(formations_path)
        window_df = aggregate_pitch_control_by_window(frame_df, formations_df)
    else:
        window_df = pd.DataFrame()
    
    return pitch_control_result_from_frames(match_id, frame_df, window_df)



def compute_obso_artifact(match_id, processed_dir, epv_grid_path=None, radius=5.0):
    """Compute OBSO and aggregate by formation windows."""
    import pandas as pd
    
    # Compute frame-level OBSO
    frame_df = compute_obso_for_match(match_id, processed_dir, epv_grid_path, radius)
    
    # Try to load formations
    match_dir_path = match_dir(match_id, processed_dir)
    formations_path = match_dir_path / "formations.csv"
    if formations_path.is_file():
        formations_df = pd.read_csv(formations_path)
        window_df = aggregate_obso_by_window(frame_df, formations_df)
    else:
        window_df = pd.DataFrame()
    
    return {"frame": frame_df, "window": window_df}



# --- Helper to create a table figure ---
def _df_to_table_figure(df, title, metric_col='mean_home_control'):
    """
    Create a bar chart for the given metric column per formation window.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
    
    # If metric_col not in df, try to find one
    if metric_col not in df.columns:
        for col in ['mean_home_control', 'mean_away_control', 'mean_obso']:
            if col in df.columns:
                metric_col = col
                break
        else:
            metric_col = None
    
    if metric_col is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "No metric column found", ha='center', va='center')
        return fig
    
    # Create a readable window label (e.g., "1-300s")
    df = df.copy()
    df['window'] = df['windowStartSec'].astype(str) + '-' + df['windowEndSec'].astype(str) + 's'
    
    # Pivot by team for side-by-side bars (if we have both home and away)
    # For now, just plot each row as a bar, color by team
    fig, ax = plt.subplots(figsize=(14, 6))
    
    teams = df['team']
    x = np.arange(len(df))
    width = 0.6
    
    colors = {'home': '#3498db', 'away': '#e74c3c'}  # blue for home, red for away
    bar_colors = [colors.get(t, 'gray') for t in teams]
    
    bars = ax.bar(x, df[metric_col], width, color=bar_colors, alpha=0.8)
    
    # Add formation labels on top of bars
    for i, (bar, formation) in enumerate(zip(bars, df['formation'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                formation, ha='center', va='bottom', fontsize=7, rotation=45)
    
    # Add period labels on x-axis
    ax.set_xticks(x)
    ax.set_xticklabels(df['window'], rotation=45, ha='right', fontsize=8)
    
    ax.set_ylabel(metric_col.replace('_', ' ').title())
    ax.set_xlabel('Window (start-end seconds)')
    ax.set_title(title, fontsize=14)
    
    # Add legend for teams
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#3498db', label='Home'),
                       Patch(facecolor='#e74c3c', label='Away')]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def pitch_control_figure(match_id, processed_dir, downsample=1):
    result = compute_pitch_control_artifact(match_id, processed_dir, downsample)
    window_df = result.window_control
    # Show mean_home_control (or you could choose mean_away_control)
    fig = _df_to_table_figure(window_df, f"Pitch Control – Match {match_id}", metric_col='mean_home_control')
    return fig, window_df

def obso_figure(match_id, processed_dir, epv_grid_path=None, radius=5.0):
    result = compute_obso_artifact(match_id, processed_dir, epv_grid_path, radius)
    window_df = result["window"]
    fig = _df_to_table_figure(window_df, f"OBSO – Match {match_id}", metric_col='mean_obso')
    return fig, window_df

