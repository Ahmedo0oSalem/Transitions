"""Pipeline helper functions built on package modules."""
from __future__ import annotations
from pathlib import Path
from typing import Iterable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..analytics.epv import das as epv_das
from ..analytics.formations import detector as formations
from ..analytics.formations import segments as formation_segments
from ..analytics.obso import compute_obso_for_match, aggregate_obso_by_window
from ..analytics.pitch_control import compute_pitch_control_for_match, aggregate_pitch_control_by_window
from ..analytics.pitch_control.artifact import PitchControlResult, pitch_control_result_from_frames
from ..artifacts import (
    EPVResult,
    FormationResult,
    epv_result_from_dataframes,
    formation_result_from_dataframe,
)
from ..io import paths as data_paths
from ..io.paths import match_dir
from ..preprocessing import preprocess
from ..ui import timeline, viewer


def preprocess_all_matches(match_id: str | None = None, raw_tracking_dir: str | None = None) -> None:
    """Run preprocessing over tracking files for one or all matches."""
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
    """Run the formation detector and segment aggregator for one or more matches."""
    if match_ids is None:
        match_ids = [
            p.name for p in Path(processed_dir).iterdir()
            if p.is_dir()
        ]
    for match_id in match_ids:
        # 1. Run standard windowed detection
        formations.process_match(
            match_id, 
            processed_dir=processed_dir,
            window_seconds=window_seconds, 
            stride_seconds=stride_seconds
        )
        # 2. Aggregate into continuous segments with rich properties
        formation_segments.build_formation_segments(match_id, processed_dir)


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
    m_dir = Path(processed_dir) / str(match_id)
    epv_df = pd.read_csv(m_dir / "epv_timeseries.csv")
    das_df = pd.read_csv(m_dir / "das_sequences.csv")
    return epv_result_from_dataframes(match_id, epv_df, das_df)


def run_timeline(match_id: str | int, processed_dir: str) -> object:
    """Render the formation timeline for a match."""
    return timeline.plot_formation_timeline(str(match_id), processed_dir=str(processed_dir))


def run_viewer(match_id: str | int, processed_dir: str, speed: float = 1.0) -> object:
    """Launch the interactive viewer for a match."""
    return viewer.run_app(str(match_id), str(processed_dir), speed)


# ---------------------------------------------------------------------------
# Pitch Control & OBSO Pipeline Helpers (Required by UI)
# ---------------------------------------------------------------------------

def compute_pitch_control_artifact(
    match_id: str | int,
    processed_dir: str,
    downsample: int = 1,
) -> PitchControlResult:
    """Compute pitch control and merge with formations."""
    frame_df = compute_pitch_control_for_match(match_id, processed_dir, downsample=downsample)
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
    frame_df = compute_obso_for_match(match_id, processed_dir, epv_grid_path, radius)
    match_dir_path = match_dir(match_id, processed_dir)
    formations_path = match_dir_path / "formations.csv"
    if formations_path.is_file():
        formations_df = pd.read_csv(formations_path)
        window_df = aggregate_obso_by_window(frame_df, formations_df)
    else:
        window_df = pd.DataFrame()
    return {"frame": frame_df, "window": window_df}


def _df_to_table_figure(df, title, metric_col='mean_home_control'):
    """Create a grouped bar chart (home vs away) per formation window."""
    if df.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')
        ax.text(0.5, 0.5, "No data available", ha='center', va='center')
        return fig
        
    df = df.copy()
    df['window'] = (
        df['period'].astype(str) + " | "
        + df['windowStartSec'].astype(str) + '-' + df['windowEndSec'].astype(str) + 's'
    )
    is_wide = 'mean_home_control' in df.columns and 'mean_away_control' in df.columns
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = {'home': '#3498db', 'away': '#e74c3c'}
    key_cols = ['period', 'windowStartSec', 'windowEndSec']
    
    if is_wide:
        windows = df[key_cols + ['window']].drop_duplicates(subset=key_cols).reset_index(drop=True)
        x = np.arange(len(windows))
        width = 0.35
        home_vals = df.groupby(key_cols)['mean_home_control'].mean()
        away_vals = df.groupby(key_cols)['mean_away_control'].mean()
        home_series = [home_vals.get((r.period, r.windowStartSec, r.windowEndSec), np.nan) for r in windows.itertuples()]
        away_series = [away_vals.get((r.period, r.windowStartSec, r.windowEndSec), np.nan) for r in windows.itertuples()]
        ax.bar(x - width / 2, home_series, width, color=colors['home'], alpha=0.85, label='Home')
        ax.bar(x + width / 2, away_series, width, color=colors['away'], alpha=0.85, label='Away')
        metric_label = 'Mean Control'
        tick_labels_full = windows['window']
    else:
        if metric_col not in df.columns:
            for col in ['mean_home_control', 'mean_away_control', 'mean_obso']:
                if col in df.columns:
                    metric_col = col
                    break
        teams_norm = df['team'].astype(str).str.strip().str.lower()
        windows = df[key_cols + ['window']].drop_duplicates(subset=key_cols).reset_index(drop=True)
        x = np.arange(len(windows))
        width = 0.35
        home_map = df[teams_norm == 'home'].groupby(key_cols)[metric_col].mean()
        away_map = df[teams_norm == 'away'].groupby(key_cols)[metric_col].mean()
        home_series = [home_map.get((r.period, r.windowStartSec, r.windowEndSec), np.nan) for r in windows.itertuples()]
        away_series = [away_map.get((r.period, r.windowStartSec, r.windowEndSec), np.nan) for r in windows.itertuples()]
        ax.bar(x - width / 2, home_series, width, color=colors['home'], alpha=0.85, label='Home')
        ax.bar(x + width / 2, away_series, width, color=colors['away'], alpha=0.85, label='Away')
        metric_label = metric_col.replace('_', ' ').title()
        tick_labels_full = windows['window']
        
    n = len(windows)
    stride = max(1, n // 20)
    tick_positions = x[::stride]
    tick_labels = tick_labels_full.iloc[::stride]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel(metric_label)
    ax.set_xlabel('Window')
    ax.set_title(title, fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig


def pitch_control_figure(match_id, processed_dir, downsample=1):
    result = compute_pitch_control_artifact(match_id, processed_dir, downsample)
    window_df = result.window_control
    fig = _df_to_table_figure(window_df, f"Pitch Control – Match {match_id}", metric_col='mean_home_control')
    return fig, window_df


def obso_figure(match_id, processed_dir, epv_grid_path=None, radius=5.0):
    result = compute_obso_artifact(match_id, processed_dir, epv_grid_path, radius)
    window_df = result["window"]
    fig = _df_to_table_figure(window_df, f"OBSO – Match {match_id}", metric_col='mean_obso')
    return fig, window_df