"""Dangerous Attacking Sequence helpers and EPV plotting."""

from __future__ import annotations

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("QtAgg" if "PyQt6" in sys.modules else ("MacOSX" if sys.platform == "darwin" else "TkAgg"))
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from ...analytics.possession import (
	attack_direction,
	detect_possession_sequences,
	epv_value,
	forward_fill_owner,
	get_base_directions,
	infer_fps,
	load_events,
	load_epv_grid,
	possession_sequences_from_events,
	smooth_owner,
	stream_ball_and_owner,
)
from .momentum import bucket_epv_by_second, compute_frame_epv, load_metadata
from ...core.config import EPV_DAS_MIN_DURATION_SECONDS, EPV_DAS_THRESHOLD, EPV_MOMENTUM_WINDOW_SECONDS, PROCESSED_DIR as PACKAGE_PROCESSED_DIR
from ...core.logger import get_logger
from ...io.paths import EPV_GRID_PATH

PROCESSED_DIR_DEFAULT = str(PACKAGE_PROCESSED_DIR)
EPV_GRID_DEFAULT = str(EPV_GRID_PATH)

DAS_EPV_THRESHOLD = EPV_DAS_THRESHOLD
DAS_MIN_DURATION_SECONDS = EPV_DAS_MIN_DURATION_SECONDS
MOMENTUM_WINDOW_SECONDS = EPV_MOMENTUM_WINDOW_SECONDS

BG_COLOR = "#0d1b2a"
GRID_COLOR = "#2a3a4a"
TEXT_COLOR = "#e6e6e6"
HOME_COLOR = "#e74c3c"
AWAY_COLOR = "#3498db"
logger = get_logger(__name__)


def evaluate_das(sequences, ball_x, ball_y, periods, elapsed, epv_grid, pitch_length, pitch_width,
                 home_dir_p1, away_dir_p1, threshold=DAS_EPV_THRESHOLD,
                 min_duration=DAS_MIN_DURATION_SECONDS):
    """Evaluate possession sequences and mark those whose peak EPV crosses the threshold."""
    rows = []
    for s in sequences:
        if s["team"] not in ("home", "away"):
            continue
        if s["duration"] < min_duration:
            continue
        mask = (periods == s["period"]) & (elapsed >= s["start_sec"]) & (elapsed <= s["end_sec"])
        idx = np.where(mask)[0]
        direction = attack_direction(s["team"], s["period"], home_dir_p1, away_dir_p1)
        peak = 0.0
        for i in idx:
            if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
                continue
            v = epv_value(epv_grid, float(ball_x[i]), float(ball_y[i]), pitch_length, pitch_width, direction)
            if v > peak:
                peak = v
        rows.append({
            "team": s["team"], "period": s["period"],
            "startSec": s["start_sec"], "endSec": s["end_sec"],
            "duration": s["duration"], "peakEPV": peak,
            "isDAS": peak >= threshold,
        })
    
    # FIX: Always return DataFrame with correct columns
    if not rows:
        return pd.DataFrame(columns=["team", "period", "startSec", "endSec", 
                                     "duration", "peakEPV", "isDAS"])
        
    return pd.DataFrame(rows)

def compute_period_offsets(metadata):
    """Convert per-period timestamps into a continuous match timeline.
    
    Handles:
    - Null/missing metadata (falls back to standard period lengths)
    - Extra time (periods 3 and 4)
    - Any number of periods
    """
    periods_meta = metadata.get("periods", {}) or {}
    offsets, boundaries = {}, []
    cursor = 0.0
    
    # Standard period lengths in seconds
    STANDARD_LENGTHS = {
        1: 2700,  # 45 min
        2: 2700,  # 45 min  
        3: 900,   # 15 min (extra time 1st half)
        4: 900,   # 15 min (extra time 2nd half)
    }
    
    for p_str in sorted(periods_meta.keys(), key=int):
        p = int(p_str)
        entry = periods_meta[p_str]
        offsets[p] = cursor
        
        # Safely get start and end times
        start_val = entry.get("start") if entry else None
        end_val = entry.get("end") if entry else None
        
        if start_val is not None and end_val is not None:
            length = float(end_val) - float(start_val)
        else:
            # Use standard length or 2700s as default
            length = STANDARD_LENGTHS.get(p, 2700)
            logger.warning(
                "Period %s has null metadata, using standard length %ss",
                p, length
            )
        
        cursor += length
        boundaries.append((cursor, p))
    
    return offsets, boundaries, cursor

def format_mmss(seconds, _pos=None):
	seconds = max(0, int(seconds))
	return f"{seconds // 60:02d}:{seconds % 60:02d}"


def plot_momentum(epv_df, das_df, offsets, boundaries, total_duration,
				   home_name, away_name, match_id, window_seconds=MOMENTUM_WINDOW_SECONDS):
	"""Plot the signed EPV momentum signal."""

	epv_df = epv_df.copy()
	epv_df["matchSec"] = epv_df["secondIntoPeriod"] + epv_df["period"].map(offsets)
	epv_df = epv_df.sort_values("matchSec")

	window = max(1, int(window_seconds))
	smoothed = epv_df["meanSignedEPV"].rolling(window, center=True, min_periods=1).mean()

	fig, ax = plt.subplots(figsize=(15, 6))
	fig.patch.set_facecolor(BG_COLOR)
	ax.set_facecolor(BG_COLOR)

	x = epv_df["matchSec"].to_numpy()
	y = smoothed.to_numpy()
	ax.fill_between(x, 0, y, where=(y >= 0), color=HOME_COLOR, alpha=0.6, zorder=2)
	ax.fill_between(x, 0, y, where=(y < 0), color=AWAY_COLOR, alpha=0.6, zorder=2)
	ax.axhline(0, color=GRID_COLOR, linewidth=1, zorder=1)

	if das_df is not None and not das_df.empty:
		das_only = das_df[das_df["isDAS"]]
		for _, row in das_only.iterrows():
			off = offsets.get(row["period"], 0.0)
			t = row["startSec"] + off
			color = HOME_COLOR if row["team"] == "home" else AWAY_COLOR
			marker = "^" if row["team"] == "home" else "v"
			y_pos = row["peakEPV"] if row["team"] == "home" else -row["peakEPV"]
			ax.scatter([t], [y_pos], marker=marker, s=70, color=color,
					   edgecolors="white", linewidths=0.8, zorder=3)

	for boundary_time, period in boundaries[:-1]:
		ax.axvline(boundary_time, color=GRID_COLOR, linewidth=1.2, linestyle="--", zorder=0)
		ax.text(boundary_time, 1.02, f"End P{period}", transform=ax.get_xaxis_transform(),
				ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

	ax.set_xlim(0, total_duration)
	ax.tick_params(colors=TEXT_COLOR)
	ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_mmss))
	for spine in ax.spines.values():
		spine.set_color(GRID_COLOR)
	ax.set_ylabel(f"<- {away_name}      EPV momentum      {home_name} ->", color=TEXT_COLOR)
	ax.set_xlabel("Match Time", color=TEXT_COLOR)
	ax.set_title(
		f"Match {match_id} â€” EPV Momentum (rolling {int(window_seconds)}s mean)\n"
		f"â–²/â–¼ markers = Dangerous Attacking Sequences (peak EPV â‰¥ {DAS_EPV_THRESHOLD})",
		color=TEXT_COLOR, fontsize=12
	)
	fig.tight_layout()
	return fig


def plot_das_timeline(das_df, offsets, boundaries, total_duration, home_name, away_name, match_id):
    """Plot dangerous attacking sequences on a match timeline."""
    fig, ax = plt.subplots(figsize=(15, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # FIX: Safely handle empty or missing DAS data
    if das_df is None or das_df.empty or "isDAS" not in das_df.columns:
        das_only = pd.DataFrame()
    else:
        das_only = das_df[das_df["isDAS"]].copy()
    
    # Add matchSec column safely
    if not das_only.empty:
        das_only["matchSec"] = das_only.apply(
            lambda r: r["startSec"] + offsets.get(r["period"], 0.0), axis=1
        )
    else:
        das_only["matchSec"] = pd.Series(dtype=float)
    
    for team, y, color, name in (("home", 1, HOME_COLOR, home_name), ("away", 0, AWAY_COLOR, away_name)):
        sub = das_only[das_only["team"] == team] if not das_only.empty else das_only
        if len(sub):
            ax.scatter(sub["matchSec"], [y] * len(sub), s=sub["peakEPV"] * 800 + 40,
                       color=color, alpha=0.85, edgecolors="white", linewidths=0.6, zorder=2)
    
    for boundary_time, period in boundaries[:-1]:
        ax.axvline(boundary_time, color=GRID_COLOR, linewidth=1.2, linestyle="--", zorder=0)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels([away_name, home_name], color=TEXT_COLOR)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xlim(0, total_duration)
    ax.tick_params(colors=TEXT_COLOR)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_mmss))
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.set_xlabel("Match Time", color=TEXT_COLOR)
    ax.set_title(f"Match {match_id} — Dangerous Attacking Sequences "
                 f"(marker size = peak EPV)", color=TEXT_COLOR, fontsize=12)
    fig.tight_layout()
    return fig


def run_analysis(match_id, processed_dir, epv_grid_path):
	"""Run the full EPV + DAS analysis pipeline for one match."""

	metadata, tracking_path, match_dir = load_metadata(match_id, processed_dir)
	pitch_length = metadata["pitch"]["length"]
	pitch_width = metadata["pitch"]["width"]
	home_name = (metadata.get("homeTeam", {}).get("shortName")
				 or metadata.get("homeTeam", {}).get("name", "Home"))
	away_name = (metadata.get("awayTeam", {}).get("shortName")
				 or metadata.get("awayTeam", {}).get("name", "Away"))

	logger.info("Loading EPV grid...")
	epv_grid = load_epv_grid(epv_grid_path)
	home_dir_p1, away_dir_p1 = get_base_directions(metadata)

	logger.info("Streaming ball position + possession...")
	periods, elapsed, ball_x, ball_y, owner = stream_ball_and_owner(
		tracking_path, pitch_length, pitch_width)
	fps = infer_fps(elapsed, periods)
	smoothed = smooth_owner(owner, periods, fps)
	sequences = detect_possession_sequences(smoothed, periods, elapsed, fps)

	events = load_events(match_dir)
	if events:
		logger.info("Using %s real events for possession sequences (ground truth).", len(events))
		home_team_id = metadata.get("homeTeam", {}).get("id")
		das_sequences_input = possession_sequences_from_events(events, home_team_id)
	else:
		logger.info("No events.json for this match -- falling back to proximity-proxy possession sequences (forward-filled).")
		das_owner = forward_fill_owner(smoothed, periods, elapsed)
		das_sequences_input = detect_possession_sequences(das_owner, periods, elapsed, fps)

	logger.info("Computing per-frame EPV...")
	signed_epv = compute_frame_epv(periods, elapsed, ball_x, ball_y, smoothed,
									epv_grid, pitch_length, pitch_width, home_dir_p1, away_dir_p1)
	epv_df = bucket_epv_by_second(periods, elapsed, signed_epv)

	logger.info("Evaluating Dangerous Attacking Sequences...")
	das_df = evaluate_das(das_sequences_input, ball_x, ball_y, periods, elapsed, epv_grid,
						   pitch_length, pitch_width, home_dir_p1, away_dir_p1)

	epv_out = match_dir / "epv_timeseries.csv"
	das_out = match_dir / "das_sequences.csv"
	epv_df.to_csv(epv_out, index=False)
	das_df.to_csv(das_out, index=False)
	logger.info("Wrote %s (%s rows)", epv_out, len(epv_df))
	logger.info("Wrote %s (%s rows)", das_out, len(das_df))

	n_das_home = int(((das_df["team"] == "home") & das_df["isDAS"]).sum()) if len(das_df) else 0
	n_das_away = int(((das_df["team"] == "away") & das_df["isDAS"]).sum()) if len(das_df) else 0

	logger.info("=== Summary ===")
	logger.info(
		"%s: %s Dangerous Attacking Sequences (out of %s possession sequences >= %ss)",
		home_name,
		n_das_home,
		int((das_df['team'] == 'home').sum()) if len(das_df) else 0,
		DAS_MIN_DURATION_SECONDS,
	)
	logger.info(
		"%s: %s Dangerous Attacking Sequences (out of %s possession sequences >= %ss)",
		away_name,
		n_das_away,
		int((das_df['team'] == 'away').sum()) if len(das_df) else 0,
		DAS_MIN_DURATION_SECONDS,
	)
	logger.info("Mean signed EPV (whole match, +ve = %s dominant): %.4f", home_name, epv_df['meanSignedEPV'].mean())

	offsets, boundaries, total_duration = compute_period_offsets(metadata)

	fig1 = plot_momentum(epv_df, das_df, offsets, boundaries, total_duration, home_name, away_name, match_id)
	fig2 = plot_das_timeline(das_df, offsets, boundaries, total_duration, home_name, away_name, match_id)
	return fig1, fig2


def main():
	"""CLI entry point for the EPV/DAS analysis package."""

	parser = argparse.ArgumentParser(description="EPV momentum + Dangerous Attacking Sequences analysis.")
	parser.add_argument("match_id")
	parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
	parser.add_argument("--epv-grid", default=EPV_GRID_DEFAULT)
	args = parser.parse_args()

	run_analysis(args.match_id, args.processed_dir, args.epv_grid)
	plt.show()
