"""
epv_das_analysis.py

Computes two things across the match timeline, from tracking data alone
(no event/pass data required):

1. EPV(t) -- Expected Possession Value over time. At every frame, whoever
   is closest to the ball (per possession.py's proximity heuristic) is
   attributed the value of a real, published EPV surface (Fernandez/
   Bornn/Cervone-style, via Laurie Shaw's Friends-of-Tracking
   implementation -- EPV_grid.csv, downloaded verbatim from
   https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking)
   looked up at the ball's location, oriented for whichever direction
   that team is attacking. This gives a continuous "momentum" signal:
   positive when the home team is holding the ball somewhere dangerous,
   negative when the away team is.

   IMPORTANT CAVEAT: this is a TRACKING-ONLY adaptation of EPV. The
   standard/textbook version (as in Laurie Shaw's tutorials) computes
   "EPV added" per pass using event data (pass start/end + a pitch
   control model) -- you don't have event data, so this instead reports
   the instantaneous value of wherever the ball currently is. It answers
   "how dangerous is this team's current situation", not "how much value
   did that specific pass add". Related, but not the same metric.

2. DAS -- Dangerous Attacking Sequences. Every possession sequence
   (from possession.py) is checked against a danger threshold: if the
   ball's EPV value (correctly oriented for the possessing team) ever
   exceeds DAS_EPV_THRESHOLD during that sequence, it's flagged as a DAS.
   This is MY assumption about what "DAS" means here, since it wasn't a
   term with an established, universal definition I could find in the
   tracking-data literature -- adjust DAS_EPV_THRESHOLD or the whole
   definition below if you had something more specific in mind.

USAGE:
    python epv_das_analysis.py <match_id> [--processed-dir DIR] [--epv-grid EPV_grid.csv]

REQUIRES:
    Processed_Tracking/<match_id>/metadata.json
    Processed_Tracking/<match_id>/tracking.jsonl.bz2
    possession.py, detect_formations.py, EPV_grid.csv in the same folder.

OUTPUTS (written into Processed_Tracking/<match_id>/):
    epv_timeseries.csv   -- 1-second-resolution EPV signal across match time
    das_sequences.csv     -- one row per possession sequence, with peak EPV
                             and whether it was flagged as a DAS
    Two plots: EPV momentum chart (with DAS moments marked), and a DAS-only
    timeline.
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import possession as pos

PROCESSED_DIR_DEFAULT = "Processed_Tracking"
EPV_GRID_DEFAULT = "EPV_grid.csv"

# ==== DAS definition (see caveat above -- my best-guess interpretation) ====
DAS_EPV_THRESHOLD = 0.15       # ball reaching this EPV value during a
                                 # possession sequence flags it as "dangerous"
DAS_MIN_DURATION_SECONDS = 2.0  # ignore possession blips shorter than this

MOMENTUM_WINDOW_SECONDS = 45.0  # smoothing window for the EPV momentum chart

BG_COLOR = "#0d1b2a"
GRID_COLOR = "#2a3a4a"
TEXT_COLOR = "#e6e6e6"
HOME_COLOR = "#e74c3c"
AWAY_COLOR = "#3498db"


def load_metadata(match_id, processed_dir):
    match_dir = os.path.join(processed_dir, str(match_id))
    metadata_path = os.path.join(match_dir, "metadata.json")
    tracking_path = os.path.join(match_dir, "tracking.jsonl.bz2")
    if not os.path.exists(metadata_path) or not os.path.exists(tracking_path):
        raise FileNotFoundError(f"Missing metadata.json / tracking.jsonl.bz2 under {match_dir}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return metadata, tracking_path, match_dir


def compute_frame_epv(periods, elapsed, ball_x, ball_y, owner_smoothed,
                       epv_grid, pitch_length, pitch_width, home_dir_p1, away_dir_p1):
    """
    Per-frame EPV attributed to whichever team's smoothed owner is
    active. Returns a single 'signed' array: +EPV when home has the
    ball, -EPV when away has it, 0 when no one does / ball untracked.
    """
    n = len(periods)
    signed_epv = np.zeros(n, dtype=np.float32)

    valid = ~np.isnan(ball_x) & ~np.isnan(ball_y) & (owner_smoothed != 0)
    idx = np.where(valid)[0]
    for i in idx:
        team = "home" if owner_smoothed[i] == 1 else "away"
        direction = pos.attack_direction(team, int(periods[i]), home_dir_p1, away_dir_p1)
        val = pos.epv_value(epv_grid, float(ball_x[i]), float(ball_y[i]),
                             pitch_length, pitch_width, direction)
        signed_epv[i] = val if team == "home" else -val

    return signed_epv


def bucket_epv_by_second(periods, elapsed, signed_epv):
    """Downsamples the per-frame signed EPV series to 1-second buckets
    per period (keeps the output CSV a sane size)."""
    rows = []
    for p in np.unique(periods):
        mask = periods == p
        e = elapsed[mask]
        s = signed_epv[mask]
        bucket = np.floor(e).astype(int)
        for b in np.unique(bucket):
            m = bucket == b
            rows.append({"period": int(p), "secondIntoPeriod": int(b),
                          "meanSignedEPV": float(np.mean(s[m]))})
    return pd.DataFrame(rows)


def evaluate_das(sequences, ball_x, ball_y, periods, epv_grid, pitch_length, pitch_width,
                  home_dir_p1, away_dir_p1, threshold=DAS_EPV_THRESHOLD,
                  min_duration=DAS_MIN_DURATION_SECONDS):
    """For every possession sequence with a clear team, finds the peak
    EPV reached during it (correctly oriented for that team) and flags
    it as a DAS if that peak clears `threshold`."""
    rows = []
    for s in sequences:
        if s["team"] not in ("home", "away"):
            continue
        if s["duration"] < min_duration:
            continue
        i0, i1 = s["start_idx"], s["end_idx"]
        direction = pos.attack_direction(s["team"], s["period"], home_dir_p1, away_dir_p1)
        peak = 0.0
        for i in range(i0, i1 + 1):
            if np.isnan(ball_x[i]) or np.isnan(ball_y[i]):
                continue
            v = pos.epv_value(epv_grid, float(ball_x[i]), float(ball_y[i]),
                               pitch_length, pitch_width, direction)
            if v > peak:
                peak = v
        rows.append({
            "team": s["team"], "period": s["period"],
            "startSec": s["start_sec"], "endSec": s["end_sec"],
            "duration": s["duration"], "peakEPV": peak,
            "isDAS": peak >= threshold,
        })
    return pd.DataFrame(rows)


def compute_period_offsets(metadata):
    periods_meta = metadata.get("periods", {}) or {}
    offsets, boundaries = {}, []
    cursor = 0.0
    for p_str in sorted(periods_meta.keys(), key=int):
        p = int(p_str)
        entry = periods_meta[p_str]
        offsets[p] = cursor
        length = float(entry["end"]) - float(entry["start"])
        cursor += length
        boundaries.append((cursor, p))
    return offsets, boundaries, cursor


def format_mmss(seconds, _pos=None):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def plot_momentum(epv_df, das_df, offsets, boundaries, total_duration,
                   home_name, away_name, match_id, window_seconds=MOMENTUM_WINDOW_SECONDS):
    epv_df = epv_df.copy()
    epv_df["matchSec"] = epv_df["secondIntoPeriod"] + epv_df["period"].map(offsets)
    epv_df = epv_df.sort_values("matchSec")

    # Rolling smoothing over a uniform 1-second grid (already 1s-bucketed,
    # so window in samples == window in seconds).
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
        f"Match {match_id} \u2014 EPV Momentum (rolling {int(window_seconds)}s mean)\n"
        f"\u25b2/\u25bc markers = Dangerous Attacking Sequences (peak EPV \u2265 {DAS_EPV_THRESHOLD})",
        color=TEXT_COLOR, fontsize=12
    )
    fig.tight_layout()
    return fig


def plot_das_timeline(das_df, offsets, boundaries, total_duration, home_name, away_name, match_id):
    fig, ax = plt.subplots(figsize=(15, 3.5))
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)

    das_only = das_df[das_df["isDAS"]].copy() if das_df is not None else pd.DataFrame()
    das_only["matchSec"] = das_only.apply(lambda r: r["startSec"] + offsets.get(r["period"], 0.0), axis=1) \
        if not das_only.empty else []

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
    ax.set_title(f"Match {match_id} \u2014 Dangerous Attacking Sequences "
                 f"(marker size = peak EPV)", color=TEXT_COLOR, fontsize=12)
    fig.tight_layout()
    return fig


def run_analysis(match_id, processed_dir, epv_grid_path):
    metadata, tracking_path, match_dir = load_metadata(match_id, processed_dir)
    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    home_name = (metadata.get("homeTeam", {}).get("shortName")
                 or metadata.get("homeTeam", {}).get("name", "Home"))
    away_name = (metadata.get("awayTeam", {}).get("shortName")
                 or metadata.get("awayTeam", {}).get("name", "Away"))

    print("Loading EPV grid...")
    epv_grid = pos.load_epv_grid(epv_grid_path)
    home_dir_p1, away_dir_p1 = pos.get_base_directions(metadata)

    print("Streaming ball position + possession...")
    periods, elapsed, ball_x, ball_y, owner = pos.stream_ball_and_owner(
        tracking_path, pitch_length, pitch_width)
    fps = pos.infer_fps(elapsed, periods)
    smoothed = pos.smooth_owner(owner, periods, fps)
    sequences = pos.detect_possession_sequences(smoothed, periods, elapsed, fps)

    # DAS specifically uses forward-filled attribution: the most dangerous
    # instant of a possession (a shot/cross in flight) is exactly when the
    # ball separates from any single player, which the raw possession
    # sequences above would treat as a break. See possession.py's
    # forward_fill_owner docstring for why this matters.
    das_owner = pos.forward_fill_owner(smoothed, periods, elapsed)
    das_sequences_input = pos.detect_possession_sequences(das_owner, periods, elapsed, fps)

    print("Computing per-frame EPV...")
    signed_epv = compute_frame_epv(periods, elapsed, ball_x, ball_y, smoothed,
                                    epv_grid, pitch_length, pitch_width, home_dir_p1, away_dir_p1)
    epv_df = bucket_epv_by_second(periods, elapsed, signed_epv)

    print("Evaluating Dangerous Attacking Sequences...")
    das_df = evaluate_das(das_sequences_input, ball_x, ball_y, periods, epv_grid,
                           pitch_length, pitch_width, home_dir_p1, away_dir_p1)

    epv_out = os.path.join(match_dir, "epv_timeseries.csv")
    das_out = os.path.join(match_dir, "das_sequences.csv")
    epv_df.to_csv(epv_out, index=False)
    das_df.to_csv(das_out, index=False)
    print(f"Wrote {epv_out} ({len(epv_df)} rows)")
    print(f"Wrote {das_out} ({len(das_df)} rows)")

    n_das_home = int(((das_df["team"] == "home") & das_df["isDAS"]).sum()) if len(das_df) else 0
    n_das_away = int(((das_df["team"] == "away") & das_df["isDAS"]).sum()) if len(das_df) else 0
    total_home_epv_time = float(das_df.loc[(das_df["team"] == "home"), "duration"].sum()) if len(das_df) else 0
    total_away_epv_time = float(das_df.loc[(das_df["team"] == "away"), "duration"].sum()) if len(das_df) else 0

    print("\n=== Summary ===")
    print(f"{home_name}: {n_das_home} Dangerous Attacking Sequences "
          f"(out of {int((das_df['team']=='home').sum()) if len(das_df) else 0} possession sequences >= {DAS_MIN_DURATION_SECONDS}s)")
    print(f"{away_name}: {n_das_away} Dangerous Attacking Sequences "
          f"(out of {int((das_df['team']=='away').sum()) if len(das_df) else 0} possession sequences >= {DAS_MIN_DURATION_SECONDS}s)")
    print(f"Mean signed EPV (whole match, +ve = {home_name} dominant): {epv_df['meanSignedEPV'].mean():.4f}")

    offsets, boundaries, total_duration = compute_period_offsets(metadata)

    fig1 = plot_momentum(epv_df, das_df, offsets, boundaries, total_duration, home_name, away_name, match_id)
    fig2 = plot_das_timeline(das_df, offsets, boundaries, total_duration, home_name, away_name, match_id)
    return fig1, fig2


def main():
    parser = argparse.ArgumentParser(description="EPV momentum + Dangerous Attacking Sequences analysis.")
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
    parser.add_argument("--epv-grid", default=EPV_GRID_DEFAULT)
    args = parser.parse_args()

    run_analysis(args.match_id, args.processed_dir, args.epv_grid)
    plt.show()


if __name__ == "__main__":
    main()
