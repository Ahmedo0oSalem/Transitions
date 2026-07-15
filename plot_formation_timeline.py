"""
plot_formation_timeline.py

Plots each team's detected formation over time as a colored "piano roll":
one row per formation, colored horizontal bars showing when that
formation was detected -- one bar per POSSESSION SEQUENCE (not per raw
sliding-window row).

CHANGE FROM THE PREVIOUS VERSION: bars now correspond to possession
sequences (via possession.py's proximity-based "who has the ball"
heuristic, run-length encoded and flicker-smoothed) rather than raw,
overlapping formations.csv rows. For each possession sequence, we look
up whichever formations.csv window was "current" at that sequence's
midpoint and use that as the sequence's formation label. This is closer
to "what shape was this team in during this spell of play" than the
previous raw/overlapping view, at the cost of depending on the
possession heuristic (see possession.py's caveat: no real possession
events exist in this data, so treat sequence boundaries as approximate).

USAGE:
    python plot_formation_timeline.py <match_id> [--processed-dir DIR]

REQUIRES:
    Processed_Tracking/<match_id>/formations.csv       (from detect_formations.py)
    Processed_Tracking/<match_id>/metadata.json
    Processed_Tracking/<match_id>/tracking.jsonl.bz2   (possession sequences are
                                                         derived directly from this)
    possession.py and detect_formations.py in the same folder as this script.
"""

import os
import sys
import json
import argparse
from collections import Counter

import matplotlib
matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

import possession as pos

PROCESSED_DIR_DEFAULT = "Processed_Tracking"

BG_COLOR = "#0d1b2a"
GRID_COLOR = "#2a3a4a"
BASELINE_COLOR = "#3a4a5a"
TEXT_COLOR = "#e6e6e6"

# Distinct, consistent color per formation (same formation = same color in
# both teams' panels). tab20 + tab20b gives 40 distinguishable colors,
# comfortably more than any match should realistically produce.
_PALETTE = [matplotlib.colors.to_hex(c) for c in
            list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)]


def load_data(match_id, processed_dir):
    folder = os.path.join(processed_dir, str(match_id))
    formations_path = os.path.join(folder, "formations.csv")
    metadata_path = os.path.join(folder, "metadata.json")
    tracking_path = os.path.join(folder, "tracking.jsonl.bz2")

    if not os.path.exists(formations_path):
        raise FileNotFoundError(
            f"{formations_path} not found -- run detect_formations.py for "
            f"match {match_id} first."
        )
    if not os.path.exists(tracking_path):
        raise FileNotFoundError(
            f"{tracking_path} not found -- possession sequences are derived "
            f"directly from the tracking file."
        )

    formations_df = pd.read_csv(formations_path)
    formations_df = formations_df[formations_df["matchId"] == int(match_id)].copy()
    formations_df = formations_df.dropna(subset=["formation"])

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    home_name = (metadata.get("homeTeam", {}).get("shortName")
                 or metadata.get("homeTeam", {}).get("name", "Home"))
    away_name = (metadata.get("awayTeam", {}).get("shortName")
                 or metadata.get("awayTeam", {}).get("name", "Away"))

    return formations_df, metadata, home_name, away_name, tracking_path


def build_possession_sequences(tracking_path, metadata):
    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]

    print("Deriving possession sequences from tracking data...")
    periods, elapsed, ball_x, ball_y, owner = pos.stream_ball_and_owner(
        tracking_path, pitch_length, pitch_width)
    fps = pos.infer_fps(elapsed, periods)
    smoothed = pos.smooth_owner(owner, periods, fps)
    sequences = pos.detect_possession_sequences(smoothed, periods, elapsed, fps)
    print(f"  -> {len(sequences)} possession sequences "
          f"({sum(1 for s in sequences if s['team'])} with a clear team, "
          f"{sum(1 for s in sequences if not s['team'])} loose-ball/no-owner)")
    return sequences


def lookup_formation(formations_df, team, period, sec):
    """Formation label whichever formations.csv window is 'current' at
    time `sec` into `period` (freshest-starting overlapping window, same
    logic as detect_formations/visualize_match use elsewhere)."""
    sub = formations_df[
        (formations_df["team"] == team)
        & (formations_df["period"] == period)
        & (formations_df["windowStartSec"] <= sec)
        & (sec < formations_df["windowEndSec"])
    ]
    if sub.empty:
        return None
    return sub.loc[sub["windowStartSec"].idxmax(), "formation"]


def infer_stride_seconds(df):
    starts = sorted(df["windowStartSec"].unique())
    diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    if not diffs:
        return 0
    return Counter(diffs).most_common(1)[0][0]


def compute_period_offsets(df, metadata=None):
    """Concatenates periods onto one continuous match-time axis. Prefers
    metadata's period start/end timestamps (exact); falls back to the
    formations.csv window grid only if metadata is missing a period."""
    stride = infer_stride_seconds(df)
    periods = sorted(df["period"].unique())
    meta_periods = (metadata or {}).get("periods", {}) or {}

    offsets = {}
    boundaries = []
    cursor = 0.0
    for p in periods:
        offsets[p] = cursor
        meta_entry = meta_periods.get(str(int(p)))
        if meta_entry and meta_entry.get("start") is not None and meta_entry.get("end") is not None:
            period_len = float(meta_entry["end"]) - float(meta_entry["start"])
        else:
            last_window_start = df.loc[df["period"] == p, "windowStartSec"].max()
            period_len = last_window_start + stride
        cursor += period_len
        boundaries.append((cursor, p))
    return offsets, boundaries, cursor


def format_mmss(seconds, _pos=None):
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def build_xticks(total_duration, window_seconds):
    if window_seconds <= 0:
        window_seconds = 300
    ticks = []
    t = 0.0
    while t < total_duration:
        ticks.append(t)
        t += window_seconds
    ticks.append(total_duration)
    cleaned = []
    for tk in ticks:
        if cleaned and (tk - cleaned[-1]) < window_seconds * 0.25:
            cleaned[-1] = tk
        else:
            cleaned.append(tk)
    return cleaned


def sequences_to_bars(sequences, team, formations_df, offsets):
    """Attaches a formation label to each of `team`'s possession
    sequences and converts sequence time -> continuous match time.
    Drops sequences with no resolvable formation (e.g. before the first
    formations.csv window starts)."""
    bars = []
    for s in sequences:
        if s["team"] != team:
            continue
        mid = (s["start_sec"] + s["end_sec"]) / 2
        formation = lookup_formation(formations_df, team, s["period"], mid)
        if formation is None:
            continue
        off = offsets.get(s["period"], 0.0)
        bars.append({
            "formation": formation,
            "matchStart": s["start_sec"] + off,
            "matchEnd": s["end_sec"] + off,
            "duration": s["duration"],
        })
    return bars


def draw_team_panel(ax, bars, color_of, total_duration):
    if not bars:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                 transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_yticks([])
        return []

    duration_by_formation = Counter()
    for b in bars:
        duration_by_formation[b["formation"]] += b["duration"]
    order_desc = [f for f, _ in duration_by_formation.most_common()]
    n = len(order_desc)
    y_of = {f: n - 1 - i for i, f in enumerate(order_desc)}

    for f, y in y_of.items():
        ax.hlines(y, 0, total_duration, colors=BASELINE_COLOR, linewidth=1,
                   linestyles=(0, (1, 3)), zorder=1)

    for b in bars:
        y = y_of[b["formation"]]
        ax.hlines(y, b["matchStart"], b["matchEnd"], colors=color_of[b["formation"]],
                   linewidth=9, alpha=0.9, capstyle="butt", zorder=2)

    ax.set_yticks(range(n))
    ax.set_yticklabels(order_desc, color=TEXT_COLOR)
    ax.set_ylim(-0.7, n - 0.3)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)
    ax.set_facecolor(BG_COLOR)
    return order_desc


def draw_period_dividers(ax_top, ax_bottom, boundaries):
    for boundary_time, period in boundaries[:-1]:
        for ax in (ax_top, ax_bottom):
            ax.axvline(boundary_time, color=GRID_COLOR, linewidth=1.2,
                       linestyle="--", zorder=0)
        ax_top.text(boundary_time, 1.02, f"End P{int(period)}",
                    transform=ax_top.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)

    end_time, _ = boundaries[-1]
    for ax in (ax_top, ax_bottom):
        ax.axvline(end_time, color=TEXT_COLOR, linewidth=1.3, linestyle="-", zorder=0)
    ax_top.text(end_time, 1.02, f"Match end {format_mmss(end_time)}",
                transform=ax_top.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=8, color=TEXT_COLOR)


def plot_formation_timeline(match_id, processed_dir=PROCESSED_DIR_DEFAULT):
    formations_df, metadata, home_name, away_name, tracking_path = load_data(match_id, processed_dir)
    offsets, boundaries, total_duration = compute_period_offsets(formations_df, metadata)
    total_duration = max(total_duration, 1.0)

    sequences = build_possession_sequences(tracking_path, metadata)

    home_bars = sequences_to_bars(sequences, "home", formations_df, offsets)
    away_bars = sequences_to_bars(sequences, "away", formations_df, offsets)

    all_formations = sorted({b["formation"] for b in home_bars + away_bars})
    color_of = {f: _PALETTE[i % len(_PALETTE)] for i, f in enumerate(all_formations)}

    fig, (ax_home, ax_away) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    fig.patch.set_facecolor(BG_COLOR)

    draw_team_panel(ax_home, home_bars, color_of, total_duration)
    draw_team_panel(ax_away, away_bars, color_of, total_duration)
    draw_period_dividers(ax_home, ax_away, boundaries)

    ax_home.set_title(f"{home_name} (home)", loc="left", fontsize=11, color=TEXT_COLOR)
    ax_away.set_title(f"{away_name} (away)", loc="left", fontsize=11, color=TEXT_COLOR)
    ax_home.set_ylabel("Formation", color=TEXT_COLOR)
    ax_away.set_ylabel("Formation", color=TEXT_COLOR)
    ax_away.set_xlabel("Match Time", color=TEXT_COLOR)

    window_seconds = 300
    xticks = build_xticks(total_duration, window_seconds)
    ax_away.set_xticks(xticks)
    ax_away.xaxis.set_major_formatter(mticker.FuncFormatter(format_mmss))
    ax_away.tick_params(axis="x", rotation=45, colors=TEXT_COLOR)
    for label in ax_away.get_xticklabels():
        label.set_ha("right")
    ax_away.set_xlim(0, total_duration)

    # Shared legend: one entry per formation, colored to match its bars.
    handles = [plt.Line2D([0], [0], color=color_of[f], linewidth=6, label=f) for f in all_formations]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(all_formations), 10),
               frameon=False, labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Match {match_id} \u2014 Formation Timeline\n"
        f"Each bar = one possession sequence  |  Dashed lines = period boundaries",
        fontsize=13, color=TEXT_COLOR
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot each team's detected formation over time, one bar per possession sequence."
    )
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
    args = parser.parse_args()

    plot_formation_timeline(args.match_id, args.processed_dir)
    plt.show()


if __name__ == "__main__":
    main()