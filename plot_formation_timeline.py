"""
plot_formation_timeline.py

Plots each team's detected formation over time, from formations.csv (the
output of detect_formation.py), as a simple "piano roll" / spectrogram-style
chart: one row per formation, horizontal bars showing when that formation
was detected.

DELIBERATE DESIGN CHOICES (see conversation for the reasoning):
  - Shows the RAW, OVERLAPPING sliding-window output as-is -- each CSV row
    becomes its own bar spanning its own [windowStartSec, windowEndSec).
    No merging/run-length-encoding into clean non-overlapping blocks (yet).
    Where several overlapping windows agree, the semi-transparent bars
    stack and visually darken; where they disagree, you'll see bars on two
    different formation rows at the same point in time -- that's useful
    signal, not noise, for a first look at how stable the detection is.
  - Periods are concatenated onto one continuous match-time x-axis (not
    reset to 0 at half-time), with a vertical divider + label at each
    period boundary. Each period's length is taken from the data itself
    (the latest windowEndSec seen in that period) rather than
    metadata["periods"], so this stays self-contained.
  - Both teams shown stacked in one figure (home on top, away below),
    sharing the same time axis.
  - One or two colors only, no per-formation color coding: a dark
    semi-transparent bar color for actual detections, light gray for the
    reference baseline and period dividers.

USAGE:
    python plot_formation_timeline.py <match_id> [--processed-dir DIR]

REQUIRES:
    Processed_Tracking/<match_id>/formations.csv  (from detect_formation.py)
    Processed_Tracking/<match_id>/metadata.json   (optional, for team names)
"""

import os
import sys
import json
import argparse

import matplotlib
# matplotlib.use() only records the preference, it doesn't validate/import
# the backend until the first draw call -- so pick by platform, not
# trial-and-error (see visualize_match.py for the long version of why).
matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

PROCESSED_DIR_DEFAULT = "Processed_Tracking"

BAR_COLOR = "#1a1a1a"
BASELINE_COLOR = "#bbbbbb"


def load_data(match_id, processed_dir):
    folder = os.path.join(processed_dir, str(match_id))
    formations_path = os.path.join(folder, "formations.csv")
    metadata_path = os.path.join(folder, "metadata.json")

    if not os.path.exists(formations_path):
        raise FileNotFoundError(
            f"{formations_path} not found -- run detect_formation.py for "
            f"match {match_id} first."
        )

    df = pd.read_csv(formations_path)
    df = df[df["matchId"] == int(match_id)].copy()
    df = df.dropna(subset=["formation"])

    home_name, away_name = "Home", "Away"
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        home_name = (metadata.get("homeTeam", {}).get("shortName")
                      or metadata.get("homeTeam", {}).get("name", "Home"))
        away_name = (metadata.get("awayTeam", {}).get("shortName")
                      or metadata.get("awayTeam", {}).get("name", "Away"))

    return df, home_name, away_name, metadata


def infer_stride_seconds(df):
    """
    Infers STRIDE_SECONDS from the CSV's own windowStartSec values,
    rather than importing the constant from detect_formation.py -- so
    this stays correct even if the CSV was generated with different
    window/stride settings than whatever's currently in that module.
    """
    from collections import Counter
    starts = sorted(df["windowStartSec"].unique())
    diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    if not diffs:
        return 0
    return Counter(diffs).most_common(1)[0][0]


def infer_window_seconds(df):
    """
    Infers WINDOW_SECONDS directly from the CSV: each row already stores
    its own windowStartSec/windowEndSec, so the window width is just
    their difference -- no need to import anything or assume a value.
    Used to space x-axis ticks at the same granularity as the windows
    themselves (e.g. every 5 min if that's the configured window size).
    """
    from collections import Counter
    diffs = (df["windowEndSec"] - df["windowStartSec"]).tolist()
    if not diffs:
        return 300
    return Counter(diffs).most_common(1)[0][0]


def build_xticks(total_duration, window_seconds):
    """
    Tick positions at every multiple of window_seconds from 0 up to
    total_duration, PLUS an explicit final tick at total_duration itself
    (so the exact end of the data is always labeled, even if it doesn't
    land on a clean multiple of the window size).
    """
    if window_seconds <= 0:
        window_seconds = 60
    ticks = []
    t = 0.0
    while t < total_duration:
        ticks.append(t)
        t += window_seconds
    ticks.append(total_duration)
    # Drop the second-to-last tick if it's basically on top of the final
    # one (avoids two overlapping/unreadable labels at the right edge).
    cleaned = []
    for tk in ticks:
        if cleaned and (tk - cleaned[-1]) < window_seconds * 0.25:
            cleaned[-1] = tk
        else:
            cleaned.append(tk)
    return cleaned


def compute_period_offsets(df, metadata=None):
    """
    Concatenates periods onto one continuous match-time axis.

    Period length source, in order of preference:
      1. metadata["periods"][str(period)]["end"] - ["start"]. This is a
         TIMESTAMP DIFFERENCE, which is valid regardless of what "zero"
         means in whatever clock start/end are recorded against (unlike
         using an absolute position from that field, which is a
         different, unverified assumption -- see the coordinate-system
         discussion earlier in this project). It's also exact, unlike
         anything derivable from formations.csv's window grid.
      2. Fallback (only if metadata/periods is missing for a period):
         last windowStartSec + inferred stride. This is coarse -- bounded
         by STRIDE_SECONDS, and further only as good as whatever windows
         survived MIN_FRAMES_PER_WINDOW trimming near the period's tail
         -- which is exactly why (1) is preferred whenever available.

    Returns:
        offsets: {period: seconds_to_add_for_this_period}
        boundaries: [(boundary_time_sec, period_that_just_ended), ...]
                    in ascending time order, one entry per period
                    (including the final one, i.e. end of match).
        total_duration: total continuous match time in seconds
    """
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


def draw_team_panel(ax, sub, offsets, total_duration=None):
    """
    Draws the raw, overlapping sliding-window formation bars for one team
    onto `ax`. If total_duration is given (the metadata-derived match
    length), baselines span that full length; otherwise falls back to
    this team's own last bar position.
    """
    if sub.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_yticks([])
        return 0.0

    sub = sub.copy()
    sub["matchStart"] = sub["windowStartSec"] + sub["period"].map(offsets)
    sub["matchEnd"] = sub["windowEndSec"] + sub["period"].map(offsets)

    # Order formations top-to-bottom by total (raw, possibly overlap-
    # counted) time spent in them -- most-worn formation at the top.
    duration_by_formation = (
        (sub["matchEnd"] - sub["matchStart"])
        .groupby(sub["formation"])
        .sum()
        .sort_values(ascending=False)
    )
    order_desc = list(duration_by_formation.index)
    n = len(order_desc)
    y_of = {f: n - 1 - i for i, f in enumerate(order_desc)}

    if total_duration is None:
        total_duration = float(sub["matchEnd"].max())

    # Baseline: thin dotted reference line across the full match for every
    # formation this team wore at some point.
    for f, y in y_of.items():
        ax.hlines(y, 0, total_duration, colors=BASELINE_COLOR, linewidth=1,
                   linestyles=(0, (1, 3)), zorder=1)

    # The actual raw (overlapping) window bars -- one per CSV row.
    ys = sub["formation"].map(y_of).to_numpy()
    ax.hlines(ys, sub["matchStart"].to_numpy(), sub["matchEnd"].to_numpy(),
              colors=BAR_COLOR, linewidth=9, alpha=0.45,
              capstyle="butt", zorder=2)

    ax.set_yticks(range(n))
    ax.set_yticklabels(order_desc)
    ax.set_ylim(-0.7, n - 0.3)
    return total_duration


def draw_period_dividers(ax_top, ax_bottom, boundaries):
    # Internal period boundaries: dashed, light gray, labeled "End P{n}".
    for boundary_time, period in boundaries[:-1]:
        for ax in (ax_top, ax_bottom):
            ax.axvline(boundary_time, color="#888888", linewidth=1,
                       linestyle="--", zorder=0)
        ax_top.text(boundary_time, 1.02, f"End P{int(period)}",
                    transform=ax_top.get_xaxis_transform(),
                    ha="center", va="bottom", fontsize=8, color="#666666")

    # Final boundary = true end of the data. Styled distinctly (solid,
    # darker) from the internal period dividers, and always labeled with
    # the exact mm:ss so it's obvious where the match actually stops.
    end_time, _last_period = boundaries[-1]
    for ax in (ax_top, ax_bottom):
        ax.axvline(end_time, color="#333333", linewidth=1.3,
                   linestyle="-", zorder=0)
    ax_top.text(end_time, 1.02, f"Match end {format_mmss(end_time)}",
                transform=ax_top.get_xaxis_transform(),
                ha="right", va="bottom", fontsize=8, color="#333333")


def plot_formation_timeline(match_id, processed_dir=PROCESSED_DIR_DEFAULT):
    df, home_name, away_name, metadata = load_data(match_id, processed_dir)
    offsets, boundaries, total_duration = compute_period_offsets(df, metadata)
    total_duration = max(total_duration, 1.0)

    fig, (ax_home, ax_away) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    home_sub = df[df["team"] == "home"]
    away_sub = df[df["team"] == "away"]

    draw_team_panel(ax_home, home_sub, offsets, total_duration)
    draw_team_panel(ax_away, away_sub, offsets, total_duration)

    draw_period_dividers(ax_home, ax_away, boundaries)

    ax_home.set_title(f"{home_name} (home)", loc="left", fontsize=11)
    ax_away.set_title(f"{away_name} (away)", loc="left", fontsize=11)
    ax_home.set_ylabel("Formation")
    ax_away.set_ylabel("Formation")
    ax_away.set_xlabel("Time")

    window_seconds = infer_window_seconds(df)
    xticks = build_xticks(total_duration, window_seconds)
    ax_away.set_xticks(xticks)
    ax_away.xaxis.set_major_formatter(mticker.FuncFormatter(format_mmss))
    ax_away.tick_params(axis="x", rotation=45)
    for label in ax_away.get_xticklabels():
        label.set_ha("right")
    ax_away.set_xlim(0, total_duration)

    fig.suptitle(
        f"Match {match_id} \u2014 formation timeline (raw, overlapping sliding windows)",
        fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot each team's detected formation over time "
                     "(raw, overlapping sliding-window output)."
    )
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
    args = parser.parse_args()

    plot_formation_timeline(args.match_id, args.processed_dir)
    plt.show()


if __name__ == "__main__":
    main()