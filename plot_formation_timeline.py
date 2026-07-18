"""
plot_formation_timeline.py

Plots each team's detected formation over time as a colored "piano roll":
one row per formation, colored horizontal bars showing when that
formation was detected -- one bar per CONTINUOUS STRETCH in that
formation (built directly from formations.csv's sliding windows,
independent of who has the ball).

CHANGE FROM THE PREVIOUS (POSSESSION-GATED) VERSION: bars used to only
appear during a team's OWN possession sequences, which meant a team's
panel went blank while the other team had the ball. That hid exactly
the thing worth seeing -- what shape a team holds while defending vs.
attacking. Bars now come straight from formations.csv for that team,
merging consecutive/overlapping sliding-window rows that agree on the
same formation into one continuous run, so each panel shows the team's
detected shape continuously through the whole match, both in and out of
possession.

ALSO: goal markers, if Processed_Tracking/<match_id>/events.json exists
(written by preprocessing.py's process_events() from Event_Data/, see
find_goals()'s docstring). A gold vertical line marks the moment on both
panels; a "MM' Scorer" label (with "(OG)" for own goals) is placed on
the SCORING team's panel. Silently skipped -- with a printed note, not
an error -- for matches with no events.json, since goal detection needs
real event data (shotOutcomeType), not the tracking-only proxy.

USAGE:
    python plot_formation_timeline.py <match_id> [--processed-dir DIR]

REQUIRES:
    Processed_Tracking/<match_id>/formations.csv       (from detect_formations.py)
    Processed_Tracking/<match_id>/metadata.json
    Processed_Tracking/<match_id>/tracking.jsonl.bz2   (still required by load_data;
                                                         no longer used for bars, but
                                                         kept as a sanity check that
                                                         this is a fully-processed match)
    Processed_Tracking/<match_id>/events.json          (optional -- enables goal
                                                         markers if present)
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

    return formations_df, metadata, home_name, away_name, tracking_path, folder


def build_possession_sequences(tracking_path, metadata):
    """No longer used to build the timeline bars (see module docstring),
    kept here in case it's useful elsewhere -- left untouched."""
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


# Outcome fields that mark a possession event as an inadvertent shot into
# the ACTING player's own net -- per the PFF FC spec, a deliberate shot
# ("SH") is never an own goal (that scenario doesn't exist in the
# taxonomy), only a clearance/touch/rebound that inadvertently beats the
# keeper. Mapping: possessionEventType -> the outcome field that carries
# the "D - Inadvertent Shot at own Goal" value for that type.
_OWN_GOAL_OUTCOME_FIELD = {
    "CL": "clearanceOutcomeType",
    "TC": "touchOutcomeType",
    "RE": "reboundOutcomeType",
}


def find_goals(events):
    """
    Scans events.json for goals. A goal is any possession event whose
    shotOutcomeType == "G" -- per the spec, this field is populated
    both for deliberate shots (possessionEventType "SH") AND for any
    other event that inadvertently results in a shot at goal (clearance,
    touch, rebound), so filtering on shotOutcomeType alone catches both
    without needing to special-case possessionEventType == "SH".

    Own goals need the SCORING team flipped relative to the ACTING
    team: gameEvents.homeTeam / teamId identify whoever performed the
    action (cleared/touched/deflected the ball), but for an own goal
    the goal counts for the opposing team. Detected via
    _OWN_GOAL_OUTCOME_FIELD -- see its comment above. ASSUMPTION:
    only verified against the spec text, not against a real own-goal
    row (I haven't seen one) -- worth checking against a match you know
    had an own goal before trusting the "(OG)" labeling specifically;
    the "a goal happened at this time" part is solid either way since
    it only depends on shotOutcomeType == "G".

    Returns a list of {period, sec, team ('home'/'away'), scorer,
    ownGoal} dicts, sorted by time. `sec` is periodElapsedTimeEstimate
    (added by preprocessing.py) -- events with no time estimate are
    skipped, since they can't be placed on the timeline.
    """
    goals = []
    for ev in events:
        if ev.get("nonEvent"):
            continue
        pe = ev.get("possessionEvents") or {}
        if pe.get("shotOutcomeType") != "G":
            continue

        sec = ev.get("periodElapsedTimeEstimate")
        period = ev.get("period")
        acting_team_is_home = ev.get("homeTeam")
        if sec is None or period is None or acting_team_is_home is None:
            continue

        possession_type = pe.get("possessionEventType")
        outcome_field = _OWN_GOAL_OUTCOME_FIELD.get(possession_type)
        own_goal = bool(outcome_field and pe.get(outcome_field) == "D")

        scoring_team_is_home = (not acting_team_is_home) if own_goal else acting_team_is_home
        scorer = pe.get("shooterPlayerName") or ev.get("playerName")

        goals.append({
            "period": period,
            "sec": sec,
            "team": "home" if scoring_team_is_home else "away",
            "scorer": scorer,
            "ownGoal": own_goal,
        })

    goals.sort(key=lambda g: (g["period"], g["sec"]))
    return goals


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


def formation_runs_from_windows(formations_df, team, offsets):
    """
    Builds continuous formation bars directly from formations.csv's
    sliding windows for `team`, WITHOUT gating on possession -- this is
    what makes the panel show the team's formation continuously, both
    while attacking and while defending. Consecutive/overlapping
    windows that agree on the same formation are merged into one
    continuous run; a change in formation, or a gap where window
    coverage stops (e.g. MIN_FRAMES_PER_WINDOW trimmed a window near a
    period's tail), starts a new bar.
    """
    sub = formations_df[formations_df["team"] == team].copy()
    if sub.empty:
        return []
    sub = sub.sort_values(["period", "windowStartSec"])

    bars = []
    current = None
    for _, row in sub.iterrows():
        off = offsets.get(row["period"], 0.0)
        start = row["windowStartSec"] + off
        end = row["windowEndSec"] + off
        formation = row["formation"]

        same_run = (
            current is not None
            and current["formation"] == formation
            and current["period"] == row["period"]
            and start <= current["matchEnd"]
        )
        if same_run:
            current["matchEnd"] = max(current["matchEnd"], end)
        else:
            if current is not None:
                bars.append(current)
            current = {
                "formation": formation,
                "period": row["period"],
                "matchStart": start,
                "matchEnd": end,
            }
    if current is not None:
        bars.append(current)

    for b in bars:
        b["duration"] = b["matchEnd"] - b["matchStart"]
        del b["period"]
    return bars


def sequences_to_bars(sequences, team, formations_df, offsets):
    """No longer used to build the timeline bars (see module docstring),
    kept here in case it's useful elsewhere -- left untouched. Attaches
    a formation label to each of `team`'s possession sequences and
    converts sequence time -> continuous match time."""
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


GOAL_COLOR = "#f1c40f"  # gold, distinct from both team/formation colors and grid lines


def draw_goals(ax_home, ax_away, goals, offsets):
    """
    Marks each goal with a full-height gold line on both panels (so
    it's visible against whichever formation bar it interrupts) plus a
    "MM' Scorer" label on the SCORING team's own panel -- own goals are
    labeled on the panel of the team that benefits from the goal, not
    the team whose player put it in their own net, since that's the
    team the goal actually counts for.
    """
    for g in goals:
        off = offsets.get(g["period"], 0.0)
        t = g["sec"] + off
        minute = int(t // 60)

        label = f"{minute}' {g['scorer']}" if g["scorer"] else f"{minute}' Goal"
        if g["ownGoal"]:
            label += " (OG)"

        for ax in (ax_home, ax_away):
            ax.axvline(t, color=GOAL_COLOR, linewidth=1.3, linestyle="-",
                       alpha=0.85, zorder=4)

        target_ax = ax_home if g["team"] == "home" else ax_away
        target_ax.annotate(
            f"\u26bd {label}", xy=(t, 1.0), xycoords=("data", "axes fraction"),
            xytext=(4, 4), textcoords="offset points",
            ha="left", va="bottom", fontsize=7.5, color=GOAL_COLOR,
            rotation=60, zorder=5,
        )


def plot_formation_timeline(match_id, processed_dir=PROCESSED_DIR_DEFAULT):
    formations_df, metadata, home_name, away_name, tracking_path, match_dir = load_data(match_id, processed_dir)
    offsets, boundaries, total_duration = compute_period_offsets(formations_df, metadata)
    total_duration = max(total_duration, 1.0)

    events = pos.load_events(match_dir)
    if events is not None:
        goals = find_goals(events)
        print(f"  -> {len(goals)} goal(s) found in events.json")
    else:
        goals = []
        print("  no events.json for this match -- skipping goal markers "
              "(goal detection requires real event data, not the proximity proxy).")

    home_bars = formation_runs_from_windows(formations_df, "home", offsets)
    away_bars = formation_runs_from_windows(formations_df, "away", offsets)

    all_formations = sorted({b["formation"] for b in home_bars + away_bars})
    color_of = {f: _PALETTE[i % len(_PALETTE)] for i, f in enumerate(all_formations)}

    fig, (ax_home, ax_away) = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True,
                                            gridspec_kw={"hspace": 0.35})
    fig.patch.set_facecolor(BG_COLOR)

    draw_team_panel(ax_home, home_bars, color_of, total_duration)
    draw_team_panel(ax_away, away_bars, color_of, total_duration)
    draw_period_dividers(ax_home, ax_away, boundaries)
    draw_goals(ax_home, ax_away, goals, offsets)

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
        f"Each bar = continuous stretch in that formation (attacking + defending)  |  Dashed lines = period boundaries"
        + ("  |  \u26bd = goal" if goals else ""),
        fontsize=13, color=TEXT_COLOR
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot each team's detected formation over time, continuously (attacking + defending)."
    )
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
    args = parser.parse_args()

    plot_formation_timeline(args.match_id, args.processed_dir)
    plt.show()


if __name__ == "__main__":
    main()