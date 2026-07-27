"""
plot_formation_timeline.py

Plots each team's detected formation over time as a colored "piano roll":
one row per formation, colored horizontal bars showing when that
formation was detected -- one bar per CONTIGUOUS WINDOW RUN where the
window-level formation matches the PERIOD-LEVEL voted formation (the
label with the highest cumulative confidence across all windows in that
period). This replaces the previous possession-sequence approach.

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
    Processed_Tracking/<match_id>/tracking.jsonl.bz2
    Processed_Tracking/<match_id>/events.json          (optional -- enables goal
                                                          markers if present)
"""

import sys
import json
import argparse
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("QtAgg" if "PyQt6" in sys.modules else ("MacOSX" if sys.platform == "darwin" else "TkAgg"))
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from ..analytics import possession as pos
from ..io.paths import PROCESSED_DIR, match_dir
from .theme import FIG_FACE, GRID, BASELINE, TEXT_PRIMARY, GOAL_COLOR

PROCESSED_DIR_DEFAULT = str(PROCESSED_DIR)

BG_COLOR = FIG_FACE
GRID_COLOR = GRID
BASELINE_COLOR = BASELINE
TEXT_COLOR = TEXT_PRIMARY

_PALETTE = [matplotlib.colors.to_hex(c) for c in
            list(plt.get_cmap("tab20").colors) + list(plt.get_cmap("tab20b").colors)]


def load_data(match_id, processed_dir):
    folder = match_dir(match_id, processed_dir)
    formations_path = folder / "formations.csv"
    metadata_path = folder / "metadata.json"
    tracking_path = folder / "tracking.jsonl.bz2"

    if not formations_path.is_file():
        raise FileNotFoundError(
            f"{formations_path} not found -- run detect_formations.py for "
            f"match {match_id} first."
        )
    if not tracking_path.is_file():
        raise FileNotFoundError(f"{tracking_path} not found.")

    formations_df = pd.read_csv(formations_path)
    formations_df = formations_df[formations_df["matchId"] == int(match_id)].copy()
    formations_df = formations_df.dropna(subset=["formation"])
    if "confidence" not in formations_df.columns:
        formations_df["confidence"] = 1.0

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    home_name = (metadata.get("homeTeam", {}).get("shortName")
                 or metadata.get("homeTeam", {}).get("name", "Home"))
    away_name = (metadata.get("awayTeam", {}).get("shortName")
                 or metadata.get("awayTeam", {}).get("name", "Away"))

    return formations_df, metadata, home_name, away_name, tracking_path, folder


_OWN_GOAL_OUTCOME_FIELD = {
    "CL": "clearanceOutcomeType",
    "TC": "touchOutcomeType",
    "RE": "reboundOutcomeType",
}


def find_goals(events):
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


def _strip_flipped(formation):
    if formation and formation.endswith("_flipped"):
        return formation[:-8]
    return formation


def resolve_formation_by_vote(df):
    """For each (team, period), pick the formation with the highest
    cumulative confidence (*base* name only -- flipped/original are
    grouped together). Returns ``{(team, period): base_formation}``."""
    groups = df.groupby(["team", "period"], sort=False)
    voted = {}
    for (team, period), grp in groups:
        by_base = defaultdict(float)
        for _, row in grp.iterrows():
            base = _strip_flipped(row["formation"])
            by_base[base] += row["confidence"]
        winner = max(by_base, key=by_base.get)
        voted[(team, period)] = winner
    return voted


def formation_runs_from_votes(df, voted, offsets):
    """Build continuous-time bar dicts from window-level formation data.

    For each (team, period), scan windows sorted by ``windowStartSec``.
    Whenever ``_strip_flipped(formation) == voted[(team, period)]``,
    extend the current run or start a new one. Runs that differ from the
    voted formation are skipped. Each run produces a bar spanning from
    the first window's start to the last window's end (in continuous
    match time via *offsets*).

    Returns a list of ``{team, formation, matchStart, matchEnd}`` dicts.
    """
    sorted_df = df.sort_values(["team", "period", "windowStartSec"])
    bars = []
    for (team, period), grp in sorted_df.groupby(["team", "period"], sort=False):
        winner = voted.get((team, period))
        if winner is None:
            continue
        off = offsets.get(period, 0.0)
        run_start = None
        run_end = None
        for _, row in grp.iterrows():
            base = _strip_flipped(row["formation"])
            if base == winner:
                ws = row["windowStartSec"] + off
                we = row["windowEndSec"] + off
                if run_start is None:
                    run_start = ws
                run_end = we
            else:
                if run_start is not None:
                    bars.append({"team": team, "formation": winner,
                                 "matchStart": run_start, "matchEnd": run_end})
                    run_start = None
        if run_start is not None:
            bars.append({"team": team, "formation": winner,
                         "matchStart": run_start, "matchEnd": run_end})
    return bars


def infer_stride_seconds(df):
    starts = sorted(df["windowStartSec"].unique())
    diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    if not diffs:
        return 0
    return Counter(diffs).most_common(1)[0][0]


def compute_period_offsets(df, metadata=None):
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


def draw_team_panel(ax, bars, color_of, total_duration):
    if not bars:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                 transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_yticks([])
        return []

    duration_by_formation = Counter()
    for b in bars:
        duration_by_formation[b["formation"]] += b["matchEnd"] - b["matchStart"]
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


def draw_goals(ax_home, ax_away, goals, offsets):
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

    voted = resolve_formation_by_vote(formations_df)
    all_bars = formation_runs_from_votes(formations_df, voted, offsets)

    home_bars = [b for b in all_bars if b["team"] == "home"]
    away_bars = [b for b in all_bars if b["team"] == "away"]

    events = pos.load_events(match_dir)
    if events is not None:
        goals = find_goals(events)
        print(f"  -> {len(goals)} goal(s) found in events.json")
    else:
        goals = []
        print("  no events.json for this match -- skipping goal markers "
              "(goal detection requires real event data, not the proximity proxy).")

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

    handles = [plt.Line2D([0], [0], color=color_of[f], linewidth=6, label=f)
               for f in all_formations]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(all_formations), 10),
               frameon=False, labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        f"Match {match_id} \u2014 Formation Timeline\n"
        f"Each bar = contiguous window run of the voted formation"
        + ("  |  \u26bd = goal" if goals else ""),
        fontsize=13, color=TEXT_COLOR
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    return fig


def main():
    parser = argparse.ArgumentParser(
        description="Plot each team's detected formation over time, one bar per voted window run."
    )
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
    args = parser.parse_args()

    plot_formation_timeline(args.match_id, args.processed_dir)
    plt.show()


if __name__ == "__main__":
    main()
