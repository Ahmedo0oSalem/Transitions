"""
plot_formation_timeline.py
Plots each team's detected formation over time as a continuous
single-lane "state strip". For each team, a rolling majority vote
(smoothing window ±45s) converts per-window formation detections into
a gap-free, non-overlapping segment sequence.

ALSO: Enriches bars with metrics from formation_segments.csv (Compactness, Possession %)
if available, and adjusts bar opacity based on detection confidence.

USAGE:
python plot_formation_timeline.py <match_id> [--processed-dir DIR]
"""
"""
plot_formation_timeline.py
Plots each team's detected formation over time.
- Single Line: Clean, gap-free state strip.
- Piano Roll: Multi-row view with hover-to-inspect metrics.
"""
import sys
import json
import argparse
from collections import Counter
import matplotlib
matplotlib.use("QtAgg" if "PyQt6" in sys.modules else ("MacOSX" if sys.platform == "darwin" else "TkAgg"))
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from ..analytics import possession as pos
from ..analytics.formations.taxonomy import (
    FAMILY_ORDER,
    build_family_color_map,
    derive_hierarchy,
    infer_family,
)
from ..io.paths import PROCESSED_DIR, match_dir
from .theme import FIG_FACE, GRID, BASELINE, TEXT_PRIMARY

PROCESSED_DIR_DEFAULT = str(PROCESSED_DIR)
BG_COLOR = FIG_FACE
GRID_COLOR = GRID
BASELINE_COLOR = BASELINE
TEXT_COLOR = TEXT_PRIMARY

SCORED_COLOR = "#2ecc71"
CONCEDED_COLOR = "#d62728"

def load_data(match_id, processed_dir):
    folder = match_dir(match_id, processed_dir)
    formations_path = folder / "formations.csv"
    segments_path = folder / "formation_segments.csv"
    metadata_path = folder / "metadata.json"
    tracking_path = folder / "tracking.jsonl.bz2"
    
    if not formations_path.is_file():
        raise FileNotFoundError(f"{formations_path} not found.")
    if not tracking_path.is_file():
        raise FileNotFoundError(f"{tracking_path} not found.")
        
    formations_df = pd.read_csv(formations_path)
    formations_df = formations_df[formations_df["matchId"] == int(match_id)].copy()
    formations_df = formations_df.dropna(subset=["formation"])
    if "confidence" not in formations_df.columns:
        formations_df["confidence"] = 1.0
        
    segments_df = None
    if segments_path.is_file():
        segments_df = pd.read_csv(segments_path)
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    home_name = (metadata.get("homeTeam", {}).get("shortName") or metadata.get("homeTeam", {}).get("name", "Home"))
    away_name = (metadata.get("awayTeam", {}).get("shortName") or metadata.get("awayTeam", {}).get("name", "Away"))
                 
    return formations_df, metadata, home_name, away_name, tracking_path, folder, segments_df

_OWN_GOAL_OUTCOME_FIELD = {"CL": "clearanceOutcomeType", "TC": "touchOutcomeType", "RE": "reboundOutcomeType"}

def find_goals(events):
    goals = []
    for ev in events:
        if ev.get("nonEvent"): continue
        pe = ev.get("possessionEvents") or {}
        if pe.get("shotOutcomeType") != "G": continue
        sec = ev.get("periodElapsedTimeEstimate")
        period = ev.get("period")
        acting_team_is_home = ev.get("homeTeam")
        if sec is None or period is None or acting_team_is_home is None: continue
        possession_type = pe.get("possessionEventType")
        outcome_field = _OWN_GOAL_OUTCOME_FIELD.get(possession_type)
        own_goal = bool(outcome_field and pe.get(outcome_field) == "D")
        scoring_team_is_home = (not acting_team_is_home) if own_goal else acting_team_is_home
        scorer = pe.get("shooterPlayerName") or ev.get("playerName")
        goals.append({"period": period, "sec": sec, "team": "home" if scoring_team_is_home else "away", "scorer": scorer, "ownGoal": own_goal})
    goals.sort(key=lambda g: (g["period"], g["sec"]))
    return goals

def _vote(df, team, period, t, level="variant"):
    sub = df[(df["team"] == team) & (df["period"] == period) & (df["windowStartSec"] <= t) & (t < df["windowEndSec"])]
    if sub.empty: return None, None
    raw = sub["formation"]
    if level == "variant": group_key = raw.str.replace(r"flat$", "", regex=True)
    elif level == "family": group_key = raw.str.replace(r"flat$", "", regex=True).apply(infer_family)
    else: group_key = raw
    weight = sub["confidence"] if "confidence" in sub.columns else pd.Series(1.0, index=sub.index)
    scores = weight.groupby(group_key).sum()
    winner = scores.idxmax()
    match_idx = group_key == winner
    winner_raw = raw[match_idx].iloc[0] if match_idx.any() else winner
    return winner, winner_raw

def build_gap_free_segments(df, offsets, min_segment_seconds=45, granularity="variant"):
    sorted_df = df.sort_values(["team", "period", "windowStartSec"])
    results = []
    for (team, period), grp in sorted_df.groupby(["team", "period"], sort=False):
        stride = infer_stride_seconds(grp)
        if stride <= 0: continue
        off = offsets.get(period, 0.0)
        t_max = float(grp["windowEndSec"].max())
        times = []
        t = 0.0
        while t <= t_max:
            times.append(t)
            t += stride
        if times and times[-1] < t_max: times.append(t_max)
        if not times: continue
        smoothed = []
        for t in times:
            merge_key, raw_fm = _vote(df, team, period, t, level=granularity)
            if merge_key is not None: smoothed.append((t, merge_key, raw_fm))
            elif smoothed:
                prev_mk, prev_raw = smoothed[-1][1], smoothed[-1][2]
                smoothed.append((t, prev_mk, prev_raw))
        if not smoothed: continue
        segments = []
        seg_start = smoothed[0][0]; seg_mk = smoothed[0][1]; seg_raw = smoothed[0][2]
        for t, mk, raw_fm in smoothed[1:]:
            if mk != seg_mk:
                segments.append({"start": seg_start, "end": t, "merge_key": seg_mk, "raw_formation": seg_raw})
                seg_start = t; seg_mk = mk; seg_raw = raw_fm
        segments.append({"start": seg_start, "end": times[-1], "merge_key": seg_mk, "raw_formation": seg_raw})
        if min_segment_seconds > 0:
            i = 0
            while i < len(segments):
                dur = segments[i]["end"] - segments[i]["start"]
                if dur >= min_segment_seconds: i += 1; continue
                left_dur = (segments[i - 1]["end"] - segments[i - 1]["start"] if i > 0 else -1.0)
                right_dur = (segments[i + 1]["end"] - segments[i + 1]["start"] if i < len(segments) - 1 else -1.0)
                if left_dur >= right_dur and left_dur > 0: segments[i - 1]["end"] = segments[i]["end"]; segments.pop(i)
                elif right_dur > 0: segments[i + 1]["start"] = segments[i]["start"]; segments.pop(i)
                else: i += 1
        for seg in segments:
            h = derive_hierarchy(seg["raw_formation"])
            results.append({"team": team, "period": period, "formation": h["formation"], "variant": h["variant"], "family": h["family"], "matchStart": seg["start"] + off, "matchEnd": seg["end"] + off})
    return results

def formation_runs_from_votes(df, offsets, granularity="variant"):
    sorted_df = df.sort_values(["team", "period", "windowStartSec"])
    bars = []
    for (team, period), grp in sorted_df.groupby(["team", "period"], sort=False):
        off = offsets.get(period, 0.0)
        starts = sorted(grp["windowStartSec"].unique())
        stride = infer_stride_seconds(grp)
        if stride <= 0 or not starts: continue
        run_start = None; run_mk = None; run_h = None
        for sec in starts:
            merge_key, raw_fm = _vote(df, team, period, sec, level=granularity)
            if merge_key is None:
                if run_start is not None:
                    bars.append({"team": team, "period": period, "formation": run_h["formation"], "variant": run_h["variant"], "family": run_h["family"], "matchStart": run_start, "matchEnd": sec + off})
                    run_start = None
                continue
            ws = sec + off
            h = derive_hierarchy(raw_fm)
            if run_start is None: run_start = ws; run_mk = merge_key; run_h = h
            elif merge_key != run_mk:
                bars.append({"team": team, "period": period, "formation": run_h["formation"], "variant": run_h["variant"], "family": run_h["family"], "matchStart": run_start, "matchEnd": ws})
                run_start = ws; run_mk = merge_key; run_h = h
        if run_start is not None:
            period_end = starts[-1] + stride + off
            bars.append({"team": team, "period": period, "formation": run_h["formation"], "variant": run_h["variant"], "family": run_h["family"], "matchStart": run_start, "matchEnd": period_end})
    return bars

def infer_stride_seconds(df):
    starts = sorted(df["windowStartSec"].unique())
    diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    if not diffs: return 0
    return Counter(diffs).most_common(1)[0][0]

def compute_period_offsets(df, metadata=None):
    stride = infer_stride_seconds(df)
    periods = sorted(df["period"].unique())
    meta_periods = (metadata or {}).get("periods", {}) or {}
    offsets = {}; boundaries = []; cursor = 0.0
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
    if window_seconds <= 0: window_seconds = 300
    ticks = []; t = 0.0
    while t < total_duration: ticks.append(t); t += window_seconds
    ticks.append(total_duration)
    cleaned = []
    for tk in ticks:
        if cleaned and (tk - cleaned[-1]) < window_seconds * 0.25: cleaned[-1] = tk
        else: cleaned.append(tk)
    return cleaned

def _lookup_metrics(segments_df, team, period, match_start, match_end, offsets):
    if segments_df is None or segments_df.empty: return None
    off = offsets.get(period, 0.0)
    start_rel = match_start - off; end_rel = match_end - off
    mask = (segments_df["team"] == team) & (segments_df["period"] == period) & (segments_df["start_sec"] < end_rel) & (segments_df["end_sec"] > start_rel)
    overlap = segments_df[mask]
    if overlap.empty: return None
    overlap = overlap.copy()
    overlap["o_start"] = overlap["start_sec"].clip(lower=start_rel)
    overlap["o_end"] = overlap["end_sec"].clip(upper=end_rel)
    overlap["dur"] = overlap["o_end"] - overlap["o_start"]
    best_row = overlap.loc[overlap["dur"].idxmax()]
    return best_row.to_dict()

def draw_team_panel(ax, segments, color_map, total_duration, color_key="variant"):
    """Single-lane state strip — CLEANED UP (no metrics text inside bars)."""
    if not segments:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_yticks([])
        return sorted({s[color_key] for s in segments})
        
    for s in segments:
        col = color_map.get(s[color_key], "#888888")
        # We keep the opacity logic for visual confidence, but remove the text clutter
        ax.hlines(0, s["matchStart"], s["matchEnd"], colors=col, linewidth=20, alpha=0.85, capstyle="butt", zorder=2)
        dur = s["matchEnd"] - s["matchStart"]
        if dur >= 30:
            mid = (s["matchStart"] + s["matchEnd"]) / 2
            # Just the formation name now
            ax.text(mid, 0, s["variant"], ha="center", va="center", fontsize=8, color="#ffffff", fontweight="bold", zorder=3)
                     
    ax.set_yticks([])
    ax.set_ylim(-1.5, 1.5)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values(): spine.set_color(GRID_COLOR)
    ax.set_facecolor(BG_COLOR)
    return sorted({s[color_key] for s in segments})

def draw_team_panel_piano_roll(ax, bars, var_color, total_duration, granularity="variant", fam_color=None, segments_df=None, offsets=None, fig=None, info_text=None):
    """Multi-row piano-roll layout with HOVER support."""
    if not bars:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color=TEXT_COLOR)
        ax.set_yticks([])
        return []
        
    # Store bars data for hover lookup
    ax._bars_data = bars
    ax._segments_df = segments_df
    ax._offsets = offsets
    
    if granularity == "family":
        present_families = []
        seen = set()
        for b in bars:
            f = b["family"]
            if f not in seen: present_families.append(f); seen.add(f)
        present_families.sort(key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 999)
        n = len(present_families)
        y_of = {fam: n - 1 - i for i, fam in enumerate(present_families)}
        for fam, y in y_of.items(): ax.hlines(y, 0, total_duration, colors=BASELINE_COLOR, linewidth=1, linestyles=(0, (1, 3)), zorder=1)
        for b in bars:
            y = y_of.get(b["family"])
            if y is None: continue
            col = (fam_color or var_color).get(b["family"], "#888888")
            ax.hlines(y, b["matchStart"], b["matchEnd"], colors=col, linewidth=9, alpha=0.9, capstyle="butt", zorder=2)
        ax.set_yticks(range(n))
        ax.set_yticklabels(present_families, color=TEXT_COLOR)
        ax.set_ylim(-0.7, n - 0.3)
    else:
        present = set()
        for b in bars: present.add((b["family"], b["variant"]))
        def _sort_key(p):
            fam, var = p
            try: fi = FAMILY_ORDER.index(fam)
            except ValueError: fi = 999
            return (fi, var)
        sorted_pairs = sorted(present, key=_sort_key)
        n = len(sorted_pairs)
        y_of = {var: n - 1 - i for i, (_, var) in enumerate(sorted_pairs)}
        for var, y in y_of.items(): ax.hlines(y, 0, total_duration, colors=BASELINE_COLOR, linewidth=1, linestyles=(0, (1, 3)), zorder=1)
        prev_family = None
        for i, (fam, var) in enumerate(sorted_pairs):
            if prev_family is not None and fam != prev_family:
                y = n - 1 - i + 0.5
                ax.axhline(y, color=GRID_COLOR, linewidth=0.8, linestyle="--", alpha=0.4, zorder=0)
                ax.text(-0.16, y, fam, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.5, fontstyle="italic", color=TEXT_COLOR, alpha=0.7)
            prev_family = fam
        for b in bars:
            y = y_of.get(b["variant"])
            if y is None: continue
            col = var_color.get(b["variant"], "#888888")
            ax.hlines(y, b["matchStart"], b["matchEnd"], colors=col, linewidth=9, alpha=0.9, capstyle="butt", zorder=2)
        ax.set_yticks(range(n))
        y_labels = [var for _, var in sorted_pairs]
        ax.set_yticklabels(y_labels, color=TEXT_COLOR)
        ax.set_ylim(-0.7, n - 0.3)
        
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values(): spine.set_color(GRID_COLOR)
    ax.set_facecolor(BG_COLOR)
    
    # Hover Logic
    if fig is not None and info_text is not None:
        def on_hover(event):
            if event.inaxes != ax: return
            x, y = event.xdata, event.ydata
            if x is None or y is None: return
            
            # Find row index
            row_idx = int(round(y))
            if row_idx < 0 or row_idx >= n: return
            
            # Find variant for this row
            if granularity == "family":
                target_key = present_families[n - 1 - row_idx]
                key_field = "family"
            else:
                target_key = y_labels[n - 1 - row_idx]
                key_field = "variant"
                
            # Find bar at this time
            for b in bars:
                if b[key_field] == target_key and b["matchStart"] <= x <= b["matchEnd"]:
                    # Found it! Update info text
                    metrics = _lookup_metrics(segments_df, b["team"], b["period"], b["matchStart"], b["matchEnd"], offsets)
                    dur_sec = b["matchEnd"] - b["matchStart"]
                    mm, ss = divmod(int(dur_sec), 60)
                    
                    info_str = f"Formation: {b['variant']} ({b['family']})\n"
                    info_str += f"Duration: {mm}m {ss}s\n"
                    if metrics:
                        info_str += f"Compactness: {metrics.get('mean_compactness', 'N/A'):.1f}m\n"
                        in_poss = metrics.get('in_possession_sec', 0)
                        total_poss = in_poss + metrics.get('out_of_possession_sec', 0) + metrics.get('loose_ball_sec', 0)
                        poss_pct = (in_poss / total_poss * 100) if total_poss > 0 else 0
                        info_str += f"Possession: {poss_pct:.0f}% Attacking\n"
                        conf = metrics.get('mean_confidence', 1.0)
                        info_str += f"Confidence: {'High' if conf > 5000 else 'Med' if conf > 2000 else 'Low'}"
                    info_text.set_text(info_str)
                    fig.canvas.draw_idle()
                    return
                    
            # If no bar found, clear text
            info_text.set_text("Hover over a bar...")
            fig.canvas.draw_idle()
            
        fig.canvas.mpl_connect('motion_notify_event', on_hover)
        
    return list(y_of.keys())

def draw_period_dividers(ax_top, ax_bottom, boundaries):
    for boundary_time, period in boundaries[:-1]:
        for ax in (ax_top, ax_bottom):
            ax.axvline(boundary_time, color=GRID_COLOR, linewidth=1.2, linestyle="--", zorder=0)
            ax_top.text(boundary_time, 1.02, f"End P{int(period)}", transform=ax_top.get_xaxis_transform(), ha="center", va="bottom", fontsize=8, color=TEXT_COLOR)
    end_time, _ = boundaries[-1]
    for ax in (ax_top, ax_bottom):
        ax.axvline(end_time, color=TEXT_COLOR, linewidth=1.3, linestyle="-", zorder=0)
        ax_top.text(end_time, 1.02, f"Match end {format_mmss(end_time)}", transform=ax_top.get_xaxis_transform(), ha="right", va="bottom", fontsize=8, color=TEXT_COLOR)

def draw_goals(ax_home, ax_away, goals, offsets):
    for g in goals:
        off = offsets.get(g["period"], 0.0)
        t = g["sec"] + off
        minute = int(t // 60)
        label = f"{minute}' {g['scorer']}" if g["scorer"] else f"{minute}' Goal"
        if g["ownGoal"]: label += " (OG)"
        for ax, panel_team in [(ax_home, "home"), (ax_away, "away")]:
            color = SCORED_COLOR if g["team"] == panel_team else CONCEDED_COLOR
            ax.axvline(t, color=color, linewidth=1.3, linestyle="-", alpha=0.85, zorder=4)
        target_ax = ax_home if g["team"] == "home" else ax_away
        target_ax.annotate(f"\u26bd {label}", xy=(t, 1.0), xycoords=("data", "axes fraction"), xytext=(4, 4), textcoords="offset points", ha="left", va="bottom", fontsize=7.5, color=SCORED_COLOR, rotation=60, zorder=5)

def plot_formation_timeline(match_id, processed_dir=PROCESSED_DIR_DEFAULT,
                            method="voted", granularity="variant"):
    formations_df, metadata, home_name, away_name, tracking_path, match_dir, segments_df = load_data(match_id, processed_dir)
    offsets, boundaries, total_duration = compute_period_offsets(formations_df, metadata)
    total_duration = max(total_duration, 1.0)
    
    events = pos.load_events(match_dir)
    goals = find_goals(events) if events is not None else []
    
    # Create Info Text for Piano Roll Hover
    info_text = None
    
    if method == "voted":
        all_segments = build_gap_free_segments(formations_df, offsets, granularity=granularity)
        home_data = [s for s in all_segments if s["team"] == "home"]
        away_data = [s for s in all_segments if s["team"] == "away"]
        var_color, fam_color = build_family_color_map(all_segments)
        color_key = "family" if granularity == "family" else "variant"
        color_map = fam_color if granularity == "family" else var_color
        n_rows = max(len({s[color_key] for s in all_segments}), 1)
        fig, (ax_home, ax_away) = plt.subplots(2, 1, figsize=(15, max(8.5, 1.8 + n_rows * 1.2)), sharex=True, gridspec_kw={"hspace": 0.18})
        fig.patch.set_facecolor(BG_COLOR)
        draw_team_panel(ax_home, home_data, color_map, total_duration, color_key=color_key)
        draw_team_panel(ax_away, away_data, color_map, total_duration, color_key=color_key)
        ax_home.set_ylabel("", color=TEXT_COLOR)
        ax_away.set_ylabel("", color=TEXT_COLOR)
        detail = "variant" if granularity == "variant" else "family"
        subtitle = f"Continuous state strip (confidence-weighted vote, {detail})"
        
    elif method == "all_formations":
        all_bars = formation_runs_from_votes(formations_df, offsets, granularity=granularity)
        home_data = [b for b in all_bars if b["team"] == "home"]
        away_data = [b for b in all_bars if b["team"] == "away"]
        var_color, fam_color = build_family_color_map(all_bars)
        
        if granularity == "family":
            unique_keys = sorted({b["family"] for b in all_bars}, key=lambda f: FAMILY_ORDER.index(f) if f in FAMILY_ORDER else 999)
        else:
            unique_keys = sorted({b["variant"] for b in all_bars})
        n_rows = max(len(unique_keys), 1)
        
        # Create figure FIRST
        fig, (ax_home, ax_away) = plt.subplots(2, 1, figsize=(15, max(8.5, 1.8 + n_rows * 1.2)), sharex=True, gridspec_kw={"hspace": 0.35})
        fig.patch.set_facecolor(BG_COLOR)
        
        # Add Info Text Box for Piano Roll (NOW fig IS DEFINED)
        info_text = fig.text(0.98, 0.98, "Hover over a bar...", ha="right", va="top", 
                             fontsize=9, color=TEXT_PRIMARY, 
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="#1e3555", edgecolor=GRID_COLOR, alpha=0.9))
        
        draw_team_panel_piano_roll(
            ax_home, home_data, var_color, total_duration,
            granularity=granularity, fam_color=fam_color, segments_df=segments_df, offsets=offsets, fig=fig, info_text=info_text,
        )
        draw_team_panel_piano_roll(
            ax_away, away_data, var_color, total_duration,
            granularity=granularity, fam_color=fam_color, segments_df=segments_df, offsets=offsets, fig=fig, info_text=info_text,
        )
        ylabel = "Family" if granularity == "family" else "Variant"
        ax_home.set_ylabel(ylabel, color=TEXT_COLOR)
        ax_away.set_ylabel(ylabel, color=TEXT_COLOR)
        detail = "family" if granularity == "family" else "variant"
        subtitle = f"Voted runs at window starts ({detail})"
        
    else:
        raise ValueError(f"Unknown method: {method!r}")
        
    draw_period_dividers(ax_home, ax_away, boundaries)
    draw_goals(ax_home, ax_away, goals, offsets)
    ax_home.set_title(f"{home_name} (home)", loc="left", fontsize=11, color=TEXT_COLOR)
    ax_away.set_title(f"{away_name} (away)", loc="left", fontsize=11, color=TEXT_COLOR)
    ax_away.set_xlabel("Match Time", color=TEXT_COLOR)
    window_seconds = 300
    xticks = build_xticks(total_duration, window_seconds)
    ax_away.set_xticks(xticks)
    ax_away.xaxis.set_major_formatter(mticker.FuncFormatter(format_mmss))
    ax_away.tick_params(axis="x", rotation=45, colors=TEXT_COLOR)
    for label in ax_away.get_xticklabels(): label.set_ha("right")
    ax_away.set_xlim(0, total_duration)
    
    present_families = [f for f in FAMILY_ORDER if f in fam_color]
    handles = [plt.Line2D([0], [0], color=fam_color[f], linewidth=6, label=f) for f in present_families]
    ncol = min(len(present_families), 6)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, frameon=False, labelcolor=TEXT_COLOR, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Match {match_id} \u2014 Formation Timeline\n{subtitle}" + ("  |  \u26bd = goal" if goals else ""), fontsize=13, color=TEXT_COLOR)
    rect = [0, 0.06, 1, 0.93] if method == "voted" else [0, 0.04, 1, 0.90]
    fig.tight_layout(rect=rect)
    return fig

def main():
    parser = argparse.ArgumentParser(description="Plot each team's detected formation over time.")
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR_DEFAULT)
    parser.add_argument("--method", choices=["voted", "all_formations"], default="voted")
    parser.add_argument("--granularity", choices=["variant", "family"], default="variant")
    args = parser.parse_args()
    plot_formation_timeline(args.match_id, args.processed_dir, method=args.method, granularity=args.granularity)
    plt.show()

if __name__ == "__main__":
    main()