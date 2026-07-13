"""
visualize_match.py

Interactive viewer for tracking data: shows both teams' player positions,
the ball, a "who has the ball" heuristic, and each team's currently
detected formation (pulled from formations.csv, produced by
detect_formations.py) -- all scrubbable with a slider.

Run:
    python3 visualize_match.py <match_id> [--processed-dir Processed_Tracking] [--speed 2.0]

Controls:
    - Drag the slider along the bottom to jump to any point in the match.
    - Click Play/Pause to auto-advance in roughly real time (--speed
      multiplies playback speed, e.g. --speed 4 runs 4x faster).

Requires the same input as detect_formations.py:
    Processed_Tracking/<match_id>/metadata.json
    Processed_Tracking/<match_id>/tracking.jsonl.bz2
and, optionally (for the formation labels to show up):
    Processed_Tracking/<match_id>/formations.csv

This script reuses detect_formations.py's parsing/config/goalkeeper logic
directly (same file, same folder) so the two tools always agree on field
names, coordinate handling, and who the goalkeepers are.
"""

import os
import sys
import bz2
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
# matplotlib.use(backend) only RECORDS the preference -- it doesn't import
# the backend module until the first plt.subplots()/pitch.draw() call. So
# wrapping it in try/except doesn't actually catch an unavailable backend;
# it always "succeeds" immediately and the real failure shows up later,
# deep in a draw() call, which is confusing. Pick by platform instead:
# MacOSX backend only exists (and is only worth using) on macOS.
matplotlib.use("MacOSX" if sys.platform == "darwin" else "TkAgg")
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mplsoccer import Pitch

# Reuse detect_formations.py's config, coordinate handling, field-name
# aliasing, and goalkeeper-identification logic so this visualizer is
# always consistent with the formation-detection output. Must be in the
# same folder as this script.
import detect_formation as df_mod


# ==========================
# Config
# ==========================

HOME_COLOR = "#d62728"       # red
AWAY_COLOR = "#1f77b4"       # blue
BALL_COLOR = "#f7e017"
POSSESSION_RING_COLOR = "#2ecc71"  # green ring around whoever "has" the ball

# Max ball-to-player distance (in meters) to count as that player having
# possession. This is a simple proximity heuristic -- your tracking data
# doesn't include real possession events, so this is an approximation, not
# ground truth (won't know about handballs, blocked shots, etc.)
POSSESSION_THRESHOLD_M = 2.5

OFF_SCREEN = -1000.0  # sentinel position used to "hide" missing points


def _extract_player_xy_lenient(player_dict):
    """
    Like detect_formations.extract_player_xy, but WITHOUT the
    confidence/visibility quality filter. That filter is correct for
    formation averaging (LOW-confidence points are noisy outliers that
    should be excluded from an average), but wrong here: dropping ~60%+
    of points per frame means most players would flicker in and out of
    existence on screen. For a single-frame display, a LOW-confidence
    point is still far more useful than no point at all.
    """
    x = df_mod._get_first(player_dict, df_mod.PLAYER_X_KEYS)
    y = df_mod._get_first(player_dict, df_mod.PLAYER_Y_KEYS)
    pid = df_mod._get_first(player_dict, df_mod.PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)


# ==========================
# Data loading
# ==========================

def load_match(match_id, processed_dir):
    match_dir = os.path.join(processed_dir, str(match_id))
    metadata_path = os.path.join(match_dir, "metadata.json")
    tracking_path = os.path.join(match_dir, "tracking.jsonl.bz2")
    formations_path = os.path.join(match_dir, "formations.csv")

    if not os.path.exists(metadata_path) or not os.path.exists(tracking_path):
        raise FileNotFoundError(
            f"Missing metadata.json / tracking.jsonl.bz2 under {match_dir}. "
            f"Run preprocessing (and detect_formations.py, optionally) first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    x_shift = pitch_length / 2 if df_mod.COORDS_ARE_CENTERED else 0.0
    y_shift = pitch_width / 2 if df_mod.COORDS_ARE_CENTERED else 0.0

    # ---- Pass 1: find every jersey/player id seen for each team, and
    #      count frames, so we can allocate fixed-size arrays up front
    #      instead of growing lists (much faster for a full match). ----
    home_ids, away_ids = set(), set()
    n_frames = 0
    print("Scanning tracking file (pass 1/2: collecting player IDs)...")
    with bz2.open(tracking_path, "rt") as f:
        for line in f:
            frame = json.loads(line)
            n_frames += 1
            for p in frame.get("homePlayers", []):
                pid = df_mod._get_first(p, df_mod.PLAYER_ID_KEYS)
                if pid is not None:
                    home_ids.add(pid)
            for p in frame.get("awayPlayers", []):
                pid = df_mod._get_first(p, df_mod.PLAYER_ID_KEYS)
                if pid is not None:
                    away_ids.add(pid)

    def _sort_key(v):
        # Sort jersey numbers numerically when possible, alphabetically otherwise.
        try:
            return (0, int(v))
        except (TypeError, ValueError):
            return (1, str(v))

    home_ids = sorted(home_ids, key=_sort_key)
    away_ids = sorted(away_ids, key=_sort_key)
    home_idx = {pid: i for i, pid in enumerate(home_ids)}
    away_idx = {pid: i for i, pid in enumerate(away_ids)}

    # ---- Pass 2: fill fixed-width arrays (NaN = player not present in
    #      that frame, e.g. not subbed on yet / already subbed off). ----
    print("Scanning tracking file (pass 2/2: loading positions)...")
    periods = np.zeros(n_frames, dtype=np.int16)
    elapsed = np.zeros(n_frames, dtype=np.float32)
    home_xy = np.full((n_frames, len(home_ids), 2), np.nan, dtype=np.float32)
    away_xy = np.full((n_frames, len(away_ids), 2), np.nan, dtype=np.float32)
    ball_xy = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with bz2.open(tracking_path, "rt") as f:
        for i, line in enumerate(f):
            frame = json.loads(line)
            periods[i] = frame.get("period", 0) or 0
            elapsed[i] = frame.get("periodElapsedTime", 0.0) or 0.0

            for p in frame.get("homePlayers", []):
                parsed = _extract_player_xy_lenient(p)
                if parsed is None:
                    continue
                pid, x, y = parsed
                home_xy[i, home_idx[pid]] = (x + x_shift, y + y_shift)

            for p in frame.get("awayPlayers", []):
                parsed = _extract_player_xy_lenient(p)
                if parsed is None:
                    continue
                pid, x, y = parsed
                away_xy[i, away_idx[pid]] = (x + x_shift, y + y_shift)

            balls = frame.get("balls", [])
            if balls:
                b = balls[0]
                bx, by = b.get("x"), b.get("y")
                if bx is not None and by is not None:
                    ball_xy[i] = (bx + x_shift, by + y_shift)

    print("Resolving goalkeepers (roster first, distance-based fallback)...")
    # resolve_goalkeepers checks metadata["goalkeepers"] (from the roster,
    # via preprocessing.py) first, and only falls back to distance-based
    # identify_goalkeepers per side if the roster didn't cover it -- same
    # logic path detect_formations.py now uses, so the two tools still
    # always agree on who the goalkeepers are. Each team maps to a small
    # set of acceptable IDs (playerId and/or shirtNumber), not a single
    # scalar, since we don't assume which key your tracking data uses.
    goalkeepers = df_mod.resolve_goalkeepers(tracking_path, metadata)

    formations_df = None
    if os.path.exists(formations_path):
        formations_df = pd.read_csv(formations_path)
        print(f"Loaded {len(formations_df)} formation-window rows from formations.csv")
    else:
        print("No formations.csv found -- formation labels will show as 'n/a'. "
              "Run detect_formations.py first if you want them.")

    return {
        "metadata": metadata,
        "pitch_length": pitch_length,
        "pitch_width": pitch_width,
        "periods": periods,
        "elapsed": elapsed,
        "home_ids": home_ids,
        "away_ids": away_ids,
        "home_xy": home_xy,
        "away_xy": away_xy,
        "ball_xy": ball_xy,
        "goalkeepers": goalkeepers,
        "formations_df": formations_df,
    }


def get_formation_label(formations_df, team, period, elapsed_sec):
    """
    Looks up the formation for `team` at `elapsed_sec` into `period`,
    using whichever formations.csv window currently covers that moment.
    If windows overlap (sliding-window output), picks the one that
    started most recently (i.e. the "freshest" reading for right now).
    """
    if formations_df is None:
        return "n/a"
    sub = formations_df[
        (formations_df["team"] == team)
        & (formations_df["period"] == period)
        & (formations_df["windowStartSec"] <= elapsed_sec)
        & (elapsed_sec < formations_df["windowEndSec"])
    ]
    if sub.empty:
        return "n/a"
    row = sub.loc[sub["windowStartSec"].idxmax()]
    return f"{row['formation']} (cost {row['avgCostPerPlayer']:.1f})"


# ==========================
# UI
# ==========================

def run_app(match_id, processed_dir, speed):
    data = load_match(match_id, processed_dir)
    metadata = data["metadata"]
    n_frames = len(data["periods"])
    fps = metadata.get("fps", 25.0)

    home_name = metadata["homeTeam"].get("shortName") or metadata["homeTeam"].get("name", "Home")
    away_name = metadata["awayTeam"].get("shortName") or metadata["awayTeam"].get("name", "Away")

    pitch = Pitch(pitch_type="custom", pitch_length=data["pitch_length"],
                   pitch_width=data["pitch_width"], pitch_color="#2b8a3e", line_color="white")
    fig, ax = pitch.draw(figsize=(12, 8))
    plt.subplots_adjust(bottom=0.2)

    home_scatter = ax.scatter([], [], s=260, c=HOME_COLOR, edgecolors="white",
                               linewidths=1.5, zorder=3, label=home_name)
    away_scatter = ax.scatter([], [], s=260, c=AWAY_COLOR, edgecolors="white",
                               linewidths=1.5, zorder=3, label=away_name)
    ball_scatter = ax.scatter([], [], s=90, c=BALL_COLOR, edgecolors="black",
                               linewidths=1, zorder=4)
    possession_ring = ax.scatter([], [], s=440, facecolors="none",
                                  edgecolors=POSSESSION_RING_COLOR, linewidths=2.5, zorder=5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=2, frameon=False, labelcolor="white")

    home_texts = [ax.text(OFF_SCREEN, OFF_SCREEN, "", ha="center", va="center",
                           fontsize=8, fontweight="bold", color="white", zorder=6)
                  for _ in data["home_ids"]]
    away_texts = [ax.text(OFF_SCREEN, OFF_SCREEN, "", ha="center", va="center",
                           fontsize=8, fontweight="bold", color="white", zorder=6)
                  for _ in data["away_ids"]]

    title_text = ax.set_title("", fontsize=11, color="white")
    fig.patch.set_facecolor("#1b1b1b")

    def nearest_owner(frame_idx):
        bx, by = data["ball_xy"][frame_idx]
        if np.isnan(bx) or np.isnan(by):
            return None, None
        best_team, best_pid, best_dist = None, None, POSSESSION_THRESHOLD_M
        for team_key, xy_arr, ids in (("home", data["home_xy"], data["home_ids"]),
                                       ("away", data["away_xy"], data["away_ids"])):
            pts = xy_arr[frame_idx]
            if pts.size == 0:
                continue
            dists = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
            if np.all(np.isnan(dists)):
                continue
            j = int(np.nanargmin(dists))
            d = dists[j]
            if not np.isnan(d) and d < best_dist:
                best_dist = d
                best_team = team_key
                best_pid = ids[j]
        return best_team, best_pid

    def update(val):
        frame_idx = int(val)
        home_pts = data["home_xy"][frame_idx]
        away_pts = data["away_xy"][frame_idx]
        ball_pt = data["ball_xy"][frame_idx]

        home_scatter.set_offsets(np.nan_to_num(home_pts, nan=OFF_SCREEN))
        away_scatter.set_offsets(np.nan_to_num(away_pts, nan=OFF_SCREEN))
        ball_scatter.set_offsets([np.nan_to_num(ball_pt, nan=OFF_SCREEN)])

        for t, pt in zip(home_texts, home_pts):
            if np.isnan(pt[0]):
                t.set_position((OFF_SCREEN, OFF_SCREEN))
            else:
                t.set_position((pt[0], pt[1]))
        for pid, t in zip(data["home_ids"], home_texts):
            t.set_text(str(pid))
        for t, pt in zip(away_texts, away_pts):
            if np.isnan(pt[0]):
                t.set_position((OFF_SCREEN, OFF_SCREEN))
            else:
                t.set_position((pt[0], pt[1]))
        for pid, t in zip(data["away_ids"], away_texts):
            t.set_text(str(pid))

        owner_team, owner_pid = nearest_owner(frame_idx)
        if owner_team is not None:
            ids = data["home_ids"] if owner_team == "home" else data["away_ids"]
            xy_arr = data["home_xy"] if owner_team == "home" else data["away_xy"]
            pt = xy_arr[frame_idx, ids.index(owner_pid)]
            possession_ring.set_offsets([pt])
            possession_str = f"{owner_team.upper()} #{owner_pid}"
        else:
            possession_ring.set_offsets([[OFF_SCREEN, OFF_SCREEN]])
            possession_str = "loose ball / unknown"

        period = int(data["periods"][frame_idx])
        elapsed_sec = float(data["elapsed"][frame_idx])
        mm, ss = divmod(int(elapsed_sec), 60)

        home_formation = get_formation_label(data["formations_df"], "home", period, elapsed_sec)
        away_formation = get_formation_label(data["formations_df"], "away", period, elapsed_sec)

        title_text.set_text(
            f"Period {period}  {mm:02d}:{ss:02d}   |   "
            f"{home_name}: {home_formation}   vs   {away_name}: {away_formation}   |   "
            f"Ball: {possession_str}"
        )

        fig.canvas.draw_idle()

    # ---- Slider ----
    ax_slider = plt.axes([0.15, 0.06, 0.65, 0.03])
    slider = Slider(ax_slider, "Frame", 0, n_frames - 1, valinit=0, valstep=1, color=POSSESSION_RING_COLOR)
    slider.on_changed(update)

    # ---- Play / Pause ----
    playing = {"on": False}
    ax_button = plt.axes([0.85, 0.055, 0.1, 0.04])
    play_button = Button(ax_button, "Play")

    step_per_tick = max(1, int(fps / 10 * speed))  # advance ~0.1s of match time per tick, scaled by --speed
    timer = fig.canvas.new_timer(interval=100)  # fires every 100ms

    def advance():
        if not playing["on"]:
            return
        next_frame = slider.val + step_per_tick
        if next_frame >= n_frames:
            playing["on"] = False
            play_button.label.set_text("Play")
            return
        slider.set_val(next_frame)

    timer.add_callback(advance)
    timer.start()

    def toggle_play(event):
        playing["on"] = not playing["on"]
        play_button.label.set_text("Pause" if playing["on"] else "Play")

    play_button.on_clicked(toggle_play)

    update(0)
    plt.show()


# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Interactive match tracking visualizer.")
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default="Processed_Tracking")
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed multiplier for Play mode (default 1.0).")
    args = parser.parse_args()
    run_app(args.match_id, args.processed_dir, args.speed)


if __name__ == "__main__":
    main()