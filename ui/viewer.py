"""
visualize_match.py

Interactive viewer for tracking data: shows both teams' player positions,
the ball, a "who has the ball" heuristic, and each team's currently
detected formation (pulled from formations.csv, produced by
detect_formations.py) -- all scrubbable with a slider.

Run:
    python3 visualize_match.py <match_id> [--processed-dir Processed_Tracking] [--speed 2.0]
"""

import sys
import bz2
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("QtAgg" if "PyQt6" in sys.modules else ("MacOSX" if sys.platform == "darwin" else "TkAgg"))
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mplsoccer import Pitch

from ..analytics.formations import detector as df_mod
from ..io.paths import match_dir, PROCESSED_DIR

# --- Modernized Theme Overrides ---
# Bypassing the old .theme import to guarantee the new dashboard look
MODERN_BG = "#121212"
MODERN_PITCH = "#016803"
MODERN_LINES = "#404040"
HOME_COLOR = "#00E5FF"       # Neon Cyan
AWAY_COLOR = "#FF007F"       # Neon Magenta
BALL_COLOR = "#FFFFFF"       # Crisp White
POSSESSION_RING_COLOR = "#FFD700" # Gold
TEXT_PRIMARY = "#FFFFFF"
TEXT_SECONDARY = "#A0A0A0"

POSSESSION_THRESHOLD_M = 2.5
OFF_SCREEN = -1000.0


def _extract_player_xy_lenient(player_dict):
    x = df_mod._get_first(player_dict, df_mod.PLAYER_X_KEYS)
    y = df_mod._get_first(player_dict, df_mod.PLAYER_Y_KEYS)
    pid = df_mod._get_first(player_dict, df_mod.PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)


def load_match(match_id, processed_dir):
    match_dir_path = match_dir(match_id, processed_dir)
    metadata_path = match_dir_path / "metadata.json"
    tracking_path = match_dir_path / "tracking.jsonl.bz2"
    formations_path = match_dir_path / "formations.csv"

    if not metadata_path.is_file() or not tracking_path.is_file():
        raise FileNotFoundError(
            f"Missing metadata.json / tracking.jsonl.bz2 under {match_dir_path}. "
            f"Run preprocessing (and detect_formations.py, optionally) first."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    x_shift = pitch_length / 2 if df_mod.COORDS_ARE_CENTERED else 0.0
    y_shift = pitch_width / 2 if df_mod.COORDS_ARE_CENTERED else 0.0

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
        try:
            return (0, int(v))
        except (TypeError, ValueError):
            return (1, str(v))

    home_ids = sorted(home_ids, key=_sort_key)
    away_ids = sorted(away_ids, key=_sort_key)
    home_idx = {pid: i for i, pid in enumerate(home_ids)}
    away_idx = {pid: i for i, pid in enumerate(away_ids)}

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
                if parsed is None: continue
                pid, x, y = parsed
                home_xy[i, home_idx[pid]] = (x + x_shift, y + y_shift)

            for p in frame.get("awayPlayers", []):
                parsed = _extract_player_xy_lenient(p)
                if parsed is None: continue
                pid, x, y = parsed
                away_xy[i, away_idx[pid]] = (x + x_shift, y + y_shift)

            balls = frame.get("balls", [])
            if balls:
                b = balls[0]
                bx, by = b.get("x"), b.get("y")
                if bx is not None and by is not None:
                    ball_xy[i] = (bx + x_shift, by + y_shift)

    print("Resolving goalkeepers (roster first, distance-based fallback)...")
    goalkeepers = df_mod.resolve_goalkeepers(tracking_path, metadata)

    formations_df = None
    segments_df = None
    segments_path = match_dir_path / "formation_segments.csv"
    if segments_path.is_file():
        segments_df = pd.read_csv(segments_path)
        print(f"Loaded {len(segments_df)} formation segments from formation_segments.csv")
        
    # Update the return dictionary to include segments_df:
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
        "segments_df": segments_df,  # <-- Add this line
    }


def get_formation_label(formations_df, segments_df, team, period, elapsed_sec):
    """Get formation label with segment properties if available."""
    # Try to find the segment first (more accurate for current moment)
    if segments_df is not None and not segments_df.empty:
        seg_sub = segments_df[
            (segments_df["team"] == team)
            & (segments_df["period"] == period)
            & (segments_df["start_sec"] <= elapsed_sec)
            & (elapsed_sec < segments_df["end_sec"])
        ]
        if not seg_sub.empty:
            row = seg_sub.iloc[0]
            # Format: "4-3-3 | 12m | Compact: 8.2m | Attacking: 65%"
            duration_min = int(row["duration"] // 60)
            compactness = row.get("mean_compactness", 0)
            total_poss = row.get("in_possession_sec", 0) + row.get("out_of_possession_sec", 0) + row.get("loose_ball_sec", 0)
            attacking_pct = int((row.get("in_possession_sec", 0) / total_poss * 100) if total_poss > 0 else 0)
            return f"{row['variant']} | {duration_min}m | Compact: {compactness:.1f}m | Attacking: {attacking_pct}%"
    
    # Fallback to window-based label
    if formations_df is None:
        return "N/A"
        
    sub = formations_df[
        (formations_df["team"] == team)
        & (formations_df["period"] == period)
        & (formations_df["windowStartSec"] <= elapsed_sec)
        & (elapsed_sec < formations_df["windowEndSec"])
    ]
    if sub.empty:
        return "Transitioning"
    row = sub.loc[sub["windowStartSec"].idxmax()]
    return f"{row['formation']}"
# ==========================
# UI
# ==========================

def run_app(match_id, processed_dir, speed, show=True, block=True):
    plt.style.use('dark_background') # Apply dark theme globally for this window
    
    data = load_match(match_id, processed_dir)
    metadata = data["metadata"]
    n_frames = len(data["periods"])
    fps = metadata.get("fps", 25.0)

    home_name = metadata["homeTeam"].get("shortName") or metadata["homeTeam"].get("name", "HOME")
    away_name = metadata["awayTeam"].get("shortName") or metadata["awayTeam"].get("name", "AWAY")
    home_name, away_name = home_name.upper(), away_name.upper()

    pitch = Pitch(
        pitch_type="custom", 
        pitch_length=data["pitch_length"],
        pitch_width=data["pitch_width"], 
        pitch_color=MODERN_PITCH, 
        line_color=MODERN_LINES,
        linewidth=1.5
    )
    
    fig, ax = pitch.draw(figsize=(14, 8))
    fig.patch.set_facecolor(MODERN_BG)
    plt.subplots_adjust(top=0.88, bottom=0.15) # Room for sleek dashboard header & controls

    # --- Modernized Dashboard Header ---
    fig.suptitle(f"MATCH TRACKING VISUALIZER", fontsize=16, fontweight="bold", color=TEXT_PRIMARY, y=0.96, ha="left", x=0.05)
    
    # Team Names & Formations
    home_header = fig.text(0.05, 0.90, f"{home_name}", fontsize=14, fontweight="heavy", color=HOME_COLOR)
    home_form_text = fig.text(0.05, 0.87, "Formation: N/A", fontsize=10, color=TEXT_SECONDARY)
    
    away_header = fig.text(0.95, 0.90, f"{away_name}", fontsize=14, fontweight="heavy", color=AWAY_COLOR, ha="right")
    away_form_text = fig.text(0.95, 0.87, "Formation: N/A", fontsize=10, color=TEXT_SECONDARY, ha="right")
    
    # Match Clock & Possession
    clock_text = fig.text(0.5, 0.92, "00:00", fontsize=18, fontweight="bold", color=TEXT_PRIMARY, ha="center")
    period_text = fig.text(0.5, 0.89, "PERIOD 1", fontsize=10, fontweight="bold", color=TEXT_SECONDARY, ha="center")
    poss_text = fig.text(0.5, 0.85, "BALL: LOOSE", fontsize=10, fontweight="bold", color=TEXT_SECONDARY, ha="center")

    # --- Plotting Elements ---
    home_scatter = ax.scatter([], [], s=220, c=HOME_COLOR, alpha=0.9, zorder=3)
    away_scatter = ax.scatter([], [], s=220, c=AWAY_COLOR, alpha=0.9, zorder=3)
    
    ball_glow = ax.scatter([], [], s=180, c=BALL_COLOR, alpha=0.2, zorder=4)
    ball_scatter = ax.scatter([], [], s=60, c=BALL_COLOR, edgecolors=MODERN_BG, linewidths=1.5, zorder=5)
    
    possession_ring = ax.scatter([], [], s=500, facecolors="none", edgecolors=POSSESSION_RING_COLOR, 
                                 linewidths=2, linestyle="--", alpha=0.8, zorder=2)

    home_texts = [ax.text(OFF_SCREEN, OFF_SCREEN, "", ha="center", va="center",
                           fontsize=8, fontweight="bold", color=MODERN_BG, zorder=6)
                  for _ in data["home_ids"]]
    away_texts = [ax.text(OFF_SCREEN, OFF_SCREEN, "", ha="center", va="center",
                           fontsize=8, fontweight="bold", color=MODERN_BG, zorder=6)
                  for _ in data["away_ids"]]

    def nearest_owner(frame_idx):
        bx, by = data["ball_xy"][frame_idx]
        if np.isnan(bx) or np.isnan(by):
            return None, None
        best_team, best_pid, best_dist = None, None, POSSESSION_THRESHOLD_M
        for team_key, xy_arr, ids in (("home", data["home_xy"], data["home_ids"]),
                                      ("away", data["away_xy"], data["away_ids"])):
            pts = xy_arr[frame_idx]
            if pts.size == 0: continue
            dists = np.hypot(pts[:, 0] - bx, pts[:, 1] - by)
            if np.all(np.isnan(dists)): continue
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
        
        safe_ball = [np.nan_to_num(ball_pt, nan=OFF_SCREEN)]
        ball_scatter.set_offsets(safe_ball)
        ball_glow.set_offsets(safe_ball)

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
            color = HOME_COLOR if owner_team == "home" else AWAY_COLOR
            poss_text.set_text(f"BALL: {owner_team.upper()} #{owner_pid}")
            poss_text.set_color(color)
        else:
            possession_ring.set_offsets([[OFF_SCREEN, OFF_SCREEN]])
            poss_text.set_text("BALL: LOOSE")
            poss_text.set_color(TEXT_SECONDARY)

        period = int(data["periods"][frame_idx])
        elapsed_sec = float(data["elapsed"][frame_idx])
        mm, ss = divmod(int(elapsed_sec), 60)

        # Update text widgets
        clock_text.set_text(f"{mm:02d}:{ss:02d}")
        period_text.set_text(f"PERIOD {period}")
        
        # UPDATED: Pass segments_df to get_formation_label
        home_form_text.set_text(f"Formation: {get_formation_label(data['formations_df'], data.get('segments_df'), 'home', period, elapsed_sec)}")
        away_form_text.set_text(f"Formation: {get_formation_label(data['formations_df'], data.get('segments_df'), 'away', period, elapsed_sec)}")

        fig.canvas.draw_idle()

    # --- Modern UI Controls ---
    ax_slider = plt.axes([0.15, 0.05, 0.65, 0.03])
    ax_slider.set_facecolor(MODERN_BG)
    slider = Slider(
        ax_slider, "Timeline ", 0, n_frames - 1, 
        valinit=0, valstep=1, 
        color=TEXT_SECONDARY,
        track_color=MODERN_LINES
    )
    slider.label.set_color(TEXT_PRIMARY)
    slider.label.set_fontweight("bold")
    slider.valtext.set_color(TEXT_PRIMARY)
    slider.on_changed(update)

    playing = {"on": False}
    ax_button = plt.axes([0.85, 0.045, 0.1, 0.04])
    play_button = Button(ax_button, "▶ PLAY", color=MODERN_LINES, hovercolor="#555555")
    play_button.label.set_color(TEXT_PRIMARY)
    play_button.label.set_fontweight("bold")

    step_per_tick = max(1, int(fps / 10 * speed))
    timer = fig.canvas.new_timer(interval=100)

    def advance():
        if not playing["on"]: return
        next_frame = slider.val + step_per_tick
        if next_frame >= n_frames:
            playing["on"] = False
            play_button.label.set_text("▶ PLAY")
            return
        slider.set_val(next_frame)

    timer.add_callback(advance)
    timer.start()

    def toggle_play(event):
        playing["on"] = not playing["on"]
        play_button.label.set_text("⏸ PAUSE" if playing["on"] else "▶ PLAY")

    play_button.on_clicked(toggle_play)

    update(0)
    fig._transitions_controls = (slider, play_button, timer)
    if show:
        plt.show(block=block)
    return fig
# ==========================
# CLI
# ==========================

def main():
    parser = argparse.ArgumentParser(description="Interactive match tracking visualizer.")
    parser.add_argument("match_id")
    parser.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    parser.add_argument("--speed", type=float, default=1.0,
                         help="Playback speed multiplier for Play mode (default 1.0).")
    args = parser.parse_args()
    run_app(args.match_id, args.processed_dir, args.speed)


if __name__ == "__main__":
    main()