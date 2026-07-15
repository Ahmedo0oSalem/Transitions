"""
possession.py

Shared building blocks for possession-sequence detection and EPV lookup,
used by both plot_formation_timeline.py (defines "one bar per possession
sequence") and epv_das_analysis.py (EPV over time + Dangerous Attacking
Sequences).

IMPORTANT CAVEAT: your tracking data has no real possession/event log.
"Possession" here means "closest player to the ball, majority-vote
smoothed to suppress frame-level flicker" -- a reasonable proxy, not
ground truth. Treat every downstream output (possession sequences, DAS,
EPV attribution) as an approximation, especially around contested/loose
balls, blocked shots, deflections, etc.

Requires detect_formations.py in the same folder (reused for field-name
aliasing so this always agrees with the rest of the pipeline).
"""

import os
import bz2
import json
import math

import numpy as np

import detect_formation as df_mod

# ==========================
# Config
# ==========================

POSSESSION_THRESHOLD_M = 2.5    # ball-to-player distance to count as "has the ball"
SMOOTH_WINDOW_SECONDS = 1.0     # majority-vote smoothing window, kills frame-level flicker
MIN_SEQUENCE_SECONDS = 2.0      # possession sequences shorter than this get merged/flagged


# ==========================
# Lenient position extraction (same reasoning as visualize_match.py:
# a single frame has nothing else to fall back on, so we don't drop
# LOW-confidence points here the way detect_formations.py does for
# formation averaging).
# ==========================

def _extract_lenient(player_dict):
    x = df_mod._get_first(player_dict, df_mod.PLAYER_X_KEYS)
    y = df_mod._get_first(player_dict, df_mod.PLAYER_Y_KEYS)
    pid = df_mod._get_first(player_dict, df_mod.PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)


# ==========================
# EPV grid
# ==========================

def load_epv_grid(path):
    """Loads the 32x50 EPV surface (Fernandez/Bornn/Cervone-style, via
    Laurie Shaw's Friends-of-Tracking implementation). Value at a cell is
    the probability a possession ends in a goal, for a team attacking
    toward increasing x (grid columns increase in value toward that end;
    rows are y-symmetric, matching the actual downloaded grid's values).
    """
    return np.loadtxt(path, delimiter=",")


def get_base_directions(metadata):
    """Each team's attacking direction in period 1: +1 = toward
    increasing x, -1 = toward decreasing x."""
    home_start_left = metadata.get("homeTeamStartLeft", True)
    home_dir_p1 = 1 if home_start_left else -1
    return home_dir_p1, -home_dir_p1


def attack_direction(team, period, home_dir_p1, away_dir_p1):
    """
    Direction a team attacks in a given period. Assumes the standard
    alternation (teams swap ends every period) -- true for regulation
    halves, and for extra time periods too under normal convention.
    """
    base = home_dir_p1 if team == "home" else away_dir_p1
    parity = 1 if (int(period) % 2 == 1) else -1
    return base * parity


def epv_value(epv_grid, x, y, pitch_length, pitch_width, direction):
    """
    EPV for a team attacking in `direction` (+1/-1) with the ball at
    (x, y) in 0..pitch_length / 0..pitch_width coordinates.
    """
    n_rows, n_cols = epv_grid.shape
    gx = x if direction == 1 else (pitch_length - x)
    gy = y  # grid is y-symmetric (verified against the downloaded data), no flip needed
    col = int(np.clip(gx / pitch_length * n_cols, 0, n_cols - 1))
    row = int(np.clip(gy / pitch_width * n_rows, 0, n_rows - 1))
    return float(epv_grid[row, col])


# ==========================
# Streaming ball position + nearest-player "owner" per frame
# ==========================

def stream_ball_and_owner(tracking_path, pitch_length, pitch_width,
                           coords_are_centered=True,
                           possession_threshold_m=POSSESSION_THRESHOLD_M):
    """
    Streams the tracking file once. Returns parallel numpy arrays:
        periods (int16), elapsed (float32, seconds into period),
        ball_x, ball_y (float32, NaN if untracked),
        owner (int8: 0 = no one within threshold / ball untracked,
                     1 = home, 2 = away)
    """
    x_shift = pitch_length / 2 if coords_are_centered else 0.0
    y_shift = pitch_width / 2 if coords_are_centered else 0.0

    periods, elapsed, ball_x, ball_y, owner = [], [], [], [], []

    n_lines = 0
    try:
        with bz2.open(tracking_path, "rt") as f:
            for line in f:
                frame = json.loads(line)
                n_lines += 1
                period = frame.get("period")
                et = frame.get("periodElapsedTime")
                if period is None or et is None:
                    continue

                balls = frame.get("balls", [])
                bx = by = float("nan")
                if balls:
                    b = balls[0]
                    if b.get("x") is not None and b.get("y") is not None:
                        bx = float(b["x"]) + x_shift
                        by = float(b["y"]) + y_shift

                best_team, best_dist = 0, possession_threshold_m
                if not math.isnan(bx):
                    for team_code, key in ((1, "homePlayers"), (2, "awayPlayers")):
                        for p in frame.get(key, []):
                            parsed = _extract_lenient(p)
                            if parsed is None:
                                continue
                            _, px, py = parsed
                            px += x_shift
                            py += y_shift
                            d = math.hypot(px - bx, py - by)
                            if d < best_dist:
                                best_dist = d
                                best_team = team_code

                periods.append(period)
                elapsed.append(et)
                ball_x.append(bx)
                ball_y.append(by)
                owner.append(best_team)
    except EOFError:
        print(f"    !! WARNING: {tracking_path} appears truncated/corrupted "
              f"(bz2 stream ended early after {n_lines} frames). Continuing "
              f"with the frames successfully read.")

    return (np.array(periods, dtype=np.int16),
            np.array(elapsed, dtype=np.float32),
            np.array(ball_x, dtype=np.float32),
            np.array(ball_y, dtype=np.float32),
            np.array(owner, dtype=np.int8))


def infer_fps(elapsed, periods):
    """Crude fps estimate from median frame spacing within the first period."""
    mask = periods == periods[0]
    diffs = np.diff(elapsed[mask])
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 25.0
    return float(1.0 / np.median(diffs))


def _rolling_sum_same(arr, window):
    kernel = np.ones(window, dtype=np.int64)
    return np.convolve(arr, kernel, mode="same")


def smooth_owner(owner, periods, fps, window_seconds=SMOOTH_WINDOW_SECONDS):
    """
    Majority-vote smoothing over a rolling window, computed separately
    per period (so the window never bleeds across the half-time gap).
    Kills single/few-frame flicker in the raw nearest-player heuristic.
    """
    window = max(1, int(round(window_seconds * fps)))
    smoothed = owner.copy()

    for p in np.unique(periods):
        idx = np.where(periods == p)[0]
        o = owner[idx]
        rn = _rolling_sum_same((o == 0).astype(np.int64), window)
        rh = _rolling_sum_same((o == 1).astype(np.int64), window)
        ra = _rolling_sum_same((o == 2).astype(np.int64), window)
        smoothed[idx] = np.argmax(np.vstack([rn, rh, ra]), axis=0)

    return smoothed


# ==========================
# Possession sequences (run-length encoding of the smoothed owner stream)
# ==========================

_TEAM_OF = {0: None, 1: "home", 2: "away"}


def detect_possession_sequences(owner_smoothed, periods, elapsed, fps,
                                 min_seconds=MIN_SEQUENCE_SECONDS):
    """
    Run-length encodes the smoothed owner stream (per period) into
    possession sequences. Returns a list of dicts:
        {team, period, start_idx, end_idx, start_sec, end_sec, duration}
    `team` is 'home', 'away', or None (no one within the possession
    threshold -- loose ball / far from any tracked player).

    Sequences shorter than `min_seconds` that are flanked on both sides
    by the SAME team are merged into that team's run (residual flicker
    the majority-vote smoothing didn't fully catch). Other short
    sequences (genuine quick changes of possession, or brief loose-ball
    gaps) are kept as-is.
    """
    sequences = []

    for p in np.unique(periods):
        idx = np.where(periods == p)[0]
        o = owner_smoothed[idx]
        e = elapsed[idx]

        change_points = np.where(np.diff(o) != 0)[0] + 1
        starts = np.concatenate(([0], change_points))
        ends = np.concatenate((change_points, [len(o)]))

        raw_seqs = []
        for s, en in zip(starts, ends):
            end_sec = float(e[en]) if en < len(e) else float(e[en - 1]) + 1.0 / fps
            raw_seqs.append({
                "team": _TEAM_OF[int(o[s])],
                "period": int(p),
                "start_idx": int(idx[s]),
                "end_idx": int(idx[en - 1]),
                "start_sec": float(e[s]),
                "end_sec": end_sec,
            })

        # Merge tiny (< min_seconds) segments flanked by the same team on
        # both sides -- residual flicker the smoothing pass left behind.
        merged = []
        i = 0
        while i < len(raw_seqs):
            seg = raw_seqs[i]
            dur = seg["end_sec"] - seg["start_sec"]
            can_merge = (
                dur < min_seconds
                and merged
                and i + 1 < len(raw_seqs)
                and merged[-1]["team"] == raw_seqs[i + 1]["team"]
                and merged[-1]["team"] is not None
            )
            if can_merge:
                merged[-1]["end_sec"] = seg["end_sec"]
                merged[-1]["end_idx"] = seg["end_idx"]
                i += 1
                continue
            merged.append(dict(seg))
            i += 1

        for seg in merged:
            seg["duration"] = seg["end_sec"] - seg["start_sec"]

        # Final coalescing pass: the merge step above can leave two
        # adjacent same-team segments un-combined (e.g. home / tiny-away
        # / home -> merging the tiny segment into the first "home" run
        # doesn't automatically fold in the second "home" run that
        # follows it). Any genuinely adjacent same-team segments should
        # always be one continuous run, so combine them here.
        coalesced = []
        for seg in merged:
            if coalesced and coalesced[-1]["team"] == seg["team"]:
                coalesced[-1]["end_sec"] = seg["end_sec"]
                coalesced[-1]["end_idx"] = seg["end_idx"]
                coalesced[-1]["duration"] = coalesced[-1]["end_sec"] - coalesced[-1]["start_sec"]
            else:
                coalesced.append(dict(seg))

        sequences.extend(coalesced)

    return sequences


def forward_fill_owner(owner_smoothed, periods, elapsed, max_gap_seconds=3.0):
    """
    Carries the last known team forward through brief gaps (owner==0,
    "no one within threshold") of up to `max_gap_seconds`, per period.

    This exists specifically for danger/EPV attribution: the single most
    dangerous instant of a possession -- a shot in flight, a ball
    bouncing loose in the box after a cross -- is exactly when the ball
    separates from any one player and the raw nearest-player heuristic
    reports "no owner". Gating a danger threshold on the raw possession
    sequences (which treat that gap as a break) systematically misses
    the moments that matter most. Forward-filling a short gap keeps
    crediting the team that produced the shot/cross for those frames.

    Longer gaps (more than max_gap_seconds) are left as 0 -- that's
    treated as a genuine break in possession (out of play, defensive
    clearance recovered later, etc.), not a shot/cross in flight.
    """
    filled = owner_smoothed.copy()

    for p in np.unique(periods):
        idx = np.where(periods == p)[0]
        o = filled[idx]
        e = elapsed[idx]

        i = 0
        n = len(o)
        while i < n:
            if o[i] == 0:
                j = i
                while j < n and o[j] == 0:
                    j += 1
                gap_duration = float(e[j - 1] - e[i]) if j > i else 0.0
                prev_team = o[i - 1] if i > 0 else 0
                if prev_team != 0 and gap_duration <= max_gap_seconds:
                    o[i:j] = prev_team
                i = j
            else:
                i += 1

        filled[idx] = o

    return filled
