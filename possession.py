"""
possession.py

Shared building blocks for possession-sequence detection and EPV lookup,
used by both plot_formation_timeline.py (defines "one bar per possession
sequence") and epv_das_analysis.py (EPV over time + Dangerous Attacking
Sequences).

TWO POSSESSION SOURCES, in order of preference:
  1. REAL events, from Processed_Tracking/<match_id>/events.json (written
     by preprocessing.py's process_events(), sourced from the dedicated
     Event_Data/<match_id>.json files -- the PFF FC Event Data
     Specification v2.5 format, game events + possession events
     pre-merged by the provider, one row per possession event). This is
     an actual provider-labeled possession log -- ground truth, not a
     proxy. Use load_events() + possession_sequences_from_events().
     (Falls back to scraping game_event/possession_event straight off
     the tracking frames only for matches with no Event_Data file --
     see preprocessing.py's process_tracking docstring.)
  2. FALLBACK proximity proxy: "closest player to the ball, majority-
     vote smoothed to suppress frame-level flicker" -- used only when
     events.json doesn't exist for a match (older data, or a provider
     without event data). Treat this path's output (possession
     sequences, DAS, EPV attribution) as an approximation, especially
     around contested/loose balls, blocked shots, deflections, etc.

  ASSUMPTION (flagging since I've only seen one example event): I don't
  know whether the event taxonomy has continuous coverage of the whole
  match (e.g. a "carry"/"open play" event filling every gap between
  passes/shots) or only tags discrete actions, leaving silent gaps in
  between. Because of that uncertainty, event-derived possession
  sequences are used for DAS (discrete danger moments -- exactly what
  events are good at), but the continuous EPV(t) momentum signal still
  uses the proximity proxy (source 2), which has no gaps by
  construction. Revisit this once you've confirmed the taxonomy.

Requires detect_formation.py in the same folder (reused for field-name
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


# ==========================
# Real events (preferred possession source -- see module docstring)
# ==========================

def load_events(match_dir):
    """
    Loads events.json (written by preprocessing.py) if present. Returns
    None if this match has no event data -- callers should fall back to
    the proximity proxy in that case, not crash.
    """
    path = os.path.join(match_dir, "events.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def possession_sequences_from_events(events, home_team_id):
    """
    Builds possession sequences directly from real event data, in the
    same {team, period, start_sec, end_sec, duration} shape that
    detect_possession_sequences() produces from the proximity proxy --
    so callers (evaluate_das, plot_formation_timeline's possession-bar
    mode) can use either source interchangeably.

    events.json (written by preprocessing.py's process_events) is now
    sourced from the PFF FC Event Data spec: one row per POSSESSION
    EVENT, not deduped by gameEventId, so a "challenge then pass" game
    event correctly shows up as two rows here. Two filters are applied
    before building sequences, both necessary for this source (neither
    applied automatically upstream):

      1. gameEventType == "OTB" only. Non-OTB game events (SUB, OFF,
         ON, OUT, END, kickoffs, FOUL, VID, G) also carry a teamId in
         this data and would otherwise get misread as that team having
         the ball -- e.g. a substitution would show up as a possession
         sequence. Only "a possession with a player on the ball" (the
         spec's own definition of OTB) should count.
      2. nonEvent == False. Possessions the provider flagged as
         disallowed after the fact (e.g. a shot later ruled offside)
         are excluded from the sequence timeline.

    GROUPING: rows are grouped by (period, sequence) rather than by
    re-derived time-adjacency. The spec defines "sequence" as "an
    uninterrupted possession by one team" -- i.e. the provider has
    already done exactly the segmentation this function used to
    reconstruct heuristically (walk sorted events, coalesce consecutive
    same-team rows). Trusting the provider's own field is more robust
    than re-deriving it -- confirmed against a real challenge-then-pass
    example where both rows shared sequence 2.0, teamId 51, but had
    different possessionEventIds (CH then PA).

    Within a group, start_sec is the min periodElapsedTimeEstimate and
    end_sec is the max (periodElapsedTimeEstimate + duration) across
    that group's rows -- necessary because startTime/duration are set
    at the GAME EVENT level and repeat identically across every
    possession event belonging to it, but a sequence can span multiple
    game events (e.g. a carry's game event, then a separate pass's game
    event, both part of the same uninterrupted possession), so taking
    the min/max across the whole group is what actually captures the
    full sequence span rather than just one game event's span.

    If a (period, sequence) group's rows disagree on teamId (shouldn't
    happen given the spec's definition of sequence, but not verified
    across a full match), the majority team is used and a warning is
    printed -- better than silently picking whichever row happened to
    be seen first.

    ASSUMPTION: each event's "teamId" reliably identifies who had the
    ball; "periodElapsedTimeEstimate" (added by preprocessing.py from
    the video-referenced startTime) is used as the time axis. Rows
    without a resolvable team, period, sequence, or time estimate are
    skipped.

    Unlike the proximity proxy, this does NOT fill silent gaps between
    sequences (see module docstring's taxonomy-coverage caveat) -- a
    gap with no covering sequence is simply not represented, rather
    than guessed at.
    """
    home_team_id = str(home_team_id)

    otb_events = [ev for ev in events
                  if ev.get("gameEventType") == "OTB" and not ev.get("nonEvent")]

    groups = {}
    for ev in otb_events:
        period = ev.get("period")
        seq = ev.get("sequence")
        team_id = ev.get("teamId")
        start = ev.get("periodElapsedTimeEstimate")
        if period is None or seq is None or team_id is None or start is None:
            continue
        groups.setdefault((period, seq), []).append(ev)

    sequences = []
    for (period, seq), evs in sorted(groups.items(),
                                      key=lambda kv: (kv[0][0], kv[0][1])):
        team_counts = {}
        for e in evs:
            t = str(e.get("teamId"))
            team_counts[t] = team_counts.get(t, 0) + 1
        if len(team_counts) > 1:
            print(f"    !! WARNING: sequence {seq} (period {period}) has "
                  f"rows from multiple teamIds {list(team_counts)} -- "
                  f"using the majority team. Check possessionEventType "
                  f"'IT' rows or a sequence-numbering edge case.")
        majority_team_id = max(team_counts, key=team_counts.get)

        starts = [e["periodElapsedTimeEstimate"] for e in evs]
        ends = [e["periodElapsedTimeEstimate"] + (e.get("duration") or 0.0) for e in evs]
        start_sec = min(starts)
        end_sec = max(ends)

        sequences.append({
            "team": "home" if majority_team_id == home_team_id else "away",
            "period": period,
            "start_sec": start_sec,
            "end_sec": end_sec,
        })

    for s in sequences:
        s["duration"] = s["end_sec"] - s["start_sec"]

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


def compute_possession_percentage(sequences, total_duration=None):
    """
    Sums sequence duration per team and returns exact possession time/%.

    total_duration: whole-match seconds (e.g. from compute_period_offsets
    in epv_das_analysis.py) to use as the denominator. If omitted, uses
    the sum of accounted-for sequence durations instead (so % excludes
    loose-ball/no-owner gaps rather than diluting against them).
    """
    home_sec = sum(s["duration"] for s in sequences if s["team"] == "home")
    away_sec = sum(s["duration"] for s in sequences if s["team"] == "away")
    accounted = home_sec + away_sec
    denom = total_duration if total_duration is not None else accounted

    return {
        "home_seconds": home_sec,
        "away_seconds": away_sec,
        "home_pct": round(100 * home_sec / denom, 2) if denom else 0.0,
        "away_pct": round(100 * away_sec / denom, 2) if denom else 0.0,
        "unaccounted_seconds": max(0.0, denom - accounted),
    }