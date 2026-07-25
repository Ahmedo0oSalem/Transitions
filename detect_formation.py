"""
detect_formations.py

Detects the formation (e.g. 4-4-2, 4-3-3, ...) each team is playing in
fixed time windows (default: 3 minutes) from tracking data, using
mplsoccer's built-in formation templates as reference shapes.

METHOD (standard approach, e.g. Bialkowski et al. 2014, and the EFPI paper
which does the same thing with mplsoccer's templates):
  1. For each window, compute each outfield player's WEIGHTED average
     (x, y) position over all frames in that window. If events.json is
     available (see frame_reliability.py), each frame gets a
     reliability weight in [0,1] first -- a frame near a foul, dead
     ball, set-piece restart, substitution, or possession turnover, or
     during unusually fast team repositioning, counts for LESS than a
     calm, settled frame, rather than counting exactly the same (the
     old plain-average behavior, still used as-is when there's no
     events.json for a match). This is what actually fixes the
     "overlapping/contradictory formations in the same few minutes"
     problem -- a contaminated frame no longer pulls the average as
     hard as an uncontaminated one.
  2. Exclude the goalkeeper. The GK for each side is read directly from
     metadata.json (populated by preprocessing.py from the match roster:
     the starting player with positionGroupType == "GK"). If a match has
     no roster-derived GK (missing roster file), we fall back to the old
     heuristic: identify via total distance covered per frame -- GKs move
     far less than outfield players.
  3. For every candidate formation template (mplsoccer has 65 "full" 11-man
     templates: 442, 433, 4231, 352, ...), solve the assignment problem
     (Hungarian algorithm) between the 10 average outfield positions and
     the 10 template positions, minimizing total squared distance.
  4. The template with the lowest total cost is the detected formation
     for that team in that window.
  5. Each window also gets a CONFIDENCE score: (sum of that window's
     frame weights) x (fit_quality, derived from the Hungarian cost --
     how well the average shape actually matched a template). Windows
     below frame_reliability.MIN_WINDOW_CONFIDENCE are dropped entirely
     -- the confidence-based replacement for the old hard
     MIN_FRAMES_PER_WINDOW-only cutoff. Downstream tools (e.g.
     plot_formation_timeline.py) use this confidence to WEIGHT-VOTE
     among overlapping windows at any given instant, rather than just
     picking whichever window started most recently.
  6. A team can be attacking left->right or right->left, and this flips
     every period (teams swap ends at half-time, and again each extra-time
     half). Rather than trying both orientations of every template and
     keeping whichever is cheaper, we compute the correct orientation
     directly from metadata's "homeTeamStartLeft" + the period number, and
     only match against that one orientation. This is faster and removes
     the (small) risk of a "mirrored" formation winning by fluke.
     !!! VERIFIED for at least one real match this session -- see
     get_orientation()'s docstring for how to re-verify on new data. !!!

REQUIRES:
    pip install mplsoccer scipy numpy pandas --break-system-packages

INPUT:
    Expects the output of your preprocessing script, i.e. for a given
    match_id:
        Processed_Tracking/<match_id>/metadata.json
        Processed_Tracking/<match_id>/tracking.jsonl.bz2
        Processed_Tracking/<match_id>/roster.json   (optional but recommended)

    metadata.json must contain (as produced by your script):
        "fps", "pitch": {"length": ..., "width": ...}, "periods": {...},
        "homeTeamStartLeft", "goalkeepers": {"home": {...}, "away": {...}}

    Each line of tracking.jsonl.bz2 must contain:
        "period", "periodElapsedTime", "homePlayers", "awayPlayers", "balls"

    homePlayers / awayPlayers are expected to be a list of dicts, one per
    player, each roughly like:
        {"playerId": <id>, "number": <shirt number>, "x": <float>, "y": <float>}

    !!! ADJUST THE `PLAYER_KEY_ALIASES` / COORD CONFIG BELOW to match your
    provider's exact field names if they differ -- print one frame of your
    tracking.jsonl.bz2 to check first. !!!

OUTPUT:
    Processed_Tracking/<match_id>/formations.csv with one row per
    (team, window), containing the detected formation and match cost.
"""

import os
import bz2
import json
import math
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from mplsoccer import Pitch

from tracking_fields import (
    PLAYER_ID_KEYS, PLAYER_X_KEYS, PLAYER_Y_KEYS, PLAYER_NUMBER_KEYS,
    REJECT_CONFIDENCE_VALUES, REJECT_VISIBILITY_VALUES,
    _get_first, extract_player_xy,
)
import frame_reliability


# ==========================
# Configuration
# ==========================

PROCESSED_DIR = "Processed_Tracking"

# Size of the detection window, in seconds.
WINDOW_SECONDS = 180  # 3 minutes

# How far the window slides forward between readings, in seconds.
# STRIDE_SECONDS == WINDOW_SECONDS -> plain non-overlapping windows (old behavior).
# STRIDE_SECONDS <  WINDOW_SECONDS -> sliding/rolling window: e.g. a 300s window
#   with a 60s stride gives you a new (overlapping) reading every minute, each
#   still averaged over a full 5-minute span -- smoother AND more frequent.
STRIDE_SECONDS = 180

# Only keep a window if at least this many frames of data were found
# for a team (protects against tiny/empty windows at the very start/end
# of a period). Tune based on your fps.
MIN_FRAMES_PER_WINDOW = 30

# Only attempt to detect a formation if we have at least this many
# outfield players with valid data in the window (out of 10). Missing
# players (subs not on, red cards, tracking dropout) below this are
# skipped rather than guessed.
MIN_OUTFIELD_PLAYERS = 9

# Field name aliases (PLAYER_ID_KEYS, PLAYER_X_KEYS, PLAYER_Y_KEYS,
# PLAYER_NUMBER_KEYS) and quality-flag filtering (REJECT_CONFIDENCE_VALUES,
# REJECT_VISIBILITY_VALUES) now live in tracking_fields.py, imported above --
# edit them there; possession.py and frame_reliability.py read the same
# values via that shared module, so this is the one place to change them.

# If your tracking coordinates are already 0..pitch_length / 0..pitch_width
# set this to False. If they are centered on (0,0), e.g. x in
# [-length/2, length/2], leave this True and we will shift them.
COORDS_ARE_CENTERED = True

# Minimum number of tracked points a player needs before they're considered
# a candidate for goalkeeper identification. Without this, a substitute who
# only played a few minutes will have artificially low TOTAL distance
# covered (not because they're a keeper, just because they were barely on
# the pitch) and get misidentified as the GK. Tune down if your matches are
# short / fps is low.
GK_MIN_FRAMES = 3000


# ==========================
# Formation templates (via mplsoccer)
# ==========================

def build_templates(pitch_length, pitch_width):
    """
    Returns:
        templates: dict[formation_name] -> {
            'names': list[str] (10 outfield position names, GK excluded),
            'normal': (10, 2) ndarray of (x, y),
            'flipped': (10, 2) ndarray of (x_flip, y_flip),
        }
    Only the 65 "full" 11-a-side formations are used (pitch.formations),
    not the extra depleted-squad variants mplsoccer also ships with.
    """
    pitch = Pitch(pitch_type="custom", pitch_length=pitch_length, pitch_width=pitch_width)
    df = pitch.formations_dataframe
    templates = {}
    for formation in pitch.formations:
        sub = df[(df["formation"] == formation) & (df["name"] != "GK")]
        if len(sub) != 10:
            # Skip anything that isn't a standard 11-a-side (1 GK + 10 outfield)
            continue
        templates[formation] = {
            "names": sub["name"].tolist(),
            "normal": sub[["x", "y"]].to_numpy(dtype=float),
            "flipped": sub[["x_flip", "y_flip"]].to_numpy(dtype=float),
        }
    return templates


def match_formation(player_xy, templates, orientation):
    """
    Given (N, 2) average outfield player positions (N==10 ideally, but
    works down to MIN_OUTFIELD_PLAYERS by matching a subset of template
    positions), find the best-fitting formation, using only the given
    orientation ("normal" or "flipped") of each template -- the caller
    (process_match, via get_orientation) has already worked out which
    orientation matches this team's actual attacking direction for this
    period, so we don't need to try both and pick the cheaper one.

    Returns: (best_formation_name, best_cost, best_assignment_names)
    """
    best_formation, best_cost, best_names = None, np.inf, None

    for formation, tmpl in templates.items():
        template_xy = tmpl[orientation]
        # linear_sum_assignment handles rectangular cost matrices fine
        # (it matches min(n_players, n_template_slots) pairs), so this
        # works whether a window has fewer players than the template
        # (missing/subbed player) or more (tracking noise/ghost
        # detections -- e.g. a staff member briefly picked up).
        cost_matrix = cdist(player_xy, template_xy)
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cost = cost_matrix[row_ind, col_ind].sum()
        # Normalize by number of matched pairs so formations aren't
        # unfairly penalized/favored by player count.
        norm_cost = cost / len(row_ind)

        if norm_cost < best_cost:
            best_cost = norm_cost
            best_formation = formation
            best_names = [tmpl["names"][i] for i in col_ind]

    return best_formation, best_cost, best_names


def get_orientation(team_key, period, home_team_start_left):
    """
    Returns "normal" or "flipped" -- which orientation of the formation
    templates matches this team's actual attacking direction in this
    period, given by metadata's "homeTeamStartLeft".

    ASSUMPTION (verify before trusting a full run!): "normal" template
    coordinates represent a team attacking left->right (increasing x,
    toward the corner-origin pitch's right edge), matching mplsoccer's
    convention. "homeTeamStartLeft" == True means the home team starts
    period 1 defending the left side / attacking left->right.

    To verify: pick one match, one frame near kickoff, average the home
    team's raw x (after the COORDS_ARE_CENTERED shift) -- it should sit
    in the defensive (low-x) half if homeTeamStartLeft is True. If it's
    backwards for your data, just swap "normal" <-> "flipped" below.

    Teams swap ends every period (half-time, and each extra-time half),
    so periods 1 and 3 share home's period-1 direction; periods 2 and 4
    are the reverse.
    """
    home_attacks_left_to_right = (
        home_team_start_left if period % 2 == 1 else not home_team_start_left
    )
    if team_key == "homePlayers":
        return "normal" if home_attacks_left_to_right else "flipped"
    else:
        return "flipped" if home_attacks_left_to_right else "normal"


# ==========================
# Helpers to read player dicts robustly -- see tracking_fields.py
# (_get_first, extract_player_xy imported above)
# ==========================


# ==========================
# Goalkeeper identification
# ==========================

def goalkeepers_from_metadata(metadata):
    """
    Preferred path: read goalkeeper identity straight from the roster
    (via preprocessing.py, which wrote metadata["goalkeepers"] = {"home":
    {"playerId", "shirtNumber"}, "away": {...}}). No estimation needed.

    Because we don't know in advance whether your tracking provider's
    homePlayers/awayPlayers entries key each player by an internal ID or
    by shirt number, we return a SET of acceptable identifiers per team
    (playerId AND shirtNumber, as strings) -- accumulate_positions checks
    a tracked pid against this set, so it matches whichever key
    PLAYER_ID_KEYS actually picked up from the raw frame.

    Returns: {"homePlayers": set[str] | None, "awayPlayers": set[str] | None}
    (None for a side means: no roster GK available for that side.)
    """
    gk_meta = metadata.get("goalkeepers", {}) or {}
    result = {}
    for team_key, side in (("homePlayers", "home"), ("awayPlayers", "away")):
        entry = gk_meta.get(side)
        if not entry:
            result[team_key] = None
            continue
        ids = {str(v) for v in entry.values() if v is not None}
        result[team_key] = ids if ids else None
    return result


def identify_goalkeepers(tracking_path, team_keys=("homePlayers", "awayPlayers")):
    """
    FALLBACK ONLY -- used when the roster wasn't available at
    preprocessing time (see goalkeepers_from_metadata). Walks the whole
    tracking file once and finds, for each team, the player with the
    smallest AVERAGE per-frame displacement (a speed proxy), among
    players tracked for at least GK_MIN_FRAMES points.

    We normalize by frame count rather than using raw total distance
    covered, because a substitute who only played a few minutes would
    otherwise have an artificially low total distance (simply from being
    on the pitch briefly) and get misidentified as the goalkeeper. GKs
    reliably have the lowest average per-frame movement of anyone who
    played meaningful minutes, regardless of how long they were on.

    Returns: {team_key: {goalkeeper_player_id} } -- a single-element set,
    for a uniform interface with goalkeepers_from_metadata.
    """
    last_xy = {team: {} for team in team_keys}
    total_dist = {team: defaultdict(float) for team in team_keys}
    frame_count = {team: defaultdict(int) for team in team_keys}

    n_lines = 0
    try:
        with bz2.open(tracking_path, "rt") as f:
            for line in f:
                frame = json.loads(line)
                n_lines += 1
                for team in team_keys:
                    for p in frame.get(team, []):
                        parsed = extract_player_xy(p)
                        if parsed is None:
                            continue
                        pid, x, y = parsed
                        frame_count[team][pid] += 1
                        if pid in last_xy[team]:
                            lx, ly = last_xy[team][pid]
                            total_dist[team][pid] += ((x - lx) ** 2 + (y - ly) ** 2) ** 0.5
                        last_xy[team][pid] = (x, y)
    except EOFError:
        print(f"    !! WARNING: {tracking_path} appears truncated/corrupted "
              f"(bz2 stream ended early after {n_lines} frames). "
              f"Continuing with the frames successfully read -- re-check/re-generate "
              f"this file if results look incomplete.")

    goalkeepers = {}
    for team in team_keys:
        candidates = {
            pid: total_dist[team][pid] / frame_count[team][pid]
            for pid in total_dist[team]
            if frame_count[team][pid] >= GK_MIN_FRAMES
        }
        if not candidates:
            # Fallback: nobody met the minimum-frames bar (e.g. very short
            # clip). Use whoever has the most frames as a last resort.
            if frame_count[team]:
                candidates = {
                    pid: total_dist[team][pid] / frame_count[team][pid]
                    for pid in total_dist[team]
                }
            else:
                goalkeepers[team] = None
                continue
        goalkeepers[team] = {str(min(candidates, key=candidates.get))}
    return goalkeepers


def resolve_goalkeepers(tracking_path, metadata):
    """
    Tries the roster first (fast, exact); falls back to the distance
    heuristic per-team for any side the roster didn't cover.

    Returns: {"homePlayers": set[str] | None, "awayPlayers": set[str] | None}
    """
    from_roster = goalkeepers_from_metadata(metadata)
    missing = [team for team, ids in from_roster.items() if ids is None]

    if not missing:
        return from_roster

    print(f"    goalkeeper(s) missing from roster for: {missing} "
          f"-- falling back to distance-based estimation for those.")
    from_distance = identify_goalkeepers(tracking_path, team_keys=tuple(missing))

    resolved = dict(from_roster)
    for team in missing:
        resolved[team] = from_distance.get(team)
    return resolved


# ==========================
# Windowing + averaging
# ==========================

def get_window_indices(elapsed_seconds):
    """
    Returns every window index k (where window k covers
    [k*STRIDE_SECONDS, k*STRIDE_SECONDS + WINDOW_SECONDS)) that this
    timestamp falls inside. With STRIDE_SECONDS == WINDOW_SECONDS this
    returns exactly one index (plain non-overlapping windows). With
    STRIDE_SECONDS < WINDOW_SECONDS it returns several indices, since a
    sliding window overlaps its neighbors.
    """
    k_max = int(elapsed_seconds // STRIDE_SECONDS)
    k_min = max(0, math.ceil((elapsed_seconds - WINDOW_SECONDS) / STRIDE_SECONDS))
    return range(k_min, k_max + 1)


def accumulate_positions(tracking_path, goalkeepers, pitch_length, pitch_width,
                          weight_lookup=None, team_keys=("homePlayers", "awayPlayers")):
    """
    Streams through the tracking file, bucketing each outfield player's
    (x, y, weight) into (team, period, window_index) buckets. With
    sliding windows (STRIDE_SECONDS < WINDOW_SECONDS) a single frame
    contributes to every overlapping window it falls inside.

    weight_lookup (from frame_reliability.compute_frame_weights), if
    given, supplies a per-frame, per-team reliability weight in [0,1] --
    a frame near a foul/stoppage/high-velocity moment counts for less
    toward the window's average position, instead of counting exactly
    as much as a calm, settled frame (the old behavior). If None, every
    frame gets weight 1.0 (equivalent to the old plain averaging).

    Returns:
        buckets[(team, period, window_index)][player_id] -> list of (x, y, weight)
        window_weight_sum[(team, period, window_index)] -> sum of frame
            weights contributed to that window (used for window
            confidence in process_match) -- tracked once per window per
            FRAME, not per player, since it's a property of the frame/
            window, not of any individual player within it.
    """
    buckets = defaultdict(lambda: defaultdict(list))
    window_weight_sum = defaultdict(float)
    x_shift = pitch_length / 2 if COORDS_ARE_CENTERED else 0.0
    y_shift = pitch_width / 2 if COORDS_ARE_CENTERED else 0.0

    n_lines = 0
    try:
        with bz2.open(tracking_path, "rt") as f:
            for line in f:
                frame = json.loads(line)
                n_lines += 1
                period = frame.get("period")
                elapsed = frame.get("periodElapsedTime")
                if period is None or elapsed is None:
                    continue

                window_indices = list(get_window_indices(elapsed))
                frame_weights = None
                if weight_lookup is not None:
                    frame_weights = weight_lookup.get(period, {}).get(elapsed)

                for team in team_keys:
                    gk_ids = goalkeepers.get(team) or set()
                    w = 1.0
                    if frame_weights is not None:
                        w = frame_weights.get(team, 1.0)

                    for k in window_indices:
                        window_weight_sum[(team, period, k)] += w

                    for p in frame.get(team, []):
                        parsed = extract_player_xy(p)
                        if parsed is None:
                            continue
                        pid, x, y = parsed
                        if str(pid) in gk_ids:
                            continue  # exclude goalkeeper
                        xyw = (x + x_shift, y + y_shift, w)
                        for k in window_indices:
                            buckets[(team, period, k)][pid].append(xyw)
    except EOFError:
        print(f"    !! WARNING: {tracking_path} appears truncated/corrupted "
              f"(bz2 stream ended early after {n_lines} frames). "
              f"Continuing with the frames successfully read.")

    return buckets, window_weight_sum


def _load_events(match_dir):
    """
    Loads events.json if present (written by preprocessing.py). This is
    a plain file load, not a call into possession.py's load_events --
    possession.py imports this module for shared primitives, so this
    module importing possession.py back would create a circular import.
    """
    path = os.path.join(match_dir, "events.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ==========================
# Main per-match pipeline
# ==========================

def process_match(match_id, processed_dir=PROCESSED_DIR):
    match_dir = os.path.join(processed_dir, str(match_id))
    metadata_path = os.path.join(match_dir, "metadata.json")
    tracking_path = os.path.join(match_dir, "tracking.jsonl.bz2")

    if not os.path.exists(metadata_path) or not os.path.exists(tracking_path):
        print(f"[{match_id}] processed metadata/tracking missing, skipping.")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    home_team_start_left = metadata["homeTeamStartLeft"]

    print(f"[{match_id}] building formation templates ({pitch_length}x{pitch_width})...")
    templates = build_templates(pitch_length, pitch_width)

    print(f"[{match_id}] resolving goalkeepers...")
    goalkeepers = resolve_goalkeepers(tracking_path, metadata)
    print(f"[{match_id}] goalkeepers: {goalkeepers}")

    events = _load_events(match_dir)
    weight_lookup = None
    if events:
        print(f"[{match_id}] computing frame-reliability weights from "
              f"{len(events)} events + tracking (foul/dead-ball/set-piece/sub/"
              f"turnover decay + centroid-velocity/surface-area scoring)...")
        weight_lookup, info = frame_reliability.compute_frame_weights(
            tracking_path, metadata, events, goalkeepers,
            coords_are_centered=COORDS_ARE_CENTERED,
        )
        print(f"[{match_id}] weighting inputs: {info}")
    else:
        print(f"[{match_id}] no events.json -- frame weighting disabled "
              f"(every frame weight 1.0, same as the old plain average).")

    print(f"[{match_id}] accumulating positions into {WINDOW_SECONDS}s windows "
          f"(stride {STRIDE_SECONDS}s)...")
    buckets, window_weight_sum = accumulate_positions(
        tracking_path, goalkeepers, pitch_length, pitch_width, weight_lookup=weight_lookup)

    n_dropped_low_confidence = 0
    rows = []
    for (team, period, window_index), players in sorted(buckets.items()):
        n_frames = sum(len(v) for v in players.values())
        if n_frames < MIN_FRAMES_PER_WINDOW:
            continue

        # Weighted average position per player over the window -- a
        # frame with low reliability weight (foul/stoppage/high
        # velocity nearby) counts less toward this average than a
        # calm, settled frame. Falls back to a plain mean for a player
        # whose every frame in this window was fully suppressed
        # (weight sum ~0), rather than dividing by zero.
        avg_xy = []
        for pid, coords in players.items():
            arr = np.array(coords, dtype=float)  # columns: x, y, weight
            w = arr[:, 2]
            wsum = w.sum()
            if wsum > 1e-9:
                avg = (arr[:, :2] * w[:, None]).sum(axis=0) / wsum
            else:
                avg = arr[:, :2].mean(axis=0)
            avg_xy.append(avg)
        avg_xy = np.array(avg_xy)

        if avg_xy.shape[0] < MIN_OUTFIELD_PLAYERS:
            continue

        orientation = get_orientation(team, period, home_team_start_left)
        formation, cost, _assigned_names = match_formation(avg_xy, templates, orientation)

        # Window confidence: how much real, reliable signal fed this
        # window's average (sum of frame weights) times how well that
        # signal actually fit a template (fit_quality, derived from the
        # existing Hungarian-match cost -- lower cost = better fit).
        # Windows below MIN_WINDOW_CONFIDENCE are dropped entirely --
        # the confidence-based replacement for the old hard
        # MIN_FRAMES_PER_WINDOW-only cutoff.
        fit_quality = 1.0 / (1.0 + cost)
        window_confidence = window_weight_sum.get((team, period, window_index), 0.0) * fit_quality
        if window_confidence < frame_reliability.MIN_WINDOW_CONFIDENCE:
            n_dropped_low_confidence += 1
            continue

        window_start = window_index * STRIDE_SECONDS
        window_end = window_start + WINDOW_SECONDS

        rows.append({
            "matchId": match_id,
            "team": "home" if team == "homePlayers" else "away",
            "period": period,
            "windowIndex": window_index,
            "windowStartSec": window_start,
            "windowEndSec": window_end,
            "nOutfieldPlayers": avg_xy.shape[0],
            "nFrames": n_frames,
            "formation": formation,
            "orientation": orientation,
            "avgCostPerPlayer": round(float(cost), 3),
            "confidence": round(float(window_confidence), 4),
        })

    if n_dropped_low_confidence:
        print(f"[{match_id}] dropped {n_dropped_low_confidence} window(s) below "
              f"MIN_WINDOW_CONFIDENCE ({frame_reliability.MIN_WINDOW_CONFIDENCE}).")

    out_df = pd.DataFrame(rows).sort_values(["team", "period", "windowIndex"])
    out_path = os.path.join(match_dir, "formations.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[{match_id}] wrote {len(out_df)} rows -> {out_path}")
    return out_df


# ==========================
# CLI
# ==========================

def main():
    global WINDOW_SECONDS, STRIDE_SECONDS

    parser = argparse.ArgumentParser(description="Detect formations per time window from tracking data.")
    parser.add_argument("match_ids", nargs="*", help="Match IDs to process (folder names under Processed_Tracking). "
                                                       "If omitted, processes every match found.")
    parser.add_argument("--processed-dir", default=PROCESSED_DIR)
    parser.add_argument("--window-seconds", type=int, default=WINDOW_SECONDS,
                         help="Length of each detection window, in seconds.")
    parser.add_argument("--stride-seconds", type=int, default=None,
                         help="How far the window slides forward between readings, in seconds. "
                              "Defaults to --window-seconds (plain non-overlapping windows). "
                              "Set smaller than --window-seconds for an overlapping/sliding window "
                              "(e.g. --window-seconds 300 --stride-seconds 60 = a 5-minute window, "
                              "re-evaluated every minute).")
    args = parser.parse_args()

    WINDOW_SECONDS = args.window_seconds
    STRIDE_SECONDS = args.stride_seconds if args.stride_seconds is not None else WINDOW_SECONDS

    if args.match_ids:
        match_ids = args.match_ids
    else:
        match_ids = [
            d for d in os.listdir(args.processed_dir)
            if os.path.isdir(os.path.join(args.processed_dir, d))
        ]

    print(f"Processing {len(match_ids)} match(es)...")
    for match_id in match_ids:
        process_match(match_id, processed_dir=args.processed_dir)

    print("Done!")


if __name__ == "__main__":
    main()