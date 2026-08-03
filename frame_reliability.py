"""
frame_reliability.py

Builds a per-frame, per-team reliability WEIGHT in [0, 1] -- how much a
given tracking frame should count toward a team's average formation
shape -- instead of the old all-or-nothing "is this frame in a good
window or not" split. A frame right on top of a foul doesn't need to be
confidently classified as "this is a Transition, throw it out" -- it
just needs to count for less. Multiple independent disruption signals
combine MULTIPLICATIVELY (see combine step below), so any single strong
disruption can drag a frame's weight toward zero on its own, rather
than being diluted by averaging against several calm signals.

WHERE EACH SIGNAL COMES FROM:
  - foul / dead-ball (OUT) / set-piece restart / substitution: exact,
    from events.json (preprocessing.py's Event_Data parse -- see that
    module for field provenance). No estimation.
  - turnover (start of a new possession sequence): also exact, from
    events.json's own (period, sequence) grouping.
  - centroid velocity / surface-area rate of change: computed here,
    directly from tracking.jsonl.bz2, no events needed.

ARCHITECTURE NOTE (why this file exists separately, and only depends on
tracking_fields.py): detect_formation.py needs this module. If this
module imported possession.py for its sequence-grouping logic, and
possession.py imports detect_formation.py (it does, for the shared
field-alias primitives), that would be a circular import:
detect_formation -> frame_reliability -> possession -> detect_formation.
So turnover-sequence grouping is reimplemented locally here (see
_turnover_instants) rather than calling possession.py directly -- keep
the two in sync if that grouping logic ever changes.
"""

import bz2
import json
from collections import defaultdict

import numpy as np

from tracking_fields import extract_player_xy


# ============================================================
# Decay parameters -- all ramp durations in seconds. Weight is 0 at/
# during the disruptive event, and recovers linearly to 1 over the
# stated number of seconds. These are deliberately FIXED, documented
# defaults rather than derived from data (unlike the velocity/area
# scales below) -- reorganization time after a stoppage is plausibly
# close to universal (a sub takes about as long to jog into position
# regardless of league or team), so a reasoned starting guess is more
# defensible here than for a raw physical quantity like speed, which
# genuinely varies match to match. Treat all of these as a first pass:
# validate by eye against real video once this is running, the same
# way DAS_EPV_THRESHOLD in epv_das_analysis.py was flagged as a
# best-guess starting point, not a validated constant.
# ============================================================

# Fouls: ramps on BOTH sides of the foul's own timestamp. A foul is
# usually preceded by a contested challenge/tackle, so positions are
# already getting chaotic in the seconds leading up to the whistle, not
# just after it -- unlike OUT/setpiece below, whose boundaries are
# clean regime changes rather than a build-up.
FOUL_DECAY_SECONDS = 3.0

# Dead-ball spans (OUT -> next restart): weight is a HARD 0 across the
# entire span (no ramp on the way in -- the ball crossing the line is a
# clean, instant regime change, not a gradual one). Only the recovery
# AFTER the restart ramps back up:
SETPIECE_RECOVERY_SECONDS = 5.0

# Safety cap: if no restart event can be matched after an OUT event
# within a period (shouldn't normally happen, but a data gap
# shouldn't be allowed to zero out the rest of the match), the dead
# span is capped at this many seconds past the OUT event.
OUT_MAX_GAP_SECONDS = 120.0

# Substitutions: longer than a foul on purpose -- a sub means a player
# physically jogging into a new position (or the team reshuffling shape
# around them), plausibly slower to settle than a foul's momentary
# stoppage. Applied ONLY to the substituting team, not the opponent
# (see _sub_instants / the per-team application in compute_frame_weights).
SUB_DECAY_SECONDS = 10.0

# Turnovers (start of a new possession sequence): ramps on both sides,
# same reasoning as fouls (usually preceded by a contested moment).
# Applied to BOTH teams at once, unlike subs -- a turnover is
# simultaneously one team's defensive transition and the other's
# offensive transition.
TURNOVER_DECAY_SECONDS = 4.0

# ============================================================
# Continuous tracking-derived scores: score = exp(-metric / scale),
# always in (0, 1], no hard threshold. `scale` is NOT a fixed constant
# -- it's derived per match from that match's own observed distribution
# (the given percentile of centroid velocity / surface-area rate seen
# across the whole match), same "infer from the data in front of you,
# don't hardcode a guess" principle already used for
# stride/window-size inference in plot_formation_timeline.py. A team's
# "normal" pace of reorganization plausibly differs by league, weather,
# even individual match tempo -- a fixed absolute number risked being
# wrong in a way we couldn't detect, whereas a percentile of the same
# match's own data self-calibrates.
#
# WHY 75th PERCENTILE specifically: the goal is "clearly above this
# team's own normal movement for this match" -- the median (50th)
# would flag literally half of all frames as at least somewhat
# suspicious, which isn't what we want (most of the match should score
# near 1). The 75th percentile means only the most active quarter of
# frames get meaningfully downweighted by this signal, leaving the
# event-derived decays above (which are exact, not scored) to do the
# heavy lifting for genuinely disruptive moments.
# ============================================================
CENTROID_VELOCITY_PERCENTILE = 75
SURFACE_AREA_RATE_PERCENTILE = 75

# ============================================================
# Window-level floor: below this confidence (see
# detect_formation.py's process_match for how window confidence is
# computed), a window contributes NOTHING to voting and isn't written
# to formations.csv at all -- same spirit as the old
# MIN_FRAMES_PER_WINDOW cutoff, but on confidence instead of raw frame
# count. Chosen as a low bar deliberately: the weighting scheme above
# already does most of the work continuously; this floor only exists
# to drop windows that contributed essentially nothing (e.g. entirely
# swallowed by a single very long stoppage), not to re-introduce a
# strict pass/fail cutoff.
# ============================================================
MIN_WINDOW_CONFIDENCE = 0.15


# ============================================================
# Event -> disruption-instant/span extraction
# ============================================================

def _turnover_instants(events):
    """
    Local reimplementation of possession.py's (period, sequence)
    grouping (OTB + not-nonEvent), simplified to just timing -- see
    module docstring for why this isn't a call into possession.py.

    Returns sorted [(period, periodElapsedTimeEstimate), ...] marking
    the start of every possession sequence.
    """
    seen = {}
    for ev in events:
        if ev.get("gameEventType") != "OTB" or ev.get("nonEvent"):
            continue
        period = ev.get("period")
        seq = ev.get("sequence")
        t = ev.get("periodElapsedTimeEstimate")
        if period is None or seq is None or t is None:
            continue
        key = (period, seq)
        if key not in seen or t < seen[key]:
            seen[key] = t
    return sorted((p, t) for (p, s), t in seen.items())


def _foul_instants(events):
    """
    Returns sorted [(period, periodElapsedTimeEstimate), ...] for every
    foul -- both "primary" fouls (truthy 'fouls' field on the
    possession event where it happened) and "additional" separate rows
    (gameEventId == 'FOUL', possessionEventId == 'FO', per the spec's
    description of multi-infringement fouls).
    """
    out = []
    for ev in events:
        is_primary = bool(ev.get("fouls"))
        is_additional = (ev.get("gameEventId") == "FOUL" and ev.get("possessionEventId") == "FO")
        if not (is_primary or is_additional):
            continue
        period = ev.get("period")
        t = ev.get("periodElapsedTimeEstimate")
        if period is None or t is None:
            continue
        out.append((period, t))
    return sorted(out)


def _setpiece_instants(events):
    """Returns sorted [(period, periodElapsedTimeEstimate), ...] for
    every event carrying a non-null setpieceType (kickoffs, corners,
    free kicks, throw-ins, ...) -- the recovery ramp applies after each."""
    out = []
    for ev in events:
        if not ev.get("setpieceType"):
            continue
        period = ev.get("period")
        t = ev.get("periodElapsedTimeEstimate")
        if period is None or t is None:
            continue
        out.append((period, t))
    return sorted(out)


def _out_spans(events, max_gap_seconds=OUT_MAX_GAP_SECONDS):
    """
    Returns [(period, start_sec, end_sec), ...] dead-ball spans: from
    an "OUT" game event to the next restart (any event with a
    setpieceType, or a FIRSTKICKOFF/SECONDKICKOFF) in the same period.
    If no restart is found, the span is capped at max_gap_seconds past
    the OUT event as a safety bound.
    """
    by_period = defaultdict(list)
    for ev in events:
        period = ev.get("period")
        t = ev.get("periodElapsedTimeEstimate")
        if period is None or t is None:
            continue
        by_period[period].append(ev)

    spans = []
    for period, evs in by_period.items():
        evs = sorted(evs, key=lambda e: e["periodElapsedTimeEstimate"])
        for i, ev in enumerate(evs):
            if ev.get("gameEventType") != "OUT":
                continue
            start = ev["periodElapsedTimeEstimate"]
            end = start + max_gap_seconds
            for later in evs[i + 1:]:
                is_restart = (bool(later.get("setpieceType"))
                              or later.get("gameEventType") in ("FIRSTKICKOFF", "SECONDKICKOFF"))
                if is_restart:
                    end = later["periodElapsedTimeEstimate"]
                    break
            spans.append((period, start, end))
    return spans


def _sub_instants(events, home_team_id):
    """Returns [(period, sec, team_key), ...] for every SUB/ON/OFF
    event -- team_key resolved from the event's own teamId vs.
    metadata's home team id, same string-comparison pattern used
    throughout this pipeline."""
    home_team_id = str(home_team_id) if home_team_id is not None else None
    out = []
    for ev in events:
        if ev.get("gameEventType") not in ("SUB", "ON", "OFF"):
            continue
        period = ev.get("period")
        t = ev.get("periodElapsedTimeEstimate")
        team_id = ev.get("teamId")
        if period is None or t is None or team_id is None:
            continue
        team_key = "homePlayers" if str(team_id) == home_team_id else "awayPlayers"
        out.append((period, t, team_key))
    return out


# ============================================================
# Decay application onto (period, elapsed) frame arrays
# ============================================================

def _period_index(elapsed, period_arr, period):
    """Indices belonging to `period`, sorted by elapsed (defensive --
    doesn't assume the raw file is period-contiguous, even though it
    normally is)."""
    idx = np.where(period_arr == period)[0]
    if len(idx) == 0:
        return idx
    return idx[np.argsort(elapsed[idx])]


def _apply_point_decay(weight_arr, elapsed, period_arr, period, center_sec, ramp_seconds):
    """In-place: multiplies-in (via elementwise MIN against this
    weight_arr's current values, so multiple instances of the SAME
    disruption type combine as "worst nearby disruption of this type",
    not compounding multiplicatively against each other) a decay
    centered at center_sec: 0 at the center, ramping linearly to 1 by
    ramp_seconds away on either side."""
    if ramp_seconds <= 0:
        return
    idx = _period_index(elapsed, period_arr, period)
    if len(idx) == 0:
        return
    e = elapsed[idx]
    lo = np.searchsorted(e, center_sec - ramp_seconds, side="left")
    hi = np.searchsorted(e, center_sec + ramp_seconds, side="right")
    sel = idx[lo:hi]
    if len(sel) == 0:
        return
    dist = np.abs(elapsed[sel] - center_sec)
    decay = np.clip(dist / ramp_seconds, 0.0, 1.0)
    weight_arr[sel] = np.minimum(weight_arr[sel], decay)


def _apply_span_decay(weight_arr, elapsed, period_arr, period, start_sec, end_sec,
                       ramp_after_seconds, ramp_before_seconds=0.0):
    """In-place, same MIN-combining rule as _apply_point_decay: 0 within
    [start_sec, end_sec], ramping from 0 to 1 over ramp_before_seconds
    before start_sec and ramp_after_seconds after end_sec."""
    idx = _period_index(elapsed, period_arr, period)
    if len(idx) == 0:
        return
    e = elapsed[idx]
    lo = np.searchsorted(e, start_sec - ramp_before_seconds, side="left")
    hi = np.searchsorted(e, end_sec + ramp_after_seconds, side="right")
    sel = idx[lo:hi]
    if len(sel) == 0:
        return
    t = elapsed[sel]
    decay = np.ones(len(sel), dtype=np.float64)
    hard = (t >= start_sec) & (t <= end_sec)
    decay[hard] = 0.0
    before = t < start_sec
    if ramp_before_seconds > 0 and np.any(before):
        decay[before] = np.clip((start_sec - t[before]) / ramp_before_seconds, 0.0, 1.0)
    after = t > end_sec
    if ramp_after_seconds > 0 and np.any(after):
        decay[after] = np.clip((t[after] - end_sec) / ramp_after_seconds, 0.0, 1.0)
    weight_arr[sel] = np.minimum(weight_arr[sel], decay)


# ============================================================
# Tracking-derived: centroid velocity + surface-area rate of change
# ============================================================

def stream_centroid_and_spread(tracking_path, goalkeepers, pitch_length, pitch_width,
                                coords_are_centered=True):
    """
    One streaming pass over tracking.jsonl.bz2. For every frame (same
    period-is-not-None/elapsed-is-not-None skip condition as
    detect_formation.py's accumulate_positions -- keep these in sync,
    see compute_frame_weights docstring for why), computes each team's
    OUTFIELD centroid (mean x,y, goalkeeper excluded via the same
    resolved `goalkeepers` sets detect_formation.py uses) and spread
    (radius of gyration: RMS distance from that frame's own centroid).

    Returns: periods (int64 array), elapsed (float64 array), and a
    dict {"homePlayers": {"cx":arr,"cy":arr,"spread":arr},
          "awayPlayers": {...}} -- NaN entries where a team had zero
    outfield players tracked that frame.
    """
    x_shift = pitch_length / 2 if coords_are_centered else 0.0
    y_shift = pitch_width / 2 if coords_are_centered else 0.0
    team_keys = ("homePlayers", "awayPlayers")

    periods, elapsed = [], []
    cx = {t: [] for t in team_keys}
    cy = {t: [] for t in team_keys}
    spread = {t: [] for t in team_keys}

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

                periods.append(period)
                elapsed.append(et)

                for team in team_keys:
                    gk_ids = goalkeepers.get(team) or set()
                    pts = []
                    for p in frame.get(team, []):
                        parsed = extract_player_xy(p)
                        if parsed is None:
                            continue
                        pid, x, y = parsed
                        if str(pid) in gk_ids:
                            continue
                        pts.append((x + x_shift, y + y_shift))

                    if pts:
                        arr = np.array(pts, dtype=np.float64)
                        mx, my = arr.mean(axis=0)
                        sp = float(np.sqrt(np.mean(np.sum((arr - [mx, my]) ** 2, axis=1))))
                    else:
                        mx = my = sp = np.nan

                    cx[team].append(mx)
                    cy[team].append(my)
                    spread[team].append(sp)
    except EOFError:
        print(f"    !! WARNING: {tracking_path} appears truncated/corrupted "
              f"(bz2 stream ended early after {n_lines} frames) while computing "
              f"frame-reliability weights. Continuing with frames read so far.")

    periods = np.array(periods, dtype=np.int64)
    elapsed = np.array(elapsed, dtype=np.float64)
    result = {}
    for team in team_keys:
        result[team] = {
            "cx": np.array(cx[team], dtype=np.float64),
            "cy": np.array(cy[team], dtype=np.float64),
            "spread": np.array(spread[team], dtype=np.float64),
        }
    return periods, elapsed, result


def _velocity_and_rate_scores(periods, elapsed, cx, cy, spread, velocity_percentile, rate_percentile):
    """
    Vectorized. Computes frame-to-frame centroid velocity and
    |spread| rate of change (respecting period boundaries -- never
    diffs across a period gap), then scores each as exp(-metric/scale)
    where scale = the given percentile of that metric's own valid
    (finite) values for this match. NaN/undefined frames (first frame
    of a period, or a frame with zero tracked outfield players) score
    1.0 (neutral -- "no evidence of disruption", not penalized for
    missing data).
    """
    n = len(elapsed)
    velocity = np.full(n, np.nan)
    rate = np.full(n, np.nan)

    same_period = np.zeros(n, dtype=bool)
    same_period[1:] = periods[1:] == periods[:-1]

    dt = np.full(n, np.nan)
    dt[1:] = elapsed[1:] - elapsed[:-1]
    valid_dt = same_period & np.isfinite(dt) & (dt > 0)

    dx = np.full(n, np.nan)
    dy = np.full(n, np.nan)
    dx[1:] = cx[1:] - cx[:-1]
    dy[1:] = cy[1:] - cy[:-1]
    ok = valid_dt & np.isfinite(dx) & np.isfinite(dy)
    velocity[ok] = np.sqrt(dx[ok] ** 2 + dy[ok] ** 2) / dt[ok]

    dspread = np.full(n, np.nan)
    dspread[1:] = spread[1:] - spread[:-1]
    ok2 = valid_dt & np.isfinite(dspread)
    rate[ok2] = np.abs(dspread[ok2]) / dt[ok2]

    def _score(metric, percentile):
        finite = metric[np.isfinite(metric)]
        score = np.ones(n, dtype=np.float64)
        if len(finite) == 0:
            return score
        scale = np.percentile(finite, percentile)
        if not np.isfinite(scale) or scale < 1e-6:
            return score
        valid = np.isfinite(metric)
        score[valid] = np.exp(-metric[valid] / scale)
        return score

    return _score(velocity, velocity_percentile), _score(rate, rate_percentile)


# ============================================================
# Main entrypoint
# ============================================================

def compute_frame_weights(tracking_path, metadata, events, goalkeepers,
                           coords_are_centered=True):
    """
    Builds the full per-frame, per-team reliability weight, combining
    all decay signals MULTIPLICATIVELY:

        weight[team][frame] = foul_decay * out_decay * setpiece_decay
                               * turnover_decay * sub_decay[team]
                               * velocity_score[team] * area_score[team]

    foul/out/setpiece/turnover apply identically to both teams (a foul
    or a turnover disrupts everyone on the pitch, not just one side);
    sub/velocity/area are team-specific.

    IMPORTANT ALIGNMENT NOTE: this runs its OWN streaming pass over
    tracking.jsonl.bz2 (via stream_centroid_and_spread), separate from
    detect_formation.py's accumulate_positions pass over the SAME file.
    Rather than relying on the two passes producing identical frame
    ORDER (fragile -- any divergence silently misaligns everything),
    the result is returned as a dict keyed by the (period, elapsed)
    VALUES themselves, which accumulate_positions looks up per frame as
    it does its own independent pass. This is safe because both passes
    parse the same raw JSON floats with no intervening arithmetic, so
    equality holds exactly.

    Returns: (weight_lookup, info)
        weight_lookup: {period: {elapsed_value: {"homePlayers": w, "awayPlayers": w}}}
        info: dict of counts (n_fouls, n_out_spans, n_setpieces, n_subs,
              n_turnovers) for logging.
    """
    home_team_id = metadata.get("homeTeam", {}).get("id")

    periods, elapsed, centroid_data = stream_centroid_and_spread(
        tracking_path, goalkeepers,
        metadata["pitch"]["length"], metadata["pitch"]["width"],
        coords_are_centered=coords_are_centered,
    )

    team_keys = ("homePlayers", "awayPlayers")
    weights = {t: np.ones(len(elapsed), dtype=np.float64) for t in team_keys}

    fouls = _foul_instants(events) if events else []
    out_spans = _out_spans(events) if events else []
    setpieces = _setpiece_instants(events) if events else []
    turnovers = _turnover_instants(events) if events else []
    subs = _sub_instants(events, home_team_id) if events else []

    # -- shared (both-team) decays, applied identically to both arrays --
    common = np.ones(len(elapsed), dtype=np.float64)
    for period, t in fouls:
        _apply_point_decay(common, elapsed, periods, period, t, FOUL_DECAY_SECONDS)
    for period, start, end in out_spans:
        _apply_span_decay(common, elapsed, periods, period, start, end,
                           ramp_after_seconds=0.0, ramp_before_seconds=0.0)
    for period, t in turnovers:
        _apply_point_decay(common, elapsed, periods, period, t, TURNOVER_DECAY_SECONDS)

    # Setpiece recovery is a one-sided ramp-up AFTER the restart, not a
    # symmetric point decay (there's nothing to penalize BEFORE a
    # restart that the OUT span above doesn't already cover) -- applied
    # as a zero-length "span" so only ramp_after takes effect.
    for period, t in setpieces:
        _apply_span_decay(common, elapsed, periods, period, t, t,
                           ramp_after_seconds=SETPIECE_RECOVERY_SECONDS, ramp_before_seconds=0.0)

    for t in team_keys:
        weights[t] *= common

    # -- team-specific: substitutions --
    for period, t, team_key in subs:
        _apply_point_decay(weights[team_key], elapsed, periods, period, t, SUB_DECAY_SECONDS)

    # -- team-specific: centroid velocity / surface-area rate scores --
    for team in team_keys:
        vel_score, area_score = _velocity_and_rate_scores(
            periods, elapsed,
            centroid_data[team]["cx"], centroid_data[team]["cy"], centroid_data[team]["spread"],
            CENTROID_VELOCITY_PERCENTILE, SURFACE_AREA_RATE_PERCENTILE,
        )
        weights[team] *= vel_score
        weights[team] *= area_score

    weight_lookup = defaultdict(dict)
    for i in range(len(elapsed)):
        weight_lookup[int(periods[i])][float(elapsed[i])] = {
            "homePlayers": float(weights["homePlayers"][i]),
            "awayPlayers": float(weights["awayPlayers"][i]),
        }

    info = {
        "n_fouls": len(fouls), "n_out_spans": len(out_spans),
        "n_setpieces": len(setpieces), "n_turnovers": len(turnovers),
        "n_subs": len(subs),
    }
    return dict(weight_lookup), info
