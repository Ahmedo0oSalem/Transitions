"""Per-frame reliability weights for formation detection.

Builds a per-frame, per-team weight in [0, 1] indicating how much each
tracking frame should count toward its team's average formation shape.
The event-derived and tracking-derived signals are combined
multiplicatively; within the event family, overlapping disruption
windows are combined with min() (the strongest suppression wins):

  1. **Event-derived** (exact, from events.json):
     - Fouls, dead-ball spans, set-piece recovery,
       substitutions (team-specific), turnovers (both teams).
     - Each disruption imposes an upper bound on the weight; a frame's
       event weight is the *minimum* of the bounds covering it.

  2. **Tracking-derived** (continuous, no threshold):
     - Centroid velocity  →  exp(-v / scale)
     - Spread rate of change  →  exp(-r / scale)
     where *scale* = 90th percentile of this match's distribution.
"""

from __future__ import annotations

import bz2
import json
import math
from collections import defaultdict

import numpy as np

from ...core.logger import get_logger
from .tracking_fields import extract_player_xy, _get_first

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Decay / recovery constants — documented starting guesses
# ---------------------------------------------------------------------------

FOUL_DECAY_SECONDS = 15.0
"""How long after a foul before the affected team's formation weight
recovers fully.  Fouls often cause clustered player positions (wall,
protesting, crowding the ref) that don't represent the team's shape."""

SETPIECE_RECOVERY_SECONDS = 10.0
"""Recovery period after a dead-ball restart or set piece.  Players
are still moving into position from a standstill, so positions during
this window are less representative."""

SUB_DECAY_SECONDS = 30.0
"""Team shape is disrupted after a substitution while the new player
gets into position and teammates adjust.  Longer than a foul because
the lineup change itself persists."""

TURNOVER_DECAY_SECONDS = 5.0
"""Brief uncertainty around the moment possession changes — both teams
are in transition, neither is set in its formation shape yet."""

OTB_SPAN_GAP_SECONDS = 15.0
"""OTB (out of bounds) events closer than this are considered part of the
same dead-ball span; events further apart start a new span.  Without this
gap check, every OTB in a period merges into one span covering the whole
period."""

MIN_WINDOW_CONFIDENCE = 0.3
"""Windows whose mean frame weight (average reliability weight across the
window's frames, in [0, 1]) falls below this threshold are dropped entirely
and never written to formations.csv."""
"""Windows whose total confidence falls below this threshold are
dropped entirely and never written to formations.csv."""

# Tracking-derived percentile for the exp(-x/scale) sigmoid.
_PERCENTILE = 90.0

_TEAM_KEYS = ("homePlayers", "awayPlayers")

# ---------------------------------------------------------------------------
# Event helpers
# ---------------------------------------------------------------------------


def _classify_events(events):
    """Parse events.json into structured disruption records.

    Returns a list of ``(period, start_sec, end_sec, affected_team,
    base_weight)`` tuples, where *affected_team* is ``"homePlayers"``,
    ``"awayPlayers"``, or ``"both"``, and *base_weight* is the weight
    that the disrupted team(s) receive during the disruption period
    (typically 0.0 for a full drop).
    """
    records: list[tuple[int, float, float, str, float]] = []

    if events is None:
        return records

    # --- Fouls ---
    for ev in events:
        if ev.get("nonEvent"):
            continue
        if ev.get("gameEventType") != "FOU":
            continue
        period = ev.get("period")
        sec = ev.get("periodElapsedTimeEstimate")
        if period is None or sec is None:
            continue
        team = "homePlayers" if ev.get("homeTeam") else "awayPlayers"
        end = sec + FOUL_DECAY_SECONDS
        records.append((period, sec, end, team, 0.0))

    # --- Substitutions ---
    for ev in events:
        if ev.get("nonEvent"):
            continue
        if ev.get("gameEventType") != "SUB":
            continue
        period = ev.get("period")
        sec = ev.get("periodElapsedTimeEstimate")
        if period is None or sec is None:
            continue
        team = "homePlayers" if ev.get("homeTeam") else "awayPlayers"
        end = sec + SUB_DECAY_SECONDS
        records.append((period, sec, end, team, 0.0))

    # --- Dead-ball (OTB) spans ---
    # Group OTB events into dead-ball spans: a span runs from the first OTB
    # event and continues while subsequent OTB events occur within
    # *OTB_SPAN_GAP_SECONDS* of the previous one.  The span ends at the last
    # grouped OTB event + *SETPIECE_RECOVERY_SECONDS*.
    otb_events = sorted(
        [
            ev
            for ev in (events or [])
            if ev.get("gameEventType") == "OTB" and not ev.get("nonEvent")
        ],
        key=lambda e: (e.get("period", 0), e.get("periodElapsedTimeEstimate", 0.0)),
    )
    i = 0
    while i < len(otb_events):
        period = otb_events[i].get("period")
        span_start = otb_events[i].get("periodElapsedTimeEstimate")
        if period is None or span_start is None:
            i += 1
            continue
        span_end = span_start
        j = i + 1
        while j < len(otb_events):
            ev = otb_events[j]
            if ev.get("period") != period:
                break
            next_sec = ev.get("periodElapsedTimeEstimate")
            if next_sec is None:
                j += 1
                continue
            if next_sec - span_end > OTB_SPAN_GAP_SECONDS:
                break
            span_end = next_sec
            j += 1
        recovery_end = span_end + SETPIECE_RECOVERY_SECONDS
        records.append((period, span_start, recovery_end, "both", 0.0))
        i = j

    return records


# ---------------------------------------------------------------------------
# Turnover detection (local, avoids circular dep with possession.py)
# ---------------------------------------------------------------------------


def _detect_turnovers(tracking_path, pitch_length, pitch_width):
    """Proximity-based ball-owner detection yielding turnover times.

    Returns a list of ``(period, elapsed_sec)`` for each frame where the
    ball owner switches between teams.  The previous owner is kept through
    no-owner (ball in flight) frames, so real turnovers are single changes,
    and a change is only registered once the new owner persists for
    *MIN_OWNER_PERSIST_FRAMES* owned frames — duel flicker between two
    nearby players is not counted.
    """
    x_shift = pitch_length / 2.0
    y_shift = pitch_width / 2.0
    POSSESSION_THRESHOLD = 2.5  # metres
    MIN_OWNER_PERSIST_FRAMES = 8  # owned frames (~0.3s at 25fps)

    periods: list[int] = []
    elapseds: list[float] = []
    owners: list[int] = []  # 0 = no owner, 1 = home, 2 = away

    try:
        with bz2.open(tracking_path, "rt") as f:
            for line in f:
                frame = json.loads(line)
                period = frame.get("period")
                elapsed = frame.get("periodElapsedTime")
                if period is None or elapsed is None:
                    continue

                balls = frame.get("balls", [])
                if not balls:
                    periods.append(period)
                    elapseds.append(float(elapsed))
                    owners.append(0)
                    continue
                bx = _get_first(balls[0], ("x", "X"))
                by = _get_first(balls[0], ("y", "Y"))
                if bx is None or by is None:
                    periods.append(period)
                    elapseds.append(float(elapsed))
                    owners.append(0)
                    continue
                bx_f, by_f = float(bx) + x_shift, float(by) + y_shift

                best_dist = POSSESSION_THRESHOLD
                best_team: str | None = None
                for team in _TEAM_KEYS:
                    for p in frame.get(team, []):
                        parsed = extract_player_xy(p)
                        if parsed is None:
                            continue
                        _pid, px, py = parsed
                        d = math.hypot(px + x_shift - bx_f, py + y_shift - by_f)
                        if d < best_dist:
                            best_dist = d
                            best_team = team

                periods.append(period)
                elapseds.append(float(elapsed))
                owners.append(0 if best_team is None else (1 if best_team == "homePlayers" else 2))
    except EOFError:
        logger.warning("tracking file truncated during turnover detection — continuing with data read so far.")

    # Detect turnovers on the raw owner stream with V1 semantics: the
    # previous owner is kept through no-owner frames, so a real turnover
    # (A --ball flight--> B) is a single change of owner even though the
    # ball is unowned in between.  A change is only registered once the
    # new owner accumulates *MIN_OWNER_PERSIST_FRAMES* owned frames, so
    # duel flicker (owner alternating faster than ~0.3s per hold) is not
    # counted.
    turnovers: list[tuple[int, float]] = []
    prev_owner = 0
    candidate = 0
    persist = 0
    for i in range(len(owners)):
        s = int(owners[i])
        if s == 0:
            continue
        if s == prev_owner:
            candidate = 0
            persist = 0
            continue
        if s == candidate:
            persist += 1
            if persist == MIN_OWNER_PERSIST_FRAMES and prev_owner != 0:
                turnovers.append((periods[i], elapseds[i]))
                prev_owner = s
                candidate = 0
                persist = 0
            continue
        # First frame of a new owner: it becomes the candidate.  When we
        # have no previous owner (kickoff or takeover after a long loose
        # phase), adopt it without registering a turnover.
        if prev_owner == 0:
            prev_owner = s
            candidate = 0
            persist = 0
        else:
            candidate = s
            persist = 1

    return turnovers


# ---------------------------------------------------------------------------
# Tracking-derived signal computation  (one streaming pass)
# ---------------------------------------------------------------------------


def _compute_tracking_signals(tracking_path, metadata, goalkeepers):
    """Pass over the tracking file to compute per-frame centroid + spread.

    Returns parallel lists: (periods, elapsed, home_centroid_v, away_centroid_v,
    home_spread_rate, away_spread_rate) along with the percentile scales.
    """
    pitch_length = metadata["pitch"]["length"]
    pitch_width = metadata["pitch"]["width"]
    x_shift = pitch_length / 2.0
    y_shift = pitch_width / 2.0
    fps = metadata.get("fps", 25.0)

    periods: list[int] = []
    elapsed_list: list[float] = []

    home_cx: list[float] = []
    home_cy: list[float] = []
    away_cx: list[float] = []
    away_cy: list[float] = []
    home_spread: list[float] = []
    away_spread: list[float] = []

    def _team_stats(players, gk_ids):
        xs, ys = [], []
        for p in players:
            parsed = extract_player_xy(p)
            if parsed is None:
                continue
            pid, px, py = parsed
            if gk_ids and str(pid) in gk_ids:
                continue
            xs.append(px + x_shift)
            ys.append(py + y_shift)
        if not xs:
            return None, None, None, None
        cx = float(np.mean(xs))
        cy = float(np.mean(ys))
        spread = float(np.std(xs, ddof=0) + np.std(ys, ddof=0)) / 2.0
        return cx, cy, spread, len(xs)

    try:
        with bz2.open(tracking_path, "rt") as f:
            for line in f:
                frame = json.loads(line)
                period = frame.get("period")
                elapsed = frame.get("periodElapsedTime")
                if period is None or elapsed is None:
                    continue
                periods.append(period)
                elapsed_list.append(float(elapsed))

                for team in _TEAM_KEYS:
                    gk_ids = goalkeepers.get(team) or set()
                    stats = _team_stats(frame.get(team, []), gk_ids)
                    cx, cy, spread, n = stats if stats else (None, None, None, None)
                    if team == "homePlayers":
                        home_cx.append(cx)
                        home_cy.append(cy)
                        home_spread.append(spread)
                    else:
                        away_cx.append(cx)
                        away_cy.append(cy)
                        away_spread.append(spread)
    except EOFError:
        logger.warning("tracking file truncated during signal extraction — continuing.")

    n = len(periods)

    # Centroid velocities
    def _velocities(cx_arr, cy_arr):
        vel = [0.0] * n
        prev_cx, prev_cy = None, None
        for i in range(n):
            if cx_arr[i] is None or cy_arr[i] is None:
                vel[i] = None
                prev_cx, prev_cy = None, None
                continue
            if prev_cx is not None:
                d = math.hypot(cx_arr[i] - prev_cx, cy_arr[i] - prev_cy)
                vel[i] = d * fps
            prev_cx, prev_cy = cx_arr[i], cy_arr[i]
        return vel

    # Spread rates
    def _spread_rates(spread_arr):
        rate = [0.0] * n
        prev = None
        for i in range(n):
            if spread_arr[i] is None:
                rate[i] = None
                prev = None
                continue
            if prev is not None:
                rate[i] = abs(spread_arr[i] - prev) * fps
            prev = spread_arr[i]
        return rate

    home_vel = _velocities(home_cx, home_cy)
    away_vel = _velocities(away_cx, away_cy)
    home_sr = _spread_rates(home_spread)
    away_sr = _spread_rates(away_spread)

    # Percentile scales (90th percentile of non-None, non-zero values)
    def _scale(values):
        clean = [v for v in values if v is not None and v > 0]
        if not clean:
            return 1.0
        return float(np.percentile(clean, _PERCENTILE))

    h_vel_scale = _scale(home_vel)
    a_vel_scale = _scale(away_vel)
    h_sr_scale = _scale(home_sr)
    a_sr_scale = _scale(away_sr)

    return (
        periods, elapsed_list,
        home_vel, away_vel, home_sr, away_sr,
        h_vel_scale, a_vel_scale, h_sr_scale, a_sr_scale,
    )


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def compute_frame_weights(tracking_path, metadata, events=None, goalkeepers=None):
    """Build per-frame, per-team reliability weights.

    Parameters
    ----------
    tracking_path : Path
        Path to ``tracking.jsonl.bz2``.
    metadata : dict
        Parsed ``metadata.json``.
    events : list[dict] or None
        Parsed ``events.json``, or None.
    goalkeepers : dict or None
        Output of ``resolve_goalkeepers()``, or None.

    Returns
    -------
    dict[int, dict[float, dict[str, float]]] or None
        ``{period: {elapsed_sec: {"homePlayers": w, "awayPlayers": w}}}``.
        Returns *None* when there are no events and the caller should use
        uniform weights (backward-compatible path).
    """
    if events is None:
        return None

    if goalkeepers is None:
        goalkeepers = {}

    # --- Event disruption timeline ---
    disruptions = _classify_events(events)
    turnovers = _detect_turnovers(tracking_path, metadata["pitch"]["length"],
                                   metadata["pitch"]["width"])

    # --- Tracking signals ---
    sigs = _compute_tracking_signals(tracking_path, metadata, goalkeepers)
    (periods_arr, elapsed_arr,
     home_vel, away_vel, home_sr, away_sr,
     h_vel_s, a_vel_s, h_sr_s, a_sr_s) = sigs

    n = len(periods_arr)

    # --- Build weight lookup ---
    result: dict[int, dict[float, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    for i in range(n):
        period = periods_arr[i]
        elapsed = elapsed_arr[i]

        # Start at 1.0
        w_home = 1.0
        w_away = 1.0

        # Event disruptions.  Each disruption imposes an upper bound on the
        # weight (linear ramp from base_weight at the event time back to 1.0
        # at the end of the decay window); overlapping disruptions are
        # combined with min(), because multiplying several near-zero ramps
        # would over-suppress frames that sit inside multiple windows.
        event_w_home = 1.0
        event_w_away = 1.0
        for rec in disruptions:
            rec_period, start, end, team, base = rec
            if rec_period != period:
                continue
            if start <= elapsed <= end:
                fraction = (elapsed - start) / (end - start) if end > start else 1.0
                weight = base + (1.0 - base) * fraction

                if team in ("both", "homePlayers"):
                    event_w_home = min(event_w_home, weight)
                if team in ("both", "awayPlayers"):
                    event_w_away = min(event_w_away, weight)

        # Turnover disruption (both teams, short decay) — same min() rule.
        turn_w_home = 1.0
        turn_w_away = 1.0
        for turn_period, turn_sec in turnovers:
            if turn_period != period:
                continue
            dt = elapsed - turn_sec
            if 0 <= dt <= TURNOVER_DECAY_SECONDS:
                fraction = dt / TURNOVER_DECAY_SECONDS
                turn_w_home = min(turn_w_home, fraction)
                turn_w_away = min(turn_w_away, fraction)

        w_home = min(event_w_home, turn_w_home)
        w_away = min(event_w_away, turn_w_away)

        # Apply tracking-derived weights
        if home_vel[i] is not None and h_vel_s > 0:
            w_home *= math.exp(-home_vel[i] / h_vel_s)
        if home_sr[i] is not None and h_sr_s > 0:
            w_home *= math.exp(-home_sr[i] / h_sr_s)
        if away_vel[i] is not None and a_vel_s > 0:
            w_away *= math.exp(-away_vel[i] / a_vel_s)
        if away_sr[i] is not None and a_sr_s > 0:
            w_away *= math.exp(-away_sr[i] / a_sr_s)

        result[period][elapsed] = {"homePlayers": w_home, "awayPlayers": w_away}

    logger.info(
        "frame weights computed: %d disruptions, %d turnovers, %d frames",
        len(disruptions), len(turnovers), n,
    )

    return dict(result)
