"""Tracking-based possession helpers."""

from __future__ import annotations

import bz2
import json
import math

import numpy as np

from ...core.config import (
    POSSESSION_MIN_SEQUENCE_SECONDS as PACKAGE_POSSESSION_MIN_SEQUENCE_SECONDS,
    POSSESSION_SMOOTH_WINDOW_SECONDS as PACKAGE_POSSESSION_SMOOTH_WINDOW_SECONDS,
    POSSESSION_THRESHOLD_M as PACKAGE_POSSESSION_THRESHOLD_M,
)
from ...core.logger import get_logger
from ...io.field_keys import (
    PLAYER_ID_KEYS as _POSSESSION_PLAYER_ID_KEYS,
    PLAYER_X_KEYS as _POSSESSION_PLAYER_X_KEYS,
    PLAYER_Y_KEYS as _POSSESSION_PLAYER_Y_KEYS,
)

POSSESSION_THRESHOLD_M = PACKAGE_POSSESSION_THRESHOLD_M
SMOOTH_WINDOW_SECONDS = PACKAGE_POSSESSION_SMOOTH_WINDOW_SECONDS
MIN_SEQUENCE_SECONDS = PACKAGE_POSSESSION_MIN_SEQUENCE_SECONDS
logger = get_logger(__name__)


def load_epv_grid(path):
    """Load the EPV grid from CSV."""

    return np.loadtxt(path, delimiter=",")


def get_base_directions(metadata):
    """Return base attacking directions for home and away teams."""

    home_start_left = metadata.get("homeTeamStartLeft", True)
    home_dir_p1 = 1 if home_start_left else -1
    return home_dir_p1, -home_dir_p1


def attack_direction(team, period, home_dir_p1, away_dir_p1):
    """Return attacking direction for a team in a specific period."""

    base = home_dir_p1 if team == "home" else away_dir_p1
    parity = 1 if (int(period) % 2 == 1) else -1
    return base * parity


def epv_value(epv_grid, x, y, pitch_length, pitch_width, direction):
    """Lookup EPV at a location, oriented for the attacking direction."""

    n_rows, n_cols = epv_grid.shape
    gx = x if direction == 1 else (pitch_length - x)
    gy = y
    col = int(np.clip(gx / pitch_length * n_cols, 0, n_cols - 1))
    row = int(np.clip(gy / pitch_width * n_rows, 0, n_rows - 1))
    return float(epv_grid[row, col])


def _get_first(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def _extract_lenient(player_dict, player_id_keys, player_x_keys, player_y_keys):
    x = _get_first(player_dict, player_x_keys)
    y = _get_first(player_dict, player_y_keys)
    pid = _get_first(player_dict, player_id_keys)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)


def stream_ball_and_owner(tracking_path, pitch_length, pitch_width,
                           coords_are_centered=True,
                           possession_threshold_m=POSSESSION_THRESHOLD_M,
                           player_id_keys=None, player_x_keys=None, player_y_keys=None):
    """Stream tracking data and infer the nearest-player owner per frame."""

    player_id_keys = player_id_keys or _POSSESSION_PLAYER_ID_KEYS
    player_x_keys = player_x_keys or _POSSESSION_PLAYER_X_KEYS
    player_y_keys = player_y_keys or _POSSESSION_PLAYER_Y_KEYS

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
                            parsed = _extract_lenient(p, player_id_keys, player_x_keys, player_y_keys)
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
        logger.warning(
            "    !! WARNING: %s appears truncated/corrupted (bz2 stream ended early after %s frames). Continuing with the frames successfully read.",
            tracking_path,
            n_lines,
        )

    return (np.array(periods, dtype=np.int16),
            np.array(elapsed, dtype=np.float32),
            np.array(ball_x, dtype=np.float32),
            np.array(ball_y, dtype=np.float32),
            np.array(owner, dtype=np.int8))


def infer_fps(elapsed, periods):
    """Infer fps from median frame spacing in the first period."""

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
    """Apply rolling majority-vote smoothing to the owner stream."""

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
