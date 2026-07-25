"""
tracking_fields.py

Field-name aliasing primitives for reading player dicts out of raw
tracking frames (homePlayers/awayPlayers entries). Used by
detect_formation.py, possession.py, and frame_reliability.py.

WHY THIS IS ITS OWN FILE: it used to live in detect_formation.py, with
possession.py importing detect_formation just to reach these 4 things.
frame_reliability.py (new) also needs them, but frame_reliability.py
also needs detect_formation.py's goalkeeper-resolution logic AND
(indirectly, for turnover timing) the same event-sequence concepts
possession.py uses -- which would create an import cycle
(detect_formation -> frame_reliability -> possession -> detect_formation)
if these primitives still lived inside detect_formation.py. Pulling
them out into a dependency-free module breaks that cycle: everything
else imports FROM here, nothing here imports anything of ours.

detect_formation.py re-imports these names at module level (see its
Configuration section), so `import detect_formation as df_mod;
df_mod.PLAYER_X_KEYS` (as possession.py already does) keeps working
unchanged -- this refactor is not a breaking change for existing code.
"""

# Field name aliases we will try (in order) when reading each player
# dict in homePlayers/awayPlayers, so this works across a few common
# provider schemas without you having to rewrite the parsing loop.
# ADD/EDIT these if your schema uses different keys.
PLAYER_ID_KEYS = ["jerseyNum", "playerId", "player_id", "id", "optaId", "ssiId"]
PLAYER_X_KEYS = ["x", "X"]
PLAYER_Y_KEYS = ["y", "Y"]
PLAYER_NUMBER_KEYS = ["number", "shirtNumber", "jerseyNumber", "num"]

# Some providers tag each point with a quality flag (e.g. "confidence":
# "HIGH"/"MEDIUM"/"LOW", "visibility": "VISIBLE"/"ESTIMATED"). LOW-confidence
# points are often noisy/extrapolated and can wreck the goalkeeper heuristic
# (spurious jumps make a stationary GK look like it's roaming) and blur the
# averaged formation shape. If your data has these fields, points matching
# any of these will be DROPPED. Set to None/[] to disable filtering.
REJECT_CONFIDENCE_VALUES = ["LOW"]
REJECT_VISIBILITY_VALUES = []  # e.g. ["ESTIMATED"] to also drop interpolated points


def _get_first(d, keys, default=None):
    for k in keys:
        if k in d:
            return d[k]
    return default


def extract_player_xy(player_dict):
    if REJECT_CONFIDENCE_VALUES and player_dict.get("confidence") in REJECT_CONFIDENCE_VALUES:
        return None
    if REJECT_VISIBILITY_VALUES and player_dict.get("visibility") in REJECT_VISIBILITY_VALUES:
        return None
    x = _get_first(player_dict, PLAYER_X_KEYS)
    y = _get_first(player_dict, PLAYER_Y_KEYS)
    pid = _get_first(player_dict, PLAYER_ID_KEYS)
    if x is None or y is None or pid is None:
        return None
    return pid, float(x), float(y)
