"""Centralized field-name constants for tracking data.

All modules that parse player/ball fields from raw tracking frames
should import these lists rather than duplicating them.
"""

PLAYER_ID_KEYS = ["jerseyNum", "playerId", "player_id", "id", "optaId", "ssiId"]
PLAYER_X_KEYS = ["x", "X"]
PLAYER_Y_KEYS = ["y", "Y"]
PLAYER_NUMBER_KEYS = ["number", "shirtNumber", "jerseyNumber", "num"]
