import os
import json
import bz2

# ==========================
# Configuration
# ==========================

RAW_TRACKING_DIR = "Tracking_Data"
RAW_METADATA_DIR = "Metadata"

# ASSUMPTION: roster files live in a folder parallel to Metadata/Tracking_Data,
# named the same way (<match_id>.json). Rename this if yours differs
# (e.g. "Rosters", "Lineups", "Squads").
RAW_ROSTERS_DIR = "Rosters"

OUTPUT_DIR = "Processed_Tracking"

# Which tracking fields to keep
TRACKING_FIELDS = [
    "period",
    "periodElapsedTime",
    "homePlayers",
    "awayPlayers",
    "balls"
]

# Rename fields in the output
FIELD_RENAME = {
}

# Which metadata fields to keep
METADATA_FIELDS = [
    "id",
    "fps",
    "homeTeam",
    "awayTeam",
    "homeTeamStartLeft",
    "stadium",
]


# ==========================
# Roster
# ==========================

def process_roster(match_id, home_team_id, away_team_id):
    """
    Reads Rosters/<match_id>.json (a flat list of
    {"player": {"id", "nickname"}, "positionGroupType", "shirtNumber",
     "started", "team": {"id", "name"}} entries -- one row per squad
    player, both teams mixed together) and returns:

        {
            "players": [
                {"playerId": str, "name": str, "position": str,
                 "shirtNumber": str, "started": bool, "side": "home"/"away"},
                ...
            ],
            "goalkeepers": {
                "home": {"playerId": str, "shirtNumber": str} or None,
                "away": {"playerId": str, "shirtNumber": str} or None,
            }
        }

    or None if the roster file is missing.

    We store BOTH playerId and shirtNumber for each goalkeeper because
    we don't know, without inspecting a raw tracking frame, whether your
    tracking provider identifies players in homePlayers/awayPlayers by
    an internal player ID or by shirt number (detect_formation.py's
    PLAYER_ID_KEYS tries several). Keeping both lets the formation
    script match on whichever key the tracking data actually uses.
    """
    input_path = os.path.join(RAW_ROSTERS_DIR, f"{match_id}.json")

    if not os.path.exists(input_path):
        print(f"    !! Roster missing: {match_id} (looked in {input_path}). "
              f"Formation detection will fall back to distance-based GK guessing.")
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        roster = json.load(f)

    if isinstance(roster, dict):
        # in case some files wrap the list, same pattern as metadata
        roster = roster.get("players", roster.get("data", [roster]))

    home_team_id = str(home_team_id)
    away_team_id = str(away_team_id)

    players = []
    goalkeepers = {"home": None, "away": None}
    unmatched_team_ids = set()

    for entry in roster:
        player = entry.get("player", {})
        team = entry.get("team", {})
        team_id = str(team.get("id"))

        if team_id == home_team_id:
            side = "home"
        elif team_id == away_team_id:
            side = "away"
        else:
            unmatched_team_ids.add(team_id)
            continue

        record = {
            "playerId": str(player.get("id")),
            "name": player.get("nickname"),
            "position": entry.get("positionGroupType"),
            "shirtNumber": str(entry.get("shirtNumber")),
            "started": bool(entry.get("started", False)),
            "side": side,
        }
        players.append(record)

        if record["started"] and record["position"] == "GK":
            goalkeepers[side] = {
                "playerId": record["playerId"],
                "shirtNumber": record["shirtNumber"],
            }

    if unmatched_team_ids:
        print(f"    !! WARNING: roster for {match_id} has team id(s) "
              f"{unmatched_team_ids} that don't match homeTeam id "
              f"({home_team_id}) or awayTeam id ({away_team_id}) from metadata. "
              f"Check the id field name/type in your raw metadata/roster.")

    for side in ("home", "away"):
        if goalkeepers[side] is None:
            print(f"    !! WARNING: no starting GK found in roster for "
                  f"{match_id} / {side} team.")

    return {"players": players, "goalkeepers": goalkeepers}


# ==========================
# Metadata
# ==========================
def process_metadata(match_id):

    input_path = os.path.join(RAW_METADATA_DIR, f"{match_id}.json")

    if not os.path.exists(input_path):
        print(f"Metadata missing: {match_id}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # --------------------------------------------------
    # Some metadata files are stored as a list containing
    # one dictionary.
    # --------------------------------------------------
    if isinstance(metadata, list):
        metadata = metadata[0]

    periods = {}

    for key, value in metadata.items():

        if key.startswith("startPeriod"):

            period = key.replace("startPeriod", "")

            periods[period] = {
                "start": value,
                "end": metadata.get(f"endPeriod{period}")
            }

    output = {

        "matchId": int(metadata["id"]),

        "competition": metadata["competition"]["name"],

        "date": metadata["date"],

        "trackingType": "smoothed",

        "fps": metadata["fps"],

        "pitch": {
            "length": metadata["stadium"]["pitches"][0]["length"],
            "width": metadata["stadium"]["pitches"][0]["width"]
        },

        "homeTeam": metadata["homeTeam"],

        "awayTeam": metadata["awayTeam"],

        "homeTeamStartLeft": metadata["homeTeamStartLeft"],

        "periods": periods
    }

    # --------------------------------------------------
    # Roster: attach goalkeeper IDs (and write the full roster
    # separately) so formation detection doesn't have to guess the GK
    # from movement.
    # --------------------------------------------------
    roster_data = process_roster(
        match_id,
        home_team_id=metadata["homeTeam"]["id"],
        away_team_id=metadata["awayTeam"]["id"],
    )

    if roster_data is not None:
        output["goalkeepers"] = roster_data["goalkeepers"]
    else:
        output["goalkeepers"] = {"home": None, "away": None}

    output_folder = os.path.join(OUTPUT_DIR, str(match_id))
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "metadata.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)

    if roster_data is not None:
        roster_path = os.path.join(output_folder, "roster.json")
        with open(roster_path, "w", encoding="utf-8") as f:
            json.dump(roster_data["players"], f, indent=4)


# ==========================
# Tracking
# ==========================

def process_tracking(match_id):

    input_path = os.path.join(
        RAW_TRACKING_DIR,
        f"{match_id}.jsonl.bz2"
    )

    if not os.path.exists(input_path):
        print(f"Tracking missing: {match_id}")
        return

    output_folder = os.path.join(OUTPUT_DIR, str(match_id))
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(
        output_folder,
        "tracking.jsonl.bz2"
    )

    with bz2.open(input_path, "rt") as fin, \
         bz2.open(output_path, "wt") as fout:

        for line in fin:

            frame = json.loads(line)

            new_frame = {}

            for field in TRACKING_FIELDS:

                if field not in frame:
                    continue

                new_name = FIELD_RENAME.get(field, field)

                new_frame[new_name] = frame[field]

            fout.write(json.dumps(new_frame))
            fout.write("\n")


# ==========================
# Main
# ==========================

def main():

    tracking_files = [
        f for f in os.listdir(RAW_TRACKING_DIR)
        if f.endswith(".jsonl.bz2")
    ]

    print(f"Found {len(tracking_files)} matches.")

    for file in tracking_files:

        match_id = os.path.splitext(
            os.path.splitext(file)[0]
        )[0]

        print(f"Processing match {match_id}")

        process_metadata(match_id)
        process_tracking(match_id)

    print("Done!")


if __name__ == "__main__":
    main()