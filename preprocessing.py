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

# Dedicated per-match event file, e.g. Event_Data/<match_id>.json -- the
# PFF FC "Event Data Specification v2.5" format (game events + possession
# events pre-merged, one row per possession event). Preferred over the
# game_event_id/possession_event_id fields embedded in the raw tracking
# frames -- see process_events() docstring for why.
RAW_EVENTS_DIR = "Event_Data"

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

    return periods


# ==========================
# Events -- PREFERRED SOURCE: dedicated Event_Data/<match_id>.json
# (PFF FC Event Data Specification v2.5). Falls back to scraping
# game_event/possession_event off the raw tracking frames (see
# _extract_event_record below) only if this file doesn't exist for a
# match.
# ==========================

def process_events(match_id, periods_meta=None):
    """
    Reads Event_Data/<match_id>.json -- a list of rows, one row per
    POSSESSION EVENT (game events and possession events are pre-merged
    by the provider: a game event with two possession events, e.g. a
    challenge then a pass, is already two separate rows sharing the
    same gameEventId, each with its own possessionEventId). Because of
    that, unlike the old frame-scraped path, NO dedup-by-gameEventId is
    needed or correct here -- deduping would silently drop the second
    possession event of a multi-event game event.

    Writes Processed_Tracking/<match_id>/events.json and returns the
    list of records, or None if Event_Data/<match_id>.json doesn't
    exist (caller should fall back to process_tracking's embedded-frame
    extraction in that case).

    Each output record has a flattened "quick access" layer (the
    fields possession.py currently uses: period, sequence, team,
    player, timing, event types, nonEvent flag) plus the full raw
    gameEvents / possessionEvents / initialTouch / fouls / grades
    sub-dicts nested as-is, so nothing from the spec is thrown away
    even though only a fraction of it is consumed today. homePlayers /
    awayPlayers / ball / stadiumMetadata are dropped from the output --
    those are broadcast-tracking snapshots duplicating what's already
    in Tracking_Data, not needed here.

    ASSUMPTION (unverified -- check before relying on it): startGameClock
    is kept in the output for future use, but possession.py still derives
    its own periodElapsedTimeEstimate from startTime (video-referenced)
    rather than switching to startGameClock, because the spec doesn't
    say whether startGameClock resets to 0 each period or climbs
    continuously across the whole match. Check one second-half row
    before trusting it over the existing derivation.
    """
    input_path = os.path.join(RAW_EVENTS_DIR, f"{match_id}.json")

    if not os.path.exists(input_path):
        return None

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict):
        # same defensive pattern as metadata/roster loading, in case
        # some files wrap the row list in a dict.
        raw = raw.get("data", raw.get("rows", raw.get("events", [raw])))

    records = []
    skipped_non_events = 0

    for row in raw:
        ge = row.get("gameEvents") or {}
        pe = row.get("possessionEvents") or {}

        # "Was the possession disallowed after the fact?" -- e.g. a
        # goal/shot later ruled offside. Flagged but NOT dropped here;
        # let callers (possession.py) decide whether to filter these
        # out, same as they decide which gameEventType to keep.
        non_event = pe.get("nonEvent")
        if non_event is None:
            non_event = ge.get("initialNonEvent")
        if non_event:
            skipped_non_events += 1

        record = {
            # -- flattened quick-access fields --
            "gameEventId": row.get("gameEventId"),
            "possessionEventId": row.get("possessionEventId"),
            "period": ge.get("period"),
            "sequence": row.get("sequence"),
            "gameEventType": ge.get("gameEventType"),
            "possessionEventType": pe.get("possessionEventType"),
            "nonEvent": bool(non_event) if non_event is not None else False,
            "startTime": row.get("startTime"),
            "endTime": row.get("endTime"),
            "duration": row.get("duration"),
            "eventTime": row.get("eventTime"),
            "startGameClock": ge.get("startGameClock"),
            "formattedGameClock": pe.get("formattedGameClock") or ge.get("startFormattedGameClock"),
            "homeTeam": ge.get("homeTeam"),
            "teamId": ge.get("teamId"),
            "teamName": ge.get("teamName"),
            "playerId": ge.get("playerId"),
            "playerName": ge.get("playerName"),
            "setpieceType": ge.get("setpieceType"),
            "videoMissing": ge.get("videoMissing"),
            # -- full nested detail, preserved for future use --
            "gameEvents": ge,
            "possessionEvents": pe,
        }

        if row.get("initialTouch"):
            record["initialTouch"] = row["initialTouch"]
        if row.get("fouls"):
            record["fouls"] = row["fouls"]
        if row.get("grades"):
            record["grades"] = row["grades"]

        records.append(record)

    records.sort(key=lambda r: (r["sequence"] is None, r["sequence"],
                                 r["startTime"] is None, r["startTime"]))

    # periodElapsedTimeEstimate: same timestamp-difference approach as
    # the old embedded-frame path (see _add_period_elapsed_estimates
    # docstring) -- startTime here is video-referenced, same as before,
    # so this is unaffected by the startGameClock reset-per-period
    # question flagged above.
    _add_period_elapsed_estimates(records, periods_meta)

    output_folder = os.path.join(OUTPUT_DIR, str(match_id))
    os.makedirs(output_folder, exist_ok=True)
    events_path = os.path.join(output_folder, "events.json")
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    print(f"    wrote {len(records)} events ({skipped_non_events} flagged "
          f"nonEvent) -> {events_path}  [source: Event_Data]")

    return records


def _extract_event_record(frame, period):
    """
    Builds one flattened event record from a raw frame's game_event_id /
    game_event / possession_event_id / possession_event fields, or
    returns None if this frame carries no event.

    These fields repeat on EVERY frame across an event's
    start_frame..end_frame span in the raw data (a pass might span 20+
    frames, all carrying the identical game_event dict) -- so this is
    only called once per unique game_event_id by the caller, not once
    per frame, to avoid writing the same event out dozens of times.

    Field names are camelCased to match this pipeline's other output
    (metadata.json, roster.json) -- the raw snake_case source name is
    the same word, just re-cased, so cross-referencing the original data
    should still be straightforward.
    """
    ge = frame.get("game_event")
    if not ge:
        return None

    record = {
        "gameEventId": frame.get("game_event_id"),
        "period": period,
        "gameEventType": ge.get("game_event_type"),
        "formattedGameClock": ge.get("formatted_game_clock"),
        "playerId": ge.get("player_id"),
        "playerName": ge.get("player_name"),
        "shirtNumber": ge.get("shirt_number"),
        "positionGroupType": ge.get("position_group_type"),
        "teamId": ge.get("team_id"),
        "teamName": ge.get("team_name"),
        "startTime": ge.get("start_time"),
        "endTime": ge.get("end_time"),
        "duration": ge.get("duration"),
        "homeTeam": ge.get("home_team"),
        "homeBall": ge.get("home_ball"),
        "sequence": ge.get("sequence"),
        "startFrame": ge.get("start_frame"),
        "endFrame": ge.get("end_frame"),
        "possessionEventId": None,
        "possessionEventType": None,
        "possessionStartTime": None,
        "possessionStartFrame": None,
    }

    pe = frame.get("possession_event")
    if pe:
        record["possessionEventId"] = frame.get("possession_event_id")
        record["possessionEventType"] = pe.get("possession_event_type")
        record["possessionStartTime"] = pe.get("start_time")
        record["possessionStartFrame"] = pe.get("start_frame")

    return record


def _add_period_elapsed_estimates(events, periods_meta):
    """
    Best-effort conversion of each event's absolute startTime into a
    periodElapsedTime-comparable value, using
    metadata["periods"][period]["start"] as that period's zero point.

    This is a TIMESTAMP DIFFERENCE (event.startTime - period.start),
    which is valid regardless of what "zero" means in the raw clock --
    same reasoning already validated for period-length calculations in
    plot_formation_timeline.py. Skips events whose period has no
    metadata entry (leaves periodElapsedTimeEstimate out) rather than
    guessing.
    """
    if not periods_meta:
        return
    for r in events:
        meta_entry = periods_meta.get(str(r.get("period")))
        if not meta_entry or meta_entry.get("start") is None:
            continue
        if r.get("startTime") is None:
            continue
        r["periodElapsedTimeEstimate"] = round(float(r["startTime"]) - float(meta_entry["start"]), 3)


def process_tracking(match_id, periods_meta=None, extract_embedded_events=True):
    """
    Single streaming pass over the raw tracking file that:
      1. Writes the slimmed tracking.jsonl.bz2 (TRACKING_FIELDS only),
         same as before.
      2. IF extract_embedded_events is True, ALSO collects the event log
         embedded in the raw frames (game_event_id / game_event /
         possession_event_id / possession_event) into events.json -- one
         row per UNIQUE gameEventId, deduped (see _extract_event_record's
         docstring for why: the raw data repeats the same event dict on
         every frame across its span). Only written if the raw data
         actually contains event fields -- older/other datasets without
         them just won't get an events.json, no error.

         extract_embedded_events should be False when process_events()
         already wrote a real events.json for this match from
         Event_Data/<match_id>.json (the preferred, provider-labeled
         source -- see process_events docstring). That source is 1
         row per possession event with no dedup pitfalls, so re-deriving
         a second, worse events.json from the tracking frames here would
         just overwrite the good one with a lossier version.

    Combined into one pass (rather than a second full read of the file)
    to avoid decompressing/parsing ~200k frames twice per match.
    """

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

    events_by_id = {}

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

            if extract_embedded_events:
                game_event_id = frame.get("game_event_id")
                if game_event_id is not None and game_event_id not in events_by_id:
                    record = _extract_event_record(frame, frame.get("period"))
                    if record is not None:
                        events_by_id[game_event_id] = record

    if extract_embedded_events and events_by_id:
        events = sorted(
            events_by_id.values(),
            key=lambda r: (r["sequence"] is None, r["sequence"])
        )
        _add_period_elapsed_estimates(events, periods_meta)

        events_path = os.path.join(output_folder, "events.json")
        with open(events_path, "w", encoding="utf-8") as f:
            json.dump(events, f, indent=2)
        print(f"    wrote {len(events)} events -> {events_path}")


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

        periods_meta = process_metadata(match_id)

        # Prefer the dedicated Event_Data/<match_id>.json (real,
        # provider-labeled possession events). Only fall back to
        # scraping game_event/possession_event off the tracking frames
        # if this match has no Event_Data file.
        real_events = process_events(match_id, periods_meta)
        if real_events is None:
            print(f"    no Event_Data/{match_id}.json -- falling back to "
                  f"embedded tracking-frame event extraction.")

        process_tracking(match_id, periods_meta,
                          extract_embedded_events=(real_events is None))

    print("Done!")


if __name__ == "__main__":
    main()