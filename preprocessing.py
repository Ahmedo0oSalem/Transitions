import os
import json
import bz2

# ==========================
# Configuration
# ==========================

RAW_TRACKING_DIR = "Tracking_Data"
RAW_METADATA_DIR = "Metadata"

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

    output_folder = os.path.join(OUTPUT_DIR, str(match_id))
    os.makedirs(output_folder, exist_ok=True)

    output_path = os.path.join(output_folder, "metadata.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4)


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