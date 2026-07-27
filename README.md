# TRANSITIONS

TRANSITIONS is a modular football tracking analytics framework for preprocessing, formation detection, possession inference, EPV analysis, dangerous attacking sequences, and match visualization.

## Modules

- `preprocessing/`: raw metadata, tracking, roster, and event conversion.
- `analytics/formations/`: formation detection from tracking windows.
- `analytics/possession/`: tracking and event-derived possession helpers.
- `analytics/epv/`: EPV momentum and Dangerous Attacking Sequence analysis.
- `ui/`: PyQt6 app window, interactive match viewer, and formation timelines.
- `pipeline/`: orchestration helpers used by the CLI.
- `domain/`: core typed match, frame, player, formation, possession, and EPV models.
- `artifacts/`: analytics result builders.

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python -m Transitions preprocess
python -m Transitions formations 12345
python -m Transitions epv 12345
python -m Transitions timeline 12345
python -m Transitions viewer 12345
python -m Transitions run 12345
python -m Transitions ui
```

The EPV grid lives at `data/resources/EPV_grid.csv` by default.

## Data Layout

Input folders expected at project root:

```text
Metadata/<match_id>.json
Tracking_Data/<match_id>.jsonl.bz2
Rosters/<match_id>.json
Event_Data/<match_id>.json
```

Preprocessing writes:

```text
Processed_Tracking/<match_id>/metadata.json
Processed_Tracking/<match_id>/tracking.jsonl.bz2
Processed_Tracking/<match_id>/roster.json
Processed_Tracking/<match_id>/events.json
```

Downstream analytics add `formations.csv`, `epv_timeseries.csv`, and `das_sequences.csv`.
