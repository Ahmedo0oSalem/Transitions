# Football Transitions Project Guide

## 1. What this project does

This project processes football tracking data and turns it into tactical and game-state insights:

- Preprocess raw match files into a clean, compact format.
- Detect team formations over time from player locations.
- Build possession sequences (event-based when available, tracking-based proxy otherwise).
- Compute an EPV momentum signal from ball location and possession.
- Detect Dangerous Attacking Sequences (DAS) from EPV peaks.
- Visualize match state interactively and as formation timelines.

In short: it is an end-to-end analytics pipeline from raw tracking to interpretable tactical outputs.

## 2. Repository files and responsibilities

- `Transitions.preprocessing.preprocess`
  - Reads raw files and writes cleaned outputs under Processed_Tracking/<match_id>/.
  - Writes metadata.json, tracking.jsonl.bz2, roster.json (if available), and events.json (if available).

- `Transitions.analytics.formations.detector`
  - Detects formations per time window using mplsoccer templates plus Hungarian matching.
  - Excludes goalkeepers using roster metadata first, then fallback movement-based GK inference.
  - Writes formations.csv.

- `Transitions.analytics.possession`
  - Shared utilities for:
    - EPV grid lookup and attack-direction handling
    - tracking-based possession proxy (closest player to ball)
    - smoothing and sequence extraction
    - event-derived possession sequence extraction

- `Transitions.analytics.epv.das`
  - Produces EPV momentum time series and DAS sequence table.
  - Uses event-derived possession for DAS when events are available.
  - Writes epv_timeseries.csv and das_sequences.csv and shows plots.

- `Transitions.ui.timeline`
  - Plots formation timeline as bars aligned to possession sequences.
  - Optionally overlays goals from events.json.

- `Transitions.ui.viewer`
  - Interactive viewer with slider/playback for players, ball, possession owner, and detected formations.

- `data/resources/EPV_grid.csv`
  - External EPV surface used for ball-location EPV lookup.

- requirements.txt
  - Python package versions.

- explore_data.ipynb
  - Notebook for ad-hoc exploration.

## 3. Data layout expected by the pipeline

Input folders expected at project root:

    Metadata/
      <match_id>.json
    Tracking_Data/
      <match_id>.jsonl.bz2
    Rosters/
      <match_id>.json         (optional but recommended)
    Event_Data/
      <match_id>.json         (optional, preferred for possession truth)

Output folder created by preprocessing:

    Processed_Tracking/
      <match_id>/
        metadata.json
        tracking.jsonl.bz2
        roster.json           (if roster exists)
        events.json           (if event source exists)
        formations.csv        (after formation detection)
        epv_timeseries.csv    (after EPV analysis)
        das_sequences.csv     (after EPV analysis)

## 4. Setup from scratch

1. Create and activate a Python virtual environment.
2. Install dependencies:

    pip install -r requirements.txt

3. Verify `data/resources/EPV_grid.csv` exists.
4. Put raw files in Metadata, Tracking_Data, and optionally Rosters and Event_Data.
5. Run commands from the folder that contains the `Transitions/` package folder, or install the package in editable mode.

    cd "D:\Studying\External\Football Transitions"

## 5. Run order (recommended)

### New recommended way: PyQt6 app window

Launch the single-window UI:

    python -m Transitions ui

The UI lets you run the app without typing each command manually:

- Enter the match id.
- Confirm `Processed dir` is `Processed_Tracking`.
- Confirm `Raw tracking dir` is `Tracking_Data`.
- Confirm `EPV grid` points to `Transitions/data/resources/EPV_grid.csv` if the default path is not found from your current working directory.
- Use `Run Preprocess` to convert raw data into `Processed_Tracking/<match_id>/`.
- Use `Detect Formations` to write `formations.csv`.
- Use `Run EPV + DAS` to write `epv_timeseries.csv` and `das_sequences.csv`, and show EPV plots.
- Use `Show Timeline` to display the formation timeline.
- Use `Open Match Viewer` to display the interactive pitch viewer.
- Use `Run Full Pipeline` to run preprocessing, formation detection, EPV/DAS, and the timeline in order for the entered match.

The UI keeps a run log on the left and opens generated plots in tabs on the right.

### CLI alternative: Step A, preprocess all matches

    python -m Transitions preprocess

What it does:
- scans Tracking_Data for all match ids
- processes metadata for each match
- prefers Event_Data/<match_id>.json for events
- falls back to embedded frame events if no Event_Data file exists

### CLI alternative: Step B, detect formations

All processed matches:

    python -m Transitions formations

Specific match:

    python -m Transitions formations 12345

Sliding windows example (5-minute context, updated every minute):

    python -m Transitions formations 12345 --window-seconds 300 --stride-seconds 60

### CLI alternative: Step C, EPV + DAS analysis

    python -m Transitions epv 12345 --epv-grid data/resources/EPV_grid.csv

### CLI alternative: Step D, possession-aligned formation timeline

    python -m Transitions timeline 12345

### CLI alternative: Step E, interactive match viewer

    python -m Transitions viewer 12345 --speed 1.5

### CLI alternative: full pipeline for one match

    python -m Transitions run 12345

## 6. Methodology details

### 6.1 Formation detection logic

For each team and time window:

1. Collect all player positions in the window.
2. Remove goalkeeper.
3. Average each outfield player position across the window.
4. Compare those average positions to each template formation from mplsoccer.
5. Use Hungarian assignment to match players to template slots minimizing total distance.
6. Pick the formation with minimum normalized cost.

Notes:
- Windowing supports overlap via stride < window length.
- Orientation is chosen from period parity and homeTeamStartLeft.

### 6.2 Possession logic

Two sources:

1. Preferred: event-driven possession sequences from events.json (gameEventType OTB, nonEvent excluded).
2. Fallback: closest-player-to-ball per frame with smoothing and run-length sequence extraction.

This fallback is useful but approximate around loose balls, deflections, and contested moments.

### 6.3 EPV logic in this project

This project uses tracking-only EPV attribution:

- Take the ball location each frame.
- Choose possessing team.
- Read EPV grid value at that location in attacking orientation.
- Sign value positive for home, negative for away.

This is an instantaneous state value (momentum-like), not pass-level EPV added.

### 6.4 DAS logic in this project

A possession sequence is flagged as dangerous if its peak EPV reaches the threshold.

Current defaults:
- DAS threshold: 0.15
- minimum sequence duration: 2.0 seconds

## 7. Reading the outputs

### 7.1 formations.csv

Typical columns:
- team, period
- windowStartSec, windowEndSec
- formation
- avgCostPerPlayer
- nOutfieldPlayers, nFrames

Interpretation example:
- If home team shows repeated 4-2-3-1 windows with low avgCostPerPlayer, structure is stable.
- A sudden shift to 3-4-3 with lower cost after substitutions may indicate tactical change.

### 7.2 epv_timeseries.csv

Typical columns:
- period
- secondIntoPeriod
- meanSignedEPV

Interpretation example:
- Positive sustained region means home held more dangerous territory.
- Negative spikes indicate away dangerous moments.

### 7.3 das_sequences.csv

Typical columns:
- team, period
- startSec, endSec, duration
- peakEPV
- isDAS

Interpretation example:
- Home has 6 DAS vs away 2 even with similar possession count: home possessions reached higher danger states.

## 8. Practical caveats

- EPV here is not event-based pass value added; it is ball-state EPV over time.
- Fallback possession proxy can misclassify brief transitions and loose-ball phases.
- Goalkeeper identification is strongest with roster files; fallback movement heuristic can fail in unusual tracking quality situations.
- Orientation assumptions should be validated on a known frame for a new data provider.

## 9. Improvement suggestions

### 9.1 High-impact analytics improvements

1. Add calibration and validation reports
- Compare detected formations to a manually labeled sample.
- Track confusion matrix across common systems (4-3-3 vs 4-2-3-1, etc.).

2. Improve possession inference when events are missing
- Add velocity and heading features around ball carrier candidates.
- Use hysteresis state machine instead of pure nearest-distance threshold.

3. Add uncertainty scores
- Formation confidence from cost margin between best and second-best templates.
- DAS confidence from persistence above threshold, not only peak.

4. Make DAS definition richer
- Combine peak EPV with dwell time above threshold.
- Optional rule: require final-third location or shot-context indicators.

### 9.2 Engineering and reliability improvements

1. Add a central CLI
- One command such as:

    python -m Transitions run 12345

- Subcommands for preprocess, formations, epv, plots.

2. Add schema validation
- Validate key fields for metadata/tracking/events at load time.
- Fail early with actionable error messages.

3. Add tests
- Unit tests for orientation, windowing, sequence extraction, and EPV indexing.
- Regression tests using a tiny fixture match.

4. Add logging and run reports
- Structured logs per match.
- Summary JSON with counts, skipped windows, warnings, and timings.

5. Make config external
- Move constants (thresholds, windows, minimum frames) to YAML or TOML.
- Keep script defaults but allow per-competition overrides.

### 9.3 Performance improvements

1. Reduce repeated file scans
- Cache lightweight per-match arrays after preprocessing.
- Reuse cached arrays for multiple downstream scripts.

2. Vectorize where possible
- Replace per-row loops in some EPV and matching stages with batch operations.

3. Parallelize match-level processing
- Process multiple match ids with multiprocessing or joblib.

## 10. Suggested next development milestones

1. Build tests and fixture data first.
2. Add confidence metrics for formation and DAS.
3. Implement a single unified CLI and config file.
4. Add an evaluation notebook comparing proxy possession vs event possession.

## 11. Quick troubleshooting

- Missing metadata.json or tracking.jsonl.bz2 under Processed_Tracking/<match_id>
  - Run `python -m Transitions preprocess` first.

- No formations.csv
  - Run `python -m Transitions formations <match_id>` after preprocessing.

- No goal markers in timeline
  - events.json is missing for that match; add Event_Data file and rerun preprocessing.

- Weird orientation results
  - Validate homeTeamStartLeft and period parity assumptions against one known frame.

- Very sparse players in windows
  - Check coordinate keys, confidence filters, and tracking completeness.

---

