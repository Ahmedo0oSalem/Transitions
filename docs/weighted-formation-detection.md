# Weighted Formation Detection — Implementation Log

## Step 1: Extract `tracking_fields.py`

**Files created:** `analytics/formations/tracking_fields.py`  
**Files modified:** `analytics/formations/goalkeeper.py`

### What changed

Moved the shared field-name aliasing primitives — `PLAYER_*_KEYS`, `REJECT_CONFIDENCE_VALUES`, `REJECT_VISIBILITY_VALUES`, `_get_first()`, and `extract_player_xy()` — out of `goalkeeper.py` into a new dependency-free module `tracking_fields.py`.

### Why

`frame_reliability.py` (coming in Step 2) needs these same primitives to parse tracking frames. If they stayed in `goalkeeper.py`, then `frame_reliability.py` importing them would create a circular import chain — because `detector.py` needs to import `frame_reliability.py`, and `frame_reliability.py` would need to import `goalkeeper.py` (which is imported by `detector.py`). Pulling the primitives into their own module breaks the cycle: `goalkeeper.py`, `windows.py`, and `frame_reliability.py` all import *from* `tracking_fields.py`; nothing imports back.

### Backward compatibility

`detector.py` still imports these names from `.goalkeeper`, which re-exports them from `.tracking_fields`. Existing consumers (`viewer.py` via `df_mod._get_first`, etc.) keep working unchanged.

### Verification

- `tracking_fields` imports cleanly with no dependencies beyond `io.field_keys`
- `goalkeeper` imports cleanly from `tracking_fields`
- All 9 existing unit tests pass

---

## Step 2: Create `frame_reliability.py`

**Files created:** `analytics/formations/frame_reliability.py`

### What changed

Added a new module that computes per-frame, per-team reliability weights in [0, 1], indicating how representative each tracking frame is of a team's true formation shape.

**Two signal types, combined multiplicatively:**

1. **Event-derived** (exact, from `events.json`):
   - Foul (`FOUL_DECAY_SECONDS = 15`) — affected team weight drops to 0, recovers linearly
   - Substitution (`SUB_DECAY_SECONDS = 30`) — team-specific, longer recovery
   - Dead-ball spans (OTB events grouped consecutively + `SETPIECE_RECOVERY_SECONDS = 10`) — both teams affected
   - Turnovers (proximity-based ball-owner change, `TURNOVER_DECAY_SECONDS = 5`) — both teams briefly disrupted

2. **Tracking-derived** (continuous, threshold-free):
   - Centroid velocity per frame → `exp(-v / scale)` where *scale* = 90th percentile of this match
   - Spread (radius of gyration) rate of change → same sigmoid
   - Both percentiles computed from a single streaming pass over the tracking file

**Key constants:**
- `MIN_WINDOW_CONFIDENCE = 0.5` — windows below this threshold are dropped entirely

**Entrypoint:**
```python
compute_frame_weights(tracking_path, metadata, events, goalkeepers)
  → dict[int, dict[float, dict[str, float]]]  or  None
```

### Backward compatibility

- If `events` is `None`, returns `None` → caller uses uniform weights (identical to old behaviour)
- `frame_reliability.py` imports from `tracking_fields.py` (not from `goalkeeper.py` or `detector.py`), so no circular dependencies
- Turnover detection reimplemented locally (proximity-based, ~8 lines) to avoid importing `possession.py`

### Verification

- `frame_reliability` imports cleanly (deps: `tracking_fields`, `core.logger`, stdlib + numpy)
- All 9 existing unit tests pass

---

## Step 3: Add `weight_lookup` to `accumulate_positions`

**Files modified:** `analytics/formations/windows.py`

### What changed

Added an optional `weight_lookup` parameter (dict or None) to `accumulate_positions()`. When provided — as returned by `compute_frame_weights()` — the function stores `(x, y, w)` triples per frame instead of the original `(x, y)` tuples. When `None` (no events available), every frame gets weight `1.0`, preserving identical behaviour to the original code.

Also switched the import of `extract_player_xy` from `.goalkeeper` to `.tracking_fields` for a cleaner dependency chain (no transitive circular risk).

**Before (one bucket entry):**
```python
buckets[(team, period, k)][pid].append(xy)         # (x, y)
```

**After:**
```python
buckets[(team, period, k)][pid].append(xyw)        # (x, y, w)
```

### Why

The caller (`detector.py`) needs per-frame weights to compute weighted player centroids and per-window confidence scores. Embedding the weight in each stored coordinate avoids maintaining a parallel data structure.

### Backward compatibility

- `weight_lookup=None` (the default) → `w=1.0` for every frame → downstream `arr.mean(axis=0)` on `(x, y, 1.0)` would be wrong, so Step 4 updates the averaging code to always do weighted averaging. When all weights are 1.0, the result is identical.

### Verification

- `analytics.formations.windows` imports cleanly
- All 9 existing unit tests pass

---

## Step 4: Weighted averaging and confidence filtering in `detector.py`

**Files modified:** `analytics/formations/detector.py`

### What changed

Rewrote `process_match()` with four additions:

1. **Load events.json** — if the file exists, load and log the count; otherwise signal that no event-based confidence is available.

2. **Call `compute_frame_weights()`** — pass the loaded events (if any), metadata, goalkeepers, and tracking path to get the `weight_lookup` dict. When no events exist, `weight_lookup = None`.

3. **Weighted player averaging** — instead of `arr.mean(axis=0)`, compute:
   ```python
   w_sum = arr[:, 2].sum()
   wx    = (arr[:, 0] * arr[:, 2]).sum() / w_sum
   wy    = (arr[:, 1] * arr[:, 2]).sum() / w_sum
   ```
   Also accumulates `window_weight_sum` across all players for the window's confidence.

4. **Per-window confidence and filtering** — confidence is defined as:
   ```
   confidence = total_weight_sum * (1 / (1 + cost))
   ```
   When events exist, windows below `MIN_WINDOW_CONFIDENCE` (0.5) are skipped with a debug log. The computed confidence is written as a new `confidence` column in `formations.csv`. When no events exist, no filtering is applied.

### Why

Frames near disruptive events (fouls, subs, set pieces, fast transitions) are less representative of a team's settled formation. Weighting de-emphasises them, and the confidence filter removes whole windows where the data was too chaotic to trust the formation match.

### Verification

- `analytics.formations.detector` imports cleanly
- All 9 existing unit tests pass
- CLI entry point unchanged — `python -m analytics.formations.detector <match_id>` still works

---

## Step 5: Vote-based timeline bars

**Files modified:** `ui/timeline.py`

### What changed

Replaced the possession-sequence-based bar building with a vote-and-run approach:

1. **`resolve_formation_by_vote(formations_df)`** — groups rows by `(team, period)`, strips the `_flipped` suffix, sums `confidence` per base formation name, and returns the winner. This is the single formation label that best describes the team's shape across the entire period.

2. **`formation_runs_from_votes(formations_df, voted, offsets)`** — scans windows sorted by `(team, period, windowStartSec)`. Whenever a window's base formation matches the voted winner, it starts or extends a run. Runs are closed when a non-matching window is encountered (or the group ends). Each run produces a `{team, formation, matchStart, matchEnd}` bar in continuous match-time (via the `offsets` mapping from `compute_period_offsets`).

3. **Removed:** `build_possession_sequences()`, `sequences_to_bars()`, `lookup_formation()` — the timeline no longer depends on the proximity-based possession heuristic for bar generation.

4. **`draw_team_panel()` updated** — bar duration computed as `matchEnd - matchStart` (was `b["duration"]` from possession sequences).

5. **Subtitle updated** to "Each bar = contiguous window run of the voted formation".

6. Backward compatibility: if `formations.csv` lacks the `confidence` column (older files), `load_data` assigns `1.0` to every row, making the vote fall back to a simple count of windows per formation.

### Why

The possession-sequence approach had three problems: (a) the proximity-based ball-owner heuristic is noisy and produced spurious micro-sequences; (b) it introduced an unnecessary dependency on the possession module; (c) it showed possession spells rather than formation spells, which is a different question. The vote-and-run approach directly answers "what formation was this team playing, and when did they switch?" at the formation-detection granularity.

### Verification

- `ui.timeline` imports cleanly
- All 9 existing unit tests pass
