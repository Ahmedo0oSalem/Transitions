# Formation Timeline — Two Algorithms

The formation timeline can plot data using two distinct algorithms, selectable from the
UI combo box (`Voted (confidence-weighted)` / `Voted runs (window granularity)`) or
the `--method {voted,all_formations}` CLI argument.

---

## Method 1: Voted (confidence-weighted)

**Internal name:** `"voted"`  
**Entrypoint:** `build_gap_free_segments()`  
**Vote function:** `resolve_by_vote_at()`  
**Plot function:** `draw_team_panel()`  
**Visual:** Single-lane state strip — one gap-free coloured track per team

### Algorithm

1. **Time grid** — for each `(team, period)`, build a uniform grid at `stride`-second
   intervals from `t=0` to `t_max` (the latest `windowEndSec` in that period).
   Typical stride is ~60s.

2. **Confidence-weighted vote at each grid point** — at every grid point `t`, query
   ALL detection windows whose `[windowStartSec, windowEndSec)` interval covers `t`.
   Among those windows:

   - Strip the `_flipped` suffix from formation names (e.g. `"442_flipped"` → `"442"`,
     pooling votes across orientations).
   - Sum the `confidence` values per base formation name.
   - The base formation with the highest total confidence wins.

   **Gap fill:** if no window covers `t` (possible during dead-ball spans where all
   windows were dropped by `MIN_WINDOW_CONFIDENCE`), the previous grid point's winner
   is extended forward. At the very start of a period, if no data exists, the point
   is skipped.

3. **Merge consecutive same-formation grid points** — walk the list of `(time, winner)`
   pairs. Whenever the winner changes, close the current segment and start a new one.

4. **Sliver filter** — any segment shorter than `min_segment_seconds` (default 45s) is
   absorbed into its longer neighbour. This removes brief, low-confidence flickers.

5. **Convert to match time** — apply the period offset (`offsets[period]`) to all
   segment boundaries.

### Key functions

#### `resolve_by_vote_at(df, team, period, t)`

```python
def resolve_by_vote_at(df, team, period, t):
    sub = df[(df["team"] == team) & (df["period"] == period)
             & (df["windowStartSec"] <= t) & (t < df["windowEndSec"])]
    if sub.empty:
        return None
    base = sub["formation"].apply(_strip_flipped)
    weight = sub["confidence"] if "confidence" in sub.columns else pd.Series(1.0, index=sub.index)
    scores = weight.groupby(base).sum()
    return scores.idxmax()
```

- Filters to windows covering `t` for the given team/period.
- **Strips** the `_flipped` suffix so both orientations pool into one candidate.
- Uses the `confidence` column as vote weight (falls back to `1.0` per window).
- Returns the base formation name with the highest summed confidence, or `None`.

#### `build_gap_free_segments(df, offsets, min_segment_seconds=45)`

```python
def build_gap_free_segments(df, offsets, min_segment_seconds=45):
```

- Iterates `(team, period)` groups.
- Builds a uniform time grid at stride intervals.
- Calls `resolve_by_vote_at()` at every grid point.
- Merges consecutive same-winner points into segments.
- Filters out segments shorter than `min_segment_seconds`.
- Returns `[{team, formation, matchStart, matchEnd}, ...]` covering the match
  with **no gaps and no overlaps**.

#### `draw_team_panel(ax, segments, color_of, total_duration)`

```python
def draw_team_panel(ax, segments, color_of, total_duration):
```

- Single-lane view: a thick `ax.hlines(0, ...)` per segment at `y=0`.
- Formation label rendered at midpoint if segment ≥ 30s.
- Y-axis hidden (single row).

### Result characteristics

| Property | Value |
|---|---|
| Coverage | **Gap-free** — every moment has a formation label |
| Orientation | `_flipped` pooled — "442" and "442_flipped" vote together |
| Smoothness | **High** — uniform stride grid + vote + sliver filter |
| Transitions | At the moment the rolling consensus changes (stride granularity) |
| Visual | One coloured bar per team, formation label for segments ≥30s |

---

## Method 2: Voted runs (window granularity)

**Internal name:** `"all_formations"`  
**Entrypoint:** `formation_runs_from_votes()`  
**Vote function:** `_resolve_formation_by_vote()`  
**Plot function:** `draw_team_panel_piano_roll()`  
**Visual:** Piano roll — one horizontal row per formation, stacked

### Algorithm

1. **Vote at each window start** — for each `(team, period)`, collect all unique
   `windowStartSec` values in chronological order. At each start time `sec`:

   - Query ALL detection windows covering `sec` (same interval filter as Method 1).
   - Vote using **raw formation names** — the `_flipped` suffix is **not** stripped.
   - Sum the `confidence` values per raw formation name.
   - The raw formation with the highest total confidence wins.

   Returns `None` if no window covers `sec`.

2. **Merge consecutive same-winner into bars** — walk `(start_time, winner)` pairs.
   On a change, close the current bar and start a new one. On `None`, close any open
   run — creating a **visible gap**.

3. **No gap-fill, no sliver filter** — gaps remain as gaps. No segments are merged
   or removed regardless of duration.

4. **Last bar** — extends to `last_window_start + stride`.

### Key functions

#### `_resolve_formation_by_vote(df, team, period, sec)`

```python
def _resolve_formation_by_vote(df, team, period, sec):
    sub = df[(df["team"] == team) & (df["period"] == period)
             & (df["windowStartSec"] <= sec) & (sec < df["windowEndSec"])]
    if sub.empty:
        return None
    weight = sub["confidence"] if "confidence" in sub.columns else pd.Series(1.0, index=sub.index)
    scores = weight.groupby(sub["formation"]).sum()
    return scores.idxmax()
```

- Identical interval filter to `resolve_by_vote_at`.
- **Does NOT strip** `_flipped` — `"442"` and `"442_flipped"` are separate candidates.
- Matches the old download's `resolve_formation_by_vote` exactly.

#### `formation_runs_from_votes(df, offsets)`

```python
def formation_runs_from_votes(df, offsets):
```

- Iterates `(team, period)` groups.
- Collects unique `windowStartSec` values in order.
- Calls `_resolve_formation_by_vote()` at each start time.
- Merges consecutive same-winner into bars; closes on `None`.
- Returns `[{team, formation, matchStart, matchEnd}, ...]` — gaps are honest
  (no vote = no bar).

#### `draw_team_panel_piano_roll(ax, bars, color_of, total_duration)`

```python
def draw_team_panel_piano_roll(ax, bars, color_of, total_duration):
```

- Multi-row view: one `ax.hlines(y, ...)` per formation, sorted by total duration.
- Each row is a thin horizontal track; bars are short `hlines` segments on that track.
- Y-axis shows formation labels.

### Result characteristics

| Property | Value |
|---|---|
| Coverage | **Gappy** — gaps where no window covers a start time |
| Orientation | `_flipped` **preserved** — each orientation is its own row |
| Smoothness | **Low** — one vote per window start, no filtering |
| Transitions | At the exact window start where the voted winner changes |
| Visual | Multi-row piano roll |

---

## Shared utility functions

### `infer_stride_seconds(df)`

```python
def infer_stride_seconds(df):
    starts = sorted(df["windowStartSec"].unique())
    diffs = [b - a for a, b in zip(starts, starts[1:]) if b > a]
    if not diffs:
        return 0
    return Counter(diffs).most_common(1)[0][0]
```

Infers the detection stride from the most common gap between consecutive
`windowStartSec` values in the DataFrame. Used by both methods.

### `compute_period_offsets(df, metadata)`

```python
def compute_period_offsets(df, metadata=None):
```

Returns `(offsets, boundaries, total_duration)`:
- `offsets: dict[int, float]` — match-time offset for each period.
- `boundaries: list[(time, period)]` — cumulative period boundaries.
- `total_duration: float` — total match length in seconds.

### `_strip_flipped(formation)`

```python
def _strip_flipped(formation):
```

Strips the `_flipped` suffix from a formation name (e.g. `"442_flipped"` → `"442"`).
Used by `resolve_by_vote_at` but not by `_resolve_formation_by_vote`.

### `load_data(match_id, processed_dir)`

Loads `formations.csv` (as a DataFrame), `metadata.json` (as a dict), home/away
team names from the tracking file, and the tracking path. Returns all five plus
the match directory path.

### `plot_formation_timeline(match_id, processed_dir, method)`

Top-level entrypoint called by the UI and CLI. Dispatches to either
`build_gap_free_segments()` or `formation_runs_from_votes()` based on `method`.
Handles goal loading, period dividers, axis formatting, legend, and suptitle.

### Goal-related functions

- `find_goals(events)` — extracts goal events from `events.json` (scored/conceded
  per team, own-goal flag, minute, scorer name).
- `draw_goals(ax_home, ax_away, goals, offsets)` — draws vertical goal markers:
  green for scored (from this panel's team perspective), red for conceded.

### `draw_period_dividers(ax_top, ax_bottom, boundaries)`

Draws vertical dashed lines at period boundaries with "End P1", "End P2" labels.

---

## Key differences

| Aspect | Voted (confidence-weighted) | Voted runs (window granularity) |
|---|---|---|
| Vote resolution | Every `stride` seconds (~60s grid) | Only at `windowStartSec` timestamps |
| `_flipped` handling | Stripped → pooled into base | Preserved → separate candidates |
| Vote function | `resolve_by_vote_at()` | `_resolve_formation_by_vote()` |
| Gap fill | Yes — extend previous winner | No — honest gaps |
| Sliver filter | Yes — segments <45s absorbed | No |
| Output type | Adjacent, non-overlapping segments | Possibly gapped bars |
| Y-axis | Single lane (1 row per team) | One row per formation (multi-row) |
| Bar end point | Next stride grid point with different winner | Next window start (or period end) |
| Matches old download | No — new algorithm | **Yes** — identical logic |

### Visual comparison

```
Voted (confidence-weighted) — single lane:
  Team A  |──────── 442 ────────|── 4231 ──|── 442 ──────────|
          ^ stride grid points   ^          ^

Voted runs (window granularity) — piano roll:
  442     |████████████████|                    |██████████████|
  442_flip                  |████████████|
  4231                                       |████████████████|
          ^ window starts                    ^ gap
```

In the smoothed mode, a quick orientation flip (`442` → `442_flipped` → `442`) is
invisible — both pool into a single `442` bar. In the granular mode, the middle
`442_flipped` bar appears as a separate row, revealing the event.
