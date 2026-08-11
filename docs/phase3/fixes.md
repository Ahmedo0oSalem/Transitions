# Phase 3 — Analytics Correctness Fixes

Date: 2026-08-12
Scope: formation reliability weighting, confidence filtering, template matching quality, event-less possession handling, EPV bucketing. Pitch control intentionally excluded.

## Motivation

A review of the analytics pipeline (see conversation log / prior analysis) found several defects that distorted formation detection and EPV/DAS outputs on the World Cup 2022 data. This document records what was wrong, what was changed, and the measured before/after.

## Fix 1 — Dead-ball (OTB) span grouping

**File:** `analytics/formations/frame_reliability.py` (`_classify_events`)

**Bug:** OTB (out of bounds) events were merged into a single span with no time-gap check. For match 3816, period 1 has 1006 OTB events spanning t=1.1s → 3079.5s, so the *entire half* became one "dead ball" span whose weight ramps 0→1 linearly across the whole period. Consequence: weight at mid-half was ~0.52 from this term alone, early-period frames were systematically de-emphasised, and formation centroids were biased toward end-of-period positions.

**Fix:** Spans continue only while OTB events are within `OTB_SPAN_GAP_SECONDS = 15.0` of the previous one. A span ends at the last grouped OTB + `SETPIECE_RECOVERY_SECONDS`.

**Measured effect (3816):** disruption records 11 → 87; span lengths median 32.9s (was: one ~3090s span per period); weight time-profile now fluctuates (0.33–0.88 by 300s block) instead of monotonically ramping.

## Fix 2 — Turnover detection

**File:** `analytics/formations/frame_reliability.py` (`_detect_turnovers`)

**Bug 2a (coordinate mismatch):** the ball position was shifted into absolute pitch coordinates (`+x_shift`) but players were left in centered coordinates, so every distance was >50m and the 2.5m ownership threshold never matched. Result: **0 turnovers detected for any match**; the turnover weight signal was dead code.

**Bug 2b (flicker):** naive owner-change counting treated duel flicker (nearest player alternating during a contest) as hundreds of turnovers, and overlapping decay windows compounded multiplicatively (0.1 × 0.3 × 0.5 ≈ 0.015), suppressing most of the match to very low weights.

**Fix:**
- Players are shifted into the same absolute frame as the ball before the distance check.
- A turnover is only registered when a *new owner* (A→B, keeping the previous owner through no-owner flight frames, so real turnovers across passes count once) accumulates `MIN_OWNER_PERSIST_FRAMES = 8` owned frames (~0.3s). Same-team regain after a loose phase is not a turnover.
- Overlapping disruption windows within the event family are combined with `min()` (strongest suppression wins) instead of multiplication, both for event spans and turnover decays.

**Measured effect (3816):** turnovers 0 → 497 total (~250/period, still generous because proximity ownership counts contested spells); weights now reach ~0.88 in settled play and drop locally around disruptions.

## Fix 3 — Confidence score normalised + filter active

**Files:** `analytics/formations/detector.py`, `analytics/formations/frame_reliability.py`

**Bug:** `confidence = window_weight_sum × fit_quality` where `window_weight_sum` ≈ thousands (10 outfield players × ~25fps × 180s of [0,1] weights). Measured confidence in the old `formations.csv` was **16.4–2170.1** against `MIN_WINDOW_CONFIDENCE = 0.5` — the "drop chaotic windows" feature could never fire.

**Fix:** `confidence = mean_frame_weight = window_weight_sum / max(1, n_frames)` — a clean [0,1] measure of how representative the window's frames are. The template-fit signal is now reported separately as `fitQuality` (1/(1+cost)) plus `avgCostPerPlayer`/`costMargin`/`ambiguous`. `MIN_WINDOW_CONFIDENCE` retuned to **0.3**.

**Measured effect (3816):** rows 180 → 112 (windows dominated by dead-ball/transition time are now actually dropped); confidence range 0.302–0.789, median 0.354. 3818: 114 rows, confidence 0.301–0.719.

Note: `confidence` semantics changed (was: weight-sum × fit; now: mean frame weight). Timeline vote weighting and segment aggregation consume it, so all outputs were regenerated (below).

## Fix 4 — Template matching: centroid alignment + ambiguity flag

**Files:** `analytics/formations/matching.py`, `analytics/formations/detector.py`, `core/config.py`

**Problem:** average in-possession shapes are ~20m narrower than textbook templates (players don't hold kickoff-style width), and template costs were inflated by positional offset. Measured old `avgCostPerPlayer` was **8.9–34.3m**, median ~30m, with best-vs-second-best margins ~0.2m (1.4%) — labels were effectively "least-wrong of 53 templates" chosen by noise, producing 14 different formations in one match.

**Fix:**
- `match_formation(..., align=True)`: each template is translated so its centroid coincides with the players' centroid before Hungarian assignment, so the cost measures shape mismatch only.
- The matcher now also returns the runner-up cost; the detector writes `costMargin = second − best` and `ambiguous = costMargin < AMBIGUOUS_MARGIN_M (0.75m)` so downstream analysis can restrict to windows where the label has a real edge.
- New columns: `costMargin`, `ambiguous`, `fitQuality`, `rawWeightSum`.

**Measured effect (3816):** median cost ~30m → **11.4m** (best 8.3); margin median 0.74m with **50% of windows flagged ambiguous**; formation variety per match fell (11 distinct labels, most flagged). 3818: median cost 10.9m, 59% ambiguous.

**Honest limitation:** even aligned, textbook-template costs remain ~8–17m/player because real fluid shapes are narrower and more central than any template. The `ambiguous` flag is the mechanism to express that; analysis conclusions should filter on `ambiguous == False` (or treat labels as low-trust). A research-grade improvement (line clustering / team-vs-team shape similarity instead of textbook templates) is out of scope for this phase.

## Fix 5 — No fabricated possession stats for event-less matches

**File:** `analytics/formations/segments.py`

**Bug:** when a match had no events (or events without usable possession sequences), `calculate_possession_overlap` received an empty list and reported **100% loose ball / 0% possession / 0 turnovers** for every segment, silently.

**Fix:** when the sequence list is empty, possession columns are written as `NaN` and a warning is logged. Also discovered during verification: match 3818's events carry `sequence = None` on all OTB rows (provider quirk), so event possession is genuinely unavailable there; the segments now say so instead of fabricating numbers.

## Fix 6 — EPV per-second buckets average valid frames only

**Files:** `analytics/epv/momentum.py`, `analytics/epv/das.py`

**Bug:** `bucket_epv_by_second` averaged over *all* frames in a second, including frames with no ball position or no owner where EPV is forced to 0. In this data 37.7% of frames have no ball and only 32% have a valid owner, so per-second means were systematically diluted.

**Fix:** the bucket mean is computed over valid frames only (`ball tracked` and `owner != 0`), reported via a new `nValidFrames` column. This makes `meanSignedEPV` the true average EPV during tracked possession.

**Measured effect (3816):** `epv_timeseries.csv` regenerated (6612 rows) with `nValidFrames` 0–45 per second (median 0 — reflects genuinely sparse ball tracking in this data, now visible instead of hidden). DAS: ARG 14, KSA 2 (matches the match narrative: Argentina dominated danger).

## Fix 7 — Formation names written as strings

**Files:** `analytics/formations/detector.py`, `analytics/formations/taxonomy.py`

**Bug exposed by regeneration:** all detected formation names in the new runs are digit-only (e.g. `1432`, `1234`, `3331`), so pandas read `formations.csv` with an `int64` column and `derive_hierarchy` crashed on `.removesuffix`. Pre-existing latent bug (earlier CSVs contained mixed names like `metodo`/`3421flat`, masking it).

**Fix:** detector writes `str(formation)`; `derive_hierarchy` coerces with `str()` for robustness. `formation` column is now `object`/`str` in the CSVs.

## Regenerated outputs

- `Processed_Tracking/3816/formations.csv` (112 rows), `formation_segments.csv` (38 segments, real possession columns), `epv_timeseries.csv`, `das_sequences.csv`
- `Processed_Tracking/3818/formations.csv` (114 rows), `formation_segments.csv` (47 segments, possession = NaN with warning)

## Summary table (match 3816, before → after)

| Quantity | Before | After |
|---|---|---|
| Turnovers detected | 0 (coordinate bug) | 497 |
| OTB disruption spans (P1) | 1 whole-period span | 87 short spans |
| Weight time-profile | monotonic ramp 0→1 | fluctuates 0.33–0.88 |
| `confidence` range | 16.4–2170.1 (filter dead) | 0.302–0.789 (filter active) |
| Detection rows | 180 | 112 (dead-ball windows dropped) |
| `avgCostPerPlayer` median | ~30m | 11.4m |
| `costMargin` median | ~0.2m (noise) | 0.74m |
| `ambiguous` | n/a | 50% of windows flagged |
| Event-less segments | 100% loose ball | NaN + warning |
| `meanSignedEPV` buckets | diluted by forced zeros | valid-frames mean + `nValidFrames` |

## Constants changed

| Constant | Old | New | Location |
|---|---|---|---|
| `OTB_SPAN_GAP_SECONDS` | — (no gap check) | 15.0 | frame_reliability.py |
| `MIN_OWNER_PERSIST_FRAMES` | — (no debounce) | 8 | frame_reliability.py |
| `MIN_WINDOW_CONFIDENCE` | 0.5 (inert) | 0.3 | frame_reliability.py |
| `AMBIGUOUS_MARGIN_M` | — | 0.75 | core/config.py |

## Known remaining limitations (not in this phase)

- Formation labels remain low-trust on real fluid shapes (see Fix 4 note); consider line-based shape detection.
- Turnover counts (~500/match) still include contested spells; acceptable for a weight heuristic.
- DAS threshold (0.15) and decay constants remain uncalibrated against shot/goal ground truth.
- The template-vs-reality gap suggests evaluating labels only where `ambiguous == False`.
