# TRANSITIONS Refactor Summary

## What Changed

TRANSITIONS was reorganized from a set of flat scripts into a package-first research framework under `TRANSITIONS/`.

### Package structure added

- `TRANSITIONS/core/` for shared configuration, constants, logging, and exceptions.
- `TRANSITIONS/io/` for shared file/path helpers and processed-match loading.
- `TRANSITIONS/analytics/` for formation, possession, EPV, and future analytics modules.
- `TRANSITIONS/artifacts/` for typed return objects such as `FormationResult` and `EPVResult`.
- `TRANSITIONS/pipeline/` for orchestration helpers.
- `TRANSITIONS/cli/` for the unified command line interface.
- `TRANSITIONS/ui/` for the PyQt6 app window, viewer, and timeline helpers.

### Shared model layer

Added shared dataclasses for the project vocabulary:

- `Match`
- `Frame`
- `PlayerPosition`
- `FormationWindow`
- `PossessionSequence`
- `EPVPoint`
- `DangerousAttackingSequence`

### Centralized behavior

- Configuration defaults now live in `TRANSITIONS/core/config.py`.
- File and folder locations now live in `TRANSITIONS/io/paths.py`.
- Logging helpers were added in `TRANSITIONS/core/logger.py`.
- A shared processed-match loader was added in `TRANSITIONS/io/loader.py`.

### Analytics extraction

- Formation logic was split into formation templates, goalkeeper inference, matching, and windowing helpers.
- Possession logic was split into tracking-based, event-based, and sequence helpers.
- EPV/DAS logic was split into momentum helpers and DAS evaluation/plotting helpers.

### Artifacts

Typed artifact builders were added so analytics outputs can be consumed as structured objects instead of only raw dataframes.

### Compatibility

The original top-level scripts remain as compatibility wrappers so existing commands still work while the package structure is adopted.

## Current Note On Top-Level Files

Some files still exist at the repository root by design:

- `detect_formation.py`
- `epv_das_analysis.py`
- `plot_formation_timeline.py`
- `possession.py`
- `preprocessing.py`
- `visualize_match.py`

These are thin wrappers around the package modules and are kept to preserve backward compatibility during the migration.
