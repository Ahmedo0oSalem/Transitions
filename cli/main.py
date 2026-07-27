"""Unified TRANSITIONS command line interface.

The current implementation is a compatibility shim over the existing
scripts so the package can be invoked immediately without changing
analytical behavior.
"""

from __future__ import annotations

import argparse

from ..io.paths import EPV_GRID_PATH, PROCESSED_DIR


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level TRANSITIONS CLI parser."""

    parser = argparse.ArgumentParser(prog="TRANSITIONS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("preprocess", help="Run preprocessing for all matches.")

    formations = subparsers.add_parser("formations", help="Detect formations for one or more matches.")
    formations.add_argument("match_ids", nargs="*", help="Match IDs to process.")
    formations.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    formations.add_argument("--window-seconds", type=int, default=None)
    formations.add_argument("--stride-seconds", type=int, default=None)

    epv = subparsers.add_parser("epv", help="Run EPV/DAS analysis for one match.")
    epv.add_argument("match_id")
    epv.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    epv.add_argument("--epv-grid", default=str(EPV_GRID_PATH))

    timeline = subparsers.add_parser("timeline", help="Plot the formation timeline for one match.")
    timeline.add_argument("match_id")
    timeline.add_argument("--processed-dir", default=str(PROCESSED_DIR))

    viewer = subparsers.add_parser("viewer", help="Launch the interactive viewer for one match.")
    viewer.add_argument("match_id")
    viewer.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    viewer.add_argument("--speed", type=float, default=1.0)

    run = subparsers.add_parser("run", help="Run the full pipeline for one match.")
    run.add_argument("match_id")
    run.add_argument("--processed-dir", default=str(PROCESSED_DIR))
    run.add_argument("--epv-grid", default=str(EPV_GRID_PATH))
    run.add_argument("--speed", type=float, default=1.0)

    subparsers.add_parser("ui", help="Launch the PyQt6 desktop app.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for `python -m TRANSITIONS`."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        from ..pipeline.runner import preprocess_all_matches

        preprocess_all_matches()
        return

    if args.command == "formations":
        from ..pipeline.runner import detect_formations

        detect_formations(
            match_ids=args.match_ids or None,
            processed_dir=args.processed_dir,
            window_seconds=args.window_seconds,
            stride_seconds=args.stride_seconds,
        )
        return

    if args.command == "epv":
        from ..pipeline.runner import run_epv

        run_epv(args.match_id, args.processed_dir, args.epv_grid)
        return

    if args.command == "timeline":
        from ..pipeline.runner import run_timeline

        run_timeline(args.match_id, args.processed_dir)
        return

    if args.command == "viewer":
        from ..pipeline.runner import run_viewer

        run_viewer(args.match_id, args.processed_dir, speed=args.speed)
        return

    if args.command == "run":
        from ..pipeline.runner import detect_formations, preprocess_all_matches, run_epv, run_timeline, run_viewer

        preprocess_all_matches()
        detect_formations([args.match_id], processed_dir=args.processed_dir)
        run_epv(args.match_id, args.processed_dir, args.epv_grid)
        run_timeline(args.match_id, args.processed_dir)
        run_viewer(args.match_id, args.processed_dir, speed=args.speed)
        return

    if args.command == "ui":
        from ..ui.main_window import main as ui_main

        ui_main()
