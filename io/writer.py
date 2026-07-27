"""Simple writing helpers used by the package scaffolding."""

from __future__ import annotations

import bz2
import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any, indent: int = 4) -> None:
    """Write a JSON document to disk."""

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent)


def write_jsonl_bz2(path: str | Path, rows: list[Any]) -> None:
    """Write a JSONL payload into a bz2-compressed file."""

    with bz2.open(path, "wt") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")
