from __future__ import annotations

import bz2
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from Transitions.domain import Match
from Transitions.io.loader import load_processed_match


class PipelineTests(unittest.TestCase):
    def test_load_processed_match_builds_match_dataclass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            match_dir = root / "123"
            match_dir.mkdir(parents=True)

            metadata = {
                "id": 123,
                "pitch": {"length": 105.0, "width": 68.0},
                "homeTeamStartLeft": True,
                "homeTeam": {"shortName": "Home"},
                "awayTeam": {"shortName": "Away"},
            }
            (match_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

            frames = [
                {
                    "period": 1,
                    "periodElapsedTime": 0.0,
                    "homePlayers": [{"playerId": 1, "x": 1.0, "y": 2.0}],
                    "awayPlayers": [{"playerId": 2, "x": 3.0, "y": 4.0}],
                    "balls": [{"x": 5.0, "y": 6.0}],
                }
            ]
            with bz2.open(match_dir / "tracking.jsonl.bz2", "wt") as handle:
                for frame in frames:
                    handle.write(json.dumps(frame))
                    handle.write("\n")

            (match_dir / "formations.csv").write_text(
                "matchId,team,period,windowIndex,windowStartSec,windowEndSec,nOutfieldPlayers,nFrames,formation,orientation,avgCostPerPlayer\n"
                "123,home,1,0,0,180,10,1,433,normal,0.0\n",
                encoding="utf-8",
            )

            loaded = load_processed_match(123, root)

            self.assertIsInstance(loaded, Match)
            self.assertEqual(loaded.metadata["id"], 123)
            self.assertEqual(loaded.tracking["pitch_length"], 105.0)
            self.assertIsInstance(loaded.tracking["formations_df"], pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
