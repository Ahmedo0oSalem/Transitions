from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Transitions.analytics.formations.detector import get_orientation
from Transitions.analytics.formations.matching import match_formation
from Transitions.analytics.formations.segments import _aggregate_obso_for_segment


class FormationTests(unittest.TestCase):
    def test_get_orientation_flips_by_period(self) -> None:
        self.assertEqual(get_orientation("homePlayers", 1, True), "normal")
        self.assertEqual(get_orientation("homePlayers", 2, True), "flipped")
        self.assertEqual(get_orientation("awayPlayers", 1, True), "flipped")

    def test_match_formation_returns_best_template(self) -> None:
        player_xy = np.array([[0.0, 0.0], [1.0, 1.0]])
        templates = {
            "test": {
                "names": ["A", "B"],
                "normal": np.array([[0.0, 0.0], [1.0, 1.0]]),
                "flipped": np.array([[2.0, 2.0], [3.0, 3.0]]),
            }
        }

        formation, cost, names = match_formation(player_xy, templates, "normal")

        self.assertEqual(formation, "test")
        self.assertEqual(names, ["A", "B"])
        self.assertAlmostEqual(cost, 0.0)

    def test_aggregate_obso_for_segment_returns_team_mean(self) -> None:
        obso_df = pd.DataFrame(
            [
                {"period": 1, "elapsed": 10, "team": "home", "obso": 0.4},
                {"period": 1, "elapsed": 20, "team": "home", "obso": 0.8},
                {"period": 1, "elapsed": 30, "team": "away", "obso": 1.0},
            ]
        )

        props = _aggregate_obso_for_segment(obso_df, "home", 1, 0, 30)

        self.assertEqual(props["mean_obso"], 0.6)

    def test_2d_team_filter_normalizes_home_and_away(self) -> None:
        from Transitions.ui.two_d_analysis import _normalize_team_filter, _normalize_team_x_for_plot

        self.assertEqual(_normalize_team_filter("Home"), "home")
        self.assertEqual(_normalize_team_filter("Away"), "away")
        self.assertEqual(_normalize_team_filter("All"), "all")
        self.assertEqual(_normalize_team_filter(" home "), "home")
        self.assertEqual(_normalize_team_x_for_plot(20.0, "away", 1, True, 105.0), 85.0)
        self.assertEqual(_normalize_team_x_for_plot(20.0, "home", 1, False, 105.0), 85.0)


if __name__ == "__main__":
    unittest.main()
