from __future__ import annotations

import unittest

import numpy as np

from Transitions.analytics.formations.detector import get_orientation
from Transitions.analytics.formations.matching import match_formation


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


if __name__ == "__main__":
    unittest.main()
