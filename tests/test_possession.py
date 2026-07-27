from __future__ import annotations

import unittest

import numpy as np

from Transitions.analytics.possession import attack_direction, detect_possession_sequences, forward_fill_owner


class PossessionTests(unittest.TestCase):
    def test_attack_direction_uses_period_parity(self) -> None:
        self.assertEqual(attack_direction("home", 1, 1, -1), 1)
        self.assertEqual(attack_direction("home", 2, 1, -1), -1)

    def test_forward_fill_owner_fills_short_gap(self) -> None:
        owner = np.array([1, 1, 0, 1, 1], dtype=np.int8)
        periods = np.array([1, 1, 1, 1, 1], dtype=np.int16)
        elapsed = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)

        filled = forward_fill_owner(owner, periods, elapsed, max_gap_seconds=3.0)

        self.assertTrue(np.array_equal(filled, np.array([1, 1, 1, 1, 1], dtype=np.int8)))

    def test_detect_possession_sequences_runs(self) -> None:
        owner = np.array([1, 1, 2, 2], dtype=np.int8)
        periods = np.array([1, 1, 1, 1], dtype=np.int16)
        elapsed = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)

        sequences = detect_possession_sequences(owner, periods, elapsed, fps=1.0)

        self.assertEqual(len(sequences), 2)
        self.assertEqual(sequences[0]["team"], "home")
        self.assertEqual(sequences[1]["team"], "away")


if __name__ == "__main__":
    unittest.main()
