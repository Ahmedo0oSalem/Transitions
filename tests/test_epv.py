from __future__ import annotations

import unittest

import numpy as np

from Transitions.analytics.epv.das import bucket_epv_by_second
from Transitions.analytics.possession import attack_direction, epv_value, get_base_directions, load_epv_grid


class EPVTests(unittest.TestCase):
    def test_epv_value_uses_direction(self) -> None:
        grid = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
        self.assertAlmostEqual(epv_value(grid, 0.0, 0.0, 10.0, 10.0, 1), 0.0)
        self.assertAlmostEqual(epv_value(grid, 0.0, 0.0, 10.0, 10.0, -1), 1.0)

    def test_bucket_epv_by_second_groups_rows(self) -> None:
        periods = np.array([1, 1, 1, 2], dtype=np.int16)
        elapsed = np.array([0.1, 0.9, 1.2, 0.2], dtype=np.float32)
        signed_epv = np.array([1.0, 3.0, 5.0, 7.0], dtype=np.float32)

        df = bucket_epv_by_second(periods, elapsed, signed_epv)

        self.assertEqual(len(df), 3)
        self.assertIn(1, df["period"].values)


if __name__ == "__main__":
    unittest.main()
