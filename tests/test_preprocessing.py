from __future__ import annotations

import unittest

from Transitions.preprocessing import preprocess


class PreprocessingSmokeTests(unittest.TestCase):
    def test_process_metadata_exists(self) -> None:
        self.assertTrue(callable(preprocess.process_metadata))
        self.assertTrue(callable(preprocess.process_tracking))


if __name__ == "__main__":
    unittest.main()
