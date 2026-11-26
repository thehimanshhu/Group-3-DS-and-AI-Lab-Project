import unittest

import numpy as np

from src.compare import holm, interval, p_value, paired_bootstrap


class CompareTests(unittest.TestCase):
    def test_holm_matches_hand_computed_values(self):
        # sorted p: 0.01 (x4) = 0.04, 0.02 (x3) = 0.06, 0.03 (x2) = 0.06 (kept monotone), 0.5 (x1) = 0.5
        adjusted, rejected = holm([0.03, 0.01, 0.5, 0.02])
        np.testing.assert_allclose(adjusted, [0.06, 0.04, 0.5, 0.06])
        self.assertEqual(rejected.tolist(), [False, True, False, False])

    def test_bootstrap_sees_a_clear_improvement(self):
        rng = np.random.default_rng(0)
        y = np.repeat([0, 1], 300)
        weak = y + rng.normal(0, 0.8, y.size)    # overlapping classes
        strong = y + rng.normal(0, 0.2, y.size)  # nearly separated
        diff, diffs = paired_bootstrap(y, weak, strong, resamples=300)
        low, high = interval(diffs)
        self.assertLess(diff, 0)
        self.assertLess(high, 0)
        self.assertLess(p_value(diffs), 0.01)

    def test_identical_runs_give_no_difference(self):
        y = np.repeat([0, 1], 50)
        s = np.linspace(0, 1, 100)
        diff, diffs = paired_bootstrap(y, s, s, resamples=50)
        self.assertEqual(diff, 0)
        self.assertEqual(p_value(diffs), 1.0)


if __name__ == "__main__":
    unittest.main()
