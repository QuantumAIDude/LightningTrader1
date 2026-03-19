import unittest

from accuracy_windows import (
    majority_direction_vote,
    price_window_threshold_dollars,
    summarize_majority_direction_hits,
    summarize_price_window_errors,
)


class AccuracyWindowTests(unittest.TestCase):
    def test_intraday_window_uses_absolute_floor(self):
        self.assertAlmostEqual(price_window_threshold_dollars("intraday", reference_price=100.0), 0.15, places=6)

    def test_weekend_window_scales_with_price(self):
        self.assertAlmostEqual(price_window_threshold_dollars("weekend", reference_price=1000.0), 2.5, places=6)

    def test_summary_reports_hit_rate_and_ratio(self):
        out = summarize_price_window_errors(
            [0.10, 0.20, 0.05],
            actual_prices=[100.0, 100.0, 100.0],
            horizon="intraday",
        )
        self.assertEqual(out["n"], 3)
        self.assertAlmostEqual(out["mae_dollars"], (0.10 + 0.20 + 0.05) / 3.0, places=6)
        self.assertAlmostEqual(out["window_hit_rate"], 2.0 / 3.0, places=6)
        self.assertGreater(out["window_ratio_mean"], 0.0)

    def test_majority_direction_vote_excludes_perfect_split(self):
        out = majority_direction_vote([101.0, 100.8, 99.2, 99.0], 100.0)
        self.assertIsNone(out["direction"])
        self.assertEqual(out["ties_excluded"], 1)

    def test_summarize_majority_direction_hits_ignores_ties(self):
        out = summarize_majority_direction_hits(
            [
                [101.0, 101.0, 99.0],
                [100.8, 101.1, 99.1],
                [99.2, 101.2, 101.0],
                [99.0, 99.1, 98.8],
            ],
            actual_prices=[101.2, 101.3, 99.1],
            reference_prices=[100.0, 100.0, 100.0],
        )
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["ties_excluded"], 1)
        self.assertAlmostEqual(out["accuracy"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
