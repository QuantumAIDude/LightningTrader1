import unittest

import pandas as pd

from model_trainers import _choose_mode_training_frame, _mode_recency_half_life_days


class TestModelTrainersLongHistory(unittest.TestCase):
    def test_choose_mode_training_frame_prefers_long_history_for_long_horizons(self):
        recent = pd.DataFrame({"close": [1.0, 2.0]})
        long_df = pd.DataFrame({"close": [10.0, 20.0, 30.0]})

        chosen, source = _choose_mode_training_frame(recent, long_df, "daily_high")

        self.assertEqual(source, "daily_long_history")
        self.assertTrue(chosen.equals(long_df))

    def test_choose_mode_training_frame_keeps_intraday_on_recent_history(self):
        recent = pd.DataFrame({"close": [1.0, 2.0]})
        long_df = pd.DataFrame({"close": [10.0, 20.0, 30.0]})

        chosen, source = _choose_mode_training_frame(recent, long_df, "4min")

        self.assertEqual(source, "recent_intraday")
        self.assertTrue(chosen.equals(recent))

    def test_mode_recency_half_life_scales_with_horizon(self):
        self.assertEqual(_mode_recency_half_life_days("4min", "recent_intraday"), 7.0)
        self.assertEqual(_mode_recency_half_life_days("daily_high", "daily_long_history"), 120.0)
        self.assertEqual(_mode_recency_half_life_days("weekend", "daily_long_history"), 180.0)


if __name__ == "__main__":
    unittest.main()
