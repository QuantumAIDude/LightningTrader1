import unittest

import pandas as pd

import symbol_pipeline as sp


class SymbolPipelineTests(unittest.TestCase):
    def test_assess_symbol_history_reports_partial_horizon_eligibility(self):
        old_build_features = sp.build_features_for_training
        old_build_daily = sp.build_next_session_dataset
        old_build_weekend = sp.build_weekend_session_dataset
        old_load_daily_long = sp.load_daily_long_history
        try:
            sp.build_features_for_training = lambda df, instrument_key=None: pd.DataFrame({"x": range(120)})
            sp.build_next_session_dataset = lambda df, instrument_key=None: pd.DataFrame({"x": range(16)})
            sp.build_weekend_session_dataset = lambda df, instrument_key=None: pd.DataFrame({"x": range(4)})
            sp.load_daily_long_history = lambda symbol: pd.DataFrame({"close": [1.0] * 40}, index=pd.date_range("2026-01-01", periods=40, tz="UTC"))

            df = pd.DataFrame(
                {"close": [100.0] * 200, "open": [100.0] * 200, "high": [101.0] * 200, "low": [99.0] * 200, "volume": [1000] * 200},
                index=pd.date_range("2026-03-01 14:30:00+00:00", periods=200, freq="5min"),
            )
            out = sp.assess_symbol_history(df, instrument_key="MSFT")

            self.assertEqual(out["status_key"], "advisory_core_eligible")
            self.assertEqual(out["eligible_modes"], ["4min", "DailyHigh"])
            self.assertEqual(out["paths"]["WeekEnd"]["status"], "partial")
            self.assertIn("weekend path reaches 6 usable rows", out["primary_next_requirement"])
        finally:
            sp.build_features_for_training = old_build_features
            sp.build_next_session_dataset = old_build_daily
            sp.build_weekend_session_dataset = old_build_weekend
            sp.load_daily_long_history = old_load_daily_long

    def test_build_training_plan_distinguishes_eligible_deferred_and_blocked_modes(self):
        history = {
            "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
            "paths": {
                "4min": {"feature_rows": 80, "feature_ready": True, "train_eligible": False, "detail": "needs more intraday rows"},
                "DailyHigh": {"feature_rows": 17, "feature_ready": True, "train_eligible": True, "detail": "ready"},
                "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "detail": "missing"},
            },
        }

        out = sp.build_training_plan(history, train_requested=True)

        self.assertEqual(out["status"], "partial")
        self.assertEqual(out["eligible_modes"], ["DailyHigh"])
        self.assertEqual(out["deferred_modes"], ["4min"])
        self.assertEqual(out["blocked_modes"], ["WeekEnd"])
        self.assertEqual(out["mode_requests"]["DailyHigh"]["status"], "eligible")
        self.assertEqual(out["mode_requests"]["4min"]["status"], "deferred")
        self.assertEqual(out["mode_requests"]["WeekEnd"]["status"], "blocked")

    def test_ensure_daily_long_history_accumulates_over_repeated_runs(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = sp.DATA_DIR
            try:
                sp.DATA_DIR = tmp
                idx_one = pd.date_range("2026-03-01 14:30:00+00:00", periods=4, freq="5min")
                df_one = pd.DataFrame(
                    {
                        "open": [100.0, 101.0, 102.0, 103.0],
                        "high": [101.0, 102.0, 103.0, 104.0],
                        "low": [99.0, 100.0, 101.0, 102.0],
                        "close": [100.5, 101.5, 102.5, 103.5],
                        "volume": [1000, 1200, 1400, 1600],
                    },
                    index=idx_one,
                )
                first_df, first_summary = sp.ensure_daily_long_history("MSFT", recent_df=df_one, persist=True)

                idx_two = pd.date_range("2026-03-02 14:30:00+00:00", periods=4, freq="5min")
                df_two = pd.DataFrame(
                    {
                        "open": [104.0, 105.0, 106.0, 107.0],
                        "high": [105.0, 106.0, 107.0, 108.0],
                        "low": [103.0, 104.0, 105.0, 106.0],
                        "close": [104.5, 105.5, 106.5, 107.5],
                        "volume": [1800, 1900, 2000, 2100],
                    },
                    index=idx_two,
                )
                second_df, second_summary = sp.ensure_daily_long_history("MSFT", recent_df=df_two, persist=True)

                self.assertEqual(len(first_df), 1)
                self.assertEqual(len(second_df), 2)
                self.assertTrue(os.path.exists(os.path.join(tmp, "prices_MSFT_daily_long.csv")))
                self.assertTrue(first_summary["updated"])
                self.assertTrue(second_summary["updated"])
                self.assertIn("recent_runtime", second_summary["sources_used"])
                self.assertEqual(second_summary["rows"], 2)
            finally:
                sp.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()
