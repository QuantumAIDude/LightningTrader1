import json
import os
import tempfile
import unittest

import pandas as pd

import readiness_tracker as rt


class ReadinessTrackerTests(unittest.TestCase):
    def test_compute_pass_streak_counts_consecutive_days(self):
        rows = [
            {"date": "2026-03-01", "passed": True, "ts": "2026-03-01T20:00:00Z"},
            {"date": "2026-03-02", "passed": True, "ts": "2026-03-02T20:00:00Z"},
            {"date": "2026-03-03", "passed": False, "ts": "2026-03-03T20:00:00Z"},
            {"date": "2026-03-04", "passed": True, "ts": "2026-03-04T20:00:00Z"},
            {"date": "2026-03-05", "passed": True, "ts": "2026-03-05T20:00:00Z"},
        ]
        streak = rt._compute_pass_streak(rows)
        self.assertEqual(streak, 2)

    def test_load_readiness_state_marks_stale_as_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            try:
                rt.DATA_DIR = tmp
                instr = "QQQ"
                path = os.path.join(tmp, f"readiness_state_{instr}.json")
                stale_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=100)).isoformat()
                with open(path, "w") as f:
                    json.dump(
                        {
                            "instrument": instr,
                            "last_update": stale_ts,
                            "live_ready": True,
                            "pass_streak_days": 9,
                            "required_streak_days": 3,
                        },
                        f,
                    )
                state = rt.load_readiness_state(instr)
                self.assertIsNotNone(state)
                self.assertFalse(bool(state.get("fresh", True)))
                self.assertFalse(bool(state.get("live_ready", True)))
            finally:
                rt.DATA_DIR = old_data_dir

    def test_walk_forward_cache_invalidates_when_model_signature_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_model_sig = rt._models_dir_signature
            try:
                rt.DATA_DIR = tmp
                rt._models_dir_signature = lambda: [("model.pkl", 1, 10)]
                df = pd.DataFrame(
                    {"close": [100.0, 101.0, 102.0]},
                    index=pd.to_datetime(
                        ["2026-03-01T14:30:00Z", "2026-03-01T14:31:00Z", "2026-03-01T14:32:00Z"],
                        utc=True,
                    ),
                )
                df.to_csv(os.path.join(tmp, "prices_QQQ.csv"))
                metrics = {"folds": 3, "price_window_ratio": [1.0]}
                rt._save_cached_walk_forward("QQQ", df, 3, metrics)
                self.assertEqual(rt._load_cached_walk_forward("QQQ", df, 3), metrics)

                rt._models_dir_signature = lambda: [("model.pkl", 2, 10)]
                self.assertEqual(rt._load_cached_walk_forward("QQQ", df, 3), {})
            finally:
                rt.DATA_DIR = old_data_dir
                rt._models_dir_signature = old_model_sig

    def test_horizon_replay_directional_accuracy_excludes_perfect_splits(self):
        idx = pd.date_range("2026-03-01T14:30:00Z", periods=305, freq="min", tz="UTC")
        prices = pd.DataFrame({"close": [100.0] * len(idx)}, index=idx)
        ds_daily = pd.DataFrame(
            {
                "close": [100.0],
                "target_high_ret": [0.01],
                "target_low_ret": [-0.01],
            },
            index=pd.DatetimeIndex([idx[-1]], tz="UTC"),
        )
        ds_week = pd.DataFrame(
            {
                "close": [100.0],
                "target_weekend_ret": [0.01],
            },
            index=pd.DatetimeIndex([idx[-1]], tz="UTC"),
        )
        old_load_models = rt.load_models
        old_predict_horizon = rt.predict_horizon
        old_build_next_session_dataset = rt.build_next_session_dataset
        old_build_weekend_session_dataset = rt.build_weekend_session_dataset
        try:
            rt.load_models = lambda include_modes=True, instrument_key=None: {"ok": True}
            rt.build_next_session_dataset = lambda df, instrument_key=None: ds_daily
            rt.build_weekend_session_dataset = lambda df, instrument_key=None: ds_week

            def _fake_predict_horizon(models, subset, instrument_key=None, mode=None, log_side_effects=False):
                if mode == "DailyHigh":
                    return {
                        "high": {
                            "base": {"pred_price": 101.0},
                            "forest": {"pred_price": 100.8},
                            "boost": {"pred_price": 99.2},
                            "extra_trees": {"pred_price": 99.0},
                        },
                        "low": {
                            "base": {"pred_price": 98.5},
                            "forest": {"pred_price": 98.8},
                            "boost": {"pred_price": 99.1},
                            "extra_trees": {"pred_price": 101.0},
                        },
                    }
                return {
                    "week": {
                        "base": {"pred_price": 101.2},
                        "forest": {"pred_price": 101.0},
                        "boost": {"pred_price": 100.9},
                        "extra_trees": {"pred_price": 99.8},
                    }
                }

            rt.predict_horizon = _fake_predict_horizon
            out = rt._horizon_replay_errors(prices, "QQQ", max_daily_samples=1, max_week_samples=1)
            self.assertEqual(out["daily_directional_samples"], 1)
            self.assertEqual(out["daily_directional_ties"], 1)
            self.assertAlmostEqual(out["daily_directional_acc"], 1.0, places=6)
            self.assertEqual(out["weekend_directional_samples"], 1)
            self.assertEqual(out["weekend_directional_ties"], 0)
            self.assertAlmostEqual(out["weekend_directional_acc"], 1.0, places=6)
        finally:
            rt.load_models = old_load_models
            rt.predict_horizon = old_predict_horizon
            rt.build_next_session_dataset = old_build_next_session_dataset
            rt.build_weekend_session_dataset = old_build_weekend_session_dataset


if __name__ == "__main__":
    unittest.main()
