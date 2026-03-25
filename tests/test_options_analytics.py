import json
import os
import tempfile
import unittest

import pandas as pd

import options_analytics as oa


class OptionsAnalyticsTests(unittest.TestCase):
    def test_max_pain_prefers_balancing_strike(self):
        calls = pd.DataFrame(
            {
                "strike": [100.0, 105.0, 110.0],
                "openInterest": [10.0, 50.0, 5.0],
            }
        )
        puts = pd.DataFrame(
            {
                "strike": [100.0, 105.0, 110.0],
                "openInterest": [5.0, 45.0, 15.0],
            }
        )
        self.assertEqual(oa._max_pain(calls, puts), 105.0)

    def test_get_options_snapshot_uses_fresh_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oa.DATA_DIR
            try:
                oa.DATA_DIR = tmp
                payload = {
                    "status": "ok",
                    "symbol": "QQQ",
                    "expiry": "2026-03-27",
                    "expected_move_dollars": 8.5,
                    "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
                with open(os.path.join(tmp, "options_snapshot_QQQ.json"), "w") as f:
                    json.dump(payload, f)
                snap = oa.get_options_snapshot("QQQ", spot_price=500.0, max_age_sec=3600)
                self.assertEqual(snap.get("source"), "cache")
                self.assertEqual(snap.get("expected_move_dollars"), 8.5)
            finally:
                oa.DATA_DIR = old_data_dir

    def test_get_options_snapshot_caches_failures_for_cooldown(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oa.DATA_DIR
            old_fetch = oa._fetch_remote_snapshot
            try:
                oa.DATA_DIR = tmp
                calls = {"n": 0}

                def _fail(symbol, spot_price=None):
                    calls["n"] += 1
                    return {"status": "error", "reason": "network_down"}

                oa._fetch_remote_snapshot = _fail
                first = oa.get_options_snapshot("QQQ", spot_price=500.0, max_age_sec=3600)
                second = oa.get_options_snapshot("QQQ", spot_price=500.0, max_age_sec=3600)
                self.assertEqual(first.get("status"), "error")
                self.assertEqual(second.get("source"), "cache_failure")
                self.assertEqual(calls["n"], 1)
            finally:
                oa.DATA_DIR = old_data_dir
                oa._fetch_remote_snapshot = old_fetch

    def test_get_options_snapshot_computes_iv_history_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oa.DATA_DIR
            try:
                oa.DATA_DIR = tmp
                hist = pd.DataFrame(
                    {
                        "fetched_at": pd.date_range("2026-01-01", periods=5, freq="7D", tz="UTC"),
                        "atm_iv_mid": [0.18, 0.20, 0.24, 0.26, 0.30],
                        "expected_move_pct": [0.01, 0.011, 0.012, 0.013, 0.014],
                    }
                )
                hist.to_csv(os.path.join(tmp, "options_snapshot_history_QQQ.csv"), index=False)
                daily = pd.DataFrame(
                    {"close": [100, 101, 100, 102, 103, 104, 103, 105, 106, 107, 108]},
                    index=pd.date_range("2026-01-01", periods=11, freq="B"),
                )
                daily.to_csv(os.path.join(tmp, "prices_QQQ_daily_long.csv"))
                payload = {
                    "status": "ok",
                    "symbol": "QQQ",
                    "expiry": "2026-03-27",
                    "expected_move_dollars": 8.5,
                    "expected_move_pct": 0.012,
                    "atm_call_iv": 0.28,
                    "atm_put_iv": 0.32,
                    "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
                }
                with open(os.path.join(tmp, "options_snapshot_QQQ.json"), "w") as f:
                    json.dump(payload, f)
                snap = oa.get_options_snapshot("QQQ", spot_price=500.0, max_age_sec=3600)
                self.assertEqual(snap.get("source_kind"), "cache")
                self.assertIsNotNone(snap.get("iv_rank"))
                self.assertIsNotNone(snap.get("iv_percentile"))
                self.assertIsNotNone(snap.get("iv_vs_rv20"))
            finally:
                oa.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()
