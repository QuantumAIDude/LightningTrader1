import unittest

import pandas as pd

import horizon_overlay as hover
from horizon_overlay import (
    _bucket_edges,
    _bucket_label,
    _date_key_from_target_key,
    _daily_outcome_specs,
    _normalize_daily_history,
    _search_price_blend,
    _weekend_outcome_specs,
)


class HorizonOverlayTests(unittest.TestCase):
    def test_date_key_from_target_key_handles_iso_ts(self):
        self.assertEqual(_date_key_from_target_key("2026-03-11T20:58:00+00:00"), "2026-03-11")
        self.assertEqual(_date_key_from_target_key("2026-03-11"), "2026-03-11")

    def test_normalize_daily_history_flattens_columns(self):
        cols = pd.MultiIndex.from_tuples([("Open", "QQQ"), ("High", "QQQ"), ("Low", "QQQ"), ("Close", "QQQ"), ("Volume", "QQQ")])
        df = pd.DataFrame([[1.0, 2.0, 0.5, 1.5, 10.0]], columns=cols, index=[pd.Timestamp("2026-03-11")])
        out = _normalize_daily_history(df)
        self.assertListEqual(list(out.columns[:5]), ["open", "high", "low", "close", "volume"])
        self.assertEqual(len(out), 1)

    def test_search_price_blend_prefers_better_direct_signal(self):
        base = [101.0, 99.0, 104.0, 96.0, 103.0]
        direct = [100.1, 100.2, 100.0, 99.9, 100.3]
        actual = [100.0, 100.0, 100.0, 100.0, 100.0]
        blend = _search_price_blend(base, direct, actual, horizon="weekend", instrument_key="QQQ")
        self.assertTrue(blend["enabled"])
        self.assertGreater(blend["weight"], 0.0)
        self.assertLess(
            float(blend["selected_metrics"]["mae_dollars"]),
            float(blend["baseline_metrics"]["mae_dollars"]),
        )

    def test_bucket_helpers_partition_regimes(self):
        edges = _bucket_edges([0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6], 2)
        self.assertEqual(len(edges), 1)
        self.assertEqual(_bucket_label(0.15, edges), "bucket_0")
        self.assertEqual(_bucket_label(0.55, edges), "bucket_1")

    def test_daily_outcome_specs_enable_2x2_regime_grid(self):
        dataset = pd.DataFrame(
            {
                "vol_20": [
                    0.10, 0.11, 0.12, 0.13,
                    0.10, 0.11, 0.12, 0.13,
                    0.30, 0.31, 0.32, 0.33,
                    0.30, 0.31, 0.32, 0.33,
                ],
                "close_loc": [
                    0.10, 0.12, 0.14, 0.16,
                    0.82, 0.84, 0.86, 0.88,
                    0.10, 0.12, 0.14, 0.16,
                    0.82, 0.84, 0.86, 0.88,
                ],
                "news_long_horizon_ok": [0] * 16,
            }
        )
        keys = [spec["candidate_key"] for spec in _daily_outcome_specs(dataset)]
        self.assertIn("daily_long_regime_2x2", keys)

    def test_weekend_outcome_specs_fall_back_to_coarse_buckets_when_thin(self):
        dataset = pd.DataFrame({"wday": [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 4, 4]})
        keys = [spec["candidate_key"] for spec in _weekend_outcome_specs(dataset)]
        self.assertIn("weekend_session_coarse3", keys)
        self.assertNotIn("weekend_session_wday", keys)

    def test_refresh_structural_snapshots_persists_daily_and_weekend_rows(self):
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = hover.DATA_DIR
            old_daily_cache = dict(hover._DAILY_HISTORY_CACHE)
            old_snapshot_cache = dict(hover._SNAPSHOT_CACHE)
            old_outcome_cache = dict(hover._OUTCOME_CACHE)
            try:
                hover.DATA_DIR = tmp
                hover._DAILY_HISTORY_CACHE.clear()
                hover._SNAPSHOT_CACHE.clear()
                hover._OUTCOME_CACHE.clear()
                idx = pd.DatetimeIndex(pd.bdate_range("2026-01-05 20:58:00", periods=18, tz="UTC"))
                prices = pd.DataFrame(
                    {
                        "open": [100.0 + i for i in range(len(idx))],
                        "high": [101.0 + i for i in range(len(idx))],
                        "low": [99.0 + i for i in range(len(idx))],
                        "close": [100.5 + i for i in range(len(idx))],
                        "volume": [10.0 + i for i in range(len(idx))],
                    },
                    index=idx,
                )
                prices.to_csv(os.path.join(tmp, "prices_QQQ.csv"))
                daily_idx = pd.date_range("2025-06-01", periods=240, freq="B", tz="UTC")
                seq = [90.0 + (0.1 * i) for i in range(240)]
                daily = pd.DataFrame(
                    {
                        "open": seq,
                        "high": [v + 1.0 for v in seq],
                        "low": [v - 1.0 for v in seq],
                        "close": [v + 0.5 for v in seq],
                        "volume": [1000.0] * 240,
                    },
                    index=daily_idx,
                )
                daily.to_csv(os.path.join(tmp, "prices_QQQ_daily_long.csv"))
                snap = hover.refresh_structural_snapshots("QQQ", prices=prices, max_daily_targets=12, max_week_targets=8)
                self.assertFalse(snap.empty)
                self.assertIn("snapshot_type", snap.columns)
                self.assertIn("gap_1", snap.columns)
                self.assertIn("news_intensity_ewm_d7", snap.columns)
                self.assertGreaterEqual(int((snap["snapshot_type"] == "daily_pair").sum()), 1)
                self.assertGreaterEqual(int((snap["snapshot_type"] == "weekend_session").sum()), 1)
                outcomes = hover._load_structural_outcomes("QQQ")
                self.assertFalse(outcomes.empty)
                self.assertIn("candidate_key", outcomes.columns)
            finally:
                hover.DATA_DIR = old_data_dir
                hover._DAILY_HISTORY_CACHE.clear()
                hover._DAILY_HISTORY_CACHE.update(old_daily_cache)
                hover._SNAPSHOT_CACHE.clear()
                hover._SNAPSHOT_CACHE.update(old_snapshot_cache)
                hover._OUTCOME_CACHE.clear()
                hover._OUTCOME_CACHE.update(old_outcome_cache)


if __name__ == "__main__":
    unittest.main()
