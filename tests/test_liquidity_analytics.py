import os
import tempfile
import unittest

import pandas as pd

import liquidity_analytics as la


class LiquidityAnalyticsTests(unittest.TestCase):
    def test_liquidity_snapshot_scores_spread_and_volume(self):
        idx = pd.date_range("2026-03-18 14:30:00+00:00", periods=20, freq="2min")
        df = pd.DataFrame(
            {
                "close": [100.0 + 0.1 * i for i in range(20)],
                "volume": [1000.0 + 50.0 * i for i in range(20)],
                "bid": [99.99 + 0.1 * i for i in range(20)],
                "ask": [100.01 + 0.1 * i for i in range(20)],
            },
            index=idx,
        )
        daily = pd.DataFrame(
            {
                "close": [100.0 + 0.5 * i for i in range(30)],
                "high": [101.0 + 0.5 * i for i in range(30)],
                "low": [99.0 + 0.5 * i for i in range(30)],
                "volume": [2_000_000.0 + 20_000.0 * i for i in range(30)],
            },
            index=pd.date_range("2026-02-01", periods=30, freq="B"),
        )
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = la.DATA_DIR
            la.DATA_DIR = tmp
            try:
                daily.to_csv(os.path.join(tmp, "prices_QQQ_daily_long.csv"))
                snap = la.get_liquidity_snapshot(df, "QQQ")
                self.assertTrue(snap["rows"])
                self.assertIsNotNone(snap["tradability_score"])
            finally:
                la.DATA_DIR = old_dir


if __name__ == "__main__":
    unittest.main()
