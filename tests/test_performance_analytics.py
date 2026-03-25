import os
import tempfile
import unittest

import execution_engine as ee
import performance_analytics as pa


class PerformanceAnalyticsTests(unittest.TestCase):
    def test_performance_snapshot_aggregates_closed_trades(self):
        old_state_path = ee._state_path
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ee._state_path = lambda: os.path.join(tmp, "paper_state.json")
                state = ee.reset_paper_state(50000)
                state["closed_trades"] = [
                    {
                        "symbol": "QQQ",
                        "pnl_dollars": 150.0,
                        "pnl_pct": 0.015,
                        "risk_multiple": 1.5,
                        "setup_type": "Long setup",
                        "regime": "Bullish",
                        "confidence_bucket": "High",
                        "time_bucket": "AM",
                    },
                    {
                        "symbol": "QQQ",
                        "pnl_dollars": -60.0,
                        "pnl_pct": -0.006,
                        "risk_multiple": -0.6,
                        "setup_type": "Long setup",
                        "regime": "Bullish",
                        "confidence_bucket": "Medium",
                        "time_bucket": "PM",
                    },
                    {
                        "symbol": "SPY",
                        "pnl_dollars": 90.0,
                        "pnl_pct": 0.009,
                        "risk_multiple": 0.9,
                        "setup_type": "Short setup",
                        "regime": "Bearish",
                        "confidence_bucket": "High",
                        "time_bucket": "AM",
                    },
                ]
                ee._save_state(state)
                snap = pa.get_performance_snapshot()
                self.assertEqual(snap["trade_count"], 3)
                self.assertIn("closed trades", snap["summary"])
                self.assertTrue(any(row[0] == "Win Rate" for row in snap["rows"]))
                self.assertTrue(any(row[0] == "Best Setup" for row in snap["rows"]))
            finally:
                ee._state_path = old_state_path


if __name__ == "__main__":
    unittest.main()
