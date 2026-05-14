import unittest

import pandas as pd

import autonomy_strategies as strategies


def _trend_df(periods: int = 60) -> pd.DataFrame:
    idx = pd.date_range("2026-04-01T14:00:00Z", periods=periods, freq="2min")
    close = [100.0 + 0.12 * i for i in range(periods)]
    return pd.DataFrame(
        {
            "open": close,
            "high": [x + 0.08 for x in close],
            "low": [x - 0.08 for x in close],
            "close": close,
            "volume": [1000 + 10 * (i % 7) for i in range(periods)],
        },
        index=idx,
    )


class AutonomyStrategyTests(unittest.TestCase):
    def test_catalog_contains_requested_common_strategy_families(self):
        keys = {item["key"] for item in strategies.list_strategy_catalog()}

        self.assertIn("trend_following", keys)
        self.assertIn("momentum", keys)
        self.assertIn("mean_reversion", keys)
        self.assertIn("breakout", keys)
        self.assertIn("vwap_reversion", keys)
        self.assertIn("opening_range_breakout", keys)
        self.assertIn("pairs_stat_arb", keys)

    def test_trend_following_generates_directional_signal(self):
        signal = strategies.evaluate_strategy_signal("trend_following", _trend_df(), symbol="QQQ")

        self.assertTrue(signal["allowed"])
        self.assertEqual(signal["side"], "buy")
        self.assertGreater(signal["confidence"], 0.54)

    def test_strategy_advisory_translates_signal_into_risk_plan(self):
        signal = strategies.evaluate_strategy_signal("trend_following", _trend_df(), symbol="QQQ")
        advisory = strategies.build_strategy_advisory({}, signal, current_price=107.0)

        self.assertEqual(advisory["plan_meta"]["bias"], "Bullish")
        self.assertEqual(advisory["pred_state"]["intraday"]["direction"], "UP")
        self.assertGreater(advisory["quality_meta"]["score"], 0.0)


if __name__ == "__main__":
    unittest.main()
