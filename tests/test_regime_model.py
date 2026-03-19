import unittest

import pandas as pd

from regime_model import _prepare_regime_input


class RegimeModelTests(unittest.TestCase):
    def test_prepare_regime_input_uses_session_close_for_daily(self):
        idx = pd.to_datetime(
            [
                "2026-03-09T14:30:00Z",
                "2026-03-09T20:58:00Z",
                "2026-03-10T14:30:00Z",
                "2026-03-10T20:58:00Z",
            ],
            utc=True,
        )
        df = pd.DataFrame(
            {
                "open": [100.0, 100.5, 101.0, 101.4],
                "high": [101.0, 101.2, 102.0, 102.1],
                "low": [99.5, 100.2, 100.7, 101.1],
                "close": [100.4, 100.9, 101.3, 101.8],
                "volume": [10.0, 12.0, 13.0, 15.0],
            },
            index=idx,
        )
        reduced, granularity = _prepare_regime_input(df, horizon_key="daily")
        self.assertEqual(granularity, "session_close")
        self.assertEqual(len(reduced), 2)
        self.assertAlmostEqual(float(reduced.iloc[0]["close"]), 100.9, places=6)

    def test_prepare_regime_input_keeps_intraday_bars_for_intraday(self):
        idx = pd.date_range("2026-03-09T14:30:00Z", periods=5, freq="4min", tz="UTC")
        df = pd.DataFrame(
            {
                "open": [100.0, 100.1, 100.2, 100.3, 100.4],
                "high": [100.2, 100.3, 100.4, 100.5, 100.6],
                "low": [99.9, 100.0, 100.1, 100.2, 100.3],
                "close": [100.1, 100.2, 100.3, 100.4, 100.5],
                "volume": [10.0, 11.0, 12.0, 13.0, 14.0],
            },
            index=idx,
        )
        reduced, granularity = _prepare_regime_input(df, horizon_key="intraday")
        self.assertEqual(granularity, "intraday_bars")
        self.assertEqual(len(reduced), len(df))


if __name__ == "__main__":
    unittest.main()
