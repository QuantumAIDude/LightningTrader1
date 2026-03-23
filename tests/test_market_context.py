import os
import tempfile
import unittest

import numpy as np
import pandas as pd

import features as feat
import market_context as mctx


class MarketContextTests(unittest.TestCase):
    def _write_market_context(self, tmp: str):
        idx = pd.date_range("2026-01-05", periods=6, freq="B", tz="UTC")
        closes = np.array([100.0, 110.0, 105.0, 115.0, 120.0, 118.0], dtype=float)
        frame = pd.DataFrame(
            {
                "open": closes - 0.5,
                "high": closes + 1.0,
                "low": closes - 1.0,
                "close": closes,
                "volume": np.full(len(idx), 1000.0),
            },
            index=idx,
        )
        frame.to_csv(os.path.join(tmp, "context_QQQ_MARKET.csv"))

    def test_intraday_context_is_shifted_one_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = mctx.DATA_DIR
            old_cache = dict(mctx._CACHE)
            old_missing = dict(mctx._MISSING_CACHE)
            old_instruments = dict(mctx.INSTRUMENTS)
            try:
                mctx.DATA_DIR = tmp
                mctx._CACHE.clear()
                mctx._MISSING_CACHE.clear()
                mctx.INSTRUMENTS.clear()
                mctx.INSTRUMENTS.update({"QQQ": {"context_daily_symbols": {"market": "SPY"}}})
                self._write_market_context(tmp)

                idx = pd.DatetimeIndex(
                    [
                        pd.Timestamp("2026-01-06 20:58:00+00:00"),
                        pd.Timestamp("2026-01-07 20:58:00+00:00"),
                    ]
                )
                aligned = mctx.load_context_features_for_index(idx, instrument_key="QQQ", shift_days=1)

                self.assertAlmostEqual(float(aligned.iloc[0]["ctx_market_ret_1"]), 0.0, places=9)
                self.assertAlmostEqual(float(aligned.iloc[1]["ctx_market_ret_1"]), 0.10, places=6)
            finally:
                mctx.DATA_DIR = old_data_dir
                mctx._CACHE.clear()
                mctx._CACHE.update(old_cache)
                mctx._MISSING_CACHE.clear()
                mctx._MISSING_CACHE.update(old_missing)
                mctx.INSTRUMENTS.clear()
                mctx.INSTRUMENTS.update(old_instruments)

    def test_same_day_context_is_available_for_session_close_horizons(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = mctx.DATA_DIR
            old_cache = dict(mctx._CACHE)
            old_missing = dict(mctx._MISSING_CACHE)
            old_instruments = dict(mctx.INSTRUMENTS)
            try:
                mctx.DATA_DIR = tmp
                mctx._CACHE.clear()
                mctx._MISSING_CACHE.clear()
                mctx.INSTRUMENTS.clear()
                mctx.INSTRUMENTS.update({"QQQ": {"context_daily_symbols": {"market": "SPY"}}})
                self._write_market_context(tmp)

                row = pd.DataFrame({"close": [105.0]}, index=pd.DatetimeIndex([pd.Timestamp("2026-01-07 20:58:00+00:00")]))
                enriched = mctx.add_context_features(row, instrument_key="QQQ", shift_days=0)

                self.assertAlmostEqual(float(enriched.iloc[0]["ctx_market_ret_1"]), (105.0 / 110.0) - 1.0, places=6)
            finally:
                mctx.DATA_DIR = old_data_dir
                mctx._CACHE.clear()
                mctx._CACHE.update(old_cache)
                mctx._MISSING_CACHE.clear()
                mctx._MISSING_CACHE.update(old_missing)
                mctx.INSTRUMENTS.clear()
                mctx.INSTRUMENTS.update(old_instruments)

    def test_next_session_dataset_overrides_to_same_day_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = mctx.DATA_DIR
            old_cache = dict(mctx._CACHE)
            old_missing = dict(mctx._MISSING_CACHE)
            old_instruments = dict(mctx.INSTRUMENTS)
            try:
                mctx.DATA_DIR = tmp
                mctx._CACHE.clear()
                mctx._MISSING_CACHE.clear()
                mctx.INSTRUMENTS.clear()
                mctx.INSTRUMENTS.update({"QQQ": {"context_daily_symbols": {"market": "SPY"}, "tz": "America/New_York"}})
                self._write_market_context(tmp)

                idx = pd.DatetimeIndex(
                    [
                        pd.Timestamp("2026-01-06 20:55:00+00:00"),
                        pd.Timestamp("2026-01-06 20:58:00+00:00"),
                        pd.Timestamp("2026-01-07 20:55:00+00:00"),
                        pd.Timestamp("2026-01-07 20:58:00+00:00"),
                    ]
                )
                prices = pd.DataFrame(
                    {
                        "open": [100.0, 100.4, 104.8, 105.1],
                        "high": [100.7, 101.0, 105.8, 106.2],
                        "low": [99.8, 100.2, 104.3, 104.9],
                        "close": [100.5, 100.8, 105.2, 105.4],
                        "volume": [1000.0, 1100.0, 1200.0, 1300.0],
                    },
                    index=idx,
                )

                ds = feat.build_next_session_dataset(prices, instrument_key="QQQ")

                self.assertEqual(len(ds), 1)
                self.assertAlmostEqual(float(ds.iloc[0]["ctx_market_ret_1"]), 0.10, places=6)
            finally:
                mctx.DATA_DIR = old_data_dir
                mctx._CACHE.clear()
                mctx._CACHE.update(old_cache)
                mctx._MISSING_CACHE.clear()
                mctx._MISSING_CACHE.update(old_missing)
                mctx.INSTRUMENTS.clear()
                mctx.INSTRUMENTS.update(old_instruments)


if __name__ == "__main__":
    unittest.main()
