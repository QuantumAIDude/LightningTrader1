import unittest

import pandas as pd

import trader_analytics as ta


class TraderAnalyticsTests(unittest.TestCase):
    def test_build_trader_dashboard_returns_levels_market_options_and_news(self):
        idx = pd.DatetimeIndex(
            [
                pd.Timestamp("2026-03-18 14:30:00+00:00"),
                pd.Timestamp("2026-03-18 20:58:00+00:00"),
                pd.Timestamp("2026-03-19 14:30:00+00:00"),
                pd.Timestamp("2026-03-19 20:58:00+00:00"),
            ]
        )
        prices = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.5, 104.0],
                "low": [99.0, 100.5, 101.5, 102.4],
                "close": [100.5, 101.8, 103.2, 103.7],
                "volume": [1000.0, 1200.0, 1400.0, 1600.0],
            },
            index=idx,
        )
        daily_long = pd.DataFrame(
            {
                "open": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
                "high": [96.0, 97.0, 98.0, 99.0, 100.0, 104.0],
                "low": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
                "close": [95.5, 96.5, 97.5, 98.5, 99.5, 103.0],
                "volume": [1000.0] * 6,
            },
            index=pd.date_range("2026-03-10", periods=6, freq="B"),
        )
        market_hist = pd.DataFrame(
            {
                "open": [500.0, 505.0, 507.0],
                "high": [505.0, 508.0, 510.0],
                "low": [499.0, 504.0, 506.0],
                "close": [504.0, 507.0, 509.0],
                "volume": [1000.0, 1000.0, 1000.0],
            },
            index=pd.date_range("2026-03-17", periods=3, freq="B"),
        )
        old_load_long = ta._load_long_daily_history
        old_load_ctx = ta.load_context_daily_history
        old_load_news = ta.load_recent_headlines
        old_get_options = ta.get_options_snapshot
        old_instruments = dict(ta.INSTRUMENTS)
        try:
            ta._load_long_daily_history = lambda instrument_key: daily_long.copy()
            ta.load_context_daily_history = lambda instrument_key, role, allow_remote=False: market_hist.copy()
            ta.load_recent_headlines = lambda symbol, limit=6: [
                {"published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2), "headline": "Chip demand improves", "label": "positive", "score": 0.4},
                {"published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4), "headline": "Treasury yields ease", "label": "neutral", "score": 0.1},
            ]
            ta.get_options_snapshot = lambda symbol, spot_price=None: {
                "status": "ok",
                "expiry": "2026-03-27",
                "days_to_expiry": 8,
                "expected_move_dollars": 7.2,
                "expected_move_pct": 0.014,
                "put_call_oi_ratio": 1.1,
                "put_call_volume_ratio": 0.9,
                "iv_skew_put_minus_call": 0.02,
                "call_wall_oi": 108.0,
                "put_wall_oi": 101.0,
                "max_pain": 104.0,
                "source": "cache",
            }
            ta.INSTRUMENTS.clear()
            ta.INSTRUMENTS.update({"QQQ": {"tz": "America/New_York", "context_daily_symbols": {"market": "SPY"}}})
            preds = {
                "intraday": {"consensus": {"direction": "UP", "confidence": 61.0}},
                "daily": {
                    "high": {"base": {"pred_price": 106.0}, "forest": {"pred_price": 105.5}, "boost": {"pred_price": 105.8}, "extra_trees": {"pred_price": 106.2}},
                    "low": {"base": {"pred_price": 101.0}, "forest": {"pred_price": 101.2}, "boost": {"pred_price": 101.1}, "extra_trees": {"pred_price": 100.9}},
                },
                "weekend": {"week": {"base": {"pred_price": 107.0}, "forest": {"pred_price": 106.8}, "boost": {"pred_price": 107.1}, "extra_trees": {"pred_price": 106.9}}},
            }
            out = ta.build_trader_dashboard(prices, "QQQ", predictions=preds)
            self.assertIn("Alignment:", out["setup_card"])
            self.assertTrue(out["market_rows"])
            self.assertTrue(out["levels_rows"])
            self.assertTrue(out["options_rows"])
            self.assertEqual(len(out["news_rows"]), 2)
        finally:
            ta._load_long_daily_history = old_load_long
            ta.load_context_daily_history = old_load_ctx
            ta.load_recent_headlines = old_load_news
            ta.get_options_snapshot = old_get_options
            ta.INSTRUMENTS.clear()
            ta.INSTRUMENTS.update(old_instruments)


if __name__ == "__main__":
    unittest.main()
