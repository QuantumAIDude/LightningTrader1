import unittest

import pandas as pd

import advisory_engine as ae


class AdvisoryEngineTests(unittest.TestCase):
    def test_build_advisory_snapshot_returns_probabilities_setup_and_reliability(self):
        idx = pd.date_range("2026-03-18 14:30:00+00:00", periods=80, freq="2min")
        df = pd.DataFrame(
            {
                "open": [100.0 + 0.03 * i for i in range(80)],
                "high": [100.2 + 0.03 * i for i in range(80)],
                "low": [99.8 + 0.03 * i for i in range(80)],
                "close": [100.05 + 0.03 * i for i in range(80)],
                "volume": [1200.0 + (i % 8) * 50.0 for i in range(80)],
                "bid": [99.99 + 0.03 * i for i in range(80)],
                "ask": [100.01 + 0.03 * i for i in range(80)],
            },
            index=idx,
        )
        daily_long = pd.DataFrame(
            {
                "open": [95.0 + 0.3 * i for i in range(40)],
                "high": [96.0 + 0.35 * i for i in range(40)],
                "low": [94.0 + 0.25 * i for i in range(40)],
                "close": [95.5 + 0.32 * i for i in range(40)],
                "volume": [1_000_000.0 + 10_000.0 * i for i in range(40)],
            },
            index=pd.date_range("2026-01-20", periods=40, freq="B"),
        )
        old_load_long = ae._load_long_daily_history
        old_event = ae.get_event_snapshot
        old_news = ae.load_recent_headlines
        old_readiness = ae.load_readiness_state
        old_macro = ae.get_macro_event_snapshot
        old_breadth = ae.get_breadth_snapshot
        old_catalysts = ae.get_catalyst_provider_snapshot
        old_liquidity = ae.get_liquidity_snapshot
        try:
            ae._load_long_daily_history = lambda instrument_key: daily_long.copy()
            ae.get_event_snapshot = lambda symbol: {"status": "ok", "days_to_earnings": 9.0}
            ae.get_macro_event_snapshot = lambda: {"risk_score": 0.20, "events": [], "provider_note": "Local macro calendar file"}
            ae.get_breadth_snapshot = lambda symbol: {"risk_on_score": 0.62, "rows": [("Proxy Breadth", "62/100", "Risk-on | context-derived")]}
            ae.get_catalyst_provider_snapshot = lambda symbol: {"rows": [("SEC Feed", "Stub", "Adapter ready")], "recent": []}
            ae.get_liquidity_snapshot = lambda df, symbol: {"rows": [("Tradability", "70/100", "Good")], "tradability_score": 0.70}
            ae.load_recent_headlines = lambda symbol, limit=8: [
                {
                    "published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1),
                    "headline": "Policy risk eases",
                    "label": "positive",
                    "score": 0.25,
                    "political_relevance": 0.45,
                }
            ]
            ae.load_readiness_state = lambda symbol: {
                "live_ready": False,
                "fresh": True,
                "pass_streak_days": 2,
                "evaluation": {
                    "metrics": {
                        "intraday_avg_discrepancy_dollars": 0.24,
                        "intraday_target_window_dollars": 0.20,
                        "intraday_window_hit_rate": 0.62,
                        "intraday_updown_acc": 0.56,
                    }
                },
            }
            preds = {
                "intraday": {
                    "consensus": {
                        "direction": "UP",
                        "pred_price": 102.7,
                        "confidence": 64.0,
                        "pred_range_70_low": 102.4,
                        "pred_range_70_high": 102.9,
                        "pred_range_95_low": 102.0,
                        "pred_range_95_high": 103.3,
                        "p_up": 0.64,
                        "edge_bps": 10.0,
                        "uncertainty_bps": 6.0,
                        "position_size": 0.45,
                        "sig_bucket": "Medium",
                    },
                    "base": {"direction": "UP", "pred_price": 102.8, "confidence": 63.0, "p_up": 0.63, "pred_range_80_low": 102.3, "pred_range_80_high": 103.0},
                    "forest": {"direction": "UP", "pred_price": 102.7, "confidence": 64.0, "p_up": 0.64, "pred_range_80_low": 102.2, "pred_range_80_high": 103.0},
                    "boost": {"direction": "UP", "pred_price": 102.9, "confidence": 62.0, "p_up": 0.62, "pred_range_80_low": 102.4, "pred_range_80_high": 103.1},
                    "extra_trees": {"direction": "UP", "pred_price": 102.6, "confidence": 61.0, "p_up": 0.61, "pred_range_80_low": 102.1, "pred_range_80_high": 102.9},
                    "regime": {"Bullish": 58.0, "Neutral": 26.0, "Bearish": 16.0, "regime_conf": 0.72},
                },
                "daily": {
                    "high": {
                        "base": {"pred_price": 103.5, "confidence": 58.0, "p_up": 0.58, "pred_range_80_low": 103.0, "pred_range_80_high": 104.2},
                        "forest": {"pred_price": 103.4, "confidence": 57.0, "p_up": 0.57, "pred_range_80_low": 102.9, "pred_range_80_high": 104.1},
                        "boost": {"pred_price": 103.6, "confidence": 59.0, "p_up": 0.59, "pred_range_80_low": 103.1, "pred_range_80_high": 104.3},
                        "extra_trees": {"pred_price": 103.5, "confidence": 58.0, "p_up": 0.58, "pred_range_80_low": 103.0, "pred_range_80_high": 104.2},
                    },
                    "low": {
                        "base": {"pred_price": 101.8, "confidence": 55.0, "p_up": 0.45, "pred_range_80_low": 101.2, "pred_range_80_high": 102.3},
                        "forest": {"pred_price": 101.9, "confidence": 54.0, "p_up": 0.46, "pred_range_80_low": 101.3, "pred_range_80_high": 102.4},
                        "boost": {"pred_price": 101.7, "confidence": 56.0, "p_up": 0.44, "pred_range_80_low": 101.1, "pred_range_80_high": 102.2},
                        "extra_trees": {"pred_price": 101.8, "confidence": 55.0, "p_up": 0.45, "pred_range_80_low": 101.2, "pred_range_80_high": 102.3},
                    },
                },
                "weekend": {
                    "week": {
                        "base": {"pred_price": 104.4, "pred_high_price": 105.2, "pred_low_price": 101.7, "confidence": 57.0, "p_up": 0.57, "pred_range_80_low": 103.9, "pred_range_80_high": 104.9},
                        "forest": {"pred_price": 104.3, "pred_high_price": 105.1, "pred_low_price": 101.8, "confidence": 56.0, "p_up": 0.56, "pred_range_80_low": 103.8, "pred_range_80_high": 104.8},
                        "boost": {"pred_price": 104.5, "pred_high_price": 105.4, "pred_low_price": 101.6, "confidence": 58.0, "p_up": 0.58, "pred_range_80_low": 104.0, "pred_range_80_high": 105.0},
                        "extra_trees": {"pred_price": 104.4, "pred_high_price": 105.3, "pred_low_price": 101.7, "confidence": 57.0, "p_up": 0.57, "pred_range_80_low": 103.9, "pred_range_80_high": 104.9},
                    }
                },
            }
            out = ae.build_advisory_snapshot(df, "QQQ", predictions=preds, options_snapshot={"expected_move_dollars": 2.1})
            self.assertIn("Action:", out["action_card"])
            self.assertIn("Quality:", out["quality_card"])
            self.assertTrue(out["probability_rows"])
            self.assertTrue(out["quality_rows"])
            self.assertTrue(out["setup_rows"])
            self.assertTrue(out["scenario_rows"])
            self.assertTrue(out["invalidation_rows"])
            self.assertTrue(out["reliability_rows"])
            self.assertTrue(out["breadth_rows"])
            self.assertTrue(out["macro_rows"])
            self.assertTrue(out["liquidity_rows"])
            self.assertTrue(out["catalyst_rows"])
            self.assertIsInstance(out["quality_meta"], dict)
            self.assertIsInstance(out["plan_meta"], dict)
        finally:
            ae._load_long_daily_history = old_load_long
            ae.get_event_snapshot = old_event
            ae.load_recent_headlines = old_news
            ae.load_readiness_state = old_readiness
            ae.get_macro_event_snapshot = old_macro
            ae.get_breadth_snapshot = old_breadth
            ae.get_catalyst_provider_snapshot = old_catalysts
            ae.get_liquidity_snapshot = old_liquidity


if __name__ == "__main__":
    unittest.main()
