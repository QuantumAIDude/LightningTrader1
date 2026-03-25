import unittest

import pandas as pd

import opportunity_scanner as oscan


class OpportunityScannerTests(unittest.TestCase):
    def test_scan_ranks_candidates_by_score(self):
        old_candidate_symbols = oscan._candidate_symbols
        old_load_cached = oscan._load_cached_intraday
        old_load_models = oscan.load_models
        old_predict_4min = oscan.predict_4min
        old_predict_horizon = oscan.predict_horizon
        old_macro = oscan.get_macro_event_snapshot
        old_breadth = oscan.get_breadth_snapshot
        old_catalyst = oscan.get_catalyst_provider_snapshot
        old_liquidity = oscan.get_liquidity_snapshot
        old_event = oscan.get_event_snapshot
        old_news = oscan.load_recent_headlines
        old_build = oscan.build_advisory_snapshot
        old_cache = dict(oscan._SCAN_CACHE)
        try:
            df = pd.DataFrame(
                {"close": [100.0, 100.5], "high": [100.6, 100.8], "low": [99.8, 100.2], "open": [99.9, 100.4], "volume": [1000.0, 1200.0]},
                index=pd.date_range("2026-03-24 14:30:00+00:00", periods=2, freq="2min"),
            )
            oscan._SCAN_CACHE.clear()
            oscan._candidate_symbols = lambda max_symbols: ["QQQ", "SPY"]
            oscan._load_cached_intraday = lambda symbol: df.copy()
            oscan.load_models = lambda instrument_key=None: {"base": {}, "forest": {}, "boost": {}, "extra_trees": {}, "regime": {"meta": {"version": "1"}}}
            oscan.predict_4min = lambda *args, **kwargs: {"consensus": {"direction": "UP", "pred_price": 101.0}}
            oscan.predict_horizon = lambda *args, **kwargs: {"week": {}, "high": {}, "low": {}}
            oscan.get_macro_event_snapshot = lambda: {"risk_score": 0.1}
            oscan.get_breadth_snapshot = lambda symbol: {"risk_on_score": 0.6}
            oscan.get_catalyst_provider_snapshot = lambda symbol: {"catalyst_risk_score": 0.2}
            oscan.get_liquidity_snapshot = lambda df, symbol: {"tradability_score": 0.8 if symbol == "QQQ" else 0.55}
            oscan.get_event_snapshot = lambda symbol: {"status": "ok"}
            oscan.load_recent_headlines = lambda symbol, limit=6: []
            oscan.build_advisory_snapshot = lambda df, symbol, **kwargs: {
                "quality_meta": {"score": 0.78 if symbol == "QQQ" else 0.59, "agreement": 0.75 if symbol == "QQQ" else 0.55, "no_trade": False, "strategy": "Breakout"},
                "liquidity_meta": {"tradability_score": 0.8 if symbol == "QQQ" else 0.55},
                "catalyst_meta": {"catalyst_risk_score": 0.2},
                "plan_meta": {"action": "Long setup"},
            }
            snap = oscan.get_opportunity_scan(profile_name="balanced")
            self.assertEqual(snap["rows"][0][0], "QQQ")
            self.assertGreater(len(snap["rows"]), 1)
        finally:
            oscan._candidate_symbols = old_candidate_symbols
            oscan._load_cached_intraday = old_load_cached
            oscan.load_models = old_load_models
            oscan.predict_4min = old_predict_4min
            oscan.predict_horizon = old_predict_horizon
            oscan.get_macro_event_snapshot = old_macro
            oscan.get_breadth_snapshot = old_breadth
            oscan.get_catalyst_provider_snapshot = old_catalyst
            oscan.get_liquidity_snapshot = old_liquidity
            oscan.get_event_snapshot = old_event
            oscan.load_recent_headlines = old_news
            oscan.build_advisory_snapshot = old_build
            oscan._SCAN_CACHE.clear()
            oscan._SCAN_CACHE.update(old_cache)


if __name__ == "__main__":
    unittest.main()
