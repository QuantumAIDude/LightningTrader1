import unittest

import pandas as pd

import advisory_engine as ae
import trader_analytics as ta


class TraderAnalyticsTests(unittest.TestCase):
    def test_levels_rows_handles_close_only_long_history(self):
        intraday = pd.DataFrame(
            {
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1200.0],
            },
            index=pd.date_range("2026-03-18 14:30:00+00:00", periods=2, freq="2min"),
        )
        close_only = pd.DataFrame(
            {"close": [99.0, 100.0, 101.0, 102.0, 103.0, 104.0]},
            index=pd.date_range("2026-03-10", periods=6, freq="B"),
        )
        old_load_long = ta._load_long_daily_history
        try:
            ta._load_long_daily_history = lambda instrument_key: close_only.copy()
            rows = ta._levels_rows(intraday, "QQQ")
            self.assertTrue(rows)
            self.assertTrue(any(row[0] == "20D / 50D SMA" for row in rows))
        finally:
            ta._load_long_daily_history = old_load_long

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
        old_get_event = ta.get_event_snapshot
        old_get_breadth = ta.get_breadth_snapshot
        old_get_macro = ta.get_macro_event_snapshot
        old_get_catalysts = ta.get_catalyst_provider_snapshot
        old_get_liquidity = ta.get_liquidity_snapshot
        old_get_news_health = ta.get_news_health
        old_instruments = dict(ta.INSTRUMENTS)
        old_adv_news = ae.load_recent_headlines
        old_adv_event = ae.get_event_snapshot
        old_adv_readiness = ae.load_readiness_state
        old_adv_macro = ae.get_macro_event_snapshot
        old_adv_breadth = ae.get_breadth_snapshot
        old_adv_catalysts = ae.get_catalyst_provider_snapshot
        old_adv_liquidity = ae.get_liquidity_snapshot
        old_adv_news_health = ae.get_news_health
        old_exec = ta.get_execution_dashboard_snapshot
        old_perf = ta.get_performance_snapshot
        old_replay = ta.get_replay_dashboard_snapshot
        old_alerts = ta.get_alert_snapshot
        old_scan = ta.get_opportunity_scan
        try:
            ta._load_long_daily_history = lambda instrument_key: daily_long.copy()
            ta.load_context_daily_history = lambda instrument_key, role, allow_remote=False: market_hist.copy()
            ta.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "high_impact_recent": 1}
            ta.load_recent_headlines = lambda symbol, limit=6: [
                {"published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2), "headline": "Chip demand improves", "label": "positive", "score": 0.4, "event_label": "Guidance", "impact_score": 0.68, "impact_horizon": "multi-day", "importance_score": 0.55, "signal_note": "Guidance changes can matter for several sessions"},
                {"published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=4), "headline": "Treasury yields ease", "label": "neutral", "score": 0.1, "event_label": "Macro", "impact_score": 0.42, "impact_horizon": "intraday", "importance_score": 0.28, "signal_note": "Macro moves need tape confirmation"},
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
            ta.get_event_snapshot = lambda symbol: {
                "status": "ok",
                "next_earnings": "2026-03-28T00:00:00+00:00",
                "days_to_earnings": 9.0,
                "source": "cache",
            }
            ta.get_breadth_snapshot = lambda symbol: {
                "rows": [("Proxy Breadth", "63/100", "Risk-on | context-derived")],
                "risk_on_score": 0.63,
            }
            ta.get_macro_event_snapshot = lambda: {
                "risk_score": 0.25,
                "provider_note": "Local macro calendar file",
                "events": [],
            }
            ta.get_catalyst_provider_snapshot = lambda symbol: {
                "rows": [("SEC Feed", "Stub", "Adapter ready"), ("Analyst Feed", "Stub", "Adapter ready")],
                "recent": [],
            }
            ta.get_liquidity_snapshot = lambda df, symbol: {
                "rows": [("Tradability", "72/100", "Good"), ("Depth Feed", "Not wired", "Real order book required")],
                "tradability_score": 0.72,
            }
            ae.load_recent_headlines = lambda symbol, limit=8: [
                {
                    "published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1),
                    "headline": "Policy outlook stabilizes",
                    "label": "positive",
                    "score": 0.3,
                    "political_relevance": 0.5,
                    "event_label": "Macro",
                    "impact_score": 0.61,
                    "impact_horizon": "intraday",
                    "importance_score": 0.46,
                    "direction_score": 0.16,
                    "signal_note": "Macro headlines can dominate intraday tape",
                }
            ]
            ae.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "high_impact_recent": 1}
            ae.get_event_snapshot = lambda symbol: {
                "status": "ok",
                "next_earnings": "2026-03-28T00:00:00+00:00",
                "days_to_earnings": 9.0,
                "source": "cache",
            }
            ae.get_macro_event_snapshot = lambda: {
                "risk_score": 0.25,
                "provider_note": "Local macro calendar file",
                "events": [],
            }
            ae.get_breadth_snapshot = lambda symbol: {
                "rows": [("Proxy Breadth", "63/100", "Risk-on | context-derived")],
                "risk_on_score": 0.63,
            }
            ae.get_catalyst_provider_snapshot = lambda symbol: {
                "rows": [("SEC Feed", "Stub", "Adapter ready")],
                "recent": [],
            }
            ae.get_liquidity_snapshot = lambda df, symbol: {
                "rows": [("Tradability", "72/100", "Good")],
                "tradability_score": 0.72,
            }
            ta.get_execution_dashboard_snapshot = lambda symbol, current_price=None, advisory=None, profile_name=None: {
                "broker_mode": "local_paper",
                "runtime": {"route_label": "Local Paper", "detail": "Orders route to local paper engine"},
                "diagnostics": {"checked_at": "2026-03-24T12:00:00+00:00"},
                "verification": {"status": "blocked"},
                "rows": [("Broker", "Paper", "Immediate fills in paper mode"), ("Next Size", "10 sh", "Stop $102.00 | target $105.00")],
                "portfolio_rows": [("Equity", "$100,000.00", "Cash $98,000.00")],
            }
            ta.get_performance_snapshot = lambda symbol=None: {
                "summary": "3 closed trades | win +67%",
                "rows": [("Win Rate", "+67.0%", "Avg win $120.00 | avg loss $40.00")],
            }
            ta.get_replay_dashboard_snapshot = lambda symbol=None: {
                "card": "Replay: paused",
                "detail": "QQQ | bar 12/50",
                "rows": [("Replay", "Paused", "Simulation only")],
            }
            ta.get_alert_snapshot = lambda symbol, current_price=None, advisory=None, profile_name=None, persist=True: {
                "rows": [("Long setup", "High", "score 78/100 | agreement 75%")],
                "summary": "1 recent alert",
            }
            ta.get_opportunity_scan = lambda profile_name=None: {
                "rows": [("QQQ", "78/100", "Long setup | Q 78/100 | T 72/100")],
                "summary": "1 ranked opportunity",
            }
            ae.load_readiness_state = lambda symbol: {
                "live_ready": False,
                "fresh": True,
                "pass_streak_days": 1,
                "evaluation": {
                    "metrics": {
                        "intraday_avg_discrepancy_dollars": 0.22,
                        "intraday_target_window_dollars": 0.20,
                        "intraday_window_hit_rate": 0.61,
                        "intraday_updown_acc": 0.57,
                        "daily_avg_discrepancy_dollars": 1.8,
                        "daily_target_window_dollars": 1.5,
                        "daily_window_hit_rate": 0.45,
                        "daily_updown_acc": 0.53,
                        "weekend_avg_discrepancy_dollars": 2.9,
                        "weekend_target_window_dollars": 2.5,
                        "weekend_window_hit_rate": 0.44,
                        "weekend_updown_acc": 0.52,
                    }
                },
            }
            ta.INSTRUMENTS.clear()
            ta.INSTRUMENTS.update({"QQQ": {"tz": "America/New_York", "context_daily_symbols": {"market": "SPY"}, "news_aliases": ["QQQ", "Invesco QQQ"]}})
            preds = {
                "intraday": {"consensus": {"direction": "UP", "confidence": 61.0, "pred_price": 104.2, "pred_range_70_low": 103.8, "pred_range_70_high": 104.5, "pred_range_95_low": 103.2, "pred_range_95_high": 104.9, "p_up": 0.61, "edge_bps": 8.0, "uncertainty_bps": 5.0, "position_size": 0.4, "sig_bucket": "Medium"}},
                "daily": {
                    "high": {"base": {"pred_price": 106.0, "confidence": 60.0, "p_up": 0.60, "pred_range_80_low": 105.0, "pred_range_80_high": 107.0}, "forest": {"pred_price": 105.5, "confidence": 58.0, "p_up": 0.58, "pred_range_80_low": 104.9, "pred_range_80_high": 106.4}, "boost": {"pred_price": 105.8, "confidence": 57.0, "p_up": 0.57, "pred_range_80_low": 105.1, "pred_range_80_high": 106.6}, "extra_trees": {"pred_price": 106.2, "confidence": 59.0, "p_up": 0.59, "pred_range_80_low": 105.3, "pred_range_80_high": 107.2}},
                    "low": {"base": {"pred_price": 101.0, "confidence": 56.0, "p_up": 0.44, "pred_range_80_low": 100.2, "pred_range_80_high": 101.8}, "forest": {"pred_price": 101.2, "confidence": 55.0, "p_up": 0.45, "pred_range_80_low": 100.4, "pred_range_80_high": 101.9}, "boost": {"pred_price": 101.1, "confidence": 54.0, "p_up": 0.46, "pred_range_80_low": 100.3, "pred_range_80_high": 101.8}, "extra_trees": {"pred_price": 100.9, "confidence": 57.0, "p_up": 0.43, "pred_range_80_low": 100.0, "pred_range_80_high": 101.7}},
                },
                "weekend": {"week": {"base": {"pred_price": 107.0, "pred_high_price": 108.5, "pred_low_price": 103.0, "confidence": 58.0, "p_up": 0.58, "pred_range_80_low": 106.0, "pred_range_80_high": 108.0}, "forest": {"pred_price": 106.8, "pred_high_price": 108.2, "pred_low_price": 103.4, "confidence": 57.0, "p_up": 0.57, "pred_range_80_low": 105.9, "pred_range_80_high": 107.7}, "boost": {"pred_price": 107.1, "pred_high_price": 108.6, "pred_low_price": 103.1, "confidence": 59.0, "p_up": 0.59, "pred_range_80_low": 106.1, "pred_range_80_high": 108.1}, "extra_trees": {"pred_price": 106.9, "pred_high_price": 108.4, "pred_low_price": 103.2, "confidence": 58.0, "p_up": 0.58, "pred_range_80_low": 106.0, "pred_range_80_high": 107.9}}},
            }
            out = ta.build_trader_dashboard(prices, "QQQ", predictions=preds)
            self.assertIn("Alignment:", out["setup_card"])
            self.assertIn("Action:", out["action_card"])
            self.assertIn("Quality:", out["quality_card"])
            self.assertTrue(out["market_rows"])
            self.assertTrue(out["breadth_rows"])
            self.assertTrue(out["macro_rows"])
            self.assertTrue(out["signal_rows"])
            self.assertEqual(out["signal_rows"][0][0], "Trader Lens")
            self.assertTrue(out["stack_rows"])
            self.assertTrue(any(row[0] == "Lifecycle" for row in out["stack_rows"]))
            self.assertTrue(any(row[0] == "Operator Posture" for row in out["stack_rows"]))
            self.assertTrue(any(row[0] == "Models" for row in out["stack_rows"]))
            self.assertTrue(out["probability_rows"])
            self.assertTrue(out["quality_rows"])
            self.assertTrue(out["liquidity_rows"])
            self.assertTrue(out["catalyst_rows"])
            self.assertTrue(out["execution_rows"])
            self.assertTrue(out["replay_rows"])
            self.assertTrue(out["portfolio_rows"])
            self.assertTrue(out["performance_rows"])
            self.assertTrue(out["alerts_rows"])
            self.assertTrue(out["opportunity_rows"])
            self.assertTrue(out["setup_rows"])
            self.assertTrue(out["scenario_rows"])
            self.assertTrue(out["invalidation_rows"])
            self.assertTrue(out["reliability_rows"])
            self.assertTrue(out["levels_rows"])
            self.assertTrue(out["options_rows"])
            self.assertEqual(len(out["news_rows"]), 2)
            self.assertIn("Profile:", out["profile_card"])
            self.assertIn("Broker:", out["broker_card"])
            self.assertIn("Replay:", out["replay_card"])
            self.assertIn("checked 2026-03-24T12:00:00+00:00", out["broker_detail"])
            self.assertIn("verify blocked", out["broker_detail"])
            self.assertIsInstance(out["advisory_meta"], dict)
        finally:
            ta._load_long_daily_history = old_load_long
            ta.load_context_daily_history = old_load_ctx
            ta.load_recent_headlines = old_load_news
            ta.get_options_snapshot = old_get_options
            ta.get_event_snapshot = old_get_event
            ta.get_breadth_snapshot = old_get_breadth
            ta.get_macro_event_snapshot = old_get_macro
            ta.get_catalyst_provider_snapshot = old_get_catalysts
            ta.get_liquidity_snapshot = old_get_liquidity
            ta.get_news_health = old_get_news_health
            ta.INSTRUMENTS.clear()
            ta.INSTRUMENTS.update(old_instruments)
            ae.load_recent_headlines = old_adv_news
            ae.get_event_snapshot = old_adv_event
            ae.load_readiness_state = old_adv_readiness
            ae.get_macro_event_snapshot = old_adv_macro
            ae.get_breadth_snapshot = old_adv_breadth
            ae.get_catalyst_provider_snapshot = old_adv_catalysts
            ae.get_liquidity_snapshot = old_adv_liquidity
            ae.get_news_health = old_adv_news_health
            ta.get_execution_dashboard_snapshot = old_exec
            ta.get_performance_snapshot = old_perf
            ta.get_replay_dashboard_snapshot = old_replay
            ta.get_alert_snapshot = old_alerts
            ta.get_opportunity_scan = old_scan

    def test_build_trader_dashboard_keeps_reason_rows_when_sections_fail(self):
        prices = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.8, 102.6],
                "volume": [1000.0, 1200.0, 1300.0],
            },
            index=pd.date_range("2026-03-19 14:30:00+00:00", periods=3, freq="2h"),
        )
        daily_long = pd.DataFrame(
            {
                "open": [95.0, 96.0, 97.0, 98.0, 99.0, 100.0],
                "high": [96.0, 97.0, 98.0, 99.0, 100.0, 103.0],
                "low": [94.0, 95.0, 96.0, 97.0, 98.0, 99.0],
                "close": [95.5, 96.5, 97.5, 98.5, 99.5, 102.0],
                "volume": [1000.0] * 6,
            },
            index=pd.date_range("2026-03-10", periods=6, freq="B"),
        )

        def _boom(*args, **kwargs):
            raise RuntimeError("feed exploded")

        old_load_long = ta._load_long_daily_history
        old_get_options = ta.get_options_snapshot
        old_load_news = ta.load_recent_headlines
        old_get_news_health = ta.get_news_health
        old_build_advisory = ta.build_advisory_snapshot
        old_exec = ta.get_execution_dashboard_snapshot
        old_perf = ta.get_performance_snapshot
        old_replay = ta.get_replay_dashboard_snapshot
        old_alerts = ta.get_alert_snapshot
        old_scan = ta.get_opportunity_scan
        try:
            ta._load_long_daily_history = lambda instrument_key: daily_long.copy()
            ta.get_options_snapshot = _boom
            ta.load_recent_headlines = _boom
            ta.get_news_health = lambda symbol: {"status": "missing", "headline_state": "missing", "history_state": "missing"}
            ta.build_advisory_snapshot = _boom
            ta.get_execution_dashboard_snapshot = _boom
            ta.get_performance_snapshot = lambda symbol=None: {"summary": "No closed paper trades yet", "rows": []}
            ta.get_replay_dashboard_snapshot = lambda symbol=None: {"card": "Replay: unavailable", "detail": "No replay session", "rows": []}
            ta.get_alert_snapshot = lambda *args, **kwargs: {"summary": "No active trader alerts", "rows": []}
            ta.get_opportunity_scan = lambda *args, **kwargs: {"summary": "No ranked opportunities", "rows": []}

            out = ta.build_trader_dashboard(prices, "QQQ", predictions={})

            self.assertTrue(out["options_rows"])
            self.assertIn("feed exploded", out["options_rows"][0][2])
            self.assertTrue(out["news_rows"])
            self.assertIn("feed exploded", out["news_rows"][0][2])
            self.assertTrue(out["quality_rows"])
            self.assertTrue(out["execution_rows"])
            self.assertIn("feed exploded", out["quality_detail"])
            self.assertIn("feed exploded", out["execution_rows"][0][2])
        finally:
            ta._load_long_daily_history = old_load_long
            ta.get_options_snapshot = old_get_options
            ta.load_recent_headlines = old_load_news
            ta.get_news_health = old_get_news_health
            ta.build_advisory_snapshot = old_build_advisory
            ta.get_execution_dashboard_snapshot = old_exec
            ta.get_performance_snapshot = old_perf
            ta.get_replay_dashboard_snapshot = old_replay
            ta.get_alert_snapshot = old_alerts
            ta.get_opportunity_scan = old_scan


if __name__ == "__main__":
    unittest.main()
