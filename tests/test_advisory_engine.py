import unittest

import pandas as pd

import advisory_engine as ae


class AdvisoryEngineTests(unittest.TestCase):
    def test_signal_quality_uses_raw_direction_when_finished_signal_is_blocked(self):
        pred_state = {
            "intraday": {
                "direction": "-",
                "raw_direction": "UP",
                "p_up": 0.50,
                "raw_p_up": 0.63,
                "agreement": 0.61,
                "point": 100.05,
                "edge_bps": -0.3,
                "raw_edge_bps": 2.6,
                "raw_pred_ret_bps": 3.4,
                "cost_bps_est": 1.7,
                "uncertainty_penalty_bps": 1.2,
                "uncertainty_bps": 6.2,
                "position_size": 0.0,
                "raw_position_size": 0.28,
                "sig_bucket": "Medium",
            }
        }
        quality = ae._signal_quality(
            100.0,
            pred_state,
            noise={"noise_dollars": 0.08},
            spread_frac=0.0001,
            volume_state=("Mixed", 1.0),
            regime={"regime_conf": 0.66, "Bullish": 58.0, "Bearish": 17.0, "Neutral": 25.0},
            event_snapshot={"days_to_earnings": 9.0},
            news_items=[],
            news_snapshot={"alignment": "neutral", "headline_state": "ok", "tape_confirmation": "no_signal"},
            macro_snapshot={"risk_score": 0.20},
            breadth_snapshot={"risk_on_score": 0.55, "status": "proxy_only"},
            liquidity_snapshot={"tradability_score": 0.75},
            catalyst_snapshot={"catalyst_risk_score": 0.05, "recent": []},
        )
        self.assertGreater(quality.get("confidence") or 0.0, 0.60)
        self.assertIn("raw signal exists, but cost and uncertainty remove the tradable edge", quality.get("reasons") or [])

    def test_signal_quality_distinguishes_cost_crushed_signal(self):
        pred_state = {
            "intraday": {
                "direction": "-",
                "p_up": 0.54,
                "agreement": 0.58,
                "point": 100.05,
                "edge_bps": -0.4,
                "raw_edge_bps": 3.8,
                "raw_pred_ret_bps": 5.2,
                "cost_bps_est": 2.0,
                "uncertainty_penalty_bps": 2.3,
                "uncertainty_bps": 8.0,
                "position_size": 0.0,
                "sig_bucket": "Medium",
            }
        }
        quality = ae._signal_quality(
            100.0,
            pred_state,
            noise={"noise_dollars": 0.08},
            spread_frac=0.0001,
            volume_state=("Mixed", 1.0),
            regime={"regime_conf": 0.62, "Bullish": 52.0, "Bearish": 18.0, "Neutral": 30.0},
            event_snapshot={"days_to_earnings": 9.0},
            news_items=[],
            news_snapshot={"alignment": "neutral", "headline_state": "ok", "tape_confirmation": "no_signal"},
            macro_snapshot={"risk_score": 0.20},
            breadth_snapshot={"risk_on_score": 0.55, "status": "proxy_only"},
            liquidity_snapshot={"tradability_score": 0.75},
            catalyst_snapshot={"catalyst_risk_score": 0.05, "recent": []},
        )
        self.assertIn("raw signal exists, but cost and uncertainty remove the tradable edge", quality.get("reasons") or [])

    def test_signal_translation_marks_cost_crushed_watchlist(self):
        translation = ae._signal_translation(
            {
                "intraday": {
                    "direction": "-",
                    "raw_pred_ret_bps": 5.2,
                    "raw_edge_bps": 3.8,
                    "edge_bps": -0.4,
                    "cost_bps_est": 2.0,
                    "uncertainty_penalty_bps": 2.3,
                    "confidence": 54.0,
                    "position_size": 0.0,
                    "shrink_factor": 0.32,
                }
            },
            {
                "score": 0.61,
                "no_trade": True,
                "reasons": ["raw signal exists, but cost and uncertainty remove the tradable edge"],
            },
            {"alignment": "neutral", "tape_confirmation": "underreacting"},
        )
        self.assertTrue(translation["watchlist"])
        self.assertEqual(translation["headline"], "Bullish Watchlist")
        self.assertEqual(translation["economics_state_label"], "Uncertainty-crushed")
        self.assertIn("raw 3.8 bps -> finished -0.4 bps", translation["detail"])

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
        old_news_health = ae.get_news_health
        try:
            ae._load_long_daily_history = lambda instrument_key: daily_long.copy()
            ae.get_event_snapshot = lambda symbol: {"status": "ok", "days_to_earnings": 9.0}
            ae.get_macro_event_snapshot = lambda: {"risk_score": 0.20, "events": [], "provider_note": "Local macro calendar file"}
            ae.get_breadth_snapshot = lambda symbol: {"risk_on_score": 0.62, "rows": [("Proxy Breadth", "62/100", "Risk-on | context-derived")]}
            ae.get_catalyst_provider_snapshot = lambda symbol: {"rows": [("SEC Feed", "Stub", "Adapter ready")], "recent": []}
            ae.get_liquidity_snapshot = lambda df, symbol: {"rows": [("Tradability", "70/100", "Good")], "tradability_score": 0.70}
            ae.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "high_impact_recent": 1}
            ae.load_recent_headlines = lambda symbol, limit=8: [
                {
                    "published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1),
                    "headline": "Policy risk eases",
                    "label": "positive",
                    "score": 0.25,
                    "political_relevance": 0.45,
                    "event_label": "Macro",
                    "impact_score": 0.62,
                    "impact_horizon": "intraday",
                    "importance_score": 0.48,
                    "direction_score": 0.18,
                    "signal_note": "Macro release can dominate intraday tape",
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
            self.assertIsInstance(out["regime_meta"], dict)
            self.assertIsInstance(out["risk_meta"], dict)
            self.assertIsInstance(out["decision_payload"], dict)
            self.assertIsInstance(out["decision_rows"], list)
            self.assertIsInstance(out["news_meta"], dict)
            self.assertIsInstance(out["expression_meta"], dict)
            self.assertIsInstance(out["signal_translation"], dict)
            self.assertEqual(out["news_meta"]["dominant_event"], "Macro")
            self.assertTrue(any(row[0] == "Trader Lens" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Regime State" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Risk Context" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "News Context" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Stock Path" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Options Path" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Deployment" for row in out["reliability_rows"]))
            self.assertIsInstance(out["deployment_meta"], dict)
            self.assertIn(out["regime_meta"].get("label"), {"trend_up", "trend_down", "high_volatility", "low_volatility", "range_bound", "event_risk_or_news_disturbed", "unknown_or_insufficient_evidence"})
            self.assertIn("blocked", out["risk_meta"])
            self.assertIn("action", out["decision_payload"])
            self.assertIn("thesis", out["decision_payload"])
            self.assertIn("blockers", out["decision_payload"])
            self.assertIn("next_best_action", out["decision_payload"])
        finally:
            ae._load_long_daily_history = old_load_long
            ae.get_event_snapshot = old_event
            ae.load_recent_headlines = old_news
            ae.load_readiness_state = old_readiness
            ae.get_macro_event_snapshot = old_macro
            ae.get_breadth_snapshot = old_breadth
            ae.get_catalyst_provider_snapshot = old_catalysts
            ae.get_liquidity_snapshot = old_liquidity
            ae.get_news_health = old_news_health

    def test_build_advisory_snapshot_surfaces_hard_readiness_block_in_action_posture(self):
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
        old_news_health = ae.get_news_health
        old_deployment = ae.summarize_deployment_readiness
        try:
            ae._load_long_daily_history = lambda instrument_key: daily_long.copy()
            ae.get_event_snapshot = lambda symbol: {"status": "ok", "days_to_earnings": 9.0}
            ae.get_macro_event_snapshot = lambda: {"risk_score": 0.20, "events": [], "provider_note": "Local macro calendar file"}
            ae.get_breadth_snapshot = lambda symbol: {"risk_on_score": 0.62, "rows": [("Proxy Breadth", "62/100", "Risk-on | context-derived")]}
            ae.get_catalyst_provider_snapshot = lambda symbol: {"rows": [("SEC Feed", "Stub", "Adapter ready")], "recent": []}
            ae.get_liquidity_snapshot = lambda df, symbol: {"rows": [("Tradability", "70/100", "Good")], "tradability_score": 0.70}
            ae.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "high_impact_recent": 1}
            ae.load_recent_headlines = lambda symbol, limit=8: [
                {
                    "published_at": pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1),
                    "headline": "Policy risk eases",
                    "label": "positive",
                    "score": 0.25,
                    "political_relevance": 0.45,
                    "event_label": "Macro",
                    "impact_score": 0.62,
                    "impact_horizon": "intraday",
                    "importance_score": 0.48,
                    "direction_score": 0.18,
                    "signal_note": "Macro release can dominate intraday tape",
                }
            ]
            ae.load_readiness_state = lambda symbol: {"fresh": True, "live_ready": False}
            ae.summarize_deployment_readiness = lambda symbol, state=None, news_health=None: {
                "stage_label": "Advisory only",
                "operator_state_key": "blocked_stale_data",
                "operator_state_label": "Blocked: stale data",
                "operator_state_detail": "Signal direction may exist, but stale market inputs block trustworthy execution decisions right now.",
                "symbol_maturity_label": "Advisory-usable",
                "symbol_maturity_detail": "This symbol has trained models plus readiness metrics, so the advisory stack is usable even though autonomy remains gated.",
                "evidence_state_label": "Stale market data",
                "evidence_state_detail": "Price inputs are stale enough to block trustworthy operator use right now.",
                "rollout_label": "Rollout listed / still gated",
                "primary_next_requirement": "Refresh market data until the critical stale-data guard clears.",
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
                }
            }
            out = ae.build_advisory_snapshot(df, "QQQ", predictions=preds, options_snapshot={"expected_move_dollars": 2.1})
            self.assertEqual(out["action_card"], "Action: Stand Aside")
            self.assertIn("execution is blocked by platform evidence", out["action_detail"])
            self.assertIn("Refresh market data until the critical stale-data guard clears.", out["action_detail"])
            self.assertIn("Blocked: stale data", out["quality_detail"])
            self.assertTrue(any(row[0] == "Execution Posture" and row[1] == "Blocked: stale data" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Symbol Maturity" and row[1] == "Advisory-usable" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Evidence State" and row[1] == "Stale market data" for row in out["quality_rows"]))
            self.assertTrue(any(row[0] == "Operator State" and row[1] == "Blocked: stale data" for row in out["setup_rows"]))
        finally:
            ae._load_long_daily_history = old_load_long
            ae.get_event_snapshot = old_event
            ae.load_recent_headlines = old_news
            ae.load_readiness_state = old_readiness
            ae.get_macro_event_snapshot = old_macro
            ae.get_breadth_snapshot = old_breadth
            ae.get_catalyst_provider_snapshot = old_catalysts
            ae.get_liquidity_snapshot = old_liquidity
            ae.get_news_health = old_news_health
            ae.summarize_deployment_readiness = old_deployment

    def test_advisory_decision_payload_marks_unknown_regime_as_watchlist_or_stand_aside(self):
        idx = pd.date_range("2026-03-18 14:30:00+00:00", periods=8, freq="2min")
        df = pd.DataFrame(
            {
                "open": [100.0 + 0.03 * i for i in range(8)],
                "high": [100.2 + 0.03 * i for i in range(8)],
                "low": [99.8 + 0.03 * i for i in range(8)],
                "close": [100.05 + 0.03 * i for i in range(8)],
                "volume": [1200.0 + (i % 8) * 50.0 for i in range(8)],
                "bid": [99.99 + 0.03 * i for i in range(8)],
                "ask": [100.01 + 0.03 * i for i in range(8)],
            },
            index=idx,
        )
        old_event = ae.get_event_snapshot
        old_news = ae.load_recent_headlines
        old_readiness = ae.load_readiness_state
        old_macro = ae.get_macro_event_snapshot
        old_breadth = ae.get_breadth_snapshot
        old_catalysts = ae.get_catalyst_provider_snapshot
        old_liquidity = ae.get_liquidity_snapshot
        old_news_health = ae.get_news_health
        try:
            ae.get_event_snapshot = lambda symbol: {"status": "ok", "days_to_earnings": 9.0}
            ae.get_macro_event_snapshot = lambda: {"risk_score": 0.20, "events": [], "provider_note": "Local macro calendar file"}
            ae.get_breadth_snapshot = lambda symbol: {"risk_on_score": 0.62, "rows": [("Proxy Breadth", "62/100", "Risk-on | context-derived")]}
            ae.get_catalyst_provider_snapshot = lambda symbol: {"rows": [("SEC Feed", "Stub", "Adapter ready")], "recent": []}
            ae.get_liquidity_snapshot = lambda df, symbol: {"rows": [("Tradability", "70/100", "Good")], "tradability_score": 0.70}
            ae.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "high_impact_recent": 0}
            ae.load_recent_headlines = lambda symbol, limit=8: []
            ae.load_readiness_state = lambda symbol: {"fresh": True, "live_ready": False}
            preds = {
                "intraday": {
                    "consensus": {
                        "direction": "UP",
                        "pred_price": 102.7,
                        "confidence": 64.0,
                        "p_up": 0.64,
                        "edge_bps": 8.0,
                        "raw_edge_bps": 10.0,
                        "raw_pred_ret_bps": 11.0,
                        "cost_bps_est": 1.4,
                        "uncertainty_penalty_bps": 0.9,
                        "uncertainty_bps": 6.0,
                        "position_size": 0.45,
                        "sig_bucket": "Medium",
                    }
                }
            }
            out = ae.build_advisory_snapshot(df, "QQQ", predictions=preds, options_snapshot={"expected_move_dollars": 2.1})
            self.assertEqual(out["decision_payload"]["market_regime"], "unknown_or_insufficient_evidence")
            self.assertIn(out["decision_payload"]["action"], {"watchlist", "stand_aside"})
            self.assertIn("regime is unknown or evidence is insufficient", out["decision_payload"]["blockers"])
        finally:
            ae.get_event_snapshot = old_event
            ae.load_recent_headlines = old_news
            ae.load_readiness_state = old_readiness
            ae.get_macro_event_snapshot = old_macro
            ae.get_breadth_snapshot = old_breadth
            ae.get_catalyst_provider_snapshot = old_catalysts
            ae.get_liquidity_snapshot = old_liquidity
            ae.get_news_health = old_news_health

    def test_advisory_decision_payload_distinguishes_cost_crushed_and_risk_blocked(self):
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
        old_event = ae.get_event_snapshot
        old_news = ae.load_recent_headlines
        old_readiness = ae.load_readiness_state
        old_macro = ae.get_macro_event_snapshot
        old_breadth = ae.get_breadth_snapshot
        old_catalysts = ae.get_catalyst_provider_snapshot
        old_liquidity = ae.get_liquidity_snapshot
        old_news_health = ae.get_news_health
        old_risk = ae.risk_summary_for_symbol
        try:
            ae.get_event_snapshot = lambda symbol: {"status": "ok", "days_to_earnings": 9.0}
            ae.get_macro_event_snapshot = lambda: {"risk_score": 0.20, "events": [], "provider_note": "Local macro calendar file"}
            ae.get_breadth_snapshot = lambda symbol: {"risk_on_score": 0.62, "rows": [("Proxy Breadth", "62/100", "Risk-on | context-derived")]}
            ae.get_catalyst_provider_snapshot = lambda symbol: {"rows": [("SEC Feed", "Stub", "Adapter ready")], "recent": []}
            ae.get_liquidity_snapshot = lambda df, symbol: {"rows": [("Tradability", "70/100", "Good")], "tradability_score": 0.70}
            ae.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "high_impact_recent": 0}
            ae.load_recent_headlines = lambda symbol, limit=8: []
            ae.load_readiness_state = lambda symbol: {"fresh": True, "live_ready": False}
            ae.risk_summary_for_symbol = lambda *args, **kwargs: {
                "blocked": True,
                "primary_blocker": "risk budget is too small",
                "detail": "risk budget is too small",
                "suggested_qty": 0,
                "risk_budget": {"risk_budget": 0.0},
            }
            preds = {
                "intraday": {
                    "consensus": {
                        "direction": "-",
                        "raw_direction": "UP",
                        "pred_price": 100.2,
                        "confidence": 54.0,
                        "raw_confidence": 61.0,
                        "p_up": 0.54,
                        "raw_edge_bps": 3.8,
                        "edge_bps": -0.4,
                        "raw_pred_ret_bps": 5.2,
                        "cost_bps_est": 2.0,
                        "uncertainty_penalty_bps": 2.3,
                        "uncertainty_bps": 8.0,
                        "position_size": 0.0,
                        "raw_position_size": 0.28,
                        "sig_bucket": "Medium",
                    }
                }
            }
            out = ae.build_advisory_snapshot(df, "QQQ", predictions=preds, options_snapshot={"expected_move_dollars": 2.1})
            self.assertEqual(out["decision_payload"]["economic_edge"]["state_key"], "uncertainty_crushed")
            self.assertEqual(out["decision_payload"]["action"], "watchlist")
            self.assertIn("risk budget is too small", out["decision_payload"]["blockers"])
            self.assertIn("Wait for a finished edge that stays positive after estimated costs and uncertainty.", out["decision_payload"]["next_best_action"])
            self.assertEqual(out["decision_payload"]["risk_context"]["guidance_key"], "risk_budget_blocked")
            self.assertIn("available risk budget is still too small", out["decision_payload"]["risk_context"]["guidance_detail"])
        finally:
            ae.get_event_snapshot = old_event
            ae.load_recent_headlines = old_news
            ae.load_readiness_state = old_readiness
            ae.get_macro_event_snapshot = old_macro
            ae.get_breadth_snapshot = old_breadth
            ae.get_catalyst_provider_snapshot = old_catalysts
            ae.get_liquidity_snapshot = old_liquidity
            ae.get_news_health = old_news_health
            ae.risk_summary_for_symbol = old_risk

    def test_advisory_decision_rows_pass_through_payload_fields(self):
        payload = {
            "action": "watchlist",
            "thesis": "Raw edge exists, but uncertainty is too high.",
            "market_regime": "range_bound",
            "regime_confidence": 0.68,
            "model_edge": {
                "direction": "UP",
                "raw_edge_bps": 4.2,
                "finished_edge_bps": -0.3,
                "confidence": 0.61,
            },
            "economic_edge": {
                "state_key": "uncertainty_crushed",
                "state_label": "Uncertainty-crushed",
                "detail": "Uncertainty overwhelms the finished trade.",
                "cost_bps_est": 1.2,
                "uncertainty_penalty_bps": 2.4,
            },
            "risk_context": {
                "blocked": True,
                "primary_blocker": "risk budget is too small",
                "guidance_detail": "The available risk budget is still too small for safe execution.",
                "suggested_qty": 0,
                "risk_budget": 0.0,
            },
            "evidence_strength": {
                "quality_score": 0.58,
                "signal_grade": "C",
                "symbol_maturity": "Advisory-usable",
                "evidence_state": "Monitored evidence",
            },
            "validation_status": "passed_advisory_only",
            "validation_method": "walk_forward",
            "validation_summary": "Validation supports advisory only | walk forward | trades 12",
            "economic_validation": {"gross_return": 0.02, "net_return_after_costs": 0.01},
            "leakage_controls_summary": "strict_oos=yes | purging=no | embargo=no | holdout=no",
            "validation_blockers": ["no_paper_trading_record"],
            "validation_next_best_action": "start shadow-mode paper ledger",
            "blockers": ["risk budget is too small"],
            "contradictory_evidence": ["news conflicts with model"],
            "next_best_action": "Wait for a cleaner finished edge.",
            "autonomy_posture": {
                "operator_state": "Advisory only",
                "stage": "Advisory only",
                "rollout": "Rollout listed / still gated",
            },
        }

        rows = ae.advisory_decision_rows(payload)
        labels = [row[0] for row in rows]
        self.assertIn("Decision Action", labels)
        self.assertIn("Economic Edge", labels)
        self.assertIn("Risk Context", labels)
        self.assertIn("Validation Evidence", labels)
        self.assertIn("Validation Controls", labels)
        self.assertTrue(any(row[0] == "Next Best Action" and row[2] == "Wait for a cleaner finished edge." for row in rows))
        self.assertTrue(any(row[0] == "Risk Context" and "available risk budget" in row[2] for row in rows))
        self.assertTrue(any(row[0] == "Validation Evidence" and row[1] == "Passed Advisory Only" for row in rows))

    def test_advisory_decision_payload_includes_validation_evidence_fields(self):
        payload = ae._advisory_decision_payload(
            {
                "kind": "actionable_signal",
                "operator_side": "buy",
                "watchlist": False,
                "economics_detail": "Finished edge remains positive after estimated costs.",
                "economics_state_key": "economically_viable",
            },
            {"label": "trend_up", "confidence": 0.72},
            {"blocked": False, "suggested_qty": 4, "risk_budget": {"risk_budget": 250.0}},
            {
                "operator_state_label": "Advisory only",
                "rollout_label": "Rollout listed / still gated",
                "stage_label": "Advisory only",
                "validation_evidence": {
                    "validation_status": "passed_advisory_only",
                    "validation_method": "walk_forward",
                    "summary": "Validation supports advisory only | walk forward | trades 12",
                    "economic_metrics": {"gross_return": 0.02, "net_return_after_costs": 0.01},
                    "leakage_controls_summary": "strict_oos=yes | purging=no | embargo=no | holdout=no",
                    "autonomy_blockers": ["no_paper_trading_record"],
                    "next_best_action": "start shadow-mode paper ledger",
                },
            },
            {"hard_block": False},
            {"score": 0.64, "grade": "B"},
            {"stop": 99.0},
            {"intraday": {"direction": "UP", "confidence": 63.0}},
        )

        self.assertEqual(payload["validation_status"], "passed_advisory_only")
        self.assertEqual(payload["validation_method"], "walk_forward")
        self.assertEqual(payload["economic_validation"]["net_return_after_costs"], 0.01)
        self.assertEqual(payload["validation_blockers"], ["no_paper_trading_record"])
        self.assertEqual(payload["validation_next_best_action"], "start shadow-mode paper ledger")


if __name__ == "__main__":
    unittest.main()
