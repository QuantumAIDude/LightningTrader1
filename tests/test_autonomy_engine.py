import json
import os
import tempfile
import unittest

import pandas as pd

import autonomy_engine as auto


def _sample_df(now_utc: pd.Timestamp) -> pd.DataFrame:
    idx = pd.date_range(end=now_utc, periods=40, freq="2min")
    return pd.DataFrame(
        {
            "open": [500.0 + 0.05 * i for i in range(len(idx))],
            "high": [500.1 + 0.05 * i for i in range(len(idx))],
            "low": [499.9 + 0.05 * i for i in range(len(idx))],
            "close": [500.02 + 0.05 * i for i in range(len(idx))],
            "volume": [1000.0 + 20.0 * (i % 5) for i in range(len(idx))],
            "bid": [499.99 + 0.05 * i for i in range(len(idx))],
            "ask": [500.01 + 0.05 * i for i in range(len(idx))],
        },
        index=idx,
    )


def _sample_advisory(direction: str = "UP") -> dict:
    side = "Bullish" if direction == "UP" else "Bearish"
    return {
        "quality_meta": {"score": 0.74, "no_trade": False},
        "plan_meta": {"bias": side, "entry": 501.0, "stop": 499.5 if direction == "UP" else 502.5, "target1": 503.0 if direction == "UP" else 499.0},
        "pred_state": {
            "intraday": {
                "direction": direction,
                "confidence": 63.0,
                "edge_bps": 8.0,
                "position_size": 0.55,
            }
        },
        "liquidity_meta": {"tradability_score": 0.82},
        "news_meta": {"alignment": "neutral", "headline_state": "ok", "fresh_high_impact_count": 0, "top_item": {}},
        "macro_meta": {"risk_score": 0.20},
        "catalyst_meta": {"catalyst_risk_score": 0.10},
        "expression_meta": {
            "stock": {"label": "Primary", "detail": "Direct QQQ stock is the primary autonomy path."},
            "options": {"label": "Context only", "detail": "Options are advisory-only here."},
        },
    }


def _sample_execution(qty: int = 10, route_mode: str = "local_paper", route_label: str = "Local Paper") -> dict:
    return {
        "recommendation": {"allowed": True, "qty": qty, "reason": "ok"},
        "route_label": route_label,
        "runtime": {
            "routing_mode": route_mode,
            "route_label": route_label,
            "live_snapshot": {"account": {"equity": 100000.0, "cash": 100000.0}, "positions": [], "orders": [], "fetched_at": "2026-04-01T15:00:00+00:00"},
        },
        "portfolio": {"equity": 100000.0},
        "diagnostics": {"checked_at": "2026-04-01T15:00:00+00:00"},
    }


def _validation_evidence(status: str = "passed_paper_candidate") -> dict:
    return {
        "validation_status": status,
        "validation_method": "walk_forward",
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "leakage_controls": {
            "uses_strict_oos": True,
            "uses_purging": False,
            "uses_embargo": False,
            "uses_untouched_holdout": status in {"passed_paper_candidate", "passed_autonomy_candidate"},
            "notes": "unit-test validation evidence",
        },
        "economic_metrics": {
            "gross_return": 0.024,
            "net_return_after_costs": 0.013,
            "estimated_spread_cost": 0.001,
            "estimated_slippage_cost": 0.001,
            "transaction_cost_assumption": "spread+slippage",
            "trade_count": 18,
            "max_drawdown": -0.04,
        },
        "calibration_metrics": {
            "directional_accuracy": 0.58,
            "prediction_error": 0.018,
            "sample_count": 90,
        },
        "drift_metrics": {"realized_error_drift": False, "regime_shift_flag": False},
    }


def _readiness_override(status: str = "passed_paper_candidate", *, live_ready: bool = True, pass_streak_days: int = 5) -> dict:
    return {
        "live_ready": live_ready,
        "fresh": True,
        "pass_streak_days": pass_streak_days,
        "required_streak_days": 3,
        "validation_evidence": _validation_evidence(status),
        "evaluation": {
            "metrics": {
                "horizon_weighted_score": 0.72,
                "intraday_window_hit_rate": 0.70,
                "daily_window_hit_rate": 0.66,
                "weekend_window_hit_rate": 0.63,
            },
            "thresholds": {
                "min_horizon_score": 0.55,
                "min_intraday_window_hit_rate": 0.60,
                "min_daily_window_hit_rate": 0.58,
                "min_weekend_window_hit_rate": 0.55,
            },
        },
    }


class AutonomyEngineTests(unittest.TestCase):
    def test_shadow_mode_logs_entry_without_submitting_order(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_shadow_only")
                auto.set_autonomy_mode("QQQ", "SHADOW")
                out = auto.evaluate_autonomy_cycle("QQQ", _sample_df(now_utc), advisory=_sample_advisory(), execution_snapshot=_sample_execution())
                self.assertEqual(out["decision"]["status"], "shadow_entry")
                with open(auto._state_path("QQQ"), "r") as handle:
                    state = json.load(handle)
                self.assertIsNotNone(state["shadow_position"])
                with open(auto._audit_path("QQQ"), "r") as handle:
                    lines = handle.readlines()
                self.assertGreaterEqual(len(lines), 3)  # mode change + shadow entry + observation
                self.assertTrue(any("shadow_entry" in line for line in lines))
                self.assertTrue(any("shadow_observation" in line for line in lines))
                obs = json.loads(lines[-1])
                self.assertEqual(obs.get("kind"), "shadow_observation")
                self.assertTrue(bool(obs.get("clean_cycle")))
                self.assertTrue(bool(obs.get("allowed")))
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready

    def test_kill_switch_blocks_new_entries(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_paper_candidate")
                auto.set_autonomy_mode("QQQ", "PAPER AUTO")
                auto.set_autonomy_kill_switch("QQQ", True, reason="test halt")
                out = auto.evaluate_autonomy_cycle("QQQ", _sample_df(now_utc), advisory=_sample_advisory(), execution_snapshot=_sample_execution())
                self.assertIn(out["decision"]["status"], {"halted", "blocked"})
                self.assertTrue(out["kill_switch"])
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready

    def test_flat_by_close_exits_shadow_position(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 19:50:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_shadow_only")
                auto.set_autonomy_mode("QQQ", "SHADOW")
                state = auto._load_state("QQQ")
                state["shadow_position"] = {"side": "buy", "qty": 5, "entry_price": 500.0, "opened_at": "2026-04-01T18:30:00+00:00"}
                auto._save_state("QQQ", state)
                out = auto.evaluate_autonomy_cycle("QQQ", _sample_df(now_utc), advisory=_sample_advisory(), execution_snapshot=_sample_execution())
                self.assertEqual(out["decision"]["status"], "shadow_exit")
                saved = auto._load_state("QQQ")
                self.assertIsNone(saved.get("shadow_position"))
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready

    def test_paper_auto_places_routed_order_when_eligible(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        old_place = auto.place_routed_order
        old_exec = auto.get_execution_dashboard_snapshot
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_paper_candidate")
                auto.place_routed_order = lambda *args, **kwargs: {"status": "filled", "order_id": "auto-1", "routing_mode": "local_paper"}
                auto.get_execution_dashboard_snapshot = lambda symbol, current_price=None, advisory=None, profile_name=None: {"rows": [("Route", "Local Paper", "Updated")], "portfolio_rows": [("Equity", "$100,000", "Cash $100,000")]}
                auto.set_autonomy_mode("QQQ", "PAPER AUTO")
                out = auto.evaluate_autonomy_cycle("QQQ", _sample_df(now_utc), advisory=_sample_advisory(), execution_snapshot=_sample_execution())
                self.assertEqual(out["decision"]["status"], "filled")
                self.assertIsInstance(out.get("execution_override"), dict)
                saved = auto._load_state("QQQ")
                self.assertIsNotNone(saved.get("open_trade"))
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready
                auto.place_routed_order = old_place
                auto.get_execution_dashboard_snapshot = old_exec

    def test_live_auto_blocks_new_short_entries(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_autonomy_candidate", pass_streak_days=6)
                auto.set_autonomy_mode("QQQ", "LIVE AUTO")
                out = auto.evaluate_autonomy_cycle(
                    "QQQ",
                    _sample_df(now_utc),
                    advisory=_sample_advisory(direction="DOWN"),
                    execution_snapshot=_sample_execution(route_mode="alpaca_live", route_label="Alpaca Live"),
                )
                self.assertEqual(out["stock_path"]["label"], "Blocked")
                self.assertIn("live short autonomy", out["stock_path"]["detail"])
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready

    def test_stock_path_uses_watchlist_label_for_interesting_but_blocked_signal(self):
        now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
        advisory = _sample_advisory()
        advisory["quality_meta"] = {"score": 0.62, "no_trade": True}
        advisory["pred_state"]["intraday"] = {
            "direction": "-",
            "confidence": 54.0,
            "edge_bps": -0.4,
            "position_size": 0.0,
        }
        advisory["signal_translation"] = {
            "headline": "Bullish Watchlist",
            "detail": "A bullish raw edge exists, but costs and uncertainty still remove the finished edge.",
            "watchlist": True,
            "operator_side": "buy",
        }
        out = auto._stock_path_status(
            "QQQ",
            "shadow",
            _sample_df(now_utc),
            advisory,
            _sample_execution(),
            now_utc=now_utc,
            readiness_override=_readiness_override("passed_shadow_only"),
        )
        self.assertFalse(out["allowed"])
        self.assertEqual(out["label"], "Watchlist Long")
        self.assertIn("blocked by", out["detail"])

    def test_non_validated_symbol_stays_shadow_only(self):
        now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
        old_supported = list(auto.AUTONOMY_SUPPORTED_SYMBOLS)
        try:
            auto.AUTONOMY_SUPPORTED_SYMBOLS[:] = ["QQQ"]
            shadow = auto._stock_path_status(
                "AAPL",
                "shadow",
                _sample_df(now_utc),
                _sample_advisory(),
                _sample_execution(),
                now_utc=now_utc,
                readiness_override=_readiness_override("passed_shadow_only"),
            )
            actual = auto._stock_path_status(
                "AAPL",
                "paper_auto",
                _sample_df(now_utc),
                _sample_advisory(),
                _sample_execution(),
                now_utc=now_utc,
                readiness_override=_readiness_override("passed_shadow_only"),
            )
            self.assertEqual(shadow["validation_label"], "Shadow only")
            self.assertFalse(any("not validated" in reason for reason in shadow["reasons"]))
            self.assertTrue(any("not validated" in reason for reason in actual["reasons"]))
        finally:
            auto.AUTONOMY_SUPPORTED_SYMBOLS[:] = old_supported

    def test_stock_path_blocks_when_regime_is_unknown(self):
        now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
        short_df = _sample_df(now_utc).tail(8)
        out = auto._stock_path_status(
            "QQQ",
            "shadow",
            short_df,
            _sample_advisory(),
            _sample_execution(),
            now_utc=now_utc,
            readiness_override=_readiness_override("passed_shadow_only"),
        )
        self.assertFalse(out["allowed"])
        self.assertEqual((out.get("risk_summary") or {}).get("primary_blocker"), "regime is unknown or evidence is insufficient")
        self.assertTrue(any("regime is unknown or evidence is insufficient" in reason for reason in out["reasons"]))

    def test_paper_auto_requires_paper_autonomy_deployment_stage(self):
        now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
        advisory = _sample_advisory()
        readiness = {
            "live_ready": False,
            "fresh": True,
            "pass_streak_days": 1,
            "required_streak_days": 3,
            "validation_evidence": _validation_evidence("passed_shadow_only"),
            "evaluation": {
                "metrics": {
                    "horizon_weighted_score": 0.62,
                    "intraday_window_hit_rate": 0.66,
                    "daily_window_hit_rate": 0.61,
                    "weekend_window_hit_rate": 0.58,
                },
                "thresholds": {
                    "min_horizon_score": 0.55,
                    "min_intraday_window_hit_rate": 0.60,
                    "min_daily_window_hit_rate": 0.58,
                    "min_weekend_window_hit_rate": 0.55,
                },
            },
        }

        out = auto._stock_path_status(
            "QQQ",
            "paper_auto",
            _sample_df(now_utc),
            advisory,
            _sample_execution(),
            now_utc=now_utc,
            readiness_override=readiness,
        )

        self.assertFalse(out["allowed"])
        self.assertEqual(out["deployment_label"], "Advisory + shadow")
        self.assertTrue(any("paper-autonomy deployment stage" in reason for reason in out["reasons"]))
        self.assertTrue(any("validation evidence blocks paper autonomy" in reason for reason in out["reasons"]))

    def test_missing_validation_evidence_blocks_shadow_autonomy_path(self):
        now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
        readiness = _readiness_override("passed_shadow_only")
        readiness.pop("validation_evidence", None)

        out = auto._stock_path_status(
            "QQQ",
            "shadow",
            _sample_df(now_utc),
            _sample_advisory(),
            _sample_execution(),
            now_utc=now_utc,
            readiness_override=readiness,
        )

        self.assertFalse(out["allowed"])
        self.assertEqual(out["validation_evidence_label"], "Validation missing")
        self.assertTrue(any("validation evidence does not support shadow mode yet" in reason for reason in out["reasons"]))

    def test_in_memory_cycle_does_not_persist_files(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                state = auto._default_state("QQQ")
                state["mode"] = "shadow"
                out = auto.evaluate_autonomy_cycle(
                    "QQQ",
                    _sample_df(now_utc),
                    advisory=_sample_advisory(),
                    execution_snapshot=_sample_execution(),
                    now_utc=now_utc,
                    state_override=state,
                    persist=False,
                    readiness_override=_readiness_override("passed_shadow_only"),
                )
                self.assertEqual(out["decision"]["status"], "shadow_entry")
                self.assertFalse(os.path.exists(auto._state_path("QQQ")))
                self.assertFalse(os.path.exists(auto._audit_path("QQQ")))
                self.assertIn("events", out)
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now

    def test_blocked_shadow_cycle_logs_observation_evidence(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_shadow_only", live_ready=False)
                auto.set_autonomy_mode("QQQ", "SHADOW")
                advisory = _sample_advisory()
                advisory["quality_meta"]["no_trade"] = True
                out = auto.evaluate_autonomy_cycle("QQQ", _sample_df(now_utc), advisory=advisory, execution_snapshot=_sample_execution())
                self.assertEqual(out["decision"]["status"], "blocked")
                with open(auto._audit_path("QQQ"), "r") as handle:
                    rows = [json.loads(line) for line in handle if line.strip()]
                obs = rows[-1]
                self.assertEqual(obs.get("kind"), "shadow_observation")
                self.assertFalse(bool(obs.get("allowed")))
                self.assertTrue(bool(obs.get("clean_cycle")))
                self.assertTrue(bool(obs.get("eligible_environment")))
                self.assertTrue(bool(obs.get("blocked_by_deployment")))
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready

    def test_strategy_autonomy_uses_selected_strategy_signal(self):
        now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
        advisory = _sample_advisory()
        advisory["pred_state"]["intraday"] = {
            "direction": "-",
            "confidence": 50.0,
            "edge_bps": 0.0,
            "position_size": 0.0,
        }
        df = _sample_df(now_utc)
        out = auto._stock_path_status(
            "QQQ",
            "shadow",
            df,
            advisory,
            _sample_execution(),
            now_utc=now_utc,
            readiness_override=_readiness_override("passed_shadow_only"),
            controller_state={"decision_style": "strategy_autonomy", "strategy_key": "trend_following"},
        )

        self.assertEqual(out["decision_style"], "strategy_autonomy")
        self.assertEqual(out["strategy_key"], "trend_following")
        self.assertEqual(out["side"], "buy")
        self.assertIn("Trend Following", out["detail"])
        self.assertIsInstance(out.get("execution_advisory"), dict)

    def test_controller_stop_loss_sets_kill_switch(self):
        old_state_path = auto._state_path
        old_audit_path = auto._audit_path
        old_now = auto._utc_now
        old_ready = auto.load_readiness_state
        with tempfile.TemporaryDirectory() as tmp:
            try:
                now_utc = pd.Timestamp("2026-04-01 15:00:00+00:00")
                auto._state_path = lambda symbol: os.path.join(tmp, f"state_{symbol}.json")
                auto._audit_path = lambda symbol: os.path.join(tmp, f"audit_{symbol}.jsonl")
                auto._utc_now = lambda: now_utc
                auto.load_readiness_state = lambda symbol: _readiness_override("passed_paper_candidate")
                state = auto._default_state("QQQ")
                state["mode"] = "paper_auto"
                state["session_date"] = "2026-04-01"
                state["realized_pnl_today"] = -125.0
                state["loss_limit_dollars"] = 100.0
                out = auto.evaluate_autonomy_cycle(
                    "QQQ",
                    _sample_df(now_utc),
                    advisory=_sample_advisory(),
                    execution_snapshot=_sample_execution(),
                    state_override=state,
                    now_utc=now_utc,
                )

                self.assertTrue(out["kill_switch"])
                self.assertEqual(out["stock_path"]["label"], "Halted")
                self.assertIn("stop-loss", out["stock_path"]["detail"])
                self.assertIn(out["decision"]["status"], {"halted", "blocked"})
            finally:
                auto._state_path = old_state_path
                auto._audit_path = old_audit_path
                auto._utc_now = old_now
                auto.load_readiness_state = old_ready


if __name__ == "__main__":
    unittest.main()
