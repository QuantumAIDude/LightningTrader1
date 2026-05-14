import json
import os
import tempfile
import unittest

import pandas as pd

import readiness_tracker as rt


def _write_price_history(tmp: str, symbol: str = "QQQ", rows: int = 40) -> None:
    idx = pd.date_range("2026-04-10 14:30:00+00:00", periods=rows, freq="2min")
    closes = [100.0 + 0.03 * i for i in range(rows)]
    df = pd.DataFrame(
        {
            "open": closes,
            "high": [v + 0.10 for v in closes],
            "low": [v - 0.10 for v in closes],
            "close": closes,
            "volume": [1200.0] * rows,
        },
        index=idx,
    )
    df.to_csv(os.path.join(tmp, f"prices_{str(symbol or '').upper()}.csv"))


def _validation_evidence(
    status: str = "passed_shadow_only",
    *,
    method: str = "walk_forward",
    strict_oos: bool = True,
    untouched_holdout: bool = False,
    costs: bool = True,
    gross: float = 0.025,
    net: float = 0.014,
    trade_count: int = 18,
    sample_count: int = 90,
    stale: bool = False,
) -> dict:
    return {
        "validation_status": status,
        "validation_method": method,
        "generated_at": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=45 if stale else 1)).isoformat(),
        "data_window": {
            "train_start": "2026-01-01",
            "train_end": "2026-03-01",
            "validation_start": "2026-03-02",
            "validation_end": "2026-03-20",
            "holdout_start": "2026-03-21" if untouched_holdout else None,
            "holdout_end": "2026-04-01" if untouched_holdout else None,
        },
        "leakage_controls": {
            "uses_strict_oos": strict_oos,
            "uses_purging": method == "purged_walk_forward",
            "uses_embargo": method == "purged_walk_forward",
            "uses_untouched_holdout": untouched_holdout,
            "notes": "unit-test validation evidence",
        },
        "economic_metrics": {
            "gross_return": gross,
            "net_return_after_costs": net,
            "estimated_spread_cost": 0.001 if costs else None,
            "estimated_slippage_cost": 0.001 if costs else None,
            "transaction_cost_assumption": "spread+slippage" if costs else "",
            "trade_count": trade_count,
            "average_trade_return": net / max(1, trade_count),
            "max_drawdown": -0.04,
            "win_rate": 0.56,
            "profit_factor": 1.25,
        },
        "calibration_metrics": {
            "prediction_error": 0.018,
            "directional_accuracy": 0.57,
            "confidence_calibration_error": 0.06,
            "sample_count": sample_count,
        },
        "drift_metrics": {
            "feature_drift_score": 0.04,
            "prediction_drift_score": 0.03,
            "realized_error_drift": False,
            "regime_shift_flag": False,
        },
    }


def _ready_state(validation_evidence: dict | None = None, *, live_ready: bool = True, pass_streak_days: int = 5) -> dict:
    state = {
        "fresh": True,
        "live_ready": live_ready,
        "pass_streak_days": pass_streak_days,
        "required_streak_days": 3,
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
    if validation_evidence is not None:
        state["validation_evidence"] = validation_evidence
    return state


def _ready_system_report() -> dict:
    return {
        "status": "ok",
        "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
        "models": {"missing": []},
        "predictions": {
            "4min": {
                "consensus": {
                    "direction": "UP",
                    "confidence": 63.0,
                    "edge_bps": 8.0,
                    "raw_edge_bps": 10.0,
                    "raw_pred_ret_bps": 11.0,
                    "cost_bps_est": 1.4,
                    "uncertainty_penalty_bps": 0.9,
                    "position_size": 0.55,
                },
                "diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}},
            }
        },
        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
        "horizon_accuracy": {"drift_flag": False},
    }


class ReadinessTrackerTests(unittest.TestCase):
    def test_summarize_deployment_readiness_marks_symbol_without_data_or_models_as_supported(self):
        old_load_models = rt.load_models
        old_instruments = dict(rt.INSTRUMENTS)
        try:
            rt.INSTRUMENTS["AAPL"] = {"type": "equity"}
            rt.load_models = lambda include_modes=True, instrument_key=None: {}
            out = rt.summarize_deployment_readiness(
                "AAPL",
                state={},
                system_report={"status": "fail", "data": {"error": "no price data"}},
                news_health={"status": "missing", "headline_state": "missing"},
            )
            self.assertEqual(out["stage_key"], "supported_not_trained")
            self.assertEqual(out["symbol_maturity_label"], "Supported only")
            self.assertEqual(out["evidence_state_label"], "Missing evidence")
            self.assertEqual(out["operator_state_label"], "Blocked: insufficient symbol maturity")
            self.assertIn("no price data", out["primary_blocker"])
            self.assertIn("Stage canonical price history", out["primary_next_requirement"])
            self.assertTrue(bool((out.get("status_flags") or {}).get("supported")))
            self.assertFalse(bool((out.get("status_flags") or {}).get("trained")))
            self.assertIn("canonical price data has not been staged", out["missing_evidence"])
        finally:
            rt.load_models = old_load_models
            rt.INSTRUMENTS.clear()
            rt.INSTRUMENTS.update(old_instruments)

    def test_summarize_deployment_readiness_distinguishes_shadow_from_paper_stage(self):
        state = {
            "instrument": "QQQ",
            "fresh": True,
            "live_ready": False,
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
        system_report = {
            "status": "ok",
            "models": {"missing": []},
            "predictions": {
                "4min": {
                    "diagnostics": {
                        "critical_stale": False,
                        "readiness_locked": False,
                        "diversity": {"degenerate": False},
                    }
                }
            },
            "calibration": {"stale": False, "drift_flag": False},
            "horizon_accuracy": {"drift_flag": False},
        }
        news_health = {"status": "ok", "headline_state": "ok"}

        out = rt.summarize_deployment_readiness("QQQ", state=state, system_report=system_report, news_health=news_health)

        self.assertEqual(out["stage_key"], "advisory_shadow_ready")
        self.assertTrue(out["shadow_ready"])
        self.assertFalse(out["paper_ready"])
        self.assertIn("pass streak", out["primary_blocker"])

    def test_compute_pass_streak_counts_consecutive_days(self):
        rows = [
            {"date": "2026-03-01", "passed": True, "ts": "2026-03-01T20:00:00Z"},
            {"date": "2026-03-02", "passed": True, "ts": "2026-03-02T20:00:00Z"},
            {"date": "2026-03-03", "passed": False, "ts": "2026-03-03T20:00:00Z"},
            {"date": "2026-03-04", "passed": True, "ts": "2026-03-04T20:00:00Z"},
            {"date": "2026-03-05", "passed": True, "ts": "2026-03-05T20:00:00Z"},
        ]
        streak = rt._compute_pass_streak(rows)
        self.assertEqual(streak, 2)

    def test_load_readiness_state_marks_stale_as_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            try:
                rt.DATA_DIR = tmp
                instr = "QQQ"
                path = os.path.join(tmp, f"readiness_state_{instr}.json")
                stale_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=100)).isoformat()
                with open(path, "w") as f:
                    json.dump(
                        {
                            "instrument": instr,
                            "last_update": stale_ts,
                            "live_ready": True,
                            "pass_streak_days": 9,
                            "required_streak_days": 3,
                        },
                        f,
                    )
                state = rt.load_readiness_state(instr)
                self.assertIsNotNone(state)
                self.assertFalse(bool(state.get("fresh", True)))
                self.assertFalse(bool(state.get("live_ready", True)))
                self.assertNotEqual(state.get("stage_key"), "paper_autonomy_candidate")
            finally:
                rt.DATA_DIR = old_data_dir

    def test_summarize_deployment_readiness_marks_stale_data_as_hard_operator_block(self):
        state = {
            "instrument": "QQQ",
            "fresh": True,
            "live_ready": False,
            "pass_streak_days": 0,
            "required_streak_days": 3,
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
        system_report = {
            "status": "ok",
            "data": {
                "rows": 10,
                "last_ts": "2026-04-10T15:00:00Z",
                "staleness": {"is_stale": True},
            },
            "models": {"missing": []},
            "predictions": {
                "4min": {
                    "diagnostics": {
                        "critical_stale": True,
                        "readiness_locked": False,
                        "diversity": {"degenerate": False},
                    }
                }
            },
            "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
            "horizon_accuracy": {"drift_flag": False},
        }
        news_health = {"status": "ok", "headline_state": "ok"}

        out = rt.summarize_deployment_readiness("QQQ", state=state, system_report=system_report, news_health=news_health)

        self.assertEqual(out["stage_key"], "advisory_only")
        self.assertEqual(out["symbol_maturity_label"], "Advisory-usable")
        self.assertEqual(out["evidence_state_label"], "Stale market data")
        self.assertEqual(out["operator_state_label"], "Blocked: stale data")
        self.assertIn("Refresh market data until the critical stale-data guard clears.", out["next_requirements"])
        self.assertEqual(out["primary_next_requirement"], "Refresh market data until the critical stale-data guard clears.")

    def test_update_readiness_state_persists_evidence_and_stage_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_eval = rt.evaluate_readiness_metrics
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["AAPL"] = {"type": "equity"}
                rt.evaluate_readiness_metrics = lambda instrument_key, online=False: {"instrument": instrument_key, "error": "no price data"}
                rt.load_models = lambda include_modes=True, instrument_key=None: {}

                state = rt.update_readiness_state("AAPL", online=False)

                self.assertEqual(state.get("stage_key"), "supported_not_trained")
                evidence_path = os.path.join(tmp, "deployment_evidence_AAPL.json")
                history_path = os.path.join(tmp, "deployment_stage_history_AAPL.jsonl")
                self.assertTrue(os.path.exists(evidence_path))
                self.assertTrue(os.path.exists(history_path))
                with open(evidence_path, "r") as f:
                    evidence = json.load(f)
                self.assertEqual(evidence.get("stage_key"), "supported_not_trained")
                self.assertIn("Stage canonical price history", " ".join(evidence.get("next_requirements") or []))
                with open(history_path, "r") as f:
                    rows = [json.loads(line) for line in f if line.strip()]
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0].get("stage_key"), "supported_not_trained")
            finally:
                rt.DATA_DIR = old_data_dir
                rt.evaluate_readiness_metrics = old_eval
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_deployment_readiness_includes_onboarding_summary_and_checklist(self):
        old_onboarding_summary = rt._onboarding_summary
        old_load_models = rt.load_models
        old_instruments = dict(rt.INSTRUMENTS)
        try:
            rt.INSTRUMENTS["AAPL"] = {"type": "equity"}
            rt.load_models = lambda include_modes=True, instrument_key=None: {
                "base": {},
                "forest": {},
                "boost": {},
                "extra_trees": {},
                "meta": {},
                "regime": object(),
            }
            rt._onboarding_summary = lambda symbol: {
                "status": "running",
                "status_label": "Running",
                "summary": "Train baseline models: fitting core stack",
                "artifacts": {"profile": {"sector": "Technology"}},
                "completion_flags": {"honest_completion": False},
            }

            out = rt.summarize_deployment_readiness(
                "AAPL",
                state={"fresh": True},
                system_report={
                    "status": "ok",
                    "data": {"rows": 10, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                    "models": {"missing": []},
                    "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                    "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                    "horizon_accuracy": {"drift_flag": False},
                },
                news_health={"status": "ok", "headline_state": "ok"},
            )

            self.assertEqual(out["onboarding_summary"]["status"], "running")
            self.assertIn("onboarding_pipeline", out["checklist"])
            self.assertIn("Train baseline models", out["checklist"]["onboarding_pipeline"]["detail"])
            self.assertIn("artifacts 1", out["checklist"]["onboarding_pipeline"]["detail"])
        finally:
            rt._onboarding_summary = old_onboarding_summary
            rt.load_models = old_load_models
            rt.INSTRUMENTS.clear()
            rt.INSTRUMENTS.update(old_instruments)

    def test_shadow_observation_summary_distinguishes_clean_from_missing_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                with open(os.path.join(tmp, "autonomy_audit_QQQ.jsonl"), "w") as f:
                    f.write(json.dumps({"kind": "shadow_observation", "observed_at": "2026-04-10T15:00:00Z", "clean_cycle": False, "eligible_environment": False, "allowed": False}) + "\n")
                with open(os.path.join(tmp, "calibration_live_QQQ.json"), "w") as f:
                    json.dump({"last_update": "2026-04-10T15:00:00Z", "last_processed": "2026-04-10T14:56:00Z", "last_resolved_target_ts": "2026-04-10T15:00:00Z", "evidence_stale": False}, f)

                out = rt.summarize_deployment_readiness(
                    "QQQ",
                    state={
                        "fresh": True,
                        "live_ready": False,
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
                    },
                    system_report={
                        "status": "ok",
                        "data": {"rows": 10, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    },
                    news_health={"status": "ok", "headline_state": "ok"},
                )

                self.assertEqual(out["observation_summary"]["shadow_observations"], 1)
                self.assertEqual(out["observation_summary"]["shadow_clean_observations"], 0)
                self.assertIn("shadow observation cycles exist, but none were clean enough", " | ".join(out["missing_evidence"]))
                self.assertIn("clean shadow observation cycle", " | ".join(out["next_requirements"]))
            finally:
                rt.DATA_DIR = old_data_dir
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_clean_shadow_observation_satisfies_observation_checklist(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                with open(os.path.join(tmp, "autonomy_audit_QQQ.jsonl"), "w") as f:
                    f.write(json.dumps({"kind": "shadow_observation", "observed_at": "2026-04-10T15:00:00Z", "clean_cycle": True, "eligible_environment": True, "allowed": False}) + "\n")
                with open(os.path.join(tmp, "calibration_live_QQQ.json"), "w") as f:
                    json.dump({"last_update": "2026-04-10T15:00:00Z", "last_processed": "2026-04-10T14:56:00Z", "last_resolved_target_ts": "2026-04-10T15:00:00Z", "evidence_stale": False}, f)

                out = rt.summarize_deployment_readiness(
                    "QQQ",
                    state={
                        "fresh": True,
                        "live_ready": False,
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
                    },
                    system_report={
                        "status": "ok",
                        "data": {"rows": 10, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    },
                    news_health={"status": "ok", "headline_state": "ok"},
                )

                self.assertTrue(bool(out["checklist"]["shadow_observations"]["ok"]))
                self.assertNotIn("shadow observation cycles exist, but none were clean enough", " | ".join(out["missing_evidence"]))
                self.assertTrue(bool((out.get("calibration_summary") or {}).get("available")))
            finally:
                rt.DATA_DIR = old_data_dir
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_calibration_summary_recomputes_stale_evidence_from_age(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            try:
                rt.DATA_DIR = tmp
                with open(os.path.join(tmp, "calibration_live_QQQ.json"), "w") as f:
                    json.dump(
                        {
                            "last_update": "2026-04-10T15:00:00Z",
                            "last_check": "2026-04-10T15:00:00Z",
                            "last_processed": "2026-03-15T15:00:00Z",
                            "evidence_stale": False,
                        },
                        f,
                    )

                summary = rt._calibration_summary(
                    "QQQ",
                    system_report={
                        "calibration": {
                            "age_days": 0.0,
                            "evidence_age_days": 26.0,
                            "evidence_stale": False,
                            "drift_flag": True,
                        }
                    },
                )

                self.assertTrue(bool(summary.get("evidence_stale")))
                self.assertEqual(summary.get("evidence_age_days"), 26.0)
            finally:
                rt.DATA_DIR = old_data_dir

    def test_onboarding_specific_next_requirement_is_carried_into_readiness(self):
        old_onboarding_summary = rt._onboarding_summary
        old_load_models = rt.load_models
        old_instruments = dict(rt.INSTRUMENTS)
        try:
            rt.INSTRUMENTS["MSFT"] = {"type": "equity"}
            rt.load_models = lambda include_modes=True, instrument_key=None: {}
            rt._onboarding_summary = lambda symbol: {
                "status": "completed",
                "status_label": "Completed",
                "summary": "Partial history staged",
                "primary_next_requirement": "Extend session-level history until the daily range path produces usable rows.",
                "completion_flags": {"honest_completion": False},
                "artifacts": {},
            }

            out = rt.summarize_deployment_readiness(
                "MSFT",
                state={},
                system_report={
                    "status": "fail",
                    "data": {"rows": 20, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                },
                news_health={"status": "missing", "headline_state": "missing"},
            )

            self.assertIn(
                "Extend session-level history until the daily range path produces usable rows.",
                out["next_requirements"],
            )
        finally:
            rt._onboarding_summary = old_onboarding_summary
            rt.load_models = old_load_models
            rt.INSTRUMENTS.clear()
            rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_consumes_persisted_training_plan_fields(self):
        old_onboarding_summary = rt._onboarding_summary
        old_load_models = rt.load_models
        old_instruments = dict(rt.INSTRUMENTS)
        try:
            rt.INSTRUMENTS["MSFT"] = {"type": "equity"}
            rt.load_models = lambda include_modes=True, instrument_key=None: {}
            rt._onboarding_summary = lambda symbol: {
                "status": "completed",
                "status_label": "Completed",
                "summary": "Partial model plan retained",
                "primary_next_requirement": "",
                "completion_flags": {"honest_completion": False},
                "artifacts": {
                    "training_plan": {
                        "status": "partial",
                        "eligible_modes": ["DailyHigh"],
                        "deferred_modes": ["4min"],
                        "blocked_modes": ["WeekEnd"],
                        "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                        "retained_prior_fields": ["eligible_modes", "deferred_modes", "blocked_modes"],
                        "retained_from_prior_run": True,
                    }
                },
            }

            out = rt.summarize_deployment_readiness(
                "MSFT",
                state={},
                system_report={
                    "status": "ok",
                    "data": {"rows": 20, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                    "models": {"missing": []},
                    "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                    "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                    "horizon_accuracy": {"drift_flag": False},
                },
                news_health={"status": "ok", "headline_state": "ok"},
            )

            self.assertIn("core 4-minute path is still below its training floor", out["blockers"])
            self.assertEqual(
                out["primary_next_requirement"],
                "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
            )
            self.assertIn(
                "the core 4-minute path can build features, but it is still below its training floor",
                out["missing_evidence"],
            )
            self.assertIn("eligible DailyHigh", out["checklist"]["onboarding_pipeline"]["detail"])
            self.assertIn("deferred 4min", out["checklist"]["onboarding_pipeline"]["detail"])
            self.assertIn("blocked WeekEnd", out["checklist"]["onboarding_pipeline"]["detail"])
            self.assertIn("plan retained", out["checklist"]["onboarding_pipeline"]["detail"])
        finally:
            rt._onboarding_summary = old_onboarding_summary
            rt.load_models = old_load_models
            rt.INSTRUMENTS.clear()
            rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_training_plan_consumption_is_backward_compatible_when_artifact_missing(self):
        old_onboarding_summary = rt._onboarding_summary
        old_load_models = rt.load_models
        old_instruments = dict(rt.INSTRUMENTS)
        try:
            rt.INSTRUMENTS["MSFT"] = {"type": "equity"}
            rt.load_models = lambda include_modes=True, instrument_key=None: {}
            rt._onboarding_summary = lambda symbol: {
                "status": "completed",
                "status_label": "Completed",
                "summary": "Partial history staged",
                "primary_next_requirement": "",
                "completion_flags": {"honest_completion": False},
                "artifacts": {},
            }

            out = rt.summarize_deployment_readiness(
                "MSFT",
                state={},
                system_report={
                    "status": "ok",
                    "data": {"rows": 20, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                    "models": {"missing": []},
                    "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                    "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                    "horizon_accuracy": {"drift_flag": False},
                },
                news_health={"status": "ok", "headline_state": "ok"},
            )

            self.assertEqual(
                out["primary_next_requirement"],
                "Train the core 4-minute model set for this symbol before treating it as trained.",
            )
            self.assertIn(
                "Train the core 4-minute model set for this symbol before treating it as trained.",
                out["next_requirements"],
            )
        finally:
            rt._onboarding_summary = old_onboarding_summary
            rt.load_models = old_load_models
            rt.INSTRUMENTS.clear()
            rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_uses_retained_model_runtime_when_training_plan_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["MSFT"] = {"type": "equity"}
                _write_price_history(tmp, "MSFT", rows=40)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "Training skipped on rerun; prior model coverage retained",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {
                        "model_runtime": {
                            "status": "skipped",
                            "detail": "Training skipped for this pass; prior onboarding model coverage was retained: DailyHigh.",
                            "trained_modes": ["DailyHigh"],
                            "retained_prior_modes": ["DailyHigh"],
                            "retained_from_prior_run": True,
                        }
                    },
                }
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "modes": {
                        "daily_high": {"base": {}},
                        "daily_low": {},
                        "weekend": {},
                    },
                    "meta": {},
                    "regime": object(),
                }

                out = rt.summarize_deployment_readiness(
                    "MSFT",
                    state={"fresh": True},
                    system_report={
                        "status": "ok",
                        "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    },
                    news_health={"status": "ok", "headline_state": "ok"},
                )

                self.assertEqual(out["symbol_maturity_label"], "Partially trained")
                self.assertEqual(
                    out["primary_blocker"],
                    "non-core model coverage exists, but the core 4-minute stack is still missing",
                )
                self.assertIn(
                    "non-core trained model artifacts exist, but the core advisory path is still missing",
                    out["missing_evidence"],
                )
                self.assertEqual(
                    out["primary_next_requirement"],
                    "Train the core 4-minute model set for this symbol before treating it as trained.",
                )
                self.assertIn("trained DailyHigh", out["checklist"]["onboarding_pipeline"]["detail"])
                self.assertIn("runtime retained", out["checklist"]["onboarding_pipeline"]["detail"])
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_distinguishes_model_missing_vs_risk_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=40)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "All onboarding artifacts present",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {},
                }
                ready_state = {
                    "fresh": True,
                    "live_ready": False,
                    "pass_streak_days": 1,
                    "required_streak_days": 3,
                    "risk_summary": {
                        "blocked": True,
                        "primary_blocker": "risk budget is too small",
                        "blockers": ["risk budget is too small"],
                        "regime": {"label": "trend_up", "unknown": False},
                    },
                    "evaluation": {
                        "metrics": {
                            "horizon_weighted_score": 0.64,
                            "intraday_window_hit_rate": 0.67,
                            "daily_window_hit_rate": 0.62,
                            "weekend_window_hit_rate": 0.59,
                        },
                        "thresholds": {
                            "min_horizon_score": 0.55,
                            "min_intraday_window_hit_rate": 0.60,
                            "min_daily_window_hit_rate": 0.58,
                            "min_weekend_window_hit_rate": 0.55,
                        },
                    },
                }
                system_report = {
                    "status": "ok",
                    "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
                    "models": {"missing": []},
                    "predictions": {
                        "4min": {
                            "consensus": {
                                "direction": "UP",
                                "confidence": 63.0,
                                "edge_bps": 8.0,
                                "raw_edge_bps": 10.0,
                                "raw_pred_ret_bps": 11.0,
                                "cost_bps_est": 1.4,
                                "uncertainty_penalty_bps": 0.9,
                                "position_size": 0.55,
                            },
                            "diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}},
                        }
                    },
                    "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                    "horizon_accuracy": {"drift_flag": False},
                }
                news_health = {"status": "ok", "headline_state": "ok"}

                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                risk_blocked = rt.summarize_deployment_readiness("QQQ", state=ready_state, system_report=system_report, news_health=news_health)
                self.assertEqual(risk_blocked["decision_state_key"], "risk_blocked")
                self.assertEqual(risk_blocked["decision_state_label"], "Risk blocked")
                self.assertNotEqual(risk_blocked["operator_state_label"], "Blocked: missing trained models")

                rt.load_models = lambda include_modes=True, instrument_key=None: {}
                model_missing = rt.summarize_deployment_readiness("QQQ", state=ready_state, system_report=system_report, news_health=news_health)
                self.assertEqual(model_missing["operator_state_label"], "Blocked: missing trained models")
                self.assertNotEqual(model_missing["decision_state_key"], "risk_blocked")
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_distinguishes_unknown_regime_vs_generic_nonreadiness(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=8)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "All onboarding artifacts present",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {},
                }
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                out = rt.summarize_deployment_readiness(
                    "QQQ",
                    state={
                        "fresh": True,
                        "live_ready": False,
                        "pass_streak_days": 1,
                        "required_streak_days": 3,
                        "validation_evidence": _validation_evidence("passed_shadow_only"),
                        "evaluation": {
                            "metrics": {
                                "horizon_weighted_score": 0.64,
                                "intraday_window_hit_rate": 0.67,
                                "daily_window_hit_rate": 0.62,
                                "weekend_window_hit_rate": 0.59,
                            },
                            "thresholds": {
                                "min_horizon_score": 0.55,
                                "min_intraday_window_hit_rate": 0.60,
                                "min_daily_window_hit_rate": 0.58,
                                "min_weekend_window_hit_rate": 0.55,
                            },
                        },
                    },
                    system_report={
                        "status": "ok",
                        "data": {"rows": 8, "last_ts": "2026-04-10T14:44:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {
                            "4min": {
                                "consensus": {
                                    "direction": "UP",
                                    "confidence": 63.0,
                                    "edge_bps": 8.0,
                                    "raw_edge_bps": 10.0,
                                    "raw_pred_ret_bps": 11.0,
                                    "cost_bps_est": 1.4,
                                    "uncertainty_penalty_bps": 0.9,
                                    "position_size": 0.55,
                                },
                                "diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}},
                            }
                        },
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    },
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(out["decision_state_key"], "unknown_regime")
                self.assertEqual(out["decision_state_label"], "Unknown regime")
                self.assertEqual((out["regime_risk_summary"].get("regime") or {}).get("label"), "unknown_or_insufficient_evidence")
                self.assertIn("Collect enough recent price action and stable context to classify the current regime safely.", out["next_requirements"])
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_identifies_advisory_usable_but_autonomy_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=40)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "All onboarding artifacts present",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {},
                }
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                out = rt.summarize_deployment_readiness(
                    "QQQ",
                    state={
                        "fresh": True,
                        "live_ready": False,
                        "pass_streak_days": 1,
                        "required_streak_days": 3,
                        "validation_evidence": _validation_evidence("passed_shadow_only"),
                        "evaluation": {
                            "metrics": {
                                "horizon_weighted_score": 0.64,
                                "intraday_window_hit_rate": 0.67,
                                "daily_window_hit_rate": 0.62,
                                "weekend_window_hit_rate": 0.59,
                            },
                            "thresholds": {
                                "min_horizon_score": 0.55,
                                "min_intraday_window_hit_rate": 0.60,
                                "min_daily_window_hit_rate": 0.58,
                                "min_weekend_window_hit_rate": 0.55,
                            },
                        },
                    },
                    system_report={
                        "status": "ok",
                        "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {
                            "4min": {
                                "consensus": {
                                    "direction": "UP",
                                    "confidence": 63.0,
                                    "edge_bps": 8.0,
                                    "raw_edge_bps": 10.0,
                                    "raw_pred_ret_bps": 11.0,
                                    "cost_bps_est": 1.4,
                                    "uncertainty_penalty_bps": 0.9,
                                    "position_size": 0.55,
                                },
                                "diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}},
                            }
                        },
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    },
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(out["decision_state_key"], "advisory_usable_autonomy_blocked")
                self.assertEqual(out["decision_state_label"], "Advisory-usable / autonomy blocked")
                self.assertEqual(out["operator_state_label"], "Shadow candidate")
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_distinguishes_economic_failure_modes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=40)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "All onboarding artifacts present",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {
                        "model_runtime": {
                            "status": "completed",
                            "detail": "4-minute model runtime captured.",
                            "trained_modes": ["4min"],
                            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        }
                    },
                }
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                state = {
                    "fresh": True,
                    "live_ready": False,
                    "pass_streak_days": 1,
                    "required_streak_days": 3,
                    "evaluation": {
                        "metrics": {
                            "horizon_weighted_score": 0.64,
                            "intraday_window_hit_rate": 0.67,
                            "daily_window_hit_rate": 0.62,
                            "weekend_window_hit_rate": 0.59,
                        },
                        "thresholds": {
                            "min_horizon_score": 0.55,
                            "min_intraday_window_hit_rate": 0.60,
                            "min_daily_window_hit_rate": 0.58,
                            "min_weekend_window_hit_rate": 0.55,
                        },
                    },
                }

                def system_report(uncertainty_penalty_bps: float, cost_bps_est: float) -> dict:
                    return {
                        "status": "ok",
                        "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {
                            "4min": {
                                "consensus": {
                                    "direction": "-",
                                    "raw_direction": "UP",
                                    "confidence": 56.0,
                                    "raw_edge_bps": 3.8,
                                    "edge_bps": -0.4,
                                    "raw_pred_ret_bps": 5.2,
                                    "cost_bps_est": cost_bps_est,
                                    "uncertainty_penalty_bps": uncertainty_penalty_bps,
                                    "position_size": 0.0,
                                },
                                "diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}},
                            }
                        },
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    }

                cost_fail = rt.summarize_deployment_readiness(
                    "QQQ",
                    state=state,
                    system_report=system_report(uncertainty_penalty_bps=0.4, cost_bps_est=2.0),
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(cost_fail["economics_state_key"], "cost_crushed")
                self.assertEqual(cost_fail["decision_state_key"], "economic_fail_costs")

                uncertainty_fail = rt.summarize_deployment_readiness(
                    "QQQ",
                    state=state,
                    system_report=system_report(uncertainty_penalty_bps=2.4, cost_bps_est=1.0),
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(uncertainty_fail["economics_state_key"], "uncertainty_crushed")
                self.assertEqual(uncertainty_fail["decision_state_key"], "economic_fail_uncertainty")
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_surfaces_model_runtime_missing_and_stale_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            runtime_artifacts = {}
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=40)
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "Onboarding completed",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": runtime_artifacts,
                }
                state = {
                    "fresh": True,
                    "live_ready": False,
                    "evaluation": {
                        "metrics": {
                            "horizon_weighted_score": 0.64,
                            "intraday_window_hit_rate": 0.67,
                            "daily_window_hit_rate": 0.62,
                            "weekend_window_hit_rate": 0.59,
                        }
                    },
                }
                system_report = {
                    "status": "ok",
                    "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
                    "models": {"missing": []},
                    "predictions": {"4min": {"diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}}}},
                    "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                    "horizon_accuracy": {"drift_flag": False},
                }

                missing = rt.summarize_deployment_readiness(
                    "QQQ",
                    state=state,
                    system_report=system_report,
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(missing["model_runtime_state_key"], "model_runtime_missing")
                self.assertIn("model runtime artifact has not been captured", missing["missing_evidence"])

                runtime_artifacts["model_runtime"] = {
                    "status": "completed",
                    "detail": "Runtime came from a stale prior run.",
                    "trained_modes": ["4min"],
                    "trained_at": "2026-01-01T00:00:00Z",
                    "stale": True,
                    "primary_next_requirement": "Refresh runtime evidence.",
                }
                stale = rt.summarize_deployment_readiness(
                    "QQQ",
                    state=state,
                    system_report=system_report,
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(stale["model_runtime_state_key"], "model_runtime_stale")
                self.assertIn("model runtime artifact is stale", stale["missing_evidence"])
                self.assertIn("runtime stale", stale["checklist"]["onboarding_pipeline"]["detail"])
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_surfaces_missing_portfolio_context_without_erasing_advisory_usefulness(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=40)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "All onboarding artifacts present",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {
                        "model_runtime": {
                            "status": "completed",
                            "detail": "4-minute model runtime captured.",
                            "trained_modes": ["4min"],
                            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        }
                    },
                }
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }
                out = rt.summarize_deployment_readiness(
                    "QQQ",
                    state={
                        "fresh": True,
                        "live_ready": False,
                        "pass_streak_days": 1,
                        "required_streak_days": 3,
                        "evaluation": {
                            "metrics": {
                                "horizon_weighted_score": 0.64,
                                "intraday_window_hit_rate": 0.67,
                                "daily_window_hit_rate": 0.62,
                                "weekend_window_hit_rate": 0.59,
                            },
                            "thresholds": {
                                "min_horizon_score": 0.55,
                                "min_intraday_window_hit_rate": 0.60,
                                "min_daily_window_hit_rate": 0.58,
                                "min_weekend_window_hit_rate": 0.55,
                            },
                        },
                    },
                    system_report={
                        "status": "ok",
                        "data": {"rows": 40, "last_ts": "2026-04-10T15:48:00Z", "staleness": {"is_stale": False}},
                        "models": {"missing": []},
                        "predictions": {
                            "4min": {
                                "consensus": {
                                    "direction": "UP",
                                    "confidence": 63.0,
                                    "edge_bps": 8.0,
                                    "raw_edge_bps": 10.0,
                                    "raw_pred_ret_bps": 11.0,
                                    "cost_bps_est": 1.4,
                                    "uncertainty_penalty_bps": 0.9,
                                    "position_size": 0.55,
                                },
                                "diagnostics": {"critical_stale": False, "readiness_locked": False, "diversity": {"degenerate": False}},
                            }
                        },
                        "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                        "horizon_accuracy": {"drift_flag": False},
                    },
                    news_health={"status": "ok", "headline_state": "ok"},
                )
                self.assertEqual(out["decision_state_key"], "advisory_usable_autonomy_blocked")
                self.assertFalse(out["portfolio_context"]["available"])
                self.assertIn("portfolio snapshot is unavailable; portfolio-aware autonomy cannot be proven", out["missing_evidence"])
                self.assertTrue(
                    any("advisory review can continue separately" in item for item in out["next_requirements"])
                )
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_validation_evidence_caps_deployment_and_surfaces_specific_blockers(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_onboarding_summary = rt._onboarding_summary
            old_load_models = rt.load_models
            old_instruments = dict(rt.INSTRUMENTS)
            try:
                rt.DATA_DIR = tmp
                rt.INSTRUMENTS["QQQ"] = {"type": "equity"}
                _write_price_history(tmp, "QQQ", rows=40)
                rt._onboarding_summary = lambda symbol: {
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "All onboarding artifacts present",
                    "completion_flags": {"honest_completion": True},
                    "artifacts": {
                        "model_runtime": {
                            "status": "completed",
                            "detail": "4-minute model runtime captured.",
                            "trained_modes": ["4min"],
                            "trained_at": pd.Timestamp.now(tz="UTC").isoformat(),
                        }
                    },
                }
                rt.load_models = lambda include_modes=True, instrument_key=None: {
                    "base": {},
                    "forest": {},
                    "boost": {},
                    "extra_trees": {},
                    "meta": {},
                    "regime": object(),
                }

                cases = [
                    (
                        "missing",
                        None,
                        "missing",
                        "validation evidence has not been captured",
                        "advisory_only",
                        False,
                        False,
                    ),
                    (
                        "stale",
                        _validation_evidence("passed_paper_candidate", stale=True, untouched_holdout=True),
                        "stale",
                        "validation evidence is stale",
                        "advisory_only",
                        False,
                        False,
                    ),
                    (
                        "no_oos",
                        _validation_evidence("passed_paper_candidate", strict_oos=False, untouched_holdout=True),
                        "passed_paper_candidate",
                        "validation evidence lacks strict out-of-sample controls",
                        "advisory_only",
                        False,
                        False,
                    ),
                    (
                        "no_cost_model",
                        _validation_evidence("passed_paper_candidate", costs=False, untouched_holdout=True),
                        "passed_paper_candidate",
                        "validation evidence lacks explicit transaction-cost assumptions",
                        "advisory_only",
                        False,
                        False,
                    ),
                    (
                        "net_negative",
                        _validation_evidence("passed_paper_candidate", gross=0.03, net=-0.01, untouched_holdout=True),
                        "passed_paper_candidate",
                        "validation gross edge is positive but net edge is negative after costs",
                        "advisory_only",
                        False,
                        False,
                    ),
                    (
                        "advisory_only",
                        _validation_evidence("passed_advisory_only"),
                        "passed_advisory_only",
                        "start shadow-mode paper ledger",
                        "advisory_only",
                        False,
                        False,
                    ),
                    (
                        "shadow_only",
                        _validation_evidence("passed_shadow_only"),
                        "passed_shadow_only",
                        "start shadow-mode paper ledger",
                        "advisory_shadow_ready",
                        True,
                        False,
                    ),
                ]
                for name, validation, status, expected_text, stage_key, shadow_ready, paper_ready in cases:
                    with self.subTest(name=name):
                        out = rt.summarize_deployment_readiness(
                            "QQQ",
                            state=_ready_state(validation),
                            system_report=_ready_system_report(),
                            news_health={"status": "ok", "headline_state": "ok"},
                        )
                        self.assertEqual(out["validation_status"], status)
                        self.assertEqual(out["stage_key"], stage_key)
                        self.assertTrue(out["advisory_ready"])
                        self.assertEqual(bool(out["shadow_ready"]), shadow_ready)
                        self.assertEqual(bool(out["paper_ready"]), paper_ready)
                        joined = " | ".join(
                            list(out.get("missing_evidence") or [])
                            + list(out.get("next_requirements") or [])
                            + list(out.get("validation_blockers") or [])
                        )
                        self.assertIn(expected_text, joined)
                        self.assertFalse(out["live_ready_candidate"])
            finally:
                rt.DATA_DIR = old_data_dir
                rt._onboarding_summary = old_onboarding_summary
                rt.load_models = old_load_models
                rt.INSTRUMENTS.clear()
                rt.INSTRUMENTS.update(old_instruments)

    def test_readiness_backward_compatible_when_regime_risk_fields_absent(self):
        old_onboarding_summary = rt._onboarding_summary
        old_load_models = rt.load_models
        old_instruments = dict(rt.INSTRUMENTS)
        try:
            rt.INSTRUMENTS["MSFT"] = {"type": "equity"}
            rt.load_models = lambda include_modes=True, instrument_key=None: {}
            rt._onboarding_summary = lambda symbol: {
                "status": "completed",
                "status_label": "Completed",
                "summary": "Partial history staged",
                "primary_next_requirement": "",
                "completion_flags": {"honest_completion": False},
                "artifacts": {},
            }
            out = rt.summarize_deployment_readiness(
                "MSFT",
                state={},
                system_report={
                    "status": "ok",
                    "data": {"rows": 20, "last_ts": "2026-04-10T15:00:00Z", "staleness": {"is_stale": False}},
                    "models": {"missing": []},
                    "predictions": {},
                    "calibration": {"drift_flag": False, "stale": False, "evidence_stale": False},
                    "horizon_accuracy": {"drift_flag": False},
                },
                news_health={"status": "ok", "headline_state": "ok"},
            )
            self.assertEqual(out["operator_state_label"], "Blocked: missing trained models")
            self.assertEqual(
                out["primary_next_requirement"],
                "Train the core 4-minute model set for this symbol before treating it as trained.",
            )
            self.assertNotEqual(out["decision_state_key"], "unknown_regime")
        finally:
            rt._onboarding_summary = old_onboarding_summary
            rt.load_models = old_load_models
            rt.INSTRUMENTS.clear()
            rt.INSTRUMENTS.update(old_instruments)

    def test_walk_forward_cache_invalidates_when_model_signature_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            old_model_sig = rt._models_dir_signature
            try:
                rt.DATA_DIR = tmp
                rt._models_dir_signature = lambda: [("model.pkl", 1, 10)]
                df = pd.DataFrame(
                    {"close": [100.0, 101.0, 102.0]},
                    index=pd.to_datetime(
                        ["2026-03-01T14:30:00Z", "2026-03-01T14:31:00Z", "2026-03-01T14:32:00Z"],
                        utc=True,
                    ),
                )
                df.to_csv(os.path.join(tmp, "prices_QQQ.csv"))
                metrics = {"folds": 3, "price_window_ratio": [1.0]}
                rt._save_cached_walk_forward("QQQ", df, 3, metrics)
                self.assertEqual(rt._load_cached_walk_forward("QQQ", df, 3), metrics)

                rt._models_dir_signature = lambda: [("model.pkl", 2, 10)]
                self.assertEqual(rt._load_cached_walk_forward("QQQ", df, 3), {})
            finally:
                rt.DATA_DIR = old_data_dir
                rt._models_dir_signature = old_model_sig

    def test_horizon_replay_directional_accuracy_excludes_perfect_splits(self):
        idx = pd.date_range("2026-03-01T14:30:00Z", periods=305, freq="min", tz="UTC")
        prices = pd.DataFrame({"close": [100.0] * len(idx)}, index=idx)
        ds_daily = pd.DataFrame(
            {
                "close": [100.0],
                "target_high_ret": [0.01],
                "target_low_ret": [-0.01],
            },
            index=pd.DatetimeIndex([idx[-1]], tz="UTC"),
        )
        ds_week = pd.DataFrame(
            {
                "close": [100.0],
                "target_weekend_ret": [0.01],
            },
            index=pd.DatetimeIndex([idx[-1]], tz="UTC"),
        )
        old_load_models = rt.load_models
        old_predict_horizon = rt.predict_horizon
        old_build_next_session_dataset = rt.build_next_session_dataset
        old_build_weekend_session_dataset = rt.build_weekend_session_dataset
        try:
            rt.load_models = lambda include_modes=True, instrument_key=None: {"ok": True}
            rt.build_next_session_dataset = lambda df, instrument_key=None: ds_daily
            rt.build_weekend_session_dataset = lambda df, instrument_key=None: ds_week

            def _fake_predict_horizon(models, subset, instrument_key=None, mode=None, log_side_effects=False):
                if mode == "DailyHigh":
                    return {
                        "high": {
                            "base": {"pred_price": 101.0},
                            "forest": {"pred_price": 100.8},
                            "boost": {"pred_price": 99.2},
                            "extra_trees": {"pred_price": 99.0},
                        },
                        "low": {
                            "base": {"pred_price": 98.5},
                            "forest": {"pred_price": 98.8},
                            "boost": {"pred_price": 99.1},
                            "extra_trees": {"pred_price": 101.0},
                        },
                    }
                return {
                    "week": {
                        "base": {"pred_price": 101.2},
                        "forest": {"pred_price": 101.0},
                        "boost": {"pred_price": 100.9},
                        "extra_trees": {"pred_price": 99.8},
                    }
                }

            rt.predict_horizon = _fake_predict_horizon
            out = rt._horizon_replay_errors(prices, "QQQ", max_daily_samples=1, max_week_samples=1)
            self.assertEqual(out["daily_directional_samples"], 1)
            self.assertEqual(out["daily_directional_ties"], 1)
            self.assertAlmostEqual(out["daily_directional_acc"], 1.0, places=6)
            self.assertEqual(out["weekend_directional_samples"], 1)
            self.assertEqual(out["weekend_directional_ties"], 0)
            self.assertAlmostEqual(out["weekend_directional_acc"], 1.0, places=6)
        finally:
            rt.load_models = old_load_models
            rt.predict_horizon = old_predict_horizon
            rt.build_next_session_dataset = old_build_next_session_dataset
            rt.build_weekend_session_dataset = old_build_weekend_session_dataset


if __name__ == "__main__":
    unittest.main()
