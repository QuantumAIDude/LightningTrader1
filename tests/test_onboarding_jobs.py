import json
import os
import tempfile
import unittest

import pandas as pd

import onboarding_jobs as oj


class _FakeHub:
    def __init__(self, df):
        self.df = df
        self.calls = []

    def refresh_symbol(self, symbol, force=False, reason=None, load_models=True):
        self.calls.append((symbol, force, reason, load_models))
        return {"symbol": symbol, "df": self.df.copy(), "data_source": "unit-test"}


class OnboardingJobTests(unittest.TestCase):
    def test_run_onboarding_job_persists_stage_state_and_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            old_register_instrument = oj.register_instrument
            old_build_instrument_meta = oj.build_instrument_meta
            old_assess_symbol_history = oj.assess_symbol_history
            old_ensure_daily_long_history = oj.ensure_daily_long_history
            old_train_baseline_models = oj.train_baseline_models
            old_open_workspace = oj.open_workspace
            try:
                oj.DATA_DIR = tmp
                oj.register_instrument = lambda symbol, meta=None, persist=True: True
                oj.open_workspace = lambda symbol, meta=None, activate=True: {"symbol": symbol, "opened": True}
                oj.build_instrument_meta = lambda symbol, overrides=None: {
                    "symbol": symbol,
                    "instrument_kind": "equity",
                    "sector": "Technology",
                }
                oj.assess_symbol_history = lambda df, instrument_key=None, daily_long_history=None: {
                    "status_key": "multi_horizon_eligible",
                    "status_label": "Multi-horizon eligible",
                    "detail": "The current staged history is sufficient for intraday, daily, and weekend training paths.",
                    "price_rows": 2000,
                    "first_ts": "2026-01-01T14:30:00+00:00",
                    "last_ts": "2026-04-10T15:56:00+00:00",
                    "span_days": 99,
                    "session_count": 60,
                    "daily_long_rows": 400,
                    "feature_ready_modes": ["4min", "DailyHigh", "WeekEnd"],
                    "eligible_modes": ["4min", "DailyHigh", "WeekEnd"],
                    "blocked_modes": [],
                    "primary_next_requirement": "Train the eligible model modes for this symbol.",
                    "paths": {
                        "4min": {"feature_rows": 180, "feature_ready": True, "train_eligible": True, "min_train_rows": 100},
                        "DailyHigh": {"feature_rows": 20, "feature_ready": True, "train_eligible": True, "min_train_rows": 15},
                        "WeekEnd": {"feature_rows": 8, "feature_ready": True, "train_eligible": True, "min_train_rows": 6},
                    },
                }
                oj.ensure_daily_long_history = lambda symbol, recent_df=None, persist=True: (
                    pd.DataFrame(
                        {
                            "open": [1.0, 2.0],
                            "high": [1.0, 2.0],
                            "low": [1.0, 2.0],
                            "close": [1.0, 2.0],
                            "volume": [1.0, 2.0],
                        },
                        index=pd.date_range("2026-01-01", periods=2, tz="UTC"),
                    ),
                    {"rows": 400, "sources_used": ["recent_runtime"], "updated": True},
                )
                oj.train_baseline_models = lambda df, instrument_key=None, train_horizons=True, modes_filter=None, progress_cb=None, max_runtime_sec=None: {
                    "family": "tabular_baseline_v2",
                    "requested_modes": list(modes_filter or []),
                    "models": {
                        "base": {"ensemble_models": {"recent": [object()]}},
                        "forest": {"ensemble_models": {"recent": [object()]}},
                        "boost": {"ensemble_models": {"recent": [object()]}},
                        "extra_trees": {"ensemble_models": {"recent": [object()]}},
                        "meta": {},
                        "regime": object(),
                        "modes": {
                            "daily_high": {"base": {"ensemble_models": {"recent": [object()]}}},
                            "daily_low": {"forest": {"ensemble_models": {"recent": [object()]}}},
                            "weekend": {"boost": {"ensemble_models": {"recent": [object()]}}},
                        },
                    }
                }

                import calibration_tracker as ct
                import horizon_accuracy_tracker as hat
                import readiness_tracker as rt

                old_update_live_calibration = ct.update_live_calibration
                old_update_horizon_accuracy = hat.update_horizon_accuracy
                old_update_readiness_state = rt.update_readiness_state
                old_load_deployment_evidence = rt.load_deployment_evidence
                ct.update_live_calibration = lambda symbol, horizon_minutes=4: {"status": "updated"}
                hat.update_horizon_accuracy = lambda symbol: {"status": "updated"}
                rt.update_readiness_state = lambda symbol, online=False: {
                    "stage_key": "advisory_only",
                    "stage_label": "Advisory only",
                    "summary": "Baseline evidence ready",
                    "checklist": {"summary": {"passed": 8, "total": 12}},
                    "validation_status": "passed_advisory_only",
                    "validation_method": "walk_forward",
                    "validation_summary": "Validation supports advisory only",
                    "validation_blockers": ["no_paper_trading_record"],
                    "validation_evidence": {
                        "validation_status": "passed_advisory_only",
                        "validation_method": "walk_forward",
                        "summary": "Validation supports advisory only",
                        "blockers": ["no_paper_trading_record"],
                    },
                }
                rt.load_deployment_evidence = lambda symbol: {
                    "stage_key": "advisory_only",
                    "stage_label": "Advisory only",
                    "summary": "Evidence captured",
                    "next_stage_key": "advisory_shadow_ready",
                    "next_stage_label": "Advisory + shadow",
                    "missing_evidence": [],
                    "next_requirements": ["Collect shadow evidence"],
                    "checklist": {"summary": {"passed": 8, "total": 12}},
                    "validation_status": "passed_advisory_only",
                    "validation_method": "walk_forward",
                    "validation_summary": "Validation supports advisory only",
                    "validation_blockers": ["no_paper_trading_record"],
                    "validation_evidence": {
                        "validation_status": "passed_advisory_only",
                        "validation_method": "walk_forward",
                        "summary": "Validation supports advisory only",
                        "blockers": ["no_paper_trading_record"],
                    },
                }
                df = pd.DataFrame(
                    [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                    index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
                )
                hub = _FakeHub(df)
                progress = []

                state = oj.run_onboarding_job(
                    "AAPL",
                    meta={"company_name": "Apple Inc"},
                    train_models=True,
                    progress_cb=lambda stage_key, stage_label, state: progress.append((stage_key, state.get("status"))),
                    data_hub=hub,
                )

                summary = oj.summarize_onboarding_state("AAPL")
                self.assertEqual(state["status"], "completed")
                self.assertEqual(summary["status"], "completed")
                self.assertEqual(summary["progress_completed"], len(oj.ONBOARDING_STAGE_ORDER))
                self.assertTrue(summary["workspace_registered"])
                self.assertTrue(summary["completion_flags"]["honest_completion"])
                self.assertEqual(summary["stage_status_counts"]["completed"], len(oj.ONBOARDING_STAGE_ORDER))
                self.assertEqual(summary["next_stage_key"], "finalize")
                self.assertEqual(summary["highest_completed_stage_key"], "finalize")
                self.assertIn("deployment_evidence", summary["artifacts"])
                self.assertIn("capture_deployment_evidence", summary["milestones"])
                self.assertEqual(summary["artifacts"]["deployment_evidence"]["validation_status"], "passed_advisory_only")
                self.assertEqual(summary["artifacts"]["deployment_evidence"]["validation_method"], "walk_forward")
                self.assertEqual(summary["artifacts"]["readiness"]["validation_status"], "passed_advisory_only")
                self.assertEqual(summary["pipeline_outcome_label"], "Completed")
                self.assertEqual(summary["artifacts"]["history_assessment"]["status_label"], "Multi-horizon eligible")
                self.assertTrue(os.path.exists(os.path.join(tmp, "onboarding_state_AAPL.json")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "onboarding_history_AAPL.jsonl")))
                self.assertTrue(progress)
                self.assertEqual(hub.calls[0][0], "AAPL")
            finally:
                oj.DATA_DIR = old_data_dir
                oj.register_instrument = old_register_instrument
                oj.open_workspace = old_open_workspace
                oj.build_instrument_meta = old_build_instrument_meta
                oj.assess_symbol_history = old_assess_symbol_history
                oj.ensure_daily_long_history = old_ensure_daily_long_history
                oj.train_baseline_models = old_train_baseline_models
                ct.update_live_calibration = old_update_live_calibration
                hat.update_horizon_accuracy = old_update_horizon_accuracy
                rt.update_readiness_state = old_update_readiness_state
                rt.load_deployment_evidence = old_load_deployment_evidence

    def test_run_onboarding_job_preserves_partial_progress_and_deferred_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            old_register_instrument = oj.register_instrument
            old_build_instrument_meta = oj.build_instrument_meta
            old_assess_symbol_history = oj.assess_symbol_history
            old_ensure_daily_long_history = oj.ensure_daily_long_history
            old_train_baseline_models = oj.train_baseline_models
            old_open_workspace = oj.open_workspace
            try:
                oj.DATA_DIR = tmp
                oj.register_instrument = lambda symbol, meta=None, persist=True: True
                oj.open_workspace = lambda symbol, meta=None, activate=True: {"symbol": symbol, "opened": True}
                oj.build_instrument_meta = lambda symbol, overrides=None: {
                    "symbol": symbol,
                    "instrument_kind": "equity",
                    "sector": "Technology",
                }
                oj.assess_symbol_history = lambda df, instrument_key=None, daily_long_history=None: {
                    "status_key": "partial_horizon_eligible",
                    "status_label": "Partial horizon eligibility",
                    "detail": "Daily range training can proceed while the core 4-minute path is still below its floor.",
                    "price_rows": 640,
                    "first_ts": "2026-02-01T14:30:00+00:00",
                    "last_ts": "2026-04-10T15:56:00+00:00",
                    "span_days": 68,
                    "session_count": 41,
                    "daily_long_rows": 42,
                    "feature_ready_modes": ["4min", "DailyHigh"],
                    "eligible_modes": ["DailyHigh"],
                    "blocked_modes": ["4min", "WeekEnd"],
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "paths": {
                        "4min": {"feature_rows": 80, "feature_ready": True, "train_eligible": False, "min_train_rows": 100, "detail": "4-minute feature build produced 80 usable rows; needs 100 for training eligibility."},
                        "DailyHigh": {"feature_rows": 17, "feature_ready": True, "train_eligible": True, "min_train_rows": 15, "detail": "Daily range feature build produced 17 usable rows; meets the 15-row training floor."},
                        "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "min_train_rows": 6, "detail": "Weekend / weekly feature build produced 0 usable rows."},
                    },
                }
                oj.ensure_daily_long_history = lambda symbol, recent_df=None, persist=True: (
                    pd.DataFrame(
                        {
                            "open": [1.0, 2.0],
                            "high": [1.0, 2.0],
                            "low": [1.0, 2.0],
                            "close": [1.0, 2.0],
                            "volume": [1.0, 2.0],
                        },
                        index=pd.date_range("2026-01-01", periods=2, tz="UTC"),
                    ),
                    {"rows": 42, "sources_used": ["recent_runtime"], "updated": True},
                )
                oj.train_baseline_models = lambda df, instrument_key=None, train_horizons=True, modes_filter=None, progress_cb=None, max_runtime_sec=None: {
                    "family": "tabular_baseline_v2",
                    "requested_modes": list(modes_filter or []),
                    "models": {
                        "modes": {
                            "daily_high": {"base": {"ensemble_models": {"recent": [object()]}}},
                            "daily_low": {"forest": {"ensemble_models": {"recent": [object()]}}},
                        }
                    },
                }

                import readiness_tracker as rt

                old_update_readiness_state = rt.update_readiness_state
                old_load_deployment_evidence = rt.load_deployment_evidence
                rt.update_readiness_state = lambda symbol, online=False: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Missing core 4-minute model evidence",
                    "primary_blocker": "core 4-minute model set is missing",
                    "primary_next_requirement": "Train the core 4-minute model set for this symbol.",
                    "next_requirements": ["Train the core 4-minute model set for this symbol."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "rollout_label": "Shadow only rollout",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False, "validated": False},
                }
                rt.load_deployment_evidence = lambda symbol: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Partial model coverage recorded",
                    "next_requirements": ["Train the core 4-minute model set for this symbol."],
                    "primary_next_requirement": "Train the core 4-minute model set for this symbol.",
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }
                df = pd.DataFrame(
                    [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                    index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
                )
                hub = _FakeHub(df)

                state = oj.run_onboarding_job("MSFT", train_models=True, data_hub=hub)
                summary = oj.summarize_onboarding_state("MSFT")

                self.assertEqual(state["status"], "completed")
                self.assertEqual(summary["pipeline_outcome_label"], "Completed with gaps")
                self.assertEqual(summary["stages"]["build_feature_readiness"]["status"], "partial")
                self.assertEqual(summary["stages"]["train_baseline_models"]["status"], "partial")
                self.assertEqual(summary["stages"]["refresh_calibration"]["status"], "deferred")
                self.assertEqual(summary["artifacts"]["model_runtime"]["trained_modes"], ["DailyHigh"])
                self.assertEqual(summary["artifacts"]["model_runtime"]["deferred_modes"], ["4min"])
                self.assertEqual(summary["artifacts"]["model_runtime"]["blocked_modes"], ["WeekEnd"])
                self.assertEqual(summary["primary_next_requirement"], "Train the core 4-minute model set for this symbol.")
                self.assertFalse(summary["completion_flags"]["honest_completion"])
            finally:
                oj.DATA_DIR = old_data_dir
                oj.register_instrument = old_register_instrument
                oj.open_workspace = old_open_workspace
                oj.build_instrument_meta = old_build_instrument_meta
                oj.assess_symbol_history = old_assess_symbol_history
                oj.ensure_daily_long_history = old_ensure_daily_long_history
                oj.train_baseline_models = old_train_baseline_models
                rt.update_readiness_state = old_update_readiness_state
                rt.load_deployment_evidence = old_load_deployment_evidence

    def test_repeated_run_preserves_prior_model_progress_when_training_is_skipped(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            old_register_instrument = oj.register_instrument
            old_build_instrument_meta = oj.build_instrument_meta
            old_assess_symbol_history = oj.assess_symbol_history
            old_ensure_daily_long_history = oj.ensure_daily_long_history
            old_build_model_runtime = oj.build_model_runtime
            old_open_workspace = oj.open_workspace
            try:
                oj.DATA_DIR = tmp
                prior_state_path = os.path.join(tmp, "onboarding_state_MSFT.json")
                with open(prior_state_path, "w") as handle:
                    json.dump(
                        {
                            "symbol": "MSFT",
                            "status": "completed",
                            "artifacts": {
                                "model_runtime": {
                                    "trained_modes": ["DailyHigh"],
                                    "available_modes": ["DailyHigh"],
                                    "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": ["DailyHigh"]},
                                },
                                "calibration": {
                                    "status": "updated",
                                    "intraday": {"status": "updated"},
                                    "horizons": {"status": "updated"},
                                },
                            },
                            "workspace_registered": True,
                        },
                        handle,
                    )
                oj.register_instrument = lambda symbol, meta=None, persist=True: True
                oj.open_workspace = lambda symbol, meta=None, activate=True: {"symbol": symbol, "opened": True}
                oj.build_instrument_meta = lambda symbol, overrides=None: {
                    "symbol": symbol,
                    "instrument_kind": "equity",
                    "sector": "Technology",
                }
                oj.assess_symbol_history = lambda df, instrument_key=None, daily_long_history=None: {
                    "status_key": "partial_horizon_eligible",
                    "status_label": "Partial horizon eligibility",
                    "detail": "Existing daily history is still stronger than intraday history.",
                    "price_rows": 120,
                    "first_ts": "2026-04-01T14:30:00+00:00",
                    "last_ts": "2026-04-10T15:56:00+00:00",
                    "span_days": 9,
                    "session_count": 8,
                    "daily_long_rows": 25,
                    "feature_ready_modes": ["DailyHigh"],
                    "eligible_modes": ["DailyHigh"],
                    "blocked_modes": ["4min", "WeekEnd"],
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "paths": {
                        "4min": {"feature_rows": 60, "feature_ready": True, "train_eligible": False, "min_train_rows": 100, "detail": "needs more intraday rows"},
                        "DailyHigh": {"feature_rows": 18, "feature_ready": True, "train_eligible": True, "min_train_rows": 15, "detail": "ready"},
                        "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "min_train_rows": 6, "detail": "missing"},
                    },
                }
                oj.ensure_daily_long_history = lambda symbol, recent_df=None, persist=True: (
                    pd.DataFrame(
                        {
                            "open": [1.0],
                            "high": [1.0],
                            "low": [1.0],
                            "close": [1.0],
                            "volume": [1.0],
                        },
                        index=pd.date_range("2026-03-01", periods=1, tz="UTC"),
                    ),
                    {"rows": 25, "sources_used": ["existing_daily_long"], "updated": False},
                )
                oj.build_model_runtime = lambda symbol, include_modes=True: {
                    "instrument": symbol,
                    "family": "tabular_baseline_v2",
                    "models": {
                        "modes": {
                            "daily_high": {"base": {"ensemble_models": {"recent": [object()]}}},
                            "daily_low": {"forest": {"ensemble_models": {"recent": [object()]}}},
                        }
                    },
                    "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": ["DailyHigh"]},
                }

                import readiness_tracker as rt

                old_update_readiness_state = rt.update_readiness_state
                old_load_deployment_evidence = rt.load_deployment_evidence
                rt.update_readiness_state = lambda symbol, online=False: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Core intraday path still missing",
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "next_requirements": ["Advance intraday history until the 4-minute path reaches 100 usable feature rows."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }
                rt.load_deployment_evidence = lambda symbol: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Daily-only model coverage retained",
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "next_requirements": ["Advance intraday history until the 4-minute path reaches 100 usable feature rows."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }

                df = pd.DataFrame(
                    [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                    index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
                )
                hub = _FakeHub(df)

                state = oj.run_onboarding_job("MSFT", train_models=False, data_hub=hub)
                summary = oj.summarize_onboarding_state("MSFT")

                self.assertEqual(state["status"], "completed")
                self.assertEqual(summary["resume_count"], 1)
                self.assertEqual(summary["resumed_from_status"], "completed")
                self.assertEqual(summary["artifacts"]["model_runtime"]["trained_modes"], ["DailyHigh"])
                self.assertEqual(summary["artifacts"]["calibration"]["status"], "skipped")
                self.assertEqual(summary["stages"]["train_baseline_models"]["status"], "skipped")
                self.assertIn("existing model coverage was retained", summary["artifacts"]["model_runtime"]["detail"])
            finally:
                oj.DATA_DIR = old_data_dir
                oj.register_instrument = old_register_instrument
                oj.open_workspace = old_open_workspace
                oj.build_instrument_meta = old_build_instrument_meta
                oj.assess_symbol_history = old_assess_symbol_history
                oj.ensure_daily_long_history = old_ensure_daily_long_history
                oj.build_model_runtime = old_build_model_runtime
                rt.update_readiness_state = old_update_readiness_state
                rt.load_deployment_evidence = old_load_deployment_evidence

    def test_repeated_run_uses_prior_model_artifact_when_runtime_lookup_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            old_register_instrument = oj.register_instrument
            old_build_instrument_meta = oj.build_instrument_meta
            old_assess_symbol_history = oj.assess_symbol_history
            old_ensure_daily_long_history = oj.ensure_daily_long_history
            old_build_model_runtime = oj.build_model_runtime
            old_open_workspace = oj.open_workspace
            try:
                oj.DATA_DIR = tmp
                prior_state_path = os.path.join(tmp, "onboarding_state_NVDA.json")
                with open(prior_state_path, "w") as handle:
                    json.dump(
                        {
                            "symbol": "NVDA",
                            "status": "completed",
                            "artifacts": {
                                "model_runtime": {
                                    "trained_modes": ["DailyHigh"],
                                    "available_modes": ["DailyHigh"],
                                    "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": ["DailyHigh"]},
                                },
                            },
                            "workspace_registered": True,
                        },
                        handle,
                    )
                oj.register_instrument = lambda symbol, meta=None, persist=True: True
                oj.open_workspace = lambda symbol, meta=None, activate=True: {"symbol": symbol, "opened": True}
                oj.build_instrument_meta = lambda symbol, overrides=None: {
                    "symbol": symbol,
                    "instrument_kind": "equity",
                    "sector": "Technology",
                }
                oj.assess_symbol_history = lambda df, instrument_key=None, daily_long_history=None: {
                    "status_key": "partial_horizon_eligible",
                    "status_label": "Partial horizon eligibility",
                    "detail": "Existing daily history is still stronger than intraday history.",
                    "price_rows": 120,
                    "first_ts": "2026-04-01T14:30:00+00:00",
                    "last_ts": "2026-04-10T15:56:00+00:00",
                    "span_days": 9,
                    "session_count": 8,
                    "daily_long_rows": 25,
                    "feature_ready_modes": ["DailyHigh"],
                    "eligible_modes": ["DailyHigh"],
                    "blocked_modes": ["4min", "WeekEnd"],
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "paths": {
                        "4min": {"feature_rows": 60, "feature_ready": True, "train_eligible": False, "min_train_rows": 100, "detail": "needs more intraday rows"},
                        "DailyHigh": {"feature_rows": 18, "feature_ready": True, "train_eligible": True, "min_train_rows": 15, "detail": "ready"},
                        "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "min_train_rows": 6, "detail": "missing"},
                    },
                }
                oj.ensure_daily_long_history = lambda symbol, recent_df=None, persist=True: (
                    pd.DataFrame(
                        {
                            "open": [1.0],
                            "high": [1.0],
                            "low": [1.0],
                            "close": [1.0],
                            "volume": [1.0],
                        },
                        index=pd.date_range("2026-03-01", periods=1, tz="UTC"),
                    ),
                    {"rows": 25, "sources_used": ["existing_daily_long"], "updated": False},
                )
                oj.build_model_runtime = lambda symbol, include_modes=True: {
                    "instrument": symbol,
                    "family": "tabular_baseline_v2",
                    "models": {},
                    "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": []},
                }

                import readiness_tracker as rt

                old_update_readiness_state = rt.update_readiness_state
                old_load_deployment_evidence = rt.load_deployment_evidence
                rt.update_readiness_state = lambda symbol, online=False: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Core intraday path still missing",
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "next_requirements": ["Advance intraday history until the 4-minute path reaches 100 usable feature rows."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }
                rt.load_deployment_evidence = lambda symbol: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Daily-only model coverage retained",
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "next_requirements": ["Advance intraday history until the 4-minute path reaches 100 usable feature rows."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }

                df = pd.DataFrame(
                    [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                    index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
                )
                hub = _FakeHub(df)

                state = oj.run_onboarding_job("NVDA", train_models=False, data_hub=hub)
                summary = oj.summarize_onboarding_state("NVDA")

                self.assertEqual(state["status"], "completed")
                self.assertEqual(summary["artifacts"]["model_runtime"]["trained_modes"], ["DailyHigh"])
                self.assertEqual(summary["artifacts"]["model_runtime"]["available_modes"], ["DailyHigh"])
                self.assertEqual(summary["artifacts"]["model_runtime"]["retained_prior_modes"], ["DailyHigh"])
                self.assertTrue(summary["artifacts"]["model_runtime"]["retained_from_prior_run"])
                self.assertIn("prior onboarding model coverage was retained", summary["artifacts"]["model_runtime"]["detail"])
            finally:
                oj.DATA_DIR = old_data_dir
                oj.register_instrument = old_register_instrument
                oj.open_workspace = old_open_workspace
                oj.build_instrument_meta = old_build_instrument_meta
                oj.assess_symbol_history = old_assess_symbol_history
                oj.ensure_daily_long_history = old_ensure_daily_long_history
                oj.build_model_runtime = old_build_model_runtime
                rt.update_readiness_state = old_update_readiness_state
                rt.load_deployment_evidence = old_load_deployment_evidence

    def test_repeated_run_preserves_prior_primary_next_requirement_when_current_run_has_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            old_register_instrument = oj.register_instrument
            old_build_instrument_meta = oj.build_instrument_meta
            old_assess_symbol_history = oj.assess_symbol_history
            old_ensure_daily_long_history = oj.ensure_daily_long_history
            old_build_model_runtime = oj.build_model_runtime
            old_open_workspace = oj.open_workspace
            try:
                oj.DATA_DIR = tmp
                prior_requirement = "Advance intraday history until the 4-minute path reaches 100 usable feature rows."
                prior_state_path = os.path.join(tmp, "onboarding_state_MSFT.json")
                with open(prior_state_path, "w") as handle:
                    json.dump(
                        {
                            "symbol": "MSFT",
                            "status": "completed",
                            "primary_next_requirement": prior_requirement,
                            "artifacts": {
                                "history_assessment": {
                                    "primary_next_requirement": prior_requirement,
                                },
                            },
                            "workspace_registered": True,
                        },
                        handle,
                    )
                oj.register_instrument = lambda symbol, meta=None, persist=True: True
                oj.open_workspace = lambda symbol, meta=None, activate=True: {"symbol": symbol, "opened": True}
                oj.build_instrument_meta = lambda symbol, overrides=None: {
                    "symbol": symbol,
                    "instrument_kind": "equity",
                    "sector": "Technology",
                }
                oj.assess_symbol_history = lambda df, instrument_key=None, daily_long_history=None: {
                    "status_key": "feature_partial",
                    "status_label": "Features partially ready",
                    "detail": "Some feature paths can build usable rows, but none have reached their training floors yet.",
                    "price_rows": 120,
                    "first_ts": "2026-04-01T14:30:00+00:00",
                    "last_ts": "2026-04-10T15:56:00+00:00",
                    "span_days": 9,
                    "session_count": 8,
                    "daily_long_rows": 25,
                    "feature_ready_modes": ["4min"],
                    "eligible_modes": [],
                    "blocked_modes": ["4min", "DailyHigh", "WeekEnd"],
                    "primary_next_requirement": "",
                    "paths": {
                        "4min": {"feature_rows": 60, "feature_ready": True, "train_eligible": False, "min_train_rows": 100, "detail": "needs more intraday rows"},
                        "DailyHigh": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "min_train_rows": 15, "detail": "missing"},
                        "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "min_train_rows": 6, "detail": "missing"},
                    },
                }
                oj.ensure_daily_long_history = lambda symbol, recent_df=None, persist=True: (
                    pd.DataFrame(
                        {
                            "open": [1.0],
                            "high": [1.0],
                            "low": [1.0],
                            "close": [1.0],
                            "volume": [1.0],
                        },
                        index=pd.date_range("2026-03-01", periods=1, tz="UTC"),
                    ),
                    {"rows": 25, "sources_used": ["existing_daily_long"], "updated": False},
                )
                oj.build_model_runtime = lambda symbol, include_modes=True: {
                    "instrument": symbol,
                    "family": "tabular_baseline_v2",
                    "models": {},
                    "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": []},
                }

                import readiness_tracker as rt

                old_update_readiness_state = rt.update_readiness_state
                old_load_deployment_evidence = rt.load_deployment_evidence
                rt.update_readiness_state = lambda symbol, online=False: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Missing model evidence",
                    "primary_next_requirement": "",
                    "next_requirements": [],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Supported only",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }
                rt.load_deployment_evidence = lambda symbol: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Missing model evidence",
                    "primary_next_requirement": "",
                    "next_requirements": [],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Supported only",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }

                df = pd.DataFrame(
                    [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                    index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
                )
                hub = _FakeHub(df)

                state = oj.run_onboarding_job("MSFT", train_models=False, data_hub=hub)
                summary = oj.summarize_onboarding_state("MSFT")

                self.assertEqual(state["status"], "completed")
                self.assertEqual(state["primary_next_requirement"], prior_requirement)
                self.assertEqual(summary["primary_next_requirement"], prior_requirement)
                self.assertEqual(summary["artifacts"]["workspace_status"]["primary_next_requirement"], prior_requirement)
            finally:
                oj.DATA_DIR = old_data_dir
                oj.register_instrument = old_register_instrument
                oj.open_workspace = old_open_workspace
                oj.build_instrument_meta = old_build_instrument_meta
                oj.assess_symbol_history = old_assess_symbol_history
                oj.ensure_daily_long_history = old_ensure_daily_long_history
                oj.build_model_runtime = old_build_model_runtime
                rt.update_readiness_state = old_update_readiness_state
                rt.load_deployment_evidence = old_load_deployment_evidence

    def test_repeated_run_preserves_prior_training_plan_modes_when_current_plan_is_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            old_register_instrument = oj.register_instrument
            old_build_instrument_meta = oj.build_instrument_meta
            old_assess_symbol_history = oj.assess_symbol_history
            old_ensure_daily_long_history = oj.ensure_daily_long_history
            old_build_model_runtime = oj.build_model_runtime
            old_open_workspace = oj.open_workspace
            try:
                oj.DATA_DIR = tmp
                prior_state_path = os.path.join(tmp, "onboarding_state_AMZN.json")
                with open(prior_state_path, "w") as handle:
                    json.dump(
                        {
                            "symbol": "AMZN",
                            "status": "completed",
                            "artifacts": {
                                "training_plan": {
                                    "eligible_modes": ["DailyHigh"],
                                    "deferred_modes": ["4min"],
                                    "blocked_modes": ["WeekEnd"],
                                },
                            },
                            "workspace_registered": True,
                        },
                        handle,
                    )
                oj.register_instrument = lambda symbol, meta=None, persist=True: True
                oj.open_workspace = lambda symbol, meta=None, activate=True: {"symbol": symbol, "opened": True}
                oj.build_instrument_meta = lambda symbol, overrides=None: {
                    "symbol": symbol,
                    "instrument_kind": "equity",
                    "sector": "Technology",
                }
                oj.assess_symbol_history = lambda df, instrument_key=None, daily_long_history=None: {
                    "status_key": "partial_horizon_eligible",
                    "status_label": "Partial horizon eligibility",
                    "detail": "Existing daily history is still stronger than intraday history.",
                    "price_rows": 120,
                    "first_ts": "2026-04-01T14:30:00+00:00",
                    "last_ts": "2026-04-10T15:56:00+00:00",
                    "span_days": 9,
                    "session_count": 8,
                    "daily_long_rows": 25,
                    "feature_ready_modes": ["DailyHigh"],
                    "eligible_modes": ["DailyHigh"],
                    "blocked_modes": ["4min", "WeekEnd"],
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "paths": {
                        "4min": {"feature_rows": 60, "feature_ready": True, "train_eligible": False, "min_train_rows": 100, "detail": "needs more intraday rows"},
                        "DailyHigh": {"feature_rows": 18, "feature_ready": True, "train_eligible": True, "min_train_rows": 15, "detail": "ready"},
                        "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False, "min_train_rows": 6, "detail": "missing"},
                    },
                }
                oj.ensure_daily_long_history = lambda symbol, recent_df=None, persist=True: (
                    pd.DataFrame(
                        {
                            "open": [1.0],
                            "high": [1.0],
                            "low": [1.0],
                            "close": [1.0],
                            "volume": [1.0],
                        },
                        index=pd.date_range("2026-03-01", periods=1, tz="UTC"),
                    ),
                    {"rows": 25, "sources_used": ["existing_daily_long"], "updated": False},
                )
                oj.build_model_runtime = lambda symbol, include_modes=True: {
                    "instrument": symbol,
                    "family": "tabular_baseline_v2",
                    "models": {},
                    "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": []},
                }

                import readiness_tracker as rt

                old_update_readiness_state = rt.update_readiness_state
                old_load_deployment_evidence = rt.load_deployment_evidence
                rt.update_readiness_state = lambda symbol, online=False: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Missing model evidence",
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "next_requirements": ["Advance intraday history until the 4-minute path reaches 100 usable feature rows."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }
                rt.load_deployment_evidence = lambda symbol: {
                    "stage_key": "supported_not_trained",
                    "stage_label": "Supported / not trained",
                    "summary": "Missing model evidence",
                    "primary_next_requirement": "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
                    "next_requirements": ["Advance intraday history until the 4-minute path reaches 100 usable feature rows."],
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Partially trained",
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                }

                df = pd.DataFrame(
                    [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                    index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
                )
                hub = _FakeHub(df)

                state = oj.run_onboarding_job("AMZN", train_models=False, data_hub=hub)
                summary = oj.summarize_onboarding_state("AMZN")
                plan_artifact = summary["artifacts"]["training_plan"]

                self.assertEqual(state["status"], "completed")
                self.assertEqual(plan_artifact["eligible_modes"], ["DailyHigh"])
                self.assertEqual(plan_artifact["deferred_modes"], ["4min"])
                self.assertEqual(plan_artifact["blocked_modes"], ["WeekEnd"])
                self.assertTrue(plan_artifact["retained_from_prior_run"])
                self.assertEqual(
                    plan_artifact["retained_prior_fields"],
                    ["eligible_modes", "deferred_modes", "blocked_modes"],
                )
            finally:
                oj.DATA_DIR = old_data_dir
                oj.register_instrument = old_register_instrument
                oj.open_workspace = old_open_workspace
                oj.build_instrument_meta = old_build_instrument_meta
                oj.assess_symbol_history = old_assess_symbol_history
                oj.ensure_daily_long_history = old_ensure_daily_long_history
                oj.build_model_runtime = old_build_model_runtime
                rt.update_readiness_state = old_update_readiness_state
                rt.load_deployment_evidence = old_load_deployment_evidence

    def test_summarize_onboarding_state_defaults_when_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = oj.DATA_DIR
            try:
                oj.DATA_DIR = tmp
                summary = oj.summarize_onboarding_state("MSFT")
                self.assertEqual(summary["status"], "not_started")
                self.assertIn("No staged onboarding job", summary["summary"])
                self.assertEqual(summary["artifacts"], {})
                self.assertEqual(summary["next_stage_key"], "validate_symbol")
                self.assertEqual(summary["stage_status_counts"]["pending"], len(oj.ONBOARDING_STAGE_ORDER))
            finally:
                oj.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()
