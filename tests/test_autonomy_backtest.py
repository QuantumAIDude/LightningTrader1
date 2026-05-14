import unittest
import os
import tempfile

import pandas as pd

import autonomy_backtest as ab
from model_experience_ledger import read_records
from model_utils import _staleness_info


def _tiny_price_frame():
    idx = pd.date_range("2026-03-02 14:30:00+00:00", periods=24, freq="2min")
    return pd.DataFrame(
        {
            "open": [500.0 + 0.01 * i for i in range(len(idx))],
            "high": [500.1 + 0.01 * i for i in range(len(idx))],
            "low": [499.9 + 0.01 * i for i in range(len(idx))],
            "close": [500.05 + 0.01 * i for i in range(len(idx))],
            "volume": [1000 + 5 * i for i in range(len(idx))],
        },
        index=idx,
    )


class AutonomyBacktestTests(unittest.TestCase):
    def test_normalize_frame_uses_datetime_column_when_present(self):
        raw = pd.DataFrame(
            {
                "Datetime": [
                    "2026-03-02 14:30:00+00:00",
                    "2026-03-02 14:32:00+00:00",
                    "2026-03-03 14:30:00+00:00",
                ],
                "open": [500.0, 500.1, 501.0],
                "high": [500.2, 500.3, 501.2],
                "low": [499.9, 500.0, 500.8],
                "close": [500.1, 500.2, 501.1],
                "volume": [1000, 1100, 1200],
            }
        )
        frame = ab._normalize_frame(raw)
        self.assertEqual(frame.index[0], pd.Timestamp("2026-03-02 14:30:00+00:00"))
        self.assertEqual(frame.index[-1], pd.Timestamp("2026-03-03 14:30:00+00:00"))
        self.assertEqual(pd.Index(ab._session_dates(frame.index).dropna().unique()).nunique(), 2)

    def test_simulation_summary_tracks_long_and_short_trades(self):
        sim = {
            "starting_cash": 100000.0,
            "ending_equity": 100250.0,
            "trades": [
                {
                    "side": "buy",
                    "net_pnl": 150.0,
                    "gross_pnl": 170.0,
                    "total_fees": 20.0,
                    "return_pct": 0.003,
                    "hold_minutes": 18.0,
                    "entry_bucket": "AM",
                    "entry_regime": "Bullish",
                    "news_alignment": "aligned",
                    "exit_reason": "target hit",
                },
                {
                    "side": "sell",
                    "net_pnl": 100.0,
                    "gross_pnl": 112.0,
                    "total_fees": 12.0,
                    "return_pct": 0.002,
                    "hold_minutes": 26.0,
                    "entry_bucket": "PM",
                    "entry_regime": "Bearish",
                    "news_alignment": "neutral",
                    "exit_reason": "flat by close",
                },
            ],
            "equity_curve": [
                {"ts": pd.Timestamp("2026-03-02T15:00:00Z"), "equity": 100000.0},
                {"ts": pd.Timestamp("2026-03-02T16:00:00Z"), "equity": 100250.0},
            ],
            "block_counts": {"Blocked": 3},
            "block_reason_counts": {"cost-adjusted edge is too small": 2},
            "block_combo_counts": {"quality score is below the autonomy threshold | intraday confidence is too low": 1},
            "action_counts": {"shadow_entry": 2, "shadow_exit": 2},
        }
        summary = ab.summarize_simulation(sim)
        self.assertEqual(summary["trade_count"], 2)
        self.assertEqual(summary["long_trades"], 1)
        self.assertEqual(summary["short_trades"], 1)
        self.assertAlmostEqual(summary["net_pnl"], 250.0)
        self.assertAlmostEqual(summary["return_pct"], 0.0025)
        self.assertIn("AM", summary["time_of_day"])
        self.assertIn("Bullish", summary["regimes"])
        self.assertIn("cost-adjusted edge is too small", summary["block_reason_counts"])

    def test_record_diagnostics_separate_raw_and_finished_signal(self):
        records = [
            {
                "advisory": {
                    "pred_state": {
                        "intraday": {
                        "direction": "-",
                        "direction_stage": "blocked",
                        "raw_direction": "UP",
                            "horizon_support_stage": "supportive",
                            "horizon_support_score": 0.44,
                            "raw_pred_ret_bps": 3.2,
                            "raw_edge_bps": 2.5,
                            "edge_bps": -0.4,
                            "confidence": 34.0,
                            "raw_confidence": 61.0,
                            "position_size": 0.0,
                            "raw_position_size": 0.27,
                            "uncertainty_bps": 6.8,
                        }
                    },
                    "quality_meta": {"score": 0.59, "no_trade": True},
                    "news_meta": {"alignment": "neutral", "headline_state": "ok"},
                },
                "predictions": {"intraday": {"_meta": {"predictability_gate": {"predictable_now": True}}}},
            }
        ]
        diag = ab.summarize_record_diagnostics(records)
        self.assertEqual(diag["evaluations"], 1)
        self.assertEqual(diag["direction_non_dash"], 0)
        self.assertEqual(diag["raw_direction_non_dash"], 1)
        self.assertEqual(diag["tentative_direction_non_dash"], 0)
        self.assertEqual(diag["horizon_supportive"], 1)
        self.assertEqual(diag["raw_edge_positive"], 1)
        self.assertEqual(diag["finished_edge_positive"], 0)
        self.assertAlmostEqual(diag["means"]["raw_confidence"], 61.0)

    def test_run_stock_autonomy_validation_splits_holdout_from_folds(self):
        old_fold_specs = ab._fold_specs
        old_build = ab._build_records_for_fold
        old_sim = ab.simulate_controller_records
        try:
            ab._fold_specs = lambda df, n_splits=5: [
                {"fold": 1, "train_sessions": [pd.Timestamp("2026-03-02")], "test_sessions": [pd.Timestamp("2026-03-03")], "train_start": "2026-03-02", "train_end": "2026-03-02", "test_start": "2026-03-03", "test_end": "2026-03-03"},
                {"fold": 2, "train_sessions": [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03")], "test_sessions": [pd.Timestamp("2026-03-04")], "train_start": "2026-03-02", "train_end": "2026-03-03", "test_start": "2026-03-04", "test_end": "2026-03-04"},
                {"fold": 3, "train_sessions": [pd.Timestamp("2026-03-02"), pd.Timestamp("2026-03-03"), pd.Timestamp("2026-03-04")], "test_sessions": [pd.Timestamp("2026-03-05")], "train_start": "2026-03-02", "train_end": "2026-03-04", "test_start": "2026-03-05", "test_end": "2026-03-05"},
            ]
            def fake_build_records(*args, **kwargs):
                records = [
                    {
                        "as_of": pd.Timestamp("2026-03-03T15:00:00Z"),
                        "current_price": 500.0,
                        "market_df": pd.DataFrame({"close": [500.0]}, index=[pd.Timestamp("2026-03-03T14:58:00Z")]),
                        "advisory": {"quality_meta": {}, "plan_meta": {}, "pred_state": {"intraday": {}}, "news_meta": {}},
                    }
                ]
                return {"status": "ok", "records": records, "models": {"regime": object()}}

            ab._build_records_for_fold = fake_build_records
            ab.simulate_controller_records = lambda *args, **kwargs: {
                "starting_cash": 100000.0,
                "ending_equity": 100100.0,
                "trades": [],
                "equity_curve": [
                    {"ts": pd.Timestamp("2026-03-03T15:00:00Z"), "equity": 100000.0},
                    {"ts": pd.Timestamp("2026-03-03T15:02:00Z"), "equity": 100100.0},
                ],
                "block_counts": {},
                "action_counts": {},
            }
            out = ab.run_stock_autonomy_validation(_tiny_price_frame(), instrument_key="QQQ", n_splits=3, holdout_folds=1)
            self.assertEqual(out["instrument"], "QQQ")
            self.assertEqual(out["walk_forward"]["folds"], 2)
            self.assertEqual(out["holdout"]["folds"], 1)
            self.assertEqual(out["folds"][0]["status"], "ok")
            self.assertIn("baseline", out["sensitivity"])
            self.assertEqual(out["options_path"]["status"], "unsupported")
        finally:
            ab._fold_specs = old_fold_specs
            ab._build_records_for_fold = old_build
            ab.simulate_controller_records = old_sim

    def test_truncated_daily_history_accepts_timezone_aware_cache(self):
        old_read_csv = ab.pd.read_csv
        try:
            def fake_read_csv(*args, **kwargs):
                idx = pd.DatetimeIndex(
                    [
                        pd.Timestamp("2026-01-14T00:00:00Z"),
                        pd.Timestamp("2026-01-15T00:00:00Z"),
                        pd.Timestamp("2026-01-16T00:00:00Z"),
                    ]
                )
                return pd.DataFrame({"close": [500.0, 501.0, 502.0]}, index=idx)

            ab.pd.read_csv = fake_read_csv
            history = ab._load_truncated_daily_history("QQQ", pd.Timestamp("2026-01-15T18:00:00Z"))
            self.assertEqual(len(history), 2)
            self.assertIsNone(history.index.tz)
            self.assertEqual(history.index[-1], pd.Timestamp("2026-01-15"))
        finally:
            ab.pd.read_csv = old_read_csv

    def test_historical_replay_context_uses_as_of_freshness_not_wall_clock(self):
        idx = pd.date_range("2020-01-02 14:30:00+00:00", periods=180, freq="2min")
        frame = pd.DataFrame({"close": [100.0 + (0.01 * i) for i in range(len(idx))]}, index=idx)
        context = ab._historical_replay_context(frame, idx[-1] + pd.Timedelta(minutes=2), instrument_key="QQQ")
        self.assertTrue(context["historical_mode"])
        self.assertTrue(context["freshness_is_as_of"])
        self.assertTrue(context["fresh_as_of"])
        self.assertFalse(context["stale_as_of"])
        self.assertEqual(context["bars_available_as_of"], 180)

    def test_historical_replay_context_marks_insufficient_prior_bars(self):
        frame = _tiny_price_frame()
        context = ab._historical_replay_context(
            frame,
            frame.index[-1] + pd.Timedelta(minutes=2),
            instrument_key="QQQ",
            min_bars=160,
        )
        self.assertFalse(context["fresh_as_of"])
        self.assertTrue(context["stale_as_of"])
        self.assertFalse(context["enough_prior_bars"])
        self.assertEqual(context["stale_severity_as_of"], "insufficient_history")

    def test_live_staleness_still_uses_wall_clock_when_not_in_replay(self):
        idx = pd.date_range("2020-01-02 14:30:00+00:00", periods=180, freq="2min")
        frame = pd.DataFrame({"close": [100.0 + (0.01 * i) for i in range(len(idx))]}, index=idx)
        live_staleness = _staleness_info(frame, instrument_key="QQQ", now_utc=pd.Timestamp("2026-04-29T16:00:00Z"))
        self.assertTrue(live_staleness["is_stale"])

    def test_historical_freshness_alone_does_not_create_autonomy_permission(self):
        frame = _tiny_price_frame()
        as_of = pd.Timestamp("2026-03-02T15:20:00Z")
        advisory = {
            "quality_meta": {"score": 0.9, "no_trade": False},
            "plan_meta": {"bias": "Bullish", "entry": 500.0, "stop": 498.0, "target1": 504.0},
            "pred_state": {"intraday": {"direction": "UP", "confidence": 90.0, "position_size": 1.0, "edge_bps": 10.0}},
            "news_meta": {"headline_state": "ok", "alignment": "neutral"},
            "liquidity_meta": {"tradability_score": 0.9},
            "macro_meta": {"risk_score": 0.0},
            "catalyst_meta": {"catalyst_risk_score": 0.0},
            "expression_meta": {"stock": {"detail": "test"}},
            "signal_translation": {"operator_side": "buy", "watchlist": False, "headline": "test"},
        }
        exec_snapshot = ab._build_execution_snapshot(
            ab.new_paper_state(starting_cash=100000.0, broker_mode="backtest_probe"),
            "QQQ",
            500.0,
            advisory,
            as_of,
        )
        status = ab.autonomy._stock_path_status(
            "QQQ",
            "shadow",
            frame,
            advisory,
            exec_snapshot,
            replay_snapshot={"active": False},
            now_utc=as_of,
            readiness_override={"live_ready": True, "fresh": True},
        )
        self.assertFalse(status["allowed"])
        self.assertIn("validation evidence does not support shadow mode yet", status["reasons"])

    def test_report_keeps_advisory_and_autonomy_sections_separate(self):
        path = "reports/qqq_backtest_advisory_vs_autonomy_report.md"
        with open(path, "r") as f:
            report = f.read()
        self.assertIn("## 4. Advisory Mode Results", report)
        self.assertIn("## 5. Autonomous Mode Results", report)

    def test_replay_experience_records_resolve_only_future_rows(self):
        records = [
            {
                "as_of": pd.Timestamp("2026-03-02T15:00:00Z"),
                "current_price": 100.0,
                "advisory": {
                    "decision_payload": {"action": "buy", "model_edge": {"direction": "UP"}},
                    "risk_meta": {},
                    "deployment_meta": {"validation_evidence": {"validation_status": "missing"}},
                },
                "predictions": {
                    "intraday": {
                        "base": {"pred_ret": 0.01},
                        "consensus": {"pred_ret": 0.01, "confidence": 60.0},
                    }
                },
            },
            {
                "as_of": pd.Timestamp("2026-03-02T15:04:00Z"),
                "current_price": 101.0,
                "advisory": {"decision_payload": {"action": "stand_aside"}, "deployment_meta": {}},
                "predictions": {"intraday": {}},
            },
        ]
        experiences = ab.build_replay_experience_records(records, instrument_key="QQQ")
        self.assertEqual(len(experiences), 2)
        self.assertAlmostEqual(experiences[0]["future_return_by_horizon"]["1_step"], 0.01)
        self.assertTrue(experiences[0]["direction_correct_by_horizon"]["1_step"])
        self.assertIsNotNone(experiences[0]["outcome_resolved_at"])
        self.assertEqual(experiences[1]["future_return_by_horizon"], {})
        self.assertIsNone(experiences[1]["outcome_resolved_at"])

    def test_replay_experience_records_do_not_mutate_advisory_decisions(self):
        advisory = {"decision_payload": {"action": "stand_aside"}, "deployment_meta": {}}
        records = [
            {
                "as_of": pd.Timestamp("2026-03-02T15:00:00Z"),
                "current_price": 100.0,
                "advisory": advisory,
                "predictions": {"intraday": {"consensus": {"pred_ret": 0.0}}},
            },
            {
                "as_of": pd.Timestamp("2026-03-02T15:04:00Z"),
                "current_price": 100.1,
                "advisory": advisory,
                "predictions": {"intraday": {"consensus": {"pred_ret": 0.0}}},
            },
        ]
        ab.build_replay_experience_records(records, instrument_key="QQQ")
        self.assertEqual(records[0]["advisory"]["decision_payload"]["action"], "stand_aside")
        self.assertNotIn("future_return_by_horizon", records[0])

    def test_run_validation_can_write_experience_ledger_without_creating_trades(self):
        old_fold_specs = ab._fold_specs
        old_build = ab._build_records_for_fold
        old_sim = ab.simulate_controller_records
        try:
            ab._fold_specs = lambda df, n_splits=5: [
                {
                    "fold": 1,
                    "train_sessions": [pd.Timestamp("2026-03-02")],
                    "test_sessions": [pd.Timestamp("2026-03-03")],
                    "train_start": "2026-03-02",
                    "train_end": "2026-03-02",
                    "test_start": "2026-03-03",
                    "test_end": "2026-03-03",
                }
            ]
            def fake_build_records(*args, **kwargs):
                records = [
                    {
                        "as_of": pd.Timestamp("2026-03-03T15:00:00Z"),
                        "current_price": 500.0,
                        "market_df": pd.DataFrame({"close": [500.0]}, index=[pd.Timestamp("2026-03-03T14:58:00Z")]),
                        "advisory": {"decision_payload": {"action": "stand_aside"}, "risk_meta": {}, "deployment_meta": {}},
                        "predictions": {"intraday": {"consensus": {"pred_ret": 0.0}}},
                    },
                    {
                        "as_of": pd.Timestamp("2026-03-03T15:04:00Z"),
                        "current_price": 501.0,
                        "market_df": pd.DataFrame({"close": [501.0]}, index=[pd.Timestamp("2026-03-03T15:02:00Z")]),
                        "advisory": {"decision_payload": {"action": "stand_aside"}, "risk_meta": {}, "deployment_meta": {}},
                        "predictions": {"intraday": {"consensus": {"pred_ret": 0.0}}},
                    },
                ]
                max_records = kwargs.get("max_records")
                if max_records is not None:
                    records = records[: int(max_records)]
                return {"status": "ok", "records": records, "models": {"regime": object()}}

            ab._build_records_for_fold = fake_build_records
            ab.simulate_controller_records = lambda *args, **kwargs: {
                "starting_cash": 100000.0,
                "ending_equity": 100000.0,
                "trades": [],
                "equity_curve": [],
                "block_counts": {"Blocked": 2},
                "action_counts": {"blocked": 2},
            }
            with tempfile.TemporaryDirectory() as tmp:
                ledger_path = os.path.join(tmp, "experience.jsonl")
                out = ab.run_stock_autonomy_validation(
                    _tiny_price_frame(),
                    instrument_key="QQQ",
                    n_splits=2,
                    holdout_folds=0,
                    include_sensitivity=False,
                    experience_ledger_path=ledger_path,
                    return_experience_records=True,
                    max_replay_records=1,
                )
                self.assertEqual(out["experience_ledger"]["records_built"], 1)
                self.assertEqual(out["experience_ledger"]["records_written"], 1)
                self.assertEqual(len(read_records(ledger_path)), 1)
                self.assertEqual(out["raw_folds"][0]["summary"]["trade_count"], 0)
                self.assertEqual(out["experience_records"][0]["advisory_action"], "stand_aside")
        finally:
            ab._fold_specs = old_fold_specs
            ab._build_records_for_fold = old_build
            ab.simulate_controller_records = old_sim


if __name__ == "__main__":
    unittest.main()
