import os
import tempfile
import unittest

from model_experience_ledger import (
    append_record,
    append_records,
    build_experience_record,
    build_outcome_record,
    format_quality_report,
    ledger_quality_summary,
    meta_model_readiness_verdict,
    format_status_report,
    read_records,
    resolve_one_step_outcome,
    summarize_ledger,
    validate_experience_record,
)


class ModelExperienceLedgerTests(unittest.TestCase):
    def test_build_minimal_record(self):
        record = build_experience_record(symbol="qqq", as_of="2026-03-02T15:00:00Z", source="test")
        self.assertEqual(record["record_type"], "experience")
        self.assertEqual(record["symbol"], "QQQ")
        self.assertEqual(record["source"], "test")
        self.assertEqual(record["base_predictions"], {})
        self.assertEqual(record["future_return_by_horizon"], {})
        self.assertIsNone(record["outcome_resolved_at"])

    def test_build_rich_record_with_context_fields(self):
        predictions = {
            "intraday": {
                "base": {"direction": "UP", "pred_ret": 0.002, "confidence": 61.0},
                "forest": {"direction": "DOWN", "pred_ret": -0.001, "confidence": 52.0},
                "consensus": {"pred_ret": 0.0012, "confidence": 58.0, "disagreement_bps": 12.5},
                "regime": {"Bullish": 60.0, "Neutral": 30.0, "Bearish": 10.0, "regime_conf": 0.42},
            }
        }
        advisory = {
            "decision_payload": {
                "action": "watchlist",
                "thesis": "edge not mature",
                "model_edge": {"direction": "UP", "confidence": 58.0},
                "economic_edge": {"state_key": "cost_crushed"},
                "blockers": ["cost-adjusted edge is too small"],
                "validation_blockers": ["no_cost_model"],
                "next_best_action": "run replay backtest with transaction costs",
            },
            "deployment_meta": {
                "validation_evidence": {
                    "validation_status": "insufficient",
                    "validation_method": "replay_backtest",
                    "blockers": ["no_cost_model"],
                }
            },
            "risk_meta": {"blocked": False},
        }
        record = build_experience_record(
            symbol="QQQ",
            as_of="2026-03-02T15:00:00Z",
            source="backtest",
            predictions=predictions,
            advisory=advisory,
            market_context={"time_bucket": "AM"},
            news_context={"alignment": "neutral"},
        )
        self.assertEqual(record["advisory_action"], "watchlist")
        self.assertEqual(record["validation_status"], "insufficient")
        self.assertIn("base", record["base_predictions"])
        self.assertAlmostEqual(record["average_prediction"], 0.0012)
        self.assertAlmostEqual(record["model_disagreement"], 12.5)
        self.assertIn("no_cost_model", record["blockers"])
        self.assertEqual(record["market_context"]["time_bucket"], "AM")

    def test_append_and_read_jsonl_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            record = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:00:00Z", source="test")
            append_record(record, path=path)
            append_records([record], path=path)
            records = read_records(path)
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["symbol"], "QQQ")

    def test_missing_optional_fields_do_not_invent_data(self):
        record = build_experience_record(symbol="QQQ", as_of=None, source="test", predictions=None, advisory=None)
        self.assertEqual(record["base_predictions"], {})
        self.assertEqual(record["risk_summary"], {})
        self.assertIsNone(record["validation_status"])
        self.assertIsNone(record["regime_label"])

    def test_outcome_resolution_and_linked_outcome_record(self):
        record = build_experience_record(
            symbol="QQQ",
            as_of="2026-03-02T15:00:00Z",
            source="test",
            predictions={
                "intraday": {
                    "base": {"pred_ret": 0.010},
                    "forest": {"pred_ret": -0.005},
                    "consensus": {"pred_ret": 0.008},
                }
            },
            advisory={"decision_payload": {"action": "buy", "model_edge": {"direction": "UP"}}},
        )
        resolved = resolve_one_step_outcome(
            record,
            current_price=100.0,
            future_price=101.0,
            future_as_of="2026-03-02T15:04:00Z",
            cost_pct=0.002,
        )
        self.assertAlmostEqual(resolved["future_return_by_horizon"]["1_step"], 0.01)
        self.assertTrue(resolved["direction_correct_by_horizon"]["1_step"])
        self.assertAlmostEqual(resolved["net_after_costs_by_horizon"]["1_step"], 0.008)
        self.assertEqual(resolved["best_base_model_by_horizon"]["1_step"], "base")
        outcome = build_outcome_record(record_id=record["record_id"], source="test", outcome={"future_return_1_step": 0.01})
        self.assertEqual(outcome["record_type"], "outcome")
        self.assertEqual(outcome["parent_record_id"], record["record_id"])

    def test_record_ids_are_unique_enough_by_default(self):
        first = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:00:00Z", source="test")
        second = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:00:00Z", source="test")
        self.assertNotEqual(first["record_id"], second["record_id"])

    def test_summarize_and_format_status_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            unresolved = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:00:00Z", source="test")
            resolved = resolve_one_step_outcome(unresolved, current_price=100.0, future_price=100.5, cost_pct=0.002)
            append_records([unresolved, resolved], path=path)
            summary = summarize_ledger(path)
            self.assertEqual(summary["experience_records"], 2)
            self.assertEqual(summary["resolved_experience_records"], 1)
            report = format_status_report(summary)
            self.assertIn("Model Experience Ledger Status", report)
            self.assertIn("QQQ", report)

    def test_quality_summary_counts_missing_and_resolved_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            complete = build_experience_record(
                symbol="QQQ",
                as_of="2026-03-02T15:00:00Z",
                source="backtest",
                predictions={"intraday": {"base": {"pred_ret": 0.001}, "consensus": {"pred_ret": 0.001}}},
                advisory={
                    "decision_payload": {"action": "stand_aside"},
                    "risk_meta": {"blocked": False},
                    "deployment_meta": {"validation_evidence": {"validation_status": "missing"}},
                },
            )
            complete = resolve_one_step_outcome(complete, current_price=100.0, future_price=100.2, cost_pct=0.002)
            malformed = {"record_type": "experience", "record_id": "bad", "source": "test"}
            append_records([complete, malformed], path=path)
            with open(path, "a", encoding="utf-8") as f:
                f.write("{bad json\n")
            summary = ledger_quality_summary(path)
            self.assertEqual(summary["total_records"], 2)
            self.assertEqual(summary["malformed_records"], 1)
            self.assertEqual(summary["resolved_outcome_count"], 1)
            self.assertEqual(summary["unresolved_outcome_count"], 1)
            self.assertEqual(summary["missing_required_identity_count"], 1)
            self.assertEqual(summary["missing_base_predictions_count"], 1)

    def test_quality_summary_detects_duplicate_ids_and_future_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            leaked = build_experience_record(
                symbol="QQQ",
                as_of="2026-03-02T15:00:00Z",
                source="backtest",
                record_id="dup",
                market_context={"future_return_by_horizon": {"1_step": 0.01}},
            )
            duplicate = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:04:00Z", source="backtest", record_id="dup")
            broken_outcome = build_outcome_record(record_id="missing-parent", source="test", outcome={"future_return_1_step": 0.01})
            append_records([leaked, duplicate, broken_outcome], path=path)
            summary = ledger_quality_summary(path)
            self.assertEqual(summary["duplicate_record_id_count"], 1)
            self.assertEqual(summary["future_outcome_input_leak_count"], 1)
            self.assertEqual(summary["broken_outcome_link_count"], 1)

    def test_format_quality_report_for_small_temp_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            record = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:00:00Z", source="test")
            append_record(record, path=path)
            report = format_quality_report(ledger_quality_summary(path), generated_by="unit-test", sample=True)
            self.assertIn("Model Experience Ledger Quality Report", report)
            self.assertIn("not enough to train", report)

    def test_validate_experience_record_warns_on_recommended_fields(self):
        record = build_experience_record(symbol="QQQ", as_of="2026-03-02T15:00:00Z", source="test")
        validation = validate_experience_record(record)
        self.assertEqual(validation["status"], "warning")
        self.assertIn("missing recommended field: base_predictions", validation["warnings"])
        self.assertFalse(validation["errors"])

    def test_validate_experience_record_invalid_on_required_fields(self):
        validation = validate_experience_record({"record_type": "experience", "source": "test"})
        self.assertEqual(validation["status"], "invalid")
        self.assertIn("missing required field: record_id", validation["errors"])
        self.assertIn("missing required field: symbol", validation["errors"])

    def test_quality_report_includes_schema_counts_and_conservative_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            record = build_experience_record(
                symbol="QQQ",
                as_of="2026-03-02T15:00:00Z",
                source="test",
                predictions={"intraday": {"base": {"pred_ret": 0.001}, "consensus": {"pred_ret": 0.001}}},
                advisory={
                    "decision_payload": {"action": "stand_aside"},
                    "risk_meta": {"blocked": False},
                    "deployment_meta": {"validation_evidence": {"validation_status": "missing"}},
                },
            )
            record = resolve_one_step_outcome(record, current_price=100.0, future_price=100.2, cost_pct=0.002)
            append_record(record, path=path)
            summary = ledger_quality_summary(path)
            self.assertEqual(summary["schema_invalid_record_count"], 0)
            self.assertEqual(meta_model_readiness_verdict(summary), "inspection_ready")
            report = format_quality_report(summary)
            self.assertIn("Schema valid / warning / invalid", report)
            self.assertIn("Suitable for training: False", report)


if __name__ == "__main__":
    unittest.main()
