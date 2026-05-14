import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from model_experience_ledger import (
    append_records,
    build_experience_record,
    ledger_quality_summary,
    read_records,
    resolve_one_step_outcome,
)
import scripts.generate_qqq_experience_ledger_sample as runner


def _write_source(path: str, rows: int = 6) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("timestamp,open,high,low,close,volume\n")
        for i in range(rows):
            minute = 30 + (2 * i)
            f.write(f"2026-01-02 14:{minute:02d}:00+00:00,100,101,99,{100 + i * 0.1},1000\n")


def _fake_replay_factory(calls: list[dict]):
    def _fake_replay(df, **kwargs):
        calls.append(dict(kwargs))
        count = int(kwargs.get("max_replay_records") or 0)
        path = kwargs.get("experience_ledger_path")
        records = []
        for i in range(count):
            rec = build_experience_record(
                symbol="QQQ",
                as_of=f"2026-01-02T14:{32 + i:02d}:00+00:00",
                source="backtest",
                predictions={
                    "intraday": {
                        "base": {"pred_ret": 0.001, "confidence": 55.0},
                        "consensus": {"pred_ret": 0.001, "confidence": 55.0},
                        "regime": {"Bullish": 10.0, "Neutral": 80.0, "Bearish": 10.0, "regime_conf": 0.4},
                    }
                },
                advisory={
                    "decision_payload": {"action": "stand_aside", "autonomy_posture": {"stage": "Advisory only"}},
                    "risk_meta": {"blocked": False},
                    "deployment_meta": {"validation_evidence": {"validation_status": "missing"}},
                },
            )
            if i < count - 1:
                rec = resolve_one_step_outcome(
                    rec,
                    current_price=100.0 + i,
                    future_price=100.5 + i,
                    future_as_of=f"2026-01-02T14:{33 + i:02d}:00+00:00",
                    cost_pct=0.002,
                )
            records.append(rec)
        append_records(records, path=path)
        return {
            "status": "ok",
            "experience_ledger": {"records_built": count, "records_written": count},
            "raw_folds": [{"summary": {"trade_count": 0}}],
        }

    return _fake_replay


def _args(tmp: str, *extra: str):
    source = os.path.join(tmp, "prices_QQQ.csv")
    ledger = os.path.join(tmp, "ledger.jsonl")
    quality = os.path.join(tmp, "quality.md")
    status = os.path.join(tmp, "status.md")
    cache = os.path.join(tmp, "cache.json")
    return runner.build_parser().parse_args(
        [
            "--source-path",
            source,
            "--ledger-path",
            ledger,
            "--quality-report-path",
            quality,
            "--status-report-path",
            status,
            "--cache-path",
            cache,
            *extra,
        ]
    )


class GenerateQQQExperienceLedgerSampleTests(unittest.TestCase):
    def test_script_imports_without_side_effects(self):
        self.assertTrue(callable(runner.main))
        self.assertEqual(runner.CACHE_SCHEMA_VERSION, 1)

    def test_missing_source_data_produces_clear_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, "--max-records", "3")
            with self.assertRaises(FileNotFoundError) as ctx:
                runner.generate_sample(args, replay_runner=_fake_replay_factory([]))
            self.assertIn("Missing local QQQ source data", str(ctx.exception))

    def test_force_bypasses_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, "--max-records", "3", "--force")
            _write_source(args.source_path)
            calls = []
            runner.generate_sample(args, replay_runner=_fake_replay_factory(calls))
            runner.generate_sample(args, replay_runner=_fake_replay_factory(calls))
            self.assertEqual(len(calls), 2)

    def test_compatible_cache_is_reused(self):
        with tempfile.TemporaryDirectory() as tmp:
            args_force = _args(tmp, "--max-records", "3", "--force")
            _write_source(args_force.source_path)
            calls = []
            runner.generate_sample(args_force, replay_runner=_fake_replay_factory(calls))
            args_cached = _args(tmp, "--max-records", "3")
            result = runner.generate_sample(
                args_cached,
                replay_runner=lambda *a, **k: self.fail("replay should not run when cache is compatible"),
            )
            self.assertTrue(result["reused_cache"])
            self.assertEqual(len(calls), 1)

    def test_cache_key_changes_when_source_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = os.path.join(tmp, "prices_QQQ.csv")
            _write_source(source, rows=3)
            first = runner.build_cache_key(source, 5)
            with open(source, "a", encoding="utf-8") as f:
                f.write("2026-01-02 15:00:00+00:00,100,101,99,101,1000\n")
            second = runner.build_cache_key(source, 5)
            self.assertNotEqual(first["source"]["size"], second["source"]["size"])

    def test_max_records_is_respected_and_reports_are_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, "--max-records", "4", "--force")
            _write_source(args.source_path)
            result = runner.generate_sample(args, replay_runner=_fake_replay_factory([]))
            self.assertEqual(result["quality"]["total_records"], 4)
            self.assertEqual(len(read_records(args.ledger_path)), 4)
            self.assertTrue(Path(args.quality_report_path).exists())
            self.assertTrue(Path(args.status_report_path).exists())
            self.assertIn("Schema valid / warning / invalid", Path(args.quality_report_path).read_text(encoding="utf-8"))

    def test_missing_recommended_fields_warn_not_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            append_records([build_experience_record(symbol="QQQ", as_of="2026-01-02T15:00:00Z", source="test")], path=path)
            summary = ledger_quality_summary(path)
            self.assertEqual(summary["schema_invalid_record_count"], 0)
            self.assertEqual(summary["schema_warning_record_count"], 1)

    def test_missing_required_fields_invalid(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ledger.jsonl")
            append_records([{"record_type": "experience", "source": "test"}], path=path)
            summary = ledger_quality_summary(path)
            self.assertEqual(summary["schema_invalid_record_count"], 1)

    def test_no_network_or_alpaca_call_is_made_with_injected_runner(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, "--max-records", "2", "--force")
            _write_source(args.source_path)
            calls = []
            runner.generate_sample(args, replay_runner=_fake_replay_factory(calls))
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["instrument_key"], "QQQ")
            self.assertEqual(calls[0]["include_sensitivity"], False)

    def test_main_returns_nonzero_for_missing_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = _args(tmp, "--max-records", "1")
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                code = runner.main(
                    [
                        "--source-path",
                        args.source_path,
                        "--ledger-path",
                        args.ledger_path,
                        "--quality-report-path",
                        args.quality_report_path,
                        "--status-report-path",
                        args.status_report_path,
                        "--cache-path",
                        args.cache_path,
                        "--max-records",
                        "1",
                    ]
                )
            self.assertNotEqual(code, 0)
            self.assertIn("Missing local QQQ source data", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
