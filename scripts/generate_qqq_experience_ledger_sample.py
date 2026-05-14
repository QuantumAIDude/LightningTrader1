from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from config import SLIPPAGE_PCT, TRANSACTION_COST_PER_TRADE
from features import BAR_MINUTES
from model_experience_ledger import (
    format_quality_report,
    format_status_report,
    ledger_quality_summary,
    meta_model_readiness_verdict,
    summarize_ledger,
)


CACHE_SCHEMA_VERSION = 1
DEFAULT_LEDGER_PATH = "data/model_experience_ledger_QQQ_sample.jsonl"
DEFAULT_QUALITY_REPORT_PATH = "reports/model_experience_ledger_quality_QQQ.md"
DEFAULT_STATUS_REPORT_PATH = "reports/model_experience_ledger_status.md"
DEFAULT_CACHE_PATH = "reports/cache/qqq_experience_ledger_sample_cache.json"
DEFAULT_SOURCE_PATH = "data/prices_QQQ.csv"


def _resolve(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def _log(message: str) -> None:
    print(message, flush=True)


def file_signature(path: str | os.PathLike[str]) -> dict:
    p = _resolve(path)
    stat = p.stat()
    return {
        "path": str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def source_date_range(path: str | os.PathLike[str]) -> dict:
    p = _resolve(path)
    try:
        df = pd.read_csv(p, index_col=0, parse_dates=True)
    except Exception:
        return {"date_start": None, "date_end": None, "rows": 0}
    if df.empty:
        return {"date_start": None, "date_end": None, "rows": 0}
    try:
        idx = pd.to_datetime(df.index, utc=True, errors="coerce")
        idx = idx[~idx.isna()]
    except Exception:
        idx = []
    if len(idx) <= 0:
        return {"date_start": None, "date_end": None, "rows": int(len(df))}
    return {
        "date_start": pd.Timestamp(idx[0]).isoformat(),
        "date_end": pd.Timestamp(idx[-1]).isoformat(),
        "rows": int(len(df)),
    }


def build_cache_key(source_path: str, max_records: int) -> dict:
    return {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "symbol": "QQQ",
        "source": file_signature(source_path),
        "source_window": source_date_range(source_path),
        "max_records": int(max_records),
        "bar_minutes": int(BAR_MINUTES),
        "slippage_pct": float(SLIPPAGE_PCT),
        "transaction_cost_per_trade": float(TRANSACTION_COST_PER_TRADE),
    }


def load_cache(path: str | os.PathLike[str]) -> dict:
    p = _resolve(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def write_cache(path: str | os.PathLike[str], payload: dict) -> None:
    p = _resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def cache_is_compatible(cache: dict, cache_key: dict, ledger_path: str, quality_report_path: str, status_report_path: str) -> bool:
    if not isinstance(cache, dict):
        return False
    if int(cache.get("cache_schema_version") or 0) != CACHE_SCHEMA_VERSION:
        return False
    if cache.get("cache_key") != cache_key:
        return False
    for path in (ledger_path, quality_report_path, status_report_path):
        if not _resolve(path).exists():
            return False
    return True


def write_reports(
    *,
    ledger_path: str,
    quality_report_path: str,
    status_report_path: str,
    generated_by: str,
) -> dict:
    quality = ledger_quality_summary(ledger_path)
    quality_text = format_quality_report(quality, generated_by=generated_by, sample=True)
    quality_file = _resolve(quality_report_path)
    quality_file.parent.mkdir(parents=True, exist_ok=True)
    quality_file.write_text(quality_text, encoding="utf-8")

    status = summarize_ledger(ledger_path)
    status_text = format_status_report(status)
    status_text += "\n## Sample Ledger Note\n\n"
    status_text += f"This status reflects the bounded QQQ sample ledger at `{ledger_path}`. "
    status_text += "The production default ledger `data/model_experience_ledger.jsonl` was not written by this script.\n"
    status_text += "\n## Schema Quality\n\n"
    status_text += (
        f"- Valid / warning / invalid: {int(quality.get('schema_valid_record_count') or 0)} / "
        f"{int(quality.get('schema_warning_record_count') or 0)} / "
        f"{int(quality.get('schema_invalid_record_count') or 0)}\n"
    )
    status_text += f"- Verdict: {meta_model_readiness_verdict(quality)}\n"
    status_file = _resolve(status_report_path)
    status_file.parent.mkdir(parents=True, exist_ok=True)
    status_file.write_text(status_text, encoding="utf-8")
    return quality


def generate_sample(args: argparse.Namespace, replay_runner=None) -> dict:
    source_path = str(args.source_path)
    ledger_path = str(args.ledger_path)
    quality_report_path = str(args.quality_report_path)
    status_report_path = str(args.status_report_path)
    cache_path = str(args.cache_path)
    max_records = int(args.max_records)
    source_file = _resolve(source_path)

    _log("checking source data")
    if not source_file.exists():
        raise FileNotFoundError(f"Missing local QQQ source data: {source_path}")

    cache_key = build_cache_key(source_path, max_records)
    cache = load_cache(cache_path)
    _log("checking cache")
    generated_by = (
        "python3 scripts/generate_qqq_experience_ledger_sample.py "
        f"--max-records {max_records} --ledger-path {ledger_path}"
    )
    if not bool(args.force) and cache_is_compatible(cache, cache_key, ledger_path, quality_report_path, status_report_path):
        _log("reusing compatible cache")
        _log("writing reports")
        quality = write_reports(
            ledger_path=ledger_path,
            quality_report_path=quality_report_path,
            status_report_path=status_report_path,
            generated_by=f"{generated_by} (cache reused)",
        )
        _log("done")
        return {"status": "ok", "reused_cache": True, "quality": quality, "cache_key": cache_key}

    _log("generating replay")
    ledger_file = _resolve(ledger_path)
    if ledger_file.exists():
        ledger_file.unlink()
    runner = replay_runner
    if runner is None:
        import autonomy_backtest

        runner = autonomy_backtest.run_stock_autonomy_validation
    df = pd.read_csv(source_file, index_col=0, parse_dates=True)
    out = runner(
        df,
        instrument_key="QQQ",
        n_splits=5,
        holdout_folds=0,
        include_sensitivity=False,
        stock_horizon_blend=True,
        experience_ledger_path=ledger_path,
        max_replay_records=max_records,
    )
    _log("writing ledger")
    if not ledger_file.exists():
        raise RuntimeError(f"Replay completed but ledger was not written: {ledger_path}")
    _log("writing reports")
    quality = write_reports(
        ledger_path=ledger_path,
        quality_report_path=quality_report_path,
        status_report_path=status_report_path,
        generated_by=generated_by,
    )
    cache_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "ledger_path": ledger_path,
        "quality_report_path": quality_report_path,
        "status_report_path": status_report_path,
        "runner_status": out.get("status") if isinstance(out, dict) else None,
        "quality": {
            "records": int(quality.get("total_records") or 0),
            "resolved": int(quality.get("resolved_outcome_count") or 0),
            "unresolved": int(quality.get("unresolved_outcome_count") or 0),
            "verdict": meta_model_readiness_verdict(quality),
        },
    }
    write_cache(cache_path, cache_payload)
    _log("done")
    return {"status": "ok", "reused_cache": False, "quality": quality, "cache_key": cache_key}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a bounded local QQQ model-experience ledger sample.")
    parser.add_argument("--max-records", type=int, default=500)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ledger-path", default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--quality-report-path", default=DEFAULT_QUALITY_REPORT_PATH)
    parser.add_argument("--status-report-path", default=DEFAULT_STATUS_REPORT_PATH)
    parser.add_argument("--cache-path", default=DEFAULT_CACHE_PATH)
    parser.add_argument("--source-path", default=DEFAULT_SOURCE_PATH)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = generate_sample(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 1
    quality = dict(result.get("quality") or {})
    print(
        json.dumps(
            {
                "status": result.get("status"),
                "reused_cache": bool(result.get("reused_cache")),
                "ledger_path": args.ledger_path,
                "cache_path": args.cache_path,
                "records": int(quality.get("total_records") or 0),
                "resolved": int(quality.get("resolved_outcome_count") or 0),
                "unresolved": int(quality.get("unresolved_outcome_count") or 0),
                "valid": int(quality.get("schema_valid_record_count") or 0),
                "warning": int(quality.get("schema_warning_record_count") or 0),
                "invalid": int(quality.get("schema_invalid_record_count") or 0),
                "verdict": meta_model_readiness_verdict(quality),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
