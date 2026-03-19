#!/usr/bin/env python3
"""
deployment_gate.py
Hard deployment gate for LightningTrader4000.

Exits 0 only when all selected instruments pass operational checks.
"""

import argparse
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    INSTRUMENTS,
    DEFAULT_INSTRUMENT,
    MODEL_DIR,
    DATA_DIR,
    READINESS_MIN_HORIZON_SCORE,
    READINESS_MIN_INTRADAY_WINDOW_HIT_RATE,
    READINESS_MIN_DAILY_WINDOW_HIT_RATE,
    READINESS_MIN_WEEKEND_WINDOW_HIT_RATE,
)
from calibration_tracker import update_live_calibration
from horizon_accuracy_tracker import update_horizon_accuracy
from readiness_tracker import update_readiness_state
from system_checkup import _check_instrument
from data_fetch import update_price_data


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _model_dir_size_bytes() -> int:
    total = 0
    try:
        for name in os.listdir(MODEL_DIR):
            if not str(name).endswith(".pkl"):
                continue
            p = os.path.join(MODEL_DIR, name)
            if os.path.isfile(p):
                total += int(os.path.getsize(p))
    except Exception:
        return 0
    return total


def _parse_instruments(raw: str) -> list[str]:
    val = str(raw or "").strip()
    if not val or val.lower() == "all":
        return list(INSTRUMENTS.keys())
    return [s.strip() for s in val.split(",") if s.strip()]


def _load_cached_prices(instr: str):
    path = os.path.join(DATA_DIR, f"prices_{instr}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
        return df
    except Exception:
        return None


def _safe_mean(values):
    try:
        vals = [float(v) for v in (values or []) if v is not None and np.isfinite(float(v))]
    except Exception:
        vals = []
    if not vals:
        return None
    return float(np.mean(vals))


def main():
    parser = argparse.ArgumentParser(description="Deployment gate for LightningTrader4000")
    parser.add_argument(
        "--instrument",
        default="all",
        help="Comma-separated instruments or 'all' (default).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Skip daily/weekend checks (faster).",
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="Use live fetch during checks.",
    )
    parser.add_argument(
        "--allow-warn",
        action="store_true",
        help="Allow warning status to pass the gate.",
    )
    parser.add_argument(
        "--max-model-dir-gb",
        type=float,
        default=6.0,
        help="Fail gate if models directory exceeds this size (GB).",
    )
    parser.add_argument(
        "--max-4min-latency-s",
        type=float,
        default=20.0,
        help="Fail gate if 4min prediction latency exceeds this threshold.",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Optional JSON report path.",
    )
    parser.add_argument(
        "--require-backtest",
        action="store_true",
        help="Run walk-forward backtest checks and fail if quality thresholds are not met.",
    )
    parser.add_argument(
        "--backtest-splits",
        type=int,
        default=8,
        help="Walk-forward splits used when --require-backtest is enabled.",
    )
    parser.add_argument(
        "--min-directional-acc",
        type=float,
        default=0.52,
        help="Legacy directional-accuracy diagnostic threshold. Price-window metrics are now the primary gate.",
    )
    parser.add_argument(
        "--max-intraday-window-ratio",
        type=float,
        default=1.0,
        help="Maximum mean 4-minute price-window ratio for backtest gate.",
    )
    parser.add_argument(
        "--min-intraday-window-hit-rate",
        type=float,
        default=float(READINESS_MIN_INTRADAY_WINDOW_HIT_RATE),
        help="Minimum 4-minute fraction of forecasts landing inside the configured price window.",
    )
    parser.add_argument(
        "--max-daily-window-ratio",
        type=float,
        default=1.0,
        help="Maximum mean daily price-window ratio for backtest gate.",
    )
    parser.add_argument(
        "--min-daily-window-hit-rate",
        type=float,
        default=float(READINESS_MIN_DAILY_WINDOW_HIT_RATE),
        help="Minimum daily fraction of forecasts landing inside the configured price window.",
    )
    parser.add_argument(
        "--max-weekend-window-ratio",
        type=float,
        default=1.0,
        help="Maximum mean weekend price-window ratio for backtest gate.",
    )
    parser.add_argument(
        "--min-weekend-window-hit-rate",
        type=float,
        default=float(READINESS_MIN_WEEKEND_WINDOW_HIT_RATE),
        help="Minimum weekend fraction of forecasts landing inside the configured price window.",
    )
    parser.add_argument(
        "--min-horizon-score",
        type=float,
        default=float(READINESS_MIN_HORIZON_SCORE),
        help="Minimum weighted horizon score after combining 4-minute, daily, and weekend window accuracy.",
    )
    parser.add_argument(
        "--min-net-return",
        type=float,
        default=0.0,
        help="Minimum mean net return for backtest gate.",
    )
    parser.add_argument(
        "--min-sharpe-like",
        type=float,
        default=0.0,
        help="Minimum mean Sharpe-like metric for backtest gate.",
    )
    parser.add_argument(
        "--min-active-fraction",
        type=float,
        default=0.03,
        help="Minimum active-trade fraction to avoid degenerate always-flat strategies.",
    )
    args = parser.parse_args()

    instruments = _parse_instruments(args.instrument)
    out = {
        "ts": _ts_now(),
        "status": "ok",
        "limits": {
            "max_model_dir_gb": float(args.max_model_dir_gb),
            "max_4min_latency_s": float(args.max_4min_latency_s),
            "allow_warn": bool(args.allow_warn),
            "max_intraday_window_ratio": float(args.max_intraday_window_ratio),
            "min_intraday_window_hit_rate": float(args.min_intraday_window_hit_rate),
            "max_daily_window_ratio": float(args.max_daily_window_ratio),
            "min_daily_window_hit_rate": float(args.min_daily_window_hit_rate),
            "max_weekend_window_ratio": float(args.max_weekend_window_ratio),
            "min_weekend_window_hit_rate": float(args.min_weekend_window_hit_rate),
            "min_horizon_score": float(args.min_horizon_score),
        },
        "instruments": {},
        "global": {},
    }

    # Global size gate
    model_dir_bytes = _model_dir_size_bytes()
    model_dir_gb = model_dir_bytes / (1024.0 ** 3)
    out["global"]["model_dir_gb"] = round(float(model_dir_gb), 3)
    if model_dir_gb > float(args.max_model_dir_gb):
        out["status"] = "fail"
        out["global"]["model_dir_gate"] = (
            f"FAIL: model dir {model_dir_gb:.2f}GB > limit {args.max_model_dir_gb:.2f}GB"
        )
    else:
        out["global"]["model_dir_gate"] = "pass"

    for instr in instruments:
        if instr not in INSTRUMENTS:
            out["instruments"][instr] = {"status": "fail", "gate_failures": ["Instrument not defined"]}
            out["status"] = "fail"
            continue

        readiness_state = None
        refresh = {}
        try:
            refresh["calibration_update"] = update_live_calibration(instr, horizon_minutes=4)
        except Exception as e:
            refresh["calibration_update"] = {"error": str(e)}
        try:
            refresh["horizon_accuracy_update"] = update_horizon_accuracy(instr)
        except Exception as e:
            refresh["horizon_accuracy_update"] = {"error": str(e)}
        try:
            readiness_state = update_readiness_state(instr, online=args.online)
            refresh["readiness_state"] = readiness_state
        except Exception as e:
            refresh["readiness_state"] = {"error": str(e)}

        rep = _check_instrument(instr, quick=args.quick, online=args.online)
        rep["readiness_refresh"] = refresh
        gate_failures = []

        status = str(rep.get("status", "fail")).lower()
        if status == "fail":
            gate_failures.append("System check status is fail")
        elif status == "warn" and not args.allow_warn:
            gate_failures.append("System check status is warn (not allowed)")

        pred = rep.get("predictions", {}).get("4min", {})
        if isinstance(pred, dict) and pred.get("error"):
            gate_failures.append(f"4min prediction error: {pred.get('error')}")

        latency = rep.get("predictions", {}).get("4min_latency_s")
        try:
            if latency is not None and float(latency) > float(args.max_4min_latency_s):
                gate_failures.append(
                    f"4min latency {float(latency):.3f}s exceeds {float(args.max_4min_latency_s):.3f}s"
                )
        except Exception:
            pass

        missing = rep.get("models", {}).get("missing", [])
        if missing:
            gate_failures.append(f"Missing models: {missing}")
        try:
            diag = rep.get("predictions", {}).get("4min", {}).get("diagnostics", {})
            critical_stale = bool(diag.get("critical_stale", False))
            readiness_locked = bool(diag.get("readiness_locked", False))
            if critical_stale:
                gate_failures.append("Critical stale-data no-trade guard triggered")
            if readiness_locked:
                gate_failures.append("Readiness gate locked (pass streak not yet satisfied)")
            div = diag.get("diversity")
            if isinstance(div, dict) and bool(div.get("degenerate", False)) and not critical_stale:
                gate_failures.append("Degenerate 4min model spread (near-identical outputs)")
        except Exception:
            pass

        readiness_eval = {}
        if isinstance(readiness_state, dict):
            readiness_eval = readiness_state.get("evaluation", {}) if isinstance(readiness_state.get("evaluation"), dict) else {}
            rep["readiness_state"] = readiness_state
        if readiness_eval.get("error"):
            gate_failures.append(f"Readiness evaluation error: {readiness_eval.get('error')}")

        rep["gate_failures"] = gate_failures

        if args.require_backtest:
            try:
                metrics = readiness_eval.get("metrics", {}) if isinstance(readiness_eval, dict) else {}
                checks = readiness_eval.get("checks", {}) if isinstance(readiness_eval, dict) else {}
                h_bt = readiness_eval.get("horizon_backtest", {}) if isinstance(readiness_eval, dict) else {}
                if not metrics:
                    gate_failures.append("Backtest gate: missing readiness evaluation metrics")
                    rep["backtest"] = {"error": "No readiness evaluation metrics"}
                else:
                    folds = int(metrics.get("folds", 0) or 0)
                    mean_active = metrics.get("active_fraction_mean")
                    intr_mae = metrics.get("intraday_avg_discrepancy_dollars", metrics.get("intraday_mae_dollars"))
                    intr_ratio = metrics.get("intraday_window_ratio")
                    intr_hit = metrics.get("intraday_window_hit_rate")
                    intr_target = metrics.get("intraday_target_window_dollars")
                    day_mae = metrics.get("daily_avg_discrepancy_dollars", metrics.get("daily_mae_dollars"))
                    daily_ratio = metrics.get("daily_window_ratio")
                    daily_hit = metrics.get("daily_window_hit_rate")
                    daily_target = metrics.get("daily_target_window_dollars")
                    week_mae = metrics.get("weekend_avg_discrepancy_dollars", metrics.get("weekend_mae_dollars"))
                    week_ratio = metrics.get("weekend_window_ratio")
                    week_hit = metrics.get("weekend_window_hit_rate")
                    week_target = metrics.get("weekend_target_window_dollars")
                    rep["backtest"] = {
                        "folds": folds,
                        "active_fraction_mean": mean_active,
                        "intraday_avg_discrepancy_dollars": intr_mae,
                        "intraday_target_window_dollars": intr_target,
                        "intraday_window_ratio_mean": intr_ratio,
                        "intraday_window_hit_rate_mean": intr_hit,
                        "daily_avg_discrepancy_dollars": day_mae,
                        "daily_target_window_dollars": daily_target,
                        "daily_window_ratio_mean": daily_ratio,
                        "daily_window_hit_rate_mean": daily_hit,
                        "weekend_avg_discrepancy_dollars": week_mae,
                        "weekend_target_window_dollars": week_target,
                        "weekend_window_ratio_mean": week_ratio,
                        "weekend_window_hit_rate_mean": week_hit,
                        "horizon_weighted_score": metrics.get("horizon_weighted_score"),
                        "checks": checks,
                        "horizon_backtest": h_bt,
                    }
                    if folds < 3:
                        gate_failures.append("Backtest gate: insufficient folds")
                    if intr_ratio is None or float(intr_ratio) > float(args.max_intraday_window_ratio):
                        gate_failures.append(
                            f"Backtest gate: intraday_window_ratio {intr_ratio} > {float(args.max_intraday_window_ratio):.3f}"
                        )
                    if intr_hit is None or float(intr_hit) < float(args.min_intraday_window_hit_rate):
                        gate_failures.append(
                            f"Backtest gate: intraday_window_hit_rate {intr_hit} < {float(args.min_intraday_window_hit_rate):.3f}"
                        )
                    if daily_ratio is None or float(daily_ratio) > float(args.max_daily_window_ratio):
                        gate_failures.append(
                            f"Backtest gate: daily_window_ratio {daily_ratio} > {float(args.max_daily_window_ratio):.3f}"
                        )
                    if daily_hit is None or float(daily_hit) < float(args.min_daily_window_hit_rate):
                        gate_failures.append(
                            f"Backtest gate: daily_window_hit_rate {daily_hit} < {float(args.min_daily_window_hit_rate):.3f}"
                        )
                    if week_ratio is None or float(week_ratio) > float(args.max_weekend_window_ratio):
                        gate_failures.append(
                            f"Backtest gate: weekend_window_ratio {week_ratio} > {float(args.max_weekend_window_ratio):.3f}"
                        )
                    if week_hit is None or float(week_hit) < float(args.min_weekend_window_hit_rate):
                        gate_failures.append(
                            f"Backtest gate: weekend_window_hit_rate {week_hit} < {float(args.min_weekend_window_hit_rate):.3f}"
                        )
                    if intr_mae is None or intr_target is None or float(intr_mae) > float(intr_target):
                        gate_failures.append(
                            f"Backtest gate: intraday_avg_discrepancy {intr_mae} > target {intr_target}"
                        )
                    if day_mae is None or daily_target is None or float(day_mae) > float(daily_target):
                        gate_failures.append(
                            f"Backtest gate: daily_avg_discrepancy {day_mae} > target {daily_target}"
                        )
                    if week_mae is None or week_target is None or float(week_mae) > float(week_target):
                        gate_failures.append(
                            f"Backtest gate: weekend_avg_discrepancy {week_mae} > target {week_target}"
                        )
                    if mean_active is None or mean_active < float(args.min_active_fraction):
                        gate_failures.append(
                            f"Backtest gate: active_fraction_mean {mean_active} < {float(args.min_active_fraction):.3f}"
                        )
            except Exception as e:
                gate_failures.append(f"Backtest gate: exception {e}")

        out["instruments"][instr] = rep
        if gate_failures:
            out["status"] = "fail"

    if args.report:
        with open(args.report, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out["status"] == "ok" else 2)


if __name__ == "__main__":
    main()
