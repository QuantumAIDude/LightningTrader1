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

from config import INSTRUMENTS, DEFAULT_INSTRUMENT, MODEL_DIR, DATA_DIR
from system_checkup import _check_instrument
from data_fetch import update_price_data
from model_trainers import walk_forward_validate


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
        help="Minimum mean directional accuracy on active trades for backtest gate.",
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

        rep = _check_instrument(instr, quick=args.quick, online=args.online)
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

        rep["gate_failures"] = gate_failures

        if args.require_backtest:
            try:
                if args.online:
                    df_bt = update_price_data(instr)
                else:
                    df_bt = _load_cached_prices(instr)
                if df_bt is None or df_bt.empty:
                    gate_failures.append("Backtest gate: missing price data")
                    rep["backtest"] = {"error": "No data"}
                else:
                    wf = walk_forward_validate(
                        df_bt,
                        n_splits=max(3, int(args.backtest_splits)),
                        include_transaction_costs=True,
                        instrument_key=instr,
                    )
                    folds = int(wf.get("folds", 0) or 0)
                    mean_dir = _safe_mean(wf.get("directional_acc", []))
                    mean_net = _safe_mean(wf.get("net_returns", []))
                    mean_sharpe = _safe_mean(wf.get("sharpe_like", []))
                    mean_active = _safe_mean(wf.get("active_fraction", []))
                    rep["backtest"] = {
                        "folds": folds,
                        "directional_acc_mean": mean_dir,
                        "net_return_mean": mean_net,
                        "sharpe_like_mean": mean_sharpe,
                        "active_fraction_mean": mean_active,
                    }
                    if folds < 3:
                        gate_failures.append("Backtest gate: insufficient folds")
                    if mean_dir is None or mean_dir < float(args.min_directional_acc):
                        gate_failures.append(
                            f"Backtest gate: directional_acc_mean {mean_dir} < {float(args.min_directional_acc):.3f}"
                        )
                    if mean_net is None or mean_net < float(args.min_net_return):
                        gate_failures.append(
                            f"Backtest gate: net_return_mean {mean_net} < {float(args.min_net_return):.6f}"
                        )
                    if mean_sharpe is None or mean_sharpe < float(args.min_sharpe_like):
                        gate_failures.append(
                            f"Backtest gate: sharpe_like_mean {mean_sharpe} < {float(args.min_sharpe_like):.4f}"
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
