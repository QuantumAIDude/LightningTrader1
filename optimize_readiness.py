#!/usr/bin/env python3
"""
optimize_readiness.py

Iterative, horizon-priority optimizer for model readiness.
Targets weekend -> daily -> 4min error windows and retrains only failing modes.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone

import pandas as pd

from bootstrap_models_fast import bootstrap_fast_models
from calibration_tracker import update_live_calibration
from config import DATA_DIR, INSTRUMENTS
from data_fetch import update_price_data
from horizon_accuracy_tracker import update_horizon_accuracy
from instrument_onboarding import ensure_runtime_instruments, list_runtime_instruments
from model_trainers import train_all_models
from model_utils import load_models
from readiness_tracker import evaluate_readiness_metrics, update_readiness_state


SELF_OPTIMIZATION_PROMPT = """
You are optimizing a live market advisor for forecast-window reliability, not vanity metrics.
Prioritize horizon usefulness in this strict order: weekend, daily, 4min.
For each instrument, do this every cycle:
1) Refresh resolved accuracy metrics from logged predictions.
2) Evaluate readiness using horizon-priority windows.
3) Retrain only failing horizons; avoid unnecessary full retrains.
4) Re-evaluate and record evidence.
5) Stop only when average discrepancy sits inside the configured horizon windows or cycle budget is exhausted.
Never optimize to random noise; require stable realized error windows.
""".strip()


def _ts_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_instruments(raw: str) -> list[str]:
    val = str(raw or "").strip()
    if not val or val.lower() == "all":
        runtime = list_runtime_instruments()
        return sorted(set(runtime or list(INSTRUMENTS.keys())))
    return sorted(set([s.strip().upper() for s in val.split(",") if s.strip()]))


def _load_cached_prices(instr: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"prices_{instr}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def _ensure_models(instr: str) -> dict:
    models = load_models(include_modes=True, instrument_key=instr)
    missing_core = [k for k in ["base", "forest", "boost", "extra_trees", "regime"] if models.get(k) is None]
    if missing_core:
        bootstrap_fast_models(instr)
        models = load_models(include_modes=True, instrument_key=instr)
    return models


def _modes_from_checks(checks: dict) -> list[str]:
    modes = []
    if not bool(checks.get("weekend_window_ok", False)):
        modes.append("weekend")
    if not bool(checks.get("daily_window_ok", False)):
        modes.extend(["daily_high", "daily_low"])
    if not bool(checks.get("intraday_window_ok", False)):
        modes.append("4min")
    if not bool(checks.get("active_ok", True)):
        modes.append("4min")
    if not modes:
        if (not bool(checks.get("horizon_score_ok", True))) or (not bool(checks.get("folds_ok", True))):
            modes = ["4min", "daily_high", "daily_low", "weekend"]
    return sorted(set(modes))


def _train_target_modes(instr: str, modes: list[str], online: bool) -> dict:
    if not modes:
        return {"trained": False, "modes": []}
    if online:
        df = update_price_data(instr)
    else:
        df = _load_cached_prices(instr)
        if df is None or df.empty:
            df = update_price_data(instr)
    if df is None or df.empty:
        return {"trained": False, "modes": modes, "error": "no price data"}
    train_all_models(
        df,
        instrument_key=instr,
        train_horizons=True,
        modes_filter=modes,
    )
    return {"trained": True, "modes": modes, "rows": int(len(df))}


def _cycle_instrument(instr: str, online: bool, update_state: bool = False) -> dict:
    step = {
        "instrument": instr,
        "ts": _ts_now(),
    }
    try:
        _ensure_models(instr)
    except Exception as e:
        step["error"] = f"model_bootstrap_failed: {e}"
        return step

    try:
        step["calibration_update"] = update_live_calibration(instr, horizon_minutes=4)
    except Exception as e:
        step["calibration_update"] = {"error": str(e)}
    try:
        step["horizon_accuracy_update"] = update_horizon_accuracy(instr)
    except Exception as e:
        step["horizon_accuracy_update"] = {"error": str(e)}

    pre = evaluate_readiness_metrics(instr, online=online)
    step["pre_readiness"] = pre
    if pre.get("error"):
        step["train_action"] = {"trained": False, "modes": [], "error": pre.get("error")}
        return step

    checks = pre.get("checks", {}) if isinstance(pre, dict) else {}
    modes = _modes_from_checks(checks)
    step["train_action"] = _train_target_modes(instr, modes, online=online)

    if step["train_action"].get("trained"):
        try:
            step["post_calibration_update"] = update_live_calibration(instr, horizon_minutes=4)
        except Exception as e:
            step["post_calibration_update"] = {"error": str(e)}
        try:
            step["post_horizon_accuracy_update"] = update_horizon_accuracy(instr)
        except Exception as e:
            step["post_horizon_accuracy_update"] = {"error": str(e)}
        post = evaluate_readiness_metrics(instr, online=online)
    else:
        post = pre
    step["post_readiness"] = post
    if update_state:
        try:
            step["readiness_state"] = update_readiness_state(instr, online=online)
        except Exception as e:
            step["readiness_state"] = {"error": str(e)}
    return step


def main():
    parser = argparse.ArgumentParser(description="Iterative readiness optimizer")
    parser.add_argument("--instrument", default="all", help="Comma list or 'all'")
    parser.add_argument("--cycles", type=int, default=3, help="Optimization cycles")
    parser.add_argument("--online", action="store_true", help="Use online fetch while evaluating/training")
    parser.add_argument("--update-state", action="store_true", help="Also persist readiness_state_<symbol>.json each cycle")
    parser.add_argument("--report", default=os.path.join(DATA_DIR, "readiness_optimization_report.json"), help="Output report path")
    parser.add_argument("--print-prompt", action="store_true", help="Print optimization prompt and continue")
    args = parser.parse_args()

    if args.print_prompt:
        print(SELF_OPTIMIZATION_PROMPT)

    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass

    instruments = _parse_instruments(args.instrument)
    report = {
        "ts": _ts_now(),
        "prompt": SELF_OPTIMIZATION_PROMPT,
        "cycles": [],
        "instruments": instruments,
    }

    max_cycles = max(1, int(args.cycles))
    for i in range(max_cycles):
        cycle_out = {"cycle": i + 1, "ts": _ts_now(), "instruments": {}}
        all_passed = True
        for instr in instruments:
            if instr not in INSTRUMENTS:
                cycle_out["instruments"][instr] = {"error": "unknown instrument"}
                all_passed = False
                continue
            step = _cycle_instrument(instr, online=bool(args.online), update_state=bool(args.update_state))
            cycle_out["instruments"][instr] = step
            post = step.get("post_readiness", {})
            if not bool(post.get("passed", False)):
                all_passed = False
        cycle_out["all_passed"] = bool(all_passed)
        report["cycles"].append(cycle_out)
        if all_passed:
            break

    report["final_cycle"] = len(report["cycles"])
    report["final_all_passed"] = bool(report["cycles"][-1].get("all_passed", False)) if report["cycles"] else False

    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
