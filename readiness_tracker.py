#!/usr/bin/env python3
"""
readiness_tracker.py

Tracks deployment readiness over time and computes pass streaks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    INSTRUMENTS,
    READINESS_MIN_PASS_STREAK_DAYS,
    READINESS_MAX_AGE_HOURS,
    READINESS_BACKTEST_SPLITS,
    READINESS_MIN_DIRECTIONAL_ACC,
    READINESS_MIN_NET_RETURN,
    READINESS_MIN_SHARPE_LIKE,
    READINESS_MIN_ACTIVE_FRACTION,
    READINESS_MIN_FOLDS,
)
from data_fetch import update_price_data
from model_trainers import walk_forward_validate


def _history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"readiness_history_{instrument_key}.jsonl")


def _state_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"readiness_state_{instrument_key}.json")


def _safe_mean(values):
    try:
        vals = [float(v) for v in (values or []) if v is not None and np.isfinite(float(v))]
    except Exception:
        vals = []
    if not vals:
        return None
    return float(np.mean(vals))


def evaluate_readiness_metrics(instrument_key: str, online: bool = False) -> dict:
    if instrument_key not in INSTRUMENTS:
        return {"instrument": instrument_key, "error": "unknown instrument"}
    if online:
        df = update_price_data(instrument_key)
    else:
        p = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
        if not os.path.exists(p):
            return {"instrument": instrument_key, "error": "missing prices csv"}
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
    if df is None or df.empty:
        return {"instrument": instrument_key, "error": "no price data"}
    wf = walk_forward_validate(
        df,
        n_splits=max(3, int(READINESS_BACKTEST_SPLITS)),
        include_transaction_costs=True,
        instrument_key=instrument_key,
    )
    folds = int(wf.get("folds", 0) or 0)
    mean_dir = _safe_mean(wf.get("directional_acc", []))
    mean_net = _safe_mean(wf.get("net_returns", []))
    mean_sharpe = _safe_mean(wf.get("sharpe_like", []))
    mean_active = _safe_mean(wf.get("active_fraction", []))
    thresholds = {
        "min_folds": int(READINESS_MIN_FOLDS),
        "min_directional_acc": float(READINESS_MIN_DIRECTIONAL_ACC),
        "min_net_return": float(READINESS_MIN_NET_RETURN),
        "min_sharpe_like": float(READINESS_MIN_SHARPE_LIKE),
        "min_active_fraction": float(READINESS_MIN_ACTIVE_FRACTION),
    }
    checks = {
        "folds_ok": bool(folds >= thresholds["min_folds"]),
        "directional_ok": bool(mean_dir is not None and mean_dir >= thresholds["min_directional_acc"]),
        "net_ok": bool(mean_net is not None and mean_net >= thresholds["min_net_return"]),
        "sharpe_ok": bool(mean_sharpe is not None and mean_sharpe >= thresholds["min_sharpe_like"]),
        "active_ok": bool(mean_active is not None and mean_active >= thresholds["min_active_fraction"]),
    }
    passed = bool(all(checks.values()))
    return {
        "instrument": instrument_key,
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "passed": passed,
        "checks": checks,
        "metrics": {
            "folds": folds,
            "directional_acc_mean": mean_dir,
            "net_return_mean": mean_net,
            "sharpe_like_mean": mean_sharpe,
            "active_fraction_mean": mean_active,
        },
        "thresholds": thresholds,
    }


def _read_history(instrument_key: str) -> list[dict]:
    path = _history_path(instrument_key)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _write_history(instrument_key: str, rows: list[dict]):
    path = _history_path(instrument_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_state(instrument_key: str, state: dict):
    path = _state_path(instrument_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _compute_pass_streak(rows: list[dict]) -> int:
    if not rows:
        return 0
    dedup = {}
    for r in rows:
        d = str(r.get("date", "")).strip()
        if not d:
            continue
        prev = dedup.get(d)
        if prev is None:
            dedup[d] = r
            continue
        ts_new = pd.to_datetime(r.get("ts"), utc=True, errors="coerce")
        ts_old = pd.to_datetime(prev.get("ts"), utc=True, errors="coerce")
        if pd.isna(ts_old) or (pd.notna(ts_new) and ts_new >= ts_old):
            dedup[d] = r
    ordered = sorted(dedup.values(), key=lambda x: str(x.get("date", "")))
    if not ordered:
        return 0
    streak = 0
    prev_date = None
    for row in reversed(ordered):
        try:
            d = pd.to_datetime(row.get("date")).date()
        except Exception:
            break
        if not bool(row.get("passed", False)):
            break
        if prev_date is None:
            streak = 1
            prev_date = d
            continue
        if (prev_date - d).days == 1:
            streak += 1
            prev_date = d
        else:
            break
    return int(streak)


def _is_state_fresh(state: dict | None) -> bool:
    if not isinstance(state, dict):
        return False
    ts = pd.to_datetime(state.get("last_update"), utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    age_hours = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0
    return bool(age_hours <= float(READINESS_MAX_AGE_HOURS))


def update_readiness_state(instrument_key: str, online: bool = False) -> dict:
    eval_row = evaluate_readiness_metrics(instrument_key, online=online)
    if eval_row.get("error"):
        state = {
            "instrument": instrument_key,
            "live_ready": False,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "pass_streak_days": 0,
            "required_streak_days": int(READINESS_MIN_PASS_STREAK_DAYS),
            "error": eval_row.get("error"),
        }
        _write_state(instrument_key, state)
        return state

    rows = _read_history(instrument_key)
    # Upsert today's entry.
    today = str(eval_row.get("date"))
    replaced = False
    for i, row in enumerate(rows):
        if str(row.get("date")) == today:
            rows[i] = eval_row
            replaced = True
            break
    if not replaced:
        rows.append(eval_row)
    rows = sorted(rows, key=lambda x: (str(x.get("date", "")), str(x.get("ts", ""))))
    # Keep recent history only.
    rows = rows[-120:]
    _write_history(instrument_key, rows)

    streak = _compute_pass_streak(rows)
    required = max(1, int(READINESS_MIN_PASS_STREAK_DAYS))
    live_ready = bool(streak >= required and bool(eval_row.get("passed", False)))
    state = {
        "instrument": instrument_key,
        "last_update": eval_row.get("ts"),
        "last_date": eval_row.get("date"),
        "last_passed": bool(eval_row.get("passed", False)),
        "pass_streak_days": int(streak),
        "required_streak_days": int(required),
        "live_ready": bool(live_ready),
        "fresh": True,
        "thresholds": eval_row.get("thresholds", {}),
        "checks": eval_row.get("checks", {}),
        "metrics": eval_row.get("metrics", {}),
    }
    _write_state(instrument_key, state)
    return state


def load_readiness_state(instrument_key: str) -> dict | None:
    path = _state_path(instrument_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            state = json.load(f)
        state["fresh"] = _is_state_fresh(state)
        if not bool(state["fresh"]):
            state["live_ready"] = False
        return state
    except Exception:
        return None
