import os
import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from config import DATA_DIR, INTERVAL, CALIBRATION_MAX_AGE_DAYS, LIVE_CALIBRATION, CALIBRATION_HALF_LIFE_DAYS, DRIFT_ACC_THRESHOLD, DRIFT_MIN_SAMPLES

logger = logging.getLogger(__name__)

def _interval_minutes(interval: str) -> int:
    interval = str(interval).strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    try:
        return int(interval)
    except Exception:
        return 1

def _load_predictions_log(instrument_key: str):
    path = os.path.join(DATA_DIR, f"predictions_{instrument_key}.csv")
    if not os.path.exists(path):
        return None, path
    try:
        # Allow schema drift (new columns appended over time).
        # We parse with csv to avoid hard failures on column count changes.
        import csv
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return None, path
            for row in reader:
                if not row:
                    continue
                if len(row) < len(header):
                    row = row + [None] * (len(header) - len(row))
                elif len(row) > len(header):
                    # Ignore extra columns (future schema extensions)
                    row = row[:len(header)]
                rows.append(row)
        df = pd.DataFrame(rows, columns=header)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        return df, path
    except Exception:
        logger.exception("Failed to read predictions log")
        return None, path

def _load_price_history(instrument_key: str):
    path = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
        return df
    except Exception:
        logger.exception("Failed to read price history")
        return None

def _calibration_state_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"calibration_live_{instrument_key}.json")

def _load_state(instrument_key: str):
    path = _calibration_state_path(instrument_key)
    if not os.path.exists(path):
        return {
            "last_processed": None,
            "last_update": None,
            "by_model": {},
            "by_model_stats": {},
            "global": {},
            "recent_outcomes": [],
            "recent_abs_errors": []
        }
    try:
        with open(path, "r") as f:
            state = json.load(f)
        state.setdefault("recent_outcomes", [])
        state.setdefault("recent_abs_errors", [])
        return state
    except Exception:
        logger.exception("Failed to read calibration state")
        return {
            "last_processed": None,
            "last_update": None,
            "by_model": {},
            "by_model_stats": {},
            "global": {},
            "recent_outcomes": [],
            "recent_abs_errors": []
        }

def _save_state(instrument_key: str, state: dict):
    path = _calibration_state_path(instrument_key)
    tmp_path = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp_path, "w") as f:
            json.dump(state, f)
        os.replace(tmp_path, path)
    except Exception:
        logger.exception("Failed to write calibration state")
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass

def _bucket_from_sig(sig):
    if sig in ("Weak", "Medium", "Strong"):
        return sig
    return "Medium"

def _apply_decay(state: dict, now: datetime):
    try:
        last_update = state.get("last_update")
        if not last_update:
            state["last_update"] = now.isoformat()
            return
        last_dt = pd.to_datetime(last_update, utc=True, errors="coerce")
        if pd.isna(last_dt):
            state["last_update"] = now.isoformat()
            return
        delta_days = max(0.0, (now - last_dt).total_seconds() / 86400.0)
        if delta_days <= 0.0:
            return
        # exponential decay with half-life
        try:
            hl = float(CALIBRATION_HALF_LIFE_DAYS)
        except Exception:
            hl = 7.0
        hl = max(1.0, hl)
        decay = float(np.exp(-np.log(2.0) * delta_days / hl))
        # decay bucket stats
        for m, buckets in state.get("by_model", {}).items():
            for b, v in buckets.items():
                v["wins"] = v.get("wins", 0.0) * decay
                v["total"] = v.get("total", 0.0) * decay
        # decay model stats
        for m, v in state.get("by_model_stats", {}).items():
            v["wins"] = v.get("wins", 0.0) * decay
            v["total"] = v.get("total", 0.0) * decay
            v["sum_error"] = v.get("sum_error", 0.0) * decay
            v["sum_abs_error"] = v.get("sum_abs_error", 0.0) * decay
            v["sum_pred"] = v.get("sum_pred", 0.0) * decay
        # decay global stats
        g = state.get("global", {})
        if g:
            g["wins"] = g.get("wins", 0.0) * decay
            g["total"] = g.get("total", 0.0) * decay
        state["last_update"] = now.isoformat()
    except Exception:
        logger.exception("Failed to apply decay")

def update_live_calibration(instrument_key: str, horizon_minutes: int = 4):
    if not LIVE_CALIBRATION:
        return {"updated": 0, "skipped": 0}
    preds, _ = _load_predictions_log(instrument_key)
    if preds is None or preds.empty:
        return {"updated": 0, "skipped": 0}
    prices = _load_price_history(instrument_key)
    if prices is None or prices.empty or "close" not in prices.columns:
        return {"updated": 0, "skipped": 0}

    state = _load_state(instrument_key)
    last_proc = state.get("last_processed")
    if last_proc:
        try:
            last_proc = pd.to_datetime(last_proc, utc=True)
        except Exception:
            last_proc = None

    # Only consider recent predictions (avoid stale calibration drift)
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=CALIBRATION_MAX_AGE_DAYS)

    preds = preds.dropna(subset=["timestamp"])
    preds = preds[preds["timestamp"] >= cutoff]
    if last_proc is not None:
        preds = preds[preds["timestamp"] > last_proc]

    if preds.empty:
        return {"updated": 0, "skipped": 0}

    model_keys = ["base", "forest", "boost", "extra_trees", "meta", "consensus"]
    updated = 0
    skipped = 0
    processed_timestamps = []
    _apply_decay(state, now)
    state.setdefault("global", {}).setdefault("wins", 0.0)
    state.setdefault("global", {}).setdefault("total", 0.0)
    state.setdefault("by_model_stats", {})
    for _, row in preds.iterrows():
        ts = row.get("timestamp")
        if pd.isna(ts):
            skipped += 1
            continue
        target_ts = ts + pd.Timedelta(minutes=horizon_minutes)
        # find closest price at or after target_ts
        try:
            idx_pos = prices.index.searchsorted(target_ts)
            if idx_pos >= len(prices.index):
                skipped += 1
                continue
            actual_price = float(prices["close"].iloc[idx_pos])
            last_price = float(row.get("last_price", actual_price))
            if last_price == 0 or not pd.notna(actual_price):
                skipped += 1
                continue
            actual_ret = (actual_price / last_price) - 1.0
        except Exception:
            skipped += 1
            continue

        row_used = False
        row_preds = []
        for k in model_keys:
            pred_ret = row.get(f"{k}_pred_ret")
            sig = row.get(f"{k}_sig_bucket")
            if pred_ret is None or pd.isna(pred_ret):
                continue
            try:
                pred_ret = float(pred_ret)
            except Exception:
                continue
            correct = (pred_ret >= 0) == (actual_ret >= 0)
            row_preds.append(pred_ret)
            bucket = _bucket_from_sig(sig)
            state.setdefault("by_model", {}).setdefault(k, {}).setdefault(bucket, {"wins": 0, "total": 0})
            state["by_model"][k][bucket]["wins"] += int(correct)
            state["by_model"][k][bucket]["total"] += 1
            # model-level stats
            state["by_model_stats"].setdefault(k, {"wins": 0, "total": 0, "sum_error": 0.0, "sum_abs_error": 0.0, "sum_pred": 0.0})
            state["by_model_stats"][k]["wins"] += int(correct)
            state["by_model_stats"][k]["total"] += 1
            err = float(actual_ret - pred_ret)
            state["by_model_stats"][k]["sum_error"] += err
            state["by_model_stats"][k]["sum_abs_error"] += abs(err)
            state["by_model_stats"][k]["sum_pred"] += float(pred_ret)
            # global stats
            state["global"]["wins"] += int(correct)
            state["global"]["total"] += 1
            row_used = True
        if row_used:
            # Keep a short, row-level correctness history for faster drift reaction.
            # Prefer consensus if present; otherwise use mean model sign.
            row_correct = None
            try:
                c_pred = row.get("consensus_pred_ret")
                if c_pred is not None and not pd.isna(c_pred):
                    c_pred = float(c_pred)
                    row_correct = int((c_pred >= 0) == (actual_ret >= 0))
            except Exception:
                row_correct = None
            if row_correct is None and row_preds:
                try:
                    row_correct = int((float(np.mean(row_preds)) >= 0) == (actual_ret >= 0))
                except Exception:
                    row_correct = None
            if row_correct is not None:
                recent = state.setdefault("recent_outcomes", [])
                recent.append(int(row_correct))
                if len(recent) > 500:
                    state["recent_outcomes"] = recent[-500:]
            try:
                c_pred = row.get("consensus_pred_ret")
                if c_pred is not None and not pd.isna(c_pred):
                    c_pred = float(c_pred)
                    abs_err = abs(float(actual_ret - c_pred))
                    if np.isfinite(abs_err):
                        errs = state.setdefault("recent_abs_errors", [])
                        errs.append(float(abs_err))
                        if len(errs) > 1000:
                            state["recent_abs_errors"] = errs[-1000:]
            except Exception:
                pass
            updated += 1
            processed_timestamps.append(ts)
        else:
            skipped += 1

    # update last_processed
    try:
        if processed_timestamps:
            latest_ts = pd.to_datetime(max(processed_timestamps), utc=True, errors="coerce")
            if pd.notna(latest_ts):
                state["last_processed"] = latest_ts.isoformat()
    except Exception:
        logger.debug("Failed to advance last_processed", exc_info=True)

    # drift detection
    try:
        total = float(state["global"].get("total", 0.0))
        wins = float(state["global"].get("wins", 0.0))
        acc = wins / total if total > 0 else 1.0
        state["drift_score"] = float(acc)
        global_drift = bool(total >= DRIFT_MIN_SAMPLES and acc < DRIFT_ACC_THRESHOLD)
        recent_outcomes = [int(x) for x in state.get("recent_outcomes", []) if str(x) in ("0", "1")]
        recent_min = max(24, int(DRIFT_MIN_SAMPLES // 3))
        recent_window = max(40, int(DRIFT_MIN_SAMPLES))
        recent_slice = recent_outcomes[-recent_window:] if recent_outcomes else []
        recent_total = len(recent_slice)
        recent_acc = (float(np.mean(recent_slice)) if recent_total > 0 else 1.0)
        # Slightly stricter threshold for recent window so we react earlier to sharp degradation.
        recent_thresh = min(0.55, float(DRIFT_ACC_THRESHOLD) + 0.05)
        recent_drift = bool(recent_total >= recent_min and recent_acc < recent_thresh)
        state["recent_acc"] = float(recent_acc)
        state["recent_total"] = int(recent_total)
        state["recent_drift_flag"] = bool(recent_drift)
        state["drift_flag"] = bool(global_drift or recent_drift)
    except Exception:
        state["drift_flag"] = False
        state["recent_drift_flag"] = False
    _save_state(instrument_key, state)
    return {"updated": updated, "skipped": skipped}
