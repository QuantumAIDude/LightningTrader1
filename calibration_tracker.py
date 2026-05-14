import os
import json
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
import numpy as np

from config import (
    DATA_DIR,
    INTERVAL,
    CALIBRATION_MAX_AGE_DAYS,
    CALIBRATION_EVIDENCE_MAX_AGE_DAYS,
    LIVE_CALIBRATION,
    CALIBRATION_HALF_LIFE_DAYS,
    DRIFT_ACC_THRESHOLD,
    DRIFT_MIN_SAMPLES,
)
from accuracy_windows import price_window_threshold_dollars

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

def _calibration_history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"calibration_history_{instrument_key}.jsonl")

def _load_state(instrument_key: str):
    path = _calibration_state_path(instrument_key)
    if not os.path.exists(path):
        return {
            "last_processed": None,
            "last_update": None,
            "last_check": None,
            "last_resolved_target_ts": None,
            "by_model": {},
            "by_model_stats": {},
            "global": {},
            "recent_outcomes": [],
            "recent_abs_errors": [],
            "recent_price_abs_errors": [],
            "recent_price_window_ratios": [],
            "recent_price_window_hits": [],
            "recent_price_window_thresholds": [],
            "pending_unresolved_rows": 0,
            "pending_unresolved_oldest_prediction_ts": None,
            "pending_unresolved_newest_prediction_ts": None,
            "pending_unresolved_oldest_target_ts": None,
            "pending_unresolved_newest_target_ts": None,
            "skip_breakdown": {},
        }
    try:
        with open(path, "r") as f:
            state = json.load(f)
        state.setdefault("last_check", None)
        state.setdefault("last_resolved_target_ts", None)
        state.setdefault("recent_outcomes", [])
        state.setdefault("recent_abs_errors", [])
        state.setdefault("recent_price_abs_errors", [])
        state.setdefault("recent_price_window_ratios", [])
        state.setdefault("recent_price_window_hits", [])
        state.setdefault("recent_price_window_thresholds", [])
        state.setdefault("pending_unresolved_rows", 0)
        state.setdefault("pending_unresolved_oldest_prediction_ts", None)
        state.setdefault("pending_unresolved_newest_prediction_ts", None)
        state.setdefault("pending_unresolved_oldest_target_ts", None)
        state.setdefault("pending_unresolved_newest_target_ts", None)
        state.setdefault("skip_breakdown", {})
        return state
    except Exception:
        logger.exception("Failed to read calibration state")
        return {
            "last_processed": None,
            "last_update": None,
            "last_check": None,
            "last_resolved_target_ts": None,
            "by_model": {},
            "by_model_stats": {},
            "global": {},
            "recent_outcomes": [],
            "recent_abs_errors": [],
            "recent_price_abs_errors": [],
            "recent_price_window_ratios": [],
            "recent_price_window_hits": [],
            "recent_price_window_thresholds": [],
            "pending_unresolved_rows": 0,
            "pending_unresolved_oldest_prediction_ts": None,
            "pending_unresolved_newest_prediction_ts": None,
            "pending_unresolved_oldest_target_ts": None,
            "pending_unresolved_newest_target_ts": None,
            "skip_breakdown": {},
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

def _append_history_row(instrument_key: str, row: dict):
    path = _calibration_history_path(instrument_key)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    except Exception:
        logger.exception("Failed to append calibration history")


def _coerce_ts(value):
    try:
        ts = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if pd.isna(ts):
        return None
    return ts


def _tail_streak(values, success: int = 1) -> int:
    streak = 0
    target = int(success)
    for raw in reversed(list(values or [])):
        try:
            value = int(raw)
        except Exception:
            break
        if value != target:
            break
        streak += 1
    return int(streak)


def _finalize_state(
    instrument_key: str,
    state: dict,
    now: datetime,
    *,
    updated: int,
    skipped: int,
    pending_prediction_ts: list | None = None,
    pending_target_ts: list | None = None,
    skip_breakdown: dict | None = None,
) -> dict:
    state = dict(state or {})
    state["last_check"] = now.isoformat()
    state["resolved_rows_last_run"] = int(updated)
    state["skipped_rows_last_run"] = int(skipped)
    state["skip_breakdown"] = {
        "flat": int((skip_breakdown or {}).get("flat", 0) or 0),
        "invalid": int((skip_breakdown or {}).get("invalid", 0) or 0),
        "unresolved": int((skip_breakdown or {}).get("unresolved", 0) or 0),
    }
    pred_ts = sorted(ts for ts in (_coerce_ts(v) for v in (pending_prediction_ts or [])) if ts is not None)
    tgt_ts = sorted(ts for ts in (_coerce_ts(v) for v in (pending_target_ts or [])) if ts is not None)
    state["pending_unresolved_rows"] = int(len(pred_ts))
    state["pending_unresolved_oldest_prediction_ts"] = pred_ts[0].isoformat() if pred_ts else None
    state["pending_unresolved_newest_prediction_ts"] = pred_ts[-1].isoformat() if pred_ts else None
    state["pending_unresolved_oldest_target_ts"] = tgt_ts[0].isoformat() if tgt_ts else None
    state["pending_unresolved_newest_target_ts"] = tgt_ts[-1].isoformat() if tgt_ts else None

    try:
        total = float(state.get("global", {}).get("total", 0.0))
        wins = float(state.get("global", {}).get("wins", 0.0))
        acc = wins / total if total > 0 else 1.0
        state["drift_score"] = float(acc)
        global_drift = bool(total >= DRIFT_MIN_SAMPLES and acc < DRIFT_ACC_THRESHOLD)
        recent_outcomes = [int(x) for x in state.get("recent_outcomes", []) if str(x) in ("0", "1")]
        recent_min = max(24, int(DRIFT_MIN_SAMPLES // 3))
        recent_window = max(40, int(DRIFT_MIN_SAMPLES))
        recent_slice = recent_outcomes[-recent_window:] if recent_outcomes else []
        recent_total = len(recent_slice)
        recent_acc = (float(np.mean(recent_slice)) if recent_total > 0 else 1.0)
        recent_thresh = min(0.55, float(DRIFT_ACC_THRESHOLD) + 0.05)
        recent_drift = bool(recent_total >= recent_min and recent_acc < recent_thresh)
        recent_hits = [int(x) for x in state.get("recent_price_window_hits", []) if str(x) in ("0", "1")]
        recent_hit_slice = recent_hits[-recent_window:] if recent_hits else []
        recent_hit_total = len(recent_hit_slice)
        recent_hit_rate = float(np.mean(recent_hit_slice)) if recent_hit_total > 0 else 1.0
        price_drift = bool(recent_hit_total >= recent_min and recent_hit_rate < 0.52)
        state["recent_acc"] = float(recent_acc)
        state["recent_total"] = int(recent_total)
        state["recent_drift_flag"] = bool(recent_drift)
        state["recent_price_window_hit_rate"] = float(recent_hit_rate)
        state["recent_price_window_total"] = int(recent_hit_total)
        state["recent_price_drift_flag"] = bool(price_drift)
        state["recent_recovery_streak"] = _tail_streak(recent_slice, success=1)
        state["recent_price_recovery_streak"] = _tail_streak(recent_hit_slice, success=1)
        state["recent_failure_streak"] = _tail_streak(recent_slice, success=0)
        state["recent_price_failure_streak"] = _tail_streak(recent_hit_slice, success=0)
        state["drift_components"] = {
            "global": bool(global_drift),
            "recent_accuracy": bool(recent_drift),
            "recent_price_window": bool(price_drift),
        }
        state["drift_flag"] = bool(global_drift or recent_drift or price_drift)
    except Exception:
        state["drift_flag"] = False
        state["recent_drift_flag"] = False
        state["recent_price_drift_flag"] = False
        state["drift_components"] = {
            "global": False,
            "recent_accuracy": False,
            "recent_price_window": False,
        }

    evidence_anchor = _coerce_ts(state.get("last_resolved_target_ts")) or _coerce_ts(state.get("last_processed"))
    evidence_age_hours = None
    if evidence_anchor is not None:
        try:
            evidence_age_hours = max(0.0, float((now - evidence_anchor.to_pydatetime()).total_seconds()) / 3600.0)
        except Exception:
            evidence_age_hours = None
    state["evidence_age_hours"] = round(float(evidence_age_hours), 2) if evidence_age_hours is not None else None
    state["evidence_stale"] = bool(
        evidence_age_hours is None or evidence_age_hours > (24.0 * float(CALIBRATION_EVIDENCE_MAX_AGE_DAYS))
    )
    state["recovery_ready"] = bool(
        not bool(state.get("recent_drift_flag"))
        and not bool(state.get("recent_price_drift_flag"))
        and int(state.get("recent_recovery_streak") or 0) >= 6
        and (
            int(state.get("recent_price_window_total") or 0) < max(24, int(DRIFT_MIN_SAMPLES // 3))
            or int(state.get("recent_price_recovery_streak") or 0) >= 4
        )
    )
    _save_state(instrument_key, state)
    _append_history_row(
        instrument_key,
        {
            "ts": now.isoformat(),
            "instrument": str(instrument_key or "").upper(),
            "updated": int(updated),
            "skipped": int(skipped),
            "last_processed": state.get("last_processed"),
            "last_resolved_target_ts": state.get("last_resolved_target_ts"),
            "evidence_age_hours": state.get("evidence_age_hours"),
            "evidence_stale": bool(state.get("evidence_stale")),
            "drift_flag": bool(state.get("drift_flag")),
            "recent_drift_flag": bool(state.get("recent_drift_flag")),
            "recent_price_drift_flag": bool(state.get("recent_price_drift_flag")),
            "drift_score": state.get("drift_score"),
            "recent_acc": state.get("recent_acc"),
            "recent_total": state.get("recent_total"),
            "recent_price_window_hit_rate": state.get("recent_price_window_hit_rate"),
            "recent_price_window_total": state.get("recent_price_window_total"),
            "recent_recovery_streak": state.get("recent_recovery_streak"),
            "recent_price_recovery_streak": state.get("recent_price_recovery_streak"),
            "pending_unresolved_rows": state.get("pending_unresolved_rows"),
            "pending_unresolved_newest_prediction_ts": state.get("pending_unresolved_newest_prediction_ts"),
            "skip_breakdown": dict(state.get("skip_breakdown") or {}),
        },
    )
    return state

def _bucket_from_sig(sig):
    if sig in ("Weak", "Medium", "Strong"):
        return sig
    return "Medium"


def _is_flat_prediction_row(row, model_keys, eps: float = 1e-9) -> bool:
    """
    Skip abstentions/no-trade snapshots from live calibration.
    We want calibration on meaningful emitted signals, not flat guard rows.
    """
    try:
        consensus_dir = str(row.get("consensus_dir", "") or "").strip()
    except Exception:
        consensus_dir = ""
    try:
        consensus_pred = row.get("consensus_pred_ret")
        consensus_pred = None if consensus_pred is None or pd.isna(consensus_pred) else float(consensus_pred)
    except Exception:
        consensus_pred = None
    if consensus_dir in ("UP", "DOWN"):
        return False
    if consensus_pred is not None and abs(consensus_pred) > float(eps):
        return False
    for k in model_keys:
        try:
            pred = row.get(f"{k}_pred_ret")
            if pred is None or pd.isna(pred):
                continue
            if abs(float(pred)) > float(eps):
                return False
        except Exception:
            continue
    return True

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

    model_keys = ["base", "forest", "boost", "extra_trees", "meta", "consensus"]
    gui_model_keys = {"base", "forest", "boost", "extra_trees"}
    updated = 0
    skipped = 0
    skip_breakdown = {"flat": 0, "invalid": 0, "unresolved": 0}
    processed_timestamps = []
    processed_target_timestamps = []
    pending_prediction_ts = []
    pending_target_ts = []
    _apply_decay(state, now)
    state.setdefault("global", {}).setdefault("wins", 0.0)
    state.setdefault("global", {}).setdefault("total", 0.0)
    state.setdefault("by_model_stats", {})
    if preds.empty:
        _finalize_state(
            instrument_key,
            state,
            now,
            updated=0,
            skipped=0,
            pending_prediction_ts=[],
            pending_target_ts=[],
            skip_breakdown=skip_breakdown,
        )
        return {"updated": 0, "skipped": 0, "unresolved": 0}
    for _, row in preds.iterrows():
        if _is_flat_prediction_row(row, model_keys):
            skipped += 1
            skip_breakdown["flat"] += 1
            continue
        ts = row.get("timestamp")
        if pd.isna(ts):
            skipped += 1
            skip_breakdown["invalid"] += 1
            continue
        target_ts = ts + pd.Timedelta(minutes=horizon_minutes)
        # find closest price at or after target_ts
        try:
            idx_pos = prices.index.searchsorted(target_ts)
            if idx_pos >= len(prices.index):
                skipped += 1
                skip_breakdown["unresolved"] += 1
                pending_prediction_ts.append(ts)
                pending_target_ts.append(target_ts)
                continue
            actual_price = float(prices["close"].iloc[idx_pos])
            last_price = float(row.get("last_price", actual_price))
            if last_price == 0 or not pd.notna(actual_price):
                skipped += 1
                skip_breakdown["invalid"] += 1
                continue
            actual_ret = (actual_price / last_price) - 1.0
        except Exception:
            skipped += 1
            skip_breakdown["invalid"] += 1
            continue

        row_used = False
        row_preds = []
        row_gui_preds = []
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
            if k in gui_model_keys:
                row_gui_preds.append(pred_ret)
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
            # Track the same equal-weight model average the GUI displays.
            row_correct = None
            try:
                if row_gui_preds:
                    row_mean_pred = float(np.mean(row_gui_preds))
                    row_correct = int((row_mean_pred >= 0) == (actual_ret >= 0))
                elif row_preds:
                    row_mean_pred = float(np.mean(row_preds))
                    row_correct = int((row_mean_pred >= 0) == (actual_ret >= 0))
            except Exception:
                row_correct = None
            if row_correct is None:
                try:
                    c_pred = row.get("consensus_pred_ret")
                    if c_pred is not None and not pd.isna(c_pred):
                        c_pred = float(c_pred)
                        row_correct = int((c_pred >= 0) == (actual_ret >= 0))
                except Exception:
                    row_correct = None
            if row_correct is not None:
                recent = state.setdefault("recent_outcomes", [])
                recent.append(int(row_correct))
                if len(recent) > 500:
                    state["recent_outcomes"] = recent[-500:]
            try:
                row_price_pred = None
                if row_gui_preds:
                    row_mean_pred = float(np.mean(row_gui_preds))
                    abs_err = abs(float(actual_ret - row_mean_pred))
                    if np.isfinite(abs_err):
                        errs = state.setdefault("recent_abs_errors", [])
                        errs.append(float(abs_err))
                        if len(errs) > 1000:
                            state["recent_abs_errors"] = errs[-1000:]
                    row_price_pred = float(last_price * (1.0 + row_mean_pred))
                elif row_preds:
                    row_mean_pred = float(np.mean(row_preds))
                    abs_err = abs(float(actual_ret - row_mean_pred))
                    if np.isfinite(abs_err):
                        errs = state.setdefault("recent_abs_errors", [])
                        errs.append(float(abs_err))
                        if len(errs) > 1000:
                            state["recent_abs_errors"] = errs[-1000:]
                    row_price_pred = float(last_price * (1.0 + row_mean_pred))
                else:
                    c_pred = row.get("consensus_pred_ret")
                    if c_pred is not None and not pd.isna(c_pred):
                        c_pred = float(c_pred)
                        abs_err = abs(float(actual_ret - c_pred))
                        if np.isfinite(abs_err):
                            errs = state.setdefault("recent_abs_errors", [])
                            errs.append(float(abs_err))
                            if len(errs) > 1000:
                                state["recent_abs_errors"] = errs[-1000:]
                        row_price_pred = float(last_price * (1.0 + c_pred))
                if row_price_pred is not None:
                    pred_price = float(row_price_pred)
                    price_abs_err = abs(float(actual_price - pred_price))
                    if np.isfinite(price_abs_err):
                        window_thr = price_window_threshold_dollars(
                            "intraday",
                            instrument_key=instrument_key,
                            reference_price=actual_price,
                        )
                        window_ratio = float(price_abs_err / max(1e-9, window_thr))
                        price_errs = state.setdefault("recent_price_abs_errors", [])
                        price_errs.append(float(price_abs_err))
                        if len(price_errs) > 1000:
                            state["recent_price_abs_errors"] = price_errs[-1000:]
                        ratios = state.setdefault("recent_price_window_ratios", [])
                        ratios.append(window_ratio)
                        if len(ratios) > 1000:
                            state["recent_price_window_ratios"] = ratios[-1000:]
                        hits = state.setdefault("recent_price_window_hits", [])
                        hits.append(int(price_abs_err <= window_thr))
                        if len(hits) > 1000:
                            state["recent_price_window_hits"] = hits[-1000:]
                        thresholds = state.setdefault("recent_price_window_thresholds", [])
                        thresholds.append(float(window_thr))
                        if len(thresholds) > 1000:
                            state["recent_price_window_thresholds"] = thresholds[-1000:]
            except Exception:
                pass
            updated += 1
            processed_timestamps.append(ts)
            try:
                processed_target_timestamps.append(prices.index[idx_pos])
            except Exception:
                pass
        else:
            skipped += 1
            skip_breakdown["invalid"] += 1

    # update last_processed
    try:
        if processed_timestamps:
            latest_ts = pd.to_datetime(max(processed_timestamps), utc=True, errors="coerce")
            if pd.notna(latest_ts):
                state["last_processed"] = latest_ts.isoformat()
        if processed_target_timestamps:
            latest_target_ts = pd.to_datetime(max(processed_target_timestamps), utc=True, errors="coerce")
            if pd.notna(latest_target_ts):
                state["last_resolved_target_ts"] = latest_target_ts.isoformat()
    except Exception:
        logger.debug("Failed to advance last_processed", exc_info=True)
    _finalize_state(
        instrument_key,
        state,
        now,
        updated=updated,
        skipped=skipped,
        pending_prediction_ts=pending_prediction_ts,
        pending_target_ts=pending_target_ts,
        skip_breakdown=skip_breakdown,
    )
    return {"updated": updated, "skipped": skipped, "unresolved": int(len(pending_prediction_ts))}
