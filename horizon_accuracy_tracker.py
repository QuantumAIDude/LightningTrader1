import csv
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_DIR, INSTRUMENTS, HORIZON_DRIFT_MIN_SAMPLES

logger = logging.getLogger(__name__)

_MODEL_KEYS = ["base", "forest", "boost", "extra_trees", "meta", "consensus"]


def _predictions_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"predictions_horizon_{instrument_key}.csv")


def _prices_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")


def _state_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_accuracy_state_{instrument_key}.json")


def _history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_accuracy_resolved_{instrument_key}.csv")


def _metrics_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_accuracy_{instrument_key}.json")


def _to_float(v):
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return None


def _load_csv_loose(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        rows = []
        with open(path, "r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if not header:
                return pd.DataFrame()
            for row in reader:
                if not row:
                    continue
                if len(row) < len(header):
                    row = row + [None] * (len(header) - len(row))
                elif len(row) > len(header):
                    row = row[: len(header)]
                rows.append(row)
        return pd.DataFrame(rows, columns=header)
    except Exception:
        logger.exception("Failed reading csv %s", path)
        return pd.DataFrame()


def _load_state(instrument_key: str) -> dict:
    path = _state_path(instrument_key)
    if not os.path.exists(path):
        return {"processed_keys": [], "last_update": None}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        data.setdefault("processed_keys", [])
        return data
    except Exception:
        logger.exception("Failed reading horizon state")
        return {"processed_keys": [], "last_update": None}


def _save_state(instrument_key: str, state: dict):
    path = _state_path(instrument_key)
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)
    except Exception:
        logger.exception("Failed writing horizon state")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _load_prices(instrument_key: str) -> pd.DataFrame:
    path = _prices_path(instrument_key)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception:
        logger.exception("Failed reading prices for %s", instrument_key)
        return pd.DataFrame()


def _instrument_meta(instrument_key: str) -> dict:
    return INSTRUMENTS.get(instrument_key, {})


def _session_dates(local_idx, session_open_minute: int, session_close_minute: int):
    local_dates = pd.to_datetime(local_idx.date)
    minutes = local_idx.hour * 60 + local_idx.minute
    if session_open_minute > session_close_minute:
        shifted = np.where(minutes >= session_open_minute, local_dates, local_dates - pd.Timedelta(days=1))
        return pd.to_datetime(shifted).date
    return pd.to_datetime(local_dates).date


def _today_session_date(instr_meta: dict, now_utc: pd.Timestamp):
    tz = instr_meta.get("tz", "America/New_York")
    session_open = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    session_close = int(instr_meta.get("session_close_minute", 16 * 60))
    now_local = now_utc.tz_convert(tz)
    minute_of_day = int(now_local.hour * 60 + now_local.minute)
    d = now_local.date()
    if session_open > session_close and minute_of_day < session_open:
        d = (now_local - pd.Timedelta(days=1)).date()
    return d, minute_of_day, session_open, session_close


def _daily_high_low_actual(prices: pd.DataFrame, instrument_key: str, target_date: str):
    if prices is None or prices.empty:
        return None
    instr_meta = _instrument_meta(instrument_key)
    tz = instr_meta.get("tz", "America/New_York")
    open_min = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    close_min = int(instr_meta.get("session_close_minute", 16 * 60))
    try:
        target = pd.to_datetime(target_date).date()
    except Exception:
        return None
    idx_local = pd.to_datetime(prices.index, utc=True, errors="coerce").tz_convert(tz)
    session_key = _session_dates(idx_local, open_min, close_min)
    mask = (session_key == target)
    if not np.any(mask):
        return None
    now_utc = pd.Timestamp.now(tz="UTC")
    today_session, minute_now, _, _ = _today_session_date(instr_meta, now_utc)
    if target > today_session:
        return None
    if target == today_session and open_min <= close_min and minute_now <= close_min:
        return None
    if target == today_session and open_min > close_min:
        # overnight sessions close on following clock date; avoid resolving during active session
        if minute_now >= open_min or minute_now <= close_min:
            return None
    block = prices.loc[mask]
    if block.empty:
        return None
    high_col = "high" if "high" in block.columns else "close"
    low_col = "low" if "low" in block.columns else "close"
    actual_high = _to_float(pd.to_numeric(block[high_col], errors="coerce").max())
    actual_low = _to_float(pd.to_numeric(block[low_col], errors="coerce").min())
    if actual_high is None or actual_low is None:
        return None
    return {"high": actual_high, "low": actual_low}


def _weekend_actual(prices: pd.DataFrame, instrument_key: str, target_week: str):
    if prices is None or prices.empty:
        return None
    instr_meta = _instrument_meta(instrument_key)
    tz = instr_meta.get("tz", "America/New_York")
    close_min = int(instr_meta.get("session_close_minute", 16 * 60))
    week_end_day = int(instr_meta.get("week_end_day", 4))
    try:
        year, week = str(target_week).split("-W")
        year = int(year)
        week = int(week)
    except Exception:
        return None
    idx_local = pd.to_datetime(prices.index, utc=True, errors="coerce").tz_convert(tz)
    iso = idx_local.isocalendar()
    mask = (iso.year.astype(int) == year) & (iso.week.astype(int) == week)
    if not np.any(mask):
        return None
    now_local = pd.Timestamp.now(tz=tz)
    now_iso = now_local.isocalendar()
    if (int(now_iso.year), int(now_iso.week)) < (year, week):
        return None
    if (int(now_iso.year), int(now_iso.week)) == (year, week):
        now_minute = int(now_local.hour * 60 + now_local.minute)
        if int(now_local.weekday()) < week_end_day:
            return None
        if int(now_local.weekday()) == week_end_day and now_minute <= close_min:
            return None
    block = prices.loc[mask]
    if block.empty:
        return None
    block_idx_local = idx_local[mask]
    block = block.copy()
    block["__weekday"] = block_idx_local.weekday
    close_col = "close"
    end_rows = block[block["__weekday"] == week_end_day]
    if end_rows.empty:
        actual_close = _to_float(pd.to_numeric(block[close_col], errors="coerce").iloc[-1])
    else:
        actual_close = _to_float(pd.to_numeric(end_rows[close_col], errors="coerce").iloc[-1])
    if actual_close is None:
        return None
    return {"weekend_close": actual_close}


def _append_history_rows(path: str, rows: list[dict]):
    if not rows:
        return
    fieldnames = sorted(set().union(*(r.keys() for r in rows)))
    write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _load_history(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "resolved_ts" in df.columns:
            df["resolved_ts"] = pd.to_datetime(df["resolved_ts"], utc=True, errors="coerce")
        for col in ("abs_error", "pct_error", "signed_error", "pred_price", "actual_price"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df
    except Exception:
        logger.exception("Failed reading horizon history")
        return pd.DataFrame()


def _compute_metrics(history: pd.DataFrame) -> dict:
    if history is None or history.empty:
        return {"by_horizon": {}, "global_drift_flag": False}
    by_h = {}
    global_drift = False
    for horizon, df_h in history.groupby("horizon"):
        h_out = {"models": {}, "drift_flag": False}
        for model, df_m in df_h.groupby("model"):
            vals = df_m.sort_values("resolved_ts")
            abs_err = pd.to_numeric(vals["abs_error"], errors="coerce").dropna()
            pct_err = pd.to_numeric(vals["pct_error"], errors="coerce").dropna()
            signed = pd.to_numeric(vals["signed_error"], errors="coerce").dropna()
            n = int(len(abs_err))
            if n == 0:
                continue
            mae = float(abs_err.mean())
            rmse = float(np.sqrt(np.mean(np.square(abs_err))))
            mape = float(pct_err.mean() * 100.0) if len(pct_err) else None
            bias = float(signed.mean()) if len(signed) else 0.0
            recent_n = min(20, n)
            recent_abs = abs_err.tail(recent_n)
            recent_mae = float(recent_abs.mean()) if len(recent_abs) else mae
            drift_flag = bool(
                n >= int(HORIZON_DRIFT_MIN_SAMPLES)
                and np.isfinite(mae)
                and np.isfinite(recent_mae)
                and recent_mae > (mae * 1.25)
            )
            if drift_flag:
                h_out["drift_flag"] = True
                global_drift = True
            h_out["models"][model] = {
                "n": n,
                "mae": mae,
                "rmse": rmse,
                "mape_pct": mape,
                "bias": bias,
                "recent_mae": recent_mae,
                "drift_flag": drift_flag,
            }
        by_h[horizon] = h_out
    return {"by_horizon": by_h, "global_drift_flag": bool(global_drift)}


def _latest_predictions_per_target(preds: pd.DataFrame) -> dict:
    out = {}
    if preds is None or preds.empty:
        return out
    preds = preds.copy()
    preds["timestamp"] = pd.to_datetime(preds.get("timestamp"), utc=True, errors="coerce")
    preds = preds.dropna(subset=["timestamp"])
    for _, row in preds.iterrows():
        mode = str(row.get("mode", "")).strip().lower()
        if "daily" in mode:
            target = str(row.get("target_session_date", "")).strip()
            if not target:
                continue
            key = ("daily", target)
        elif "week" in mode:
            target = str(row.get("target_week", "")).strip()
            if not target:
                continue
            key = ("weekend", target)
        else:
            continue
        prev = out.get(key)
        if prev is None:
            out[key] = row
            continue
        ts_new = row.get("timestamp")
        ts_old = prev.get("timestamp")
        if pd.notna(ts_new) and (pd.isna(ts_old) or ts_new >= ts_old):
            out[key] = row
    return out


def update_horizon_accuracy(instrument_key: str) -> dict:
    preds = _load_csv_loose(_predictions_path(instrument_key))
    prices = _load_prices(instrument_key)
    state = _load_state(instrument_key)
    processed = set(str(x) for x in state.get("processed_keys", []))
    resolved_rows = []

    latest = _latest_predictions_per_target(preds)
    now_iso = datetime.now(timezone.utc).isoformat()
    for (mode, target), row in latest.items():
        if mode == "daily":
            actuals = _daily_high_low_actual(prices, instrument_key, target)
            if not actuals:
                continue
            for model in _MODEL_KEYS:
                pred_high = _to_float(row.get(f"{model}_high_pred_price"))
                pred_low = _to_float(row.get(f"{model}_low_pred_price"))
                if pred_high is not None:
                    key = f"daily_high|{model}|{target}"
                    if key not in processed:
                        err = float(pred_high - actuals["high"])
                        resolved_rows.append(
                            {
                                "resolved_ts": now_iso,
                                "instrument": instrument_key,
                                "horizon": "daily_high",
                                "model": model,
                                "target_key": target,
                                "pred_price": pred_high,
                                "actual_price": actuals["high"],
                                "signed_error": err,
                                "abs_error": abs(err),
                                "pct_error": abs(err) / max(1e-9, abs(actuals["high"])),
                            }
                        )
                        processed.add(key)
                if pred_low is not None:
                    key = f"daily_low|{model}|{target}"
                    if key not in processed:
                        err = float(pred_low - actuals["low"])
                        resolved_rows.append(
                            {
                                "resolved_ts": now_iso,
                                "instrument": instrument_key,
                                "horizon": "daily_low",
                                "model": model,
                                "target_key": target,
                                "pred_price": pred_low,
                                "actual_price": actuals["low"],
                                "signed_error": err,
                                "abs_error": abs(err),
                                "pct_error": abs(err) / max(1e-9, abs(actuals["low"])),
                            }
                        )
                        processed.add(key)
        elif mode == "weekend":
            actuals = _weekend_actual(prices, instrument_key, target)
            if not actuals:
                continue
            for model in _MODEL_KEYS:
                pred_week = _to_float(row.get(f"{model}_week_pred_price"))
                if pred_week is None:
                    continue
                key = f"weekend|{model}|{target}"
                if key in processed:
                    continue
                err = float(pred_week - actuals["weekend_close"])
                resolved_rows.append(
                    {
                        "resolved_ts": now_iso,
                        "instrument": instrument_key,
                        "horizon": "weekend",
                        "model": model,
                        "target_key": target,
                        "pred_price": pred_week,
                        "actual_price": actuals["weekend_close"],
                        "signed_error": err,
                        "abs_error": abs(err),
                        "pct_error": abs(err) / max(1e-9, abs(actuals["weekend_close"])),
                    }
                )
                processed.add(key)

    history_path = _history_path(instrument_key)
    _append_history_rows(history_path, resolved_rows)
    hist_df = _load_history(history_path)
    metrics = _compute_metrics(hist_df)
    metrics["last_update"] = now_iso
    metrics["newly_resolved_rows"] = int(len(resolved_rows))
    metrics_path = _metrics_path(instrument_key)
    try:
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2, sort_keys=True)
    except Exception:
        logger.exception("Failed writing horizon metrics")
    state["processed_keys"] = list(sorted(processed))[-8000:]
    state["last_update"] = now_iso
    _save_state(instrument_key, state)
    return metrics

