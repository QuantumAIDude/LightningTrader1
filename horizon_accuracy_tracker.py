import csv
import json
import logging
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    DATA_DIR,
    INSTRUMENTS,
    HORIZON_DRIFT_MIN_SAMPLES,
    HORIZON_ADAPT_MIN_TOTAL_SAMPLES,
    HORIZON_ADAPT_MIN_SAMPLES_PER_HORIZON,
)
from accuracy_windows import summarize_price_window_errors
from features import build_next_session_dataset, build_weekend_session_dataset
from horizon_overlay import fit_horizon_overlay, refresh_structural_snapshots
from model_utils import _models_dir_signature, load_models, predict_horizon

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


def _backfill_history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_accuracy_backfill_resolved_{instrument_key}.csv")


def _backfill_metrics_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_accuracy_backfill_{instrument_key}.json")


def _backfill_state_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_accuracy_backfill_state_{instrument_key}.json")


def _to_float(v):
    try:
        out = float(v)
        if np.isfinite(out):
            return out
    except Exception:
        pass
    return None


def _file_signature(path: str):
    try:
        st = os.stat(path)
        return [int(st.st_mtime_ns), int(st.st_size)]
    except Exception:
        return None


def _backfill_signature(instrument_key: str) -> dict:
    root = os.path.dirname(__file__)
    return {
        "models": [list(x) for x in _models_dir_signature()],
        "prices": _file_signature(_prices_path(instrument_key)),
        "prices_daily_long": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}_daily_long.csv")),
        "structural_snapshots": _file_signature(os.path.join(DATA_DIR, f"horizon_structural_snapshots_{instrument_key}.csv")),
        "structural_outcomes": _file_signature(os.path.join(DATA_DIR, f"horizon_structural_outcomes_{instrument_key}.csv")),
        "overlay": _file_signature(os.path.join(DATA_DIR, f"horizon_overlay_{instrument_key}.json")),
        "code": {
            "horizon_accuracy_tracker": _file_signature(__file__),
            "model_utils": _file_signature(os.path.join(root, "model_utils.py")),
            "horizon_overlay": _file_signature(os.path.join(root, "horizon_overlay.py")),
        },
    }


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


def _load_json_dict(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        logger.exception("Failed reading json %s", path)
        return {}


def _save_json_dict(path: str, payload: dict):
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        logger.exception("Failed writing json %s", path)
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


def _write_history_rows(path: str, rows: list[dict]):
    fieldnames = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
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


def _compute_metrics(history: pd.DataFrame, instrument_key: str = None) -> dict:
    if history is None or history.empty:
        return {"by_horizon": {}, "global_drift_flag": False, "sample_summary": {"total_samples": 0, "per_horizon_max_n": {}, "sample_sufficient": False}}
    by_h = {}
    global_drift = False
    per_h_n = {}
    total_samples = 0
    for horizon, df_h in history.groupby("horizon"):
        h_out = {"models": {}, "drift_flag": False}
        for model, df_m in df_h.groupby("model"):
            vals = df_m.sort_values("resolved_ts")
            abs_err = pd.to_numeric(vals["abs_error"], errors="coerce").dropna()
            pct_err = pd.to_numeric(vals["pct_error"], errors="coerce").dropna()
            signed = pd.to_numeric(vals["signed_error"], errors="coerce").dropna()
            actual_price = pd.to_numeric(vals.get("actual_price"), errors="coerce")
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
            window_metrics = summarize_price_window_errors(
                abs_err.values,
                actual_prices=actual_price.values if actual_price is not None else None,
                horizon=horizon,
                instrument_key=instrument_key,
            )
            recent_window_metrics = summarize_price_window_errors(
                recent_abs.values,
                actual_prices=actual_price.tail(recent_n).values if actual_price is not None else None,
                horizon=horizon,
                instrument_key=instrument_key,
            )
            recent_ratio = _to_float(recent_window_metrics.get("window_ratio_mean"))
            recent_hit = _to_float(recent_window_metrics.get("window_hit_rate"))
            overall_ratio = _to_float(window_metrics.get("window_ratio_mean"))
            min_hit_req = _to_float(window_metrics.get("min_window_hit_rate")) or 0.0
            drift_flag = bool(
                n >= int(HORIZON_DRIFT_MIN_SAMPLES)
                and np.isfinite(mae)
                and np.isfinite(recent_mae)
                and (
                    recent_mae > (mae * 1.15)
                    or (
                        recent_ratio is not None
                        and recent_ratio > max(1.10, (overall_ratio or 1.0) * 1.08)
                    )
                    or (
                        recent_hit is not None
                        and recent_hit < max(0.0, min_hit_req - 0.10)
                    )
                )
            )
            if drift_flag:
                h_out["drift_flag"] = True
                global_drift = True
            h_out["models"][model] = {
                "n": n,
                "mae": mae,
                "mae_dollars": window_metrics.get("mae_dollars", mae),
                "rmse": rmse,
                "rmse_dollars": window_metrics.get("rmse_dollars", rmse),
                "mape_pct": mape,
                "bias": bias,
                "recent_mae": recent_mae,
                "recent_window_ratio_mean": recent_ratio,
                "recent_window_hit_rate": recent_hit,
                "window_hit_rate": window_metrics.get("window_hit_rate"),
                "window_ratio_mean": window_metrics.get("window_ratio_mean"),
                "window_threshold_mean_dollars": window_metrics.get("window_threshold_mean_dollars"),
                "min_window_hit_rate": window_metrics.get("min_window_hit_rate"),
                "drift_flag": drift_flag,
            }
        try:
            n_vals = [int(v.get("n", 0)) for v in h_out["models"].values() if isinstance(v, dict)]
            h_n = max(n_vals) if n_vals else 0
        except Exception:
            h_n = 0
        per_h_n[horizon] = int(h_n)
        total_samples += int(h_n)
        by_h[horizon] = h_out
    min_total = max(1, int(HORIZON_ADAPT_MIN_TOTAL_SAMPLES))
    min_per_h = max(1, int(HORIZON_ADAPT_MIN_SAMPLES_PER_HORIZON))
    sample_sufficient = bool(total_samples >= min_total)
    if sample_sufficient and per_h_n:
        sample_sufficient = bool(any(int(v) >= min_per_h for v in per_h_n.values()))
    return {
        "by_horizon": by_h,
        "global_drift_flag": bool(global_drift),
        "sample_summary": {
            "total_samples": int(total_samples),
            "per_horizon_max_n": per_h_n,
            "sample_sufficient": bool(sample_sufficient),
        },
    }


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


def _pred_price_by_model(block: dict, suffix: str = "pred_price") -> dict:
    out = {}
    if not isinstance(block, dict):
        return out
    for model in _MODEL_KEYS:
        mv = block.get(model, {})
        if not isinstance(mv, dict):
            continue
        pv = _to_float(mv.get(suffix))
        if pv is not None:
            out[model] = pv
    return out


def backfill_horizon_accuracy(instrument_key: str, max_daily_targets: int = 32,
                              max_week_targets: int = 28, min_history_rows: int = 300) -> dict:
    prices = _load_prices(instrument_key)
    if prices is None or prices.empty:
        return {"error": "missing prices", "instrument": instrument_key}
    try:
        models = load_models(include_modes=True, instrument_key=instrument_key)
    except Exception:
        models = {}
    if not isinstance(models, dict) or not models:
        return {"error": "missing models", "instrument": instrument_key}

    signature = _backfill_signature(instrument_key)
    state = _load_json_dict(_backfill_state_path(instrument_key))
    reset_cache = state.get("signature") != signature
    processed = set(str(x) for x in state.get("processed_keys", []))
    if reset_cache:
        processed = set()
    rows = []
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        refresh_structural_snapshots(
            instrument_key,
            prices=prices,
            max_daily_targets=max_daily_targets,
            max_week_targets=max_week_targets,
        )
    except Exception:
        logger.exception("Failed refreshing structural snapshots for %s", instrument_key)

    try:
        ds_daily = build_next_session_dataset(prices, instrument_key=instrument_key).sort_index().tail(max(2, int(max_daily_targets)))
    except Exception:
        ds_daily = pd.DataFrame()
    for ts, row in ds_daily.iterrows():
        try:
            target_key_ts = pd.to_datetime(ts, utc=True, errors="coerce")
            target_key = target_key_ts.isoformat() if pd.notna(target_key_ts) else str(ts)
            close_v = _to_float(row.get("close"))
            th = _to_float(row.get("target_high_ret"))
            tl = _to_float(row.get("target_low_ret"))
            need_high = bool(
                th is not None
                and any(f"backfill|daily_high|{model}|{target_key}" not in processed for model in _MODEL_KEYS)
            )
            need_low = bool(
                tl is not None
                and any(f"backfill|daily_low|{model}|{target_key}" not in processed for model in _MODEL_KEYS)
            )
            if close_v is None or (not need_high and not need_low):
                continue
            subset = prices[prices.index <= ts]
            if subset is None or subset.empty or len(subset) < int(min_history_rows):
                continue
            res = predict_horizon(
                models,
                subset,
                instrument_key=instrument_key,
                mode="DailyHigh",
                log_side_effects=False,
            )
            if not isinstance(res, dict) or res.get("error"):
                continue
            high_prices = _pred_price_by_model(res.get("high", {}))
            low_prices = _pred_price_by_model(res.get("low", {}))
            if need_high:
                act_h = float(close_v * (1.0 + th))
                for model, pred_price in high_prices.items():
                    key = f"backfill|daily_high|{model}|{target_key}"
                    if key in processed:
                        continue
                    err = float(pred_price - act_h)
                    rows.append(
                        {
                            "resolved_ts": now_iso,
                            "instrument": instrument_key,
                            "horizon": "daily_high",
                            "model": model,
                            "target_key": target_key,
                            "pred_price": pred_price,
                            "actual_price": act_h,
                            "signed_error": err,
                            "abs_error": abs(err),
                            "pct_error": abs(err) / max(1e-9, abs(act_h)),
                            "source": "historical_replay",
                        }
                    )
                    processed.add(key)
            if need_low:
                act_l = float(close_v * (1.0 + tl))
                for model, pred_price in low_prices.items():
                    key = f"backfill|daily_low|{model}|{target_key}"
                    if key in processed:
                        continue
                    err = float(pred_price - act_l)
                    rows.append(
                        {
                            "resolved_ts": now_iso,
                            "instrument": instrument_key,
                            "horizon": "daily_low",
                            "model": model,
                            "target_key": target_key,
                            "pred_price": pred_price,
                            "actual_price": act_l,
                            "signed_error": err,
                            "abs_error": abs(err),
                            "pct_error": abs(err) / max(1e-9, abs(act_l)),
                            "source": "historical_replay",
                        }
                    )
                    processed.add(key)
        except Exception:
            continue

    try:
        ds_week = build_weekend_session_dataset(prices, instrument_key=instrument_key).sort_index().tail(max(2, int(max_week_targets)))
    except Exception:
        ds_week = pd.DataFrame()
    for ts, row in ds_week.iterrows():
        try:
            target_key_ts = pd.to_datetime(ts, utc=True, errors="coerce")
            target_key = target_key_ts.isoformat() if pd.notna(target_key_ts) else str(ts)
            close_v = _to_float(row.get("close"))
            tw = _to_float(row.get("target_weekend_ret"))
            need_week = bool(
                close_v is not None
                and tw is not None
                and any(f"backfill|weekend|{model}|{target_key}" not in processed for model in _MODEL_KEYS)
            )
            if not need_week:
                continue
            subset = prices[prices.index <= ts]
            if subset is None or subset.empty or len(subset) < int(min_history_rows):
                continue
            res = predict_horizon(
                models,
                subset,
                instrument_key=instrument_key,
                mode="WeekEnd",
                log_side_effects=False,
            )
            if not isinstance(res, dict) or res.get("error"):
                continue
            act_w = float(close_v * (1.0 + tw))
            week_prices = _pred_price_by_model(res.get("week", {}))
            for model, pred_price in week_prices.items():
                key = f"backfill|weekend|{model}|{target_key}"
                if key in processed:
                    continue
                err = float(pred_price - act_w)
                rows.append(
                    {
                        "resolved_ts": now_iso,
                        "instrument": instrument_key,
                        "horizon": "weekend",
                        "model": model,
                        "target_key": target_key,
                        "pred_price": pred_price,
                        "actual_price": act_w,
                        "signed_error": err,
                        "abs_error": abs(err),
                        "pct_error": abs(err) / max(1e-9, abs(act_w)),
                        "source": "historical_replay",
                    }
                )
                processed.add(key)
        except Exception:
            continue

    history_path = _backfill_history_path(instrument_key)
    if reset_cache:
        _write_history_rows(history_path, rows)
    else:
        _append_history_rows(history_path, rows)
    hist_df = _load_history(history_path)
    metrics = _compute_metrics(hist_df, instrument_key=instrument_key)
    try:
        metrics["overlay"] = fit_horizon_overlay(instrument_key, prices=prices)
    except Exception:
        logger.exception("Failed fitting horizon overlay for %s", instrument_key)
    metrics["last_update"] = now_iso
    metrics["newly_resolved_rows"] = int(len(rows))
    metrics["source"] = "historical_replay"
    metrics["signature"] = signature
    _save_json_dict(_backfill_metrics_path(instrument_key), metrics)
    _save_json_dict(
        _backfill_state_path(instrument_key),
        {"processed_keys": list(sorted(processed))[-20000:], "last_update": now_iso, "signature": signature},
    )
    return metrics


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
                                "source": "logged_prediction",
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
                                "source": "logged_prediction",
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
                        "source": "logged_prediction",
                    }
                )
                processed.add(key)

    history_path = _history_path(instrument_key)
    _append_history_rows(history_path, resolved_rows)
    hist_df = _load_history(history_path)
    metrics = _compute_metrics(hist_df, instrument_key=instrument_key)
    try:
        metrics["overlay"] = fit_horizon_overlay(instrument_key, prices=prices)
    except Exception:
        logger.exception("Failed fitting horizon overlay for %s", instrument_key)
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
