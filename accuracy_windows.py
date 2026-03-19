from __future__ import annotations

import numpy as np

from config import (
    INSTRUMENTS,
    READINESS_INTRADAY_WINDOW_DOLLARS,
    READINESS_INTRADAY_WINDOW_PCT,
    READINESS_DAILY_WINDOW_DOLLARS,
    READINESS_DAILY_WINDOW_PCT,
    READINESS_WEEKEND_WINDOW_DOLLARS,
    READINESS_WEEKEND_WINDOW_PCT,
    READINESS_MIN_INTRADAY_WINDOW_HIT_RATE,
    READINESS_MIN_DAILY_WINDOW_HIT_RATE,
    READINESS_MIN_WEEKEND_WINDOW_HIT_RATE,
)


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _horizon_key(horizon: str) -> str:
    key = str(horizon or "").strip().lower()
    if key in ("4min", "intraday", "4m"):
        return "intraday"
    if key in ("daily", "daily_high", "daily_low", "day"):
        return "daily"
    if key in ("weekend", "weekly", "week", "week_end", "weekend_close"):
        return "weekend"
    return key or "intraday"


def price_window_threshold_dollars(horizon: str, instrument_key: str = None, reference_price=None) -> float:
    hk = _horizon_key(horizon)
    instr_meta = INSTRUMENTS.get(str(instrument_key or "").upper(), {}) if instrument_key else {}
    if hk == "daily":
        abs_floor = _safe_float(instr_meta.get("readiness_daily_window_dollars"))
        pct_scale = _safe_float(instr_meta.get("readiness_daily_window_pct"))
        abs_floor = float(abs_floor if abs_floor is not None else READINESS_DAILY_WINDOW_DOLLARS)
        pct_scale = float(pct_scale if pct_scale is not None else READINESS_DAILY_WINDOW_PCT)
    elif hk == "weekend":
        abs_floor = _safe_float(instr_meta.get("readiness_weekend_window_dollars"))
        pct_scale = _safe_float(instr_meta.get("readiness_weekend_window_pct"))
        abs_floor = float(abs_floor if abs_floor is not None else READINESS_WEEKEND_WINDOW_DOLLARS)
        pct_scale = float(pct_scale if pct_scale is not None else READINESS_WEEKEND_WINDOW_PCT)
    else:
        abs_floor = _safe_float(instr_meta.get("readiness_intraday_window_dollars"))
        pct_scale = _safe_float(instr_meta.get("readiness_intraday_window_pct"))
        abs_floor = float(abs_floor if abs_floor is not None else READINESS_INTRADAY_WINDOW_DOLLARS)
        pct_scale = float(pct_scale if pct_scale is not None else READINESS_INTRADAY_WINDOW_PCT)
    ref_price = _safe_float(reference_price)
    rel_window = (abs(ref_price) * pct_scale) if (ref_price is not None and pct_scale > 0.0) else 0.0
    tick_size = _safe_float(instr_meta.get("tick_size"))
    return float(max(abs_floor, rel_window, tick_size or 0.01))


def min_window_hit_rate(horizon: str, instrument_key: str = None) -> float:
    hk = _horizon_key(horizon)
    instr_meta = INSTRUMENTS.get(str(instrument_key or "").upper(), {}) if instrument_key else {}
    if hk == "daily":
        value = _safe_float(instr_meta.get("readiness_min_daily_window_hit_rate"))
        return float(value if value is not None else READINESS_MIN_DAILY_WINDOW_HIT_RATE)
    if hk == "weekend":
        value = _safe_float(instr_meta.get("readiness_min_weekend_window_hit_rate"))
        return float(value if value is not None else READINESS_MIN_WEEKEND_WINDOW_HIT_RATE)
    value = _safe_float(instr_meta.get("readiness_min_intraday_window_hit_rate"))
    return float(value if value is not None else READINESS_MIN_INTRADAY_WINDOW_HIT_RATE)


def window_ratio_score(window_ratio_mean) -> float | None:
    ratio = _safe_float(window_ratio_mean)
    if ratio is None:
        return None
    return float(max(0.0, 1.0 - (min(2.0, ratio) / 2.0)))


def _to_float_array(values):
    if values is None:
        return np.asarray([], dtype=float)
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        try:
            arr = np.asarray(list(values), dtype=float)
        except Exception:
            return np.asarray([], dtype=float)
    return np.ravel(arr).astype(float, copy=False)


def summarize_price_window_errors(abs_errors, actual_prices=None, horizon: str = "intraday",
                                  instrument_key: str = None) -> dict:
    err_arr = _to_float_array(abs_errors)
    if err_arr.size == 0:
        return {
            "n": 0,
            "mae_dollars": None,
            "rmse_dollars": None,
            "window_hit_rate": None,
            "window_ratio_mean": None,
            "window_threshold_mean_dollars": None,
            "min_window_hit_rate": min_window_hit_rate(horizon, instrument_key=instrument_key),
        }
    err_arr = err_arr[np.isfinite(err_arr)]
    if err_arr.size == 0:
        return {
            "n": 0,
            "mae_dollars": None,
            "rmse_dollars": None,
            "window_hit_rate": None,
            "window_ratio_mean": None,
            "window_threshold_mean_dollars": None,
            "min_window_hit_rate": min_window_hit_rate(horizon, instrument_key=instrument_key),
        }
    if actual_prices is None:
        price_arr = np.full(err_arr.shape, np.nan, dtype=float)
    else:
        price_arr = _to_float_array(actual_prices)
        if price_arr.size != err_arr.size:
            price_arr = np.resize(price_arr, err_arr.size)
    thresholds = np.asarray(
        [price_window_threshold_dollars(horizon, instrument_key=instrument_key, reference_price=px) for px in price_arr],
        dtype=float,
    )
    thresholds = np.where(np.isfinite(thresholds) & (thresholds > 0.0), thresholds, 0.01)
    ratios = err_arr / thresholds
    hits = err_arr <= thresholds
    return {
        "n": int(err_arr.size),
        "mae_dollars": float(np.mean(err_arr)),
        "rmse_dollars": float(np.sqrt(np.mean(np.square(err_arr)))),
        "window_hit_rate": float(np.mean(hits)),
        "window_ratio_mean": float(np.mean(ratios)),
        "window_threshold_mean_dollars": float(np.mean(thresholds)),
        "min_window_hit_rate": min_window_hit_rate(horizon, instrument_key=instrument_key),
    }
