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
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    READINESS_REQUIRE_DIRECTIONAL,
    READINESS_REQUIRE_PROFITABILITY,
    READINESS_REQUIRE_SHARPE,
    READINESS_REQUIRE_ACTIVE,
    READINESS_USE_HORIZON_PRIORITY,
    READINESS_MIN_HORIZON_SCORE,
    READINESS_WEIGHT_WEEKEND,
    READINESS_WEIGHT_DAILY,
    READINESS_WEIGHT_INTRADAY,
    READINESS_INTRADAY_WINDOW_DOLLARS,
    READINESS_DAILY_WINDOW_DOLLARS,
    READINESS_WEEKEND_WINDOW_DOLLARS,
    READINESS_INTRADAY_WINDOW_PCT,
    READINESS_DAILY_WINDOW_PCT,
    READINESS_WEEKEND_WINDOW_PCT,
    READINESS_MIN_INTRADAY_WINDOW_HIT_RATE,
    READINESS_MIN_DAILY_WINDOW_HIT_RATE,
    READINESS_MIN_WEEKEND_WINDOW_HIT_RATE,
    READINESS_MAX_WEEKEND_ERR_PCT,
    READINESS_MAX_DAILY_ERR_PCT,
    READINESS_MAX_INTRADAY_ERR_BPS,
    READINESS_MIN_WEEKEND_SAMPLES,
    READINESS_MIN_DAILY_SAMPLES,
    READINESS_MIN_INTRADAY_SAMPLES,
    TS_SPLIT_GAP_BARS,
)
from data_fetch import update_price_data
from features import build_features_for_training, build_next_session_dataset, build_weekend_session_dataset, DEFAULT_FEATURES
from horizon_overlay import load_horizon_overlay
from model_trainers import walk_forward_validate
from model_utils import _models_dir_signature, load_models, predict_horizon
from accuracy_windows import (
    majority_direction_vote,
    price_window_threshold_dollars,
    summarize_price_window_errors,
    min_window_hit_rate,
    window_ratio_score,
)


def _history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"readiness_history_{instrument_key}.jsonl")


def _state_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"readiness_state_{instrument_key}.json")


def _wf_cache_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"walk_forward_cache_{instrument_key}.json")


def _safe_mean(values):
    try:
        vals = [float(v) for v in (values or []) if v is not None and np.isfinite(float(v))]
    except Exception:
        vals = []
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return float(x)


def _load_price_df(instrument_key: str, online: bool = False) -> pd.DataFrame:
    if online:
        df = update_price_data(instrument_key)
    else:
        p = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
        if not os.path.exists(p):
            return pd.DataFrame()
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json(path: str, payload: dict):
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _file_signature(path: str):
    try:
        st = os.stat(path)
        return [int(st.st_mtime_ns), int(st.st_size)]
    except Exception:
        return None


def _walk_forward_signature(instrument_key: str, df: pd.DataFrame, n_splits: int) -> dict:
    root = os.path.dirname(__file__)
    try:
        rows = int(len(df))
        last_ts = pd.to_datetime(df.index.max(), utc=True, errors="coerce")
        first_ts = pd.to_datetime(df.index.min(), utc=True, errors="coerce")
    except Exception:
        rows = 0
        last_ts = pd.NaT
        first_ts = pd.NaT
    return {
        "instrument": instrument_key,
        "n_splits": int(n_splits),
        "prices": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")),
        "models": [list(x) for x in _models_dir_signature()],
        "rows": rows,
        "first_ts": first_ts.isoformat() if pd.notna(first_ts) else None,
        "last_ts": last_ts.isoformat() if pd.notna(last_ts) else None,
        "code": {
            "readiness_tracker": _file_signature(__file__),
            "model_trainers": _file_signature(os.path.join(root, "model_trainers.py")),
            "accuracy_windows": _file_signature(os.path.join(root, "accuracy_windows.py")),
            "forecast_calibration": _file_signature(os.path.join(root, "forecast_calibration.py")),
        },
    }


def _horizon_backfill_signature(instrument_key: str) -> dict:
    root = os.path.dirname(__file__)
    return {
        "models": [list(x) for x in _models_dir_signature()],
        "prices": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")),
        "prices_daily_long": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}_daily_long.csv")),
        "structural_snapshots": _file_signature(os.path.join(DATA_DIR, f"horizon_structural_snapshots_{instrument_key}.csv")),
        "structural_outcomes": _file_signature(os.path.join(DATA_DIR, f"horizon_structural_outcomes_{instrument_key}.csv")),
        "overlay": _file_signature(os.path.join(DATA_DIR, f"horizon_overlay_{instrument_key}.json")),
        "code": {
            "horizon_accuracy_tracker": _file_signature(os.path.join(root, "horizon_accuracy_tracker.py")),
            "model_utils": _file_signature(os.path.join(root, "model_utils.py")),
            "horizon_overlay": _file_signature(os.path.join(root, "horizon_overlay.py")),
        },
    }


def _ts_split(n_splits: int):
    gap = max(0, int(TS_SPLIT_GAP_BARS))
    try:
        return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    except TypeError:
        return TimeSeriesSplit(n_splits=n_splits)


def _ridge_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])


def _cv_abs_ret_metrics(dataset: pd.DataFrame, target_norm_col: str, target_ret_col: str,
                        horizon: str, instrument_key: str, min_rows: int = 80) -> dict:
    if dataset is None or dataset.empty or len(dataset) < max(6, int(min_rows)):
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    if target_norm_col not in dataset.columns or target_ret_col not in dataset.columns:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    feats = [f for f in DEFAULT_FEATURES if f in dataset.columns]
    if not feats:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    X = dataset[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_norm = pd.to_numeric(dataset[target_norm_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_ret = pd.to_numeric(dataset[target_ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    close_base = pd.to_numeric(dataset.get("close"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    vol = pd.to_numeric(dataset.get("ewm_vol"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.001).clip(lower=1e-6)
    n = len(X)
    scale = max(6, int(max(8, min_rows) // 2))
    n_splits = min(max(2, int(READINESS_BACKTEST_SPLITS)), max(2, n // scale))
    if n_splits < 2:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    tss = _ts_split(n_splits)
    maes_ret = []
    maes_dollars = []
    ratio_vals = []
    hit_vals = []
    threshold_vals = []
    for tr_idx, te_idx in tss.split(X):
        try:
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr = y_norm.iloc[tr_idx]
            yte_ret = y_ret.iloc[te_idx].values.astype(float)
            if len(Xtr) < max(6, int(min_rows // 4)) or len(Xte) < 2:
                continue
            model = _ridge_pipeline()
            model.fit(Xtr, ytr)
            pred_norm = np.asarray(model.predict(Xte), dtype=float)
            pred_ret = pred_norm * vol.iloc[te_idx].values.astype(float)
            err = np.abs(pred_ret - yte_ret)
            maes_ret.append(float(np.nanmean(err)))
            if "close" in dataset.columns:
                base_prices = close_base.iloc[te_idx].values.astype(float)
                actual_prices = base_prices * (1.0 + yte_ret)
                pred_prices = base_prices * (1.0 + pred_ret)
                window_metrics = summarize_price_window_errors(
                    np.abs(pred_prices - actual_prices),
                    actual_prices=actual_prices,
                    horizon=horizon,
                    instrument_key=instrument_key,
                )
                if window_metrics.get("mae_dollars") is not None:
                    maes_dollars.append(float(window_metrics["mae_dollars"]))
                if window_metrics.get("window_ratio_mean") is not None:
                    ratio_vals.append(float(window_metrics["window_ratio_mean"]))
                if window_metrics.get("window_hit_rate") is not None:
                    hit_vals.append(float(window_metrics["window_hit_rate"]))
                if window_metrics.get("window_threshold_mean_dollars") is not None:
                    threshold_vals.append(float(window_metrics["window_threshold_mean_dollars"]))
        except Exception:
            continue
    if not maes_ret:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    mae_ret = float(np.nanmean(maes_ret))
    return {
        "folds": int(len(maes_ret)),
        "mae_ret": mae_ret,
        "mae_pct": float(mae_ret * 100.0),
        "mae_dollars": float(np.nanmean(maes_dollars)) if maes_dollars else None,
        "window_ratio_mean": float(np.nanmean(ratio_vals)) if ratio_vals else None,
        "window_hit_rate": float(np.nanmean(hit_vals)) if hit_vals else None,
        "window_threshold_mean_dollars": float(np.nanmean(threshold_vals)) if threshold_vals else None,
    }


def _aggregate_horizon_payload(payload: dict) -> dict:
    by_h = payload.get("by_horizon", {}) if isinstance(payload, dict) else {}

    def _agg_h(h_name: str):
        block = by_h.get(h_name, {}) if isinstance(by_h, dict) else {}
        models = block.get("models", {}) if isinstance(block, dict) else {}
        preferred = []
        for name in ("consensus", "meta"):
            mv = models.get(name)
            if isinstance(mv, dict) and int(mv.get("n", 0) or 0) > 0:
                preferred.append(mv)
                break
        if not preferred:
            preferred = [v for v in models.values() if isinstance(v, dict) and int(v.get("n", 0) or 0) > 0]
        if not preferred:
            return {"n": 0, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
        if len(preferred) == 1:
            block_out = preferred[0]
            return {
                "n": int(block_out.get("n", 0) or 0),
                "mae_dollars": _safe_float(block_out.get("mae_dollars", block_out.get("mae"))),
                "window_ratio_mean": _safe_float(block_out.get("window_ratio_mean")),
                "window_hit_rate": _safe_float(block_out.get("window_hit_rate")),
            }
        maes = [_safe_float(v.get("mae_dollars", v.get("mae"))) for v in preferred]
        ratios = [_safe_float(v.get("window_ratio_mean")) for v in preferred]
        hits = [_safe_float(v.get("window_hit_rate")) for v in preferred]
        ns = [int(v.get("n", 0) or 0) for v in preferred]
        maes = [v for v in maes if v is not None]
        ratios = [v for v in ratios if v is not None]
        hits = [v for v in hits if v is not None]
        return {
            "n": int(max(ns) if ns else 0),
            "mae_dollars": float(np.nanmedian(maes)) if maes else None,
            "window_ratio_mean": float(np.nanmedian(ratios)) if ratios else None,
            "window_hit_rate": float(np.nanmedian(hits)) if hits else None,
        }

    week = _agg_h("weekend")
    dh = _agg_h("daily_high")
    dl = _agg_h("daily_low")
    daily_maes = [v for v in (dh.get("mae_dollars"), dl.get("mae_dollars")) if v is not None]
    daily_ratios = [v for v in (dh.get("window_ratio_mean"), dl.get("window_ratio_mean")) if v is not None]
    daily_hits = [v for v in (dh.get("window_hit_rate"), dl.get("window_hit_rate")) if v is not None]
    daily_n = min([n for n in (dh.get("n", 0), dl.get("n", 0)) if n > 0], default=max(int(dh.get("n", 0) or 0), int(dl.get("n", 0) or 0)))
    return {
        "weekend_mae_dollars": week.get("mae_dollars"),
        "weekend_window_ratio": week.get("window_ratio_mean"),
        "weekend_window_hit_rate": week.get("window_hit_rate"),
        "weekend_n": int(week.get("n", 0) or 0),
        "daily_mae_dollars": float(np.mean(daily_maes)) if daily_maes else None,
        "daily_window_ratio": float(np.mean(daily_ratios)) if daily_ratios else None,
        "daily_window_hit_rate": float(np.mean(daily_hits)) if daily_hits else None,
        "daily_n": int(daily_n or 0),
    }


def _horizon_live_errors(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"horizon_accuracy_{instrument_key}.json")
    payload = _load_json(path)
    return _aggregate_horizon_payload(payload)


def _overlay_backtest_errors(instrument_key: str) -> dict:
    payload = load_horizon_overlay(instrument_key)
    horizons = payload.get("horizons", {}) if isinstance(payload, dict) else {}
    daily_pair = payload.get("daily_pair", {}) if isinstance(payload, dict) else {}
    structural_pair = payload.get("daily_pair_structural", {}) if isinstance(payload, dict) else {}
    structural_weekend = payload.get("weekend_structural", {}) if isinstance(payload, dict) else {}

    def _obj(block: dict, horizon_name: str):
        if not isinstance(block, dict):
            return None
        ratio = _safe_float(block.get("window_ratio_mean"))
        mae = _safe_float(block.get("mae_dollars"))
        hit = _safe_float(block.get("window_hit_rate"))
        if ratio is None or mae is None or hit is None:
            return None
        hit_req = float(min_window_hit_rate("daily" if "daily" in horizon_name else horizon_name, instrument_key=instrument_key))
        return float(ratio + (1.25 * max(0.0, hit_req - hit)) + (0.05 * mae))

    def _agg(h_name: str):
        models = horizons.get(h_name, {}) if isinstance(horizons, dict) else {}
        rows = []
        for entry in models.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("selected") or "").strip().lower() == "insufficient_week_context":
                continue
            selected = entry.get("selected_metrics", {})
            rows.append(
                {
                    "n": int(entry.get("n", 0) or 0),
                    "mae_dollars": _safe_float(selected.get("mae_dollars")),
                    "window_ratio_mean": _safe_float(selected.get("window_ratio_mean")),
                    "window_hit_rate": _safe_float(selected.get("window_hit_rate")),
                }
            )
        rows = [r for r in rows if r["mae_dollars"] is not None]
        if not rows:
            return {"n": 0, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
        return {
            "n": int(max(r["n"] for r in rows)),
            "mae_dollars": float(np.mean([r["mae_dollars"] for r in rows])),
            "window_ratio_mean": float(np.mean([r["window_ratio_mean"] for r in rows if r["window_ratio_mean"] is not None])),
            "window_hit_rate": float(np.mean([r["window_hit_rate"] for r in rows if r["window_hit_rate"] is not None])),
        }

    week = _agg("weekend")
    if isinstance(structural_weekend, dict):
        agg = structural_weekend.get("aggregate_metrics", {})
        candidate = {
            "n": int(structural_weekend.get("n", 0) or 0),
            "mae_dollars": _safe_float(agg.get("mae_dollars")),
            "window_ratio_mean": _safe_float(agg.get("window_ratio_mean")),
            "window_hit_rate": _safe_float(agg.get("window_hit_rate")),
        }
        if _obj(candidate, "weekend") is not None:
            if _obj(week, "weekend") is None or float(_obj(candidate, "weekend")) + 1e-9 < float(_obj(week, "weekend")):
                week = candidate
    dh = _agg("daily_high")
    dl = _agg("daily_low")
    pair_rows = []
    structural_rows = []
    if isinstance(structural_pair, dict):
        agg = structural_pair.get("aggregate_metrics", {})
        if _safe_float(agg.get("mae_dollars")) is not None:
            structural_rows.append(
                {
                    "n": int(structural_pair.get("n", 0) or 0),
                    "mae_dollars": _safe_float(agg.get("mae_dollars")),
                    "window_ratio_mean": _safe_float(agg.get("window_ratio_mean")),
                    "window_hit_rate": _safe_float(agg.get("window_hit_rate")),
                }
            )
    for entry in daily_pair.values() if isinstance(daily_pair, dict) else []:
        if not isinstance(entry, dict):
            continue
        hm = entry.get("high_metrics", {})
        lm = entry.get("low_metrics", {})
        hi_mae = _safe_float(hm.get("mae_dollars"))
        lo_mae = _safe_float(lm.get("mae_dollars"))
        hi_ratio = _safe_float(hm.get("window_ratio_mean"))
        lo_ratio = _safe_float(lm.get("window_ratio_mean"))
        hi_hit = _safe_float(hm.get("window_hit_rate"))
        lo_hit = _safe_float(lm.get("window_hit_rate"))
        if hi_mae is None or lo_mae is None:
            continue
        pair_rows.append(
            {
                "n": int(entry.get("n", 0) or 0),
                "mae_dollars": 0.5 * (hi_mae + lo_mae),
                "window_ratio_mean": 0.5 * (hi_ratio + lo_ratio) if hi_ratio is not None and lo_ratio is not None else None,
                "window_hit_rate": 0.5 * (hi_hit + lo_hit) if hi_hit is not None and lo_hit is not None else None,
            }
        )
    if structural_rows:
        best_struct = min(
            structural_rows,
            key=lambda row: _obj(row, "daily") if _obj(row, "daily") is not None else float("inf"),
        )
        if not pair_rows:
            pair_rows = [best_struct]
        else:
            existing_pair = {
                "n": int(max(int(r.get("n", 0) or 0) for r in pair_rows)),
                "mae_dollars": float(np.mean([r["mae_dollars"] for r in pair_rows if r["mae_dollars"] is not None])),
                "window_ratio_mean": float(np.mean([r["window_ratio_mean"] for r in pair_rows if r["window_ratio_mean"] is not None])),
                "window_hit_rate": float(np.mean([r["window_hit_rate"] for r in pair_rows if r["window_hit_rate"] is not None])),
            }
            pair_obj = _obj(existing_pair, "daily")
            struct_obj = _obj(best_struct, "daily")
            if struct_obj is not None and (pair_obj is None or float(struct_obj) + 1e-9 < float(pair_obj)):
                pair_rows = [best_struct]
    if pair_rows:
        daily_maes = [r["mae_dollars"] for r in pair_rows if r["mae_dollars"] is not None]
        daily_ratios = [r["window_ratio_mean"] for r in pair_rows if r["window_ratio_mean"] is not None]
        daily_hits = [r["window_hit_rate"] for r in pair_rows if r["window_hit_rate"] is not None]
        daily_n = max(int(r["n"] or 0) for r in pair_rows)
    else:
        daily_maes = [v for v in (dh.get("mae_dollars"), dl.get("mae_dollars")) if v is not None]
        daily_ratios = [v for v in (dh.get("window_ratio_mean"), dl.get("window_ratio_mean")) if v is not None]
        daily_hits = [v for v in (dh.get("window_hit_rate"), dl.get("window_hit_rate")) if v is not None]
        daily_n = min([n for n in (dh.get("n", 0), dl.get("n", 0)) if n > 0], default=max(int(dh.get("n", 0) or 0), int(dl.get("n", 0) or 0)))
    return {
        "daily_mae_dollars": float(np.mean(daily_maes)) if daily_maes else None,
        "daily_window_ratio": float(np.mean(daily_ratios)) if daily_ratios else None,
        "daily_window_hit_rate": float(np.mean(daily_hits)) if daily_hits else None,
        "daily_samples": int(daily_n or 0),
        "weekend_mae_dollars": week.get("mae_dollars"),
        "weekend_window_ratio": week.get("window_ratio_mean"),
        "weekend_window_hit_rate": week.get("window_hit_rate"),
        "weekend_samples": int(week.get("n", 0) or 0),
    }


def _cached_horizon_backtest_errors(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"horizon_accuracy_backfill_{instrument_key}.json")
    payload = _load_json(path)
    out = {}
    if payload and payload.get("signature") == _horizon_backfill_signature(instrument_key):
        agg = _aggregate_horizon_payload(payload)
        out = {
            "daily_mae_dollars": agg.get("daily_mae_dollars"),
            "daily_window_ratio": agg.get("daily_window_ratio"),
            "daily_window_hit_rate": agg.get("daily_window_hit_rate"),
            "daily_samples": int(agg.get("daily_n", 0) or 0),
            "weekend_mae_dollars": agg.get("weekend_mae_dollars"),
            "weekend_window_ratio": agg.get("weekend_window_ratio"),
            "weekend_window_hit_rate": agg.get("weekend_window_hit_rate"),
            "weekend_samples": int(agg.get("weekend_n", 0) or 0),
        }
    overlay = _overlay_backtest_errors(instrument_key)
    if overlay.get("daily_mae_dollars") is not None:
        out["daily_mae_dollars"] = overlay.get("daily_mae_dollars")
        out["daily_window_ratio"] = overlay.get("daily_window_ratio")
        out["daily_window_hit_rate"] = overlay.get("daily_window_hit_rate")
        out["daily_samples"] = int(overlay.get("daily_samples", 0) or 0)
    if overlay.get("weekend_mae_dollars") is not None:
        out["weekend_mae_dollars"] = overlay.get("weekend_mae_dollars")
        out["weekend_window_ratio"] = overlay.get("weekend_window_ratio")
        out["weekend_window_hit_rate"] = overlay.get("weekend_window_hit_rate")
        out["weekend_samples"] = int(overlay.get("weekend_samples", 0) or 0)
    return out


def _load_cached_walk_forward(instrument_key: str, df: pd.DataFrame, n_splits: int) -> dict:
    payload = _load_json(_wf_cache_path(instrument_key))
    if not payload:
        return {}
    if payload.get("signature") != _walk_forward_signature(instrument_key, df, n_splits):
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _save_cached_walk_forward(instrument_key: str, df: pd.DataFrame, n_splits: int, metrics: dict):
    payload = {
        "signature": _walk_forward_signature(instrument_key, df, n_splits),
        "metrics": metrics if isinstance(metrics, dict) else {},
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(_wf_cache_path(instrument_key), payload)


def _intraday_live_error(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"calibration_live_{instrument_key}.json")
    payload = _load_json(path)
    errs = payload.get("recent_price_abs_errors", []) if isinstance(payload, dict) else []
    ratios = payload.get("recent_price_window_ratios", []) if isinstance(payload, dict) else []
    hits = payload.get("recent_price_window_hits", []) if isinstance(payload, dict) else []
    thresholds = payload.get("recent_price_window_thresholds", []) if isinstance(payload, dict) else []

    err_vals = [float(v) for v in (_safe_float(e) for e in errs) if v is not None]
    ratio_vals = [float(v) for v in (_safe_float(e) for e in ratios) if v is not None]
    hit_vals = [float(v) for v in (_safe_float(e) for e in hits) if v is not None]
    threshold_vals = [float(v) for v in (_safe_float(e) for e in thresholds) if v is not None]
    if err_vals:
        return {
            "n": int(len(err_vals)),
            "mae_dollars": float(np.mean(err_vals)),
            "window_ratio_mean": float(np.mean(ratio_vals)) if ratio_vals else None,
            "window_hit_rate": float(np.mean(hit_vals)) if hit_vals else None,
            "window_threshold_mean_dollars": float(np.mean(threshold_vals)) if threshold_vals else None,
        }
    return {"n": 0, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None, "window_threshold_mean_dollars": None}


def _horizon_backtest_errors(df: pd.DataFrame, instrument_key: str, wf: dict | None = None) -> dict:
    out = {
        "daily_mae_dollars": None,
        "daily_window_ratio": None,
        "daily_window_hit_rate": None,
        "daily_folds": 0,
        "daily_samples": 0,
        "weekend_mae_dollars": None,
        "weekend_window_ratio": None,
        "weekend_window_hit_rate": None,
        "weekend_folds": 0,
        "weekend_samples": 0,
        "intraday_mae_dollars": None,
        "intraday_window_ratio": None,
        "intraday_window_hit_rate": None,
        "intraday_folds": 0,
    }
    if df is None or df.empty:
        return out
    cached = _cached_horizon_backtest_errors(instrument_key)
    if int(cached.get("daily_samples", 0) or 0) < max(8, int(READINESS_MIN_DAILY_SAMPLES)) or int(cached.get("weekend_samples", 0) or 0) < max(4, int(READINESS_MIN_WEEKEND_SAMPLES)):
        try:
            from horizon_accuracy_tracker import backfill_horizon_accuracy

            backfill_horizon_accuracy(
                instrument_key,
                max_daily_targets=max(48, int(READINESS_MIN_DAILY_SAMPLES) * 6),
                max_week_targets=max(32, int(READINESS_MIN_WEEKEND_SAMPLES) * 8),
            )
            cached = _cached_horizon_backtest_errors(instrument_key)
        except Exception:
            pass
    try:
        if int(cached.get("daily_samples", 0) or 0) >= max(8, int(READINESS_MIN_DAILY_SAMPLES)):
            out["daily_mae_dollars"] = _safe_float(cached.get("daily_mae_dollars"))
            out["daily_window_ratio"] = _safe_float(cached.get("daily_window_ratio"))
            out["daily_window_hit_rate"] = _safe_float(cached.get("daily_window_hit_rate"))
            out["daily_samples"] = int(cached.get("daily_samples", 0) or 0)
            out["daily_folds"] = out["daily_samples"]
        if int(cached.get("weekend_samples", 0) or 0) >= max(4, int(READINESS_MIN_WEEKEND_SAMPLES)):
            out["weekend_mae_dollars"] = _safe_float(cached.get("weekend_mae_dollars"))
            out["weekend_window_ratio"] = _safe_float(cached.get("weekend_window_ratio"))
            out["weekend_window_hit_rate"] = _safe_float(cached.get("weekend_window_hit_rate"))
            out["weekend_samples"] = int(cached.get("weekend_samples", 0) or 0)
            out["weekend_folds"] = out["weekend_samples"]
    except Exception:
        pass
    # Prefer replay against the actual horizon prediction stack.
    try:
        if out["daily_mae_dollars"] is None or out["weekend_mae_dollars"] is None:
            rep = _horizon_replay_errors(df, instrument_key=instrument_key)
        else:
            rep = {}
        if rep.get("daily_mae_dollars") is not None:
            out["daily_mae_dollars"] = float(rep["daily_mae_dollars"])
            out["daily_window_ratio"] = _safe_float(rep.get("daily_window_ratio"))
            out["daily_window_hit_rate"] = _safe_float(rep.get("daily_window_hit_rate"))
            out["daily_samples"] = int(rep.get("daily_samples", 0) or 0)
            out["daily_folds"] = out["daily_samples"]
        if rep.get("weekend_mae_dollars") is not None:
            out["weekend_mae_dollars"] = float(rep["weekend_mae_dollars"])
            out["weekend_window_ratio"] = _safe_float(rep.get("weekend_window_ratio"))
            out["weekend_window_hit_rate"] = _safe_float(rep.get("weekend_window_hit_rate"))
            out["weekend_samples"] = int(rep.get("weekend_samples", 0) or 0)
            out["weekend_folds"] = out["weekend_samples"]
    except Exception:
        pass
    try:
        ds_daily = build_next_session_dataset(df, instrument_key=instrument_key)
        dh = _cv_abs_ret_metrics(ds_daily, "target_high_ret_norm", "target_high_ret", horizon="daily_high", instrument_key=instrument_key, min_rows=20)
        dl = _cv_abs_ret_metrics(ds_daily, "target_low_ret_norm", "target_low_ret", horizon="daily_low", instrument_key=instrument_key, min_rows=20)
        maes = [v for v in (dh.get("mae_dollars"), dl.get("mae_dollars")) if v is not None]
        ratios = [v for v in (dh.get("window_ratio_mean"), dl.get("window_ratio_mean")) if v is not None]
        hits = [v for v in (dh.get("window_hit_rate"), dl.get("window_hit_rate")) if v is not None]
        if maes and out["daily_mae_dollars"] is None:
            out["daily_mae_dollars"] = float(np.mean(maes))
            out["daily_window_ratio"] = float(np.mean(ratios)) if ratios else None
            out["daily_window_hit_rate"] = float(np.mean(hits)) if hits else None
            out["daily_folds"] = int(min(dh.get("folds", 0), dl.get("folds", 0)))
    except Exception:
        pass
    try:
        ds_week = build_weekend_session_dataset(df, instrument_key=instrument_key)
        wk = _cv_abs_ret_metrics(ds_week, "target_weekend_ret_norm", "target_weekend_ret", horizon="weekend", instrument_key=instrument_key, min_rows=20)
        if out["weekend_mae_dollars"] is None:
            out["weekend_mae_dollars"] = wk.get("mae_dollars")
            out["weekend_window_ratio"] = wk.get("window_ratio_mean")
            out["weekend_window_hit_rate"] = wk.get("window_hit_rate")
            out["weekend_folds"] = int(wk.get("folds", 0) or 0)
    except Exception:
        pass
    if isinstance(wf, dict):
        out["intraday_mae_dollars"] = _safe_mean(wf.get("price_mae_dollars"))
        out["intraday_window_ratio"] = _safe_mean(wf.get("price_window_ratio"))
        out["intraday_window_hit_rate"] = _safe_mean(wf.get("price_window_hit_rate"))
        out["intraday_folds"] = int(wf.get("folds", 0) or 0)
    return out


def _pick_metric(live_val, live_n: int, live_min_n: int, bt_val):
    lv = _safe_float(live_val)
    bv = _safe_float(bt_val)
    n = max(0, int(live_n or 0))
    min_n = max(1, int(live_min_n))
    if lv is not None and n >= min_n:
        return lv, "live", n
    if bv is not None:
        return bv, "backtest", 0
    if lv is not None and n > 0:
        return lv, "live_low_sample", n
    return None, "missing", n


def _pick_horizon_block(live_block: dict, live_min_n: int, bt_block: dict) -> tuple[dict, str, int]:
    live_block = live_block if isinstance(live_block, dict) else {}
    bt_block = bt_block if isinstance(bt_block, dict) else {}
    live_n = int(live_block.get("n", 0) or 0)
    min_n = max(1, int(live_min_n))
    live_ready = (
        live_n >= min_n
        and _safe_float(live_block.get("window_ratio_mean")) is not None
        and _safe_float(live_block.get("window_hit_rate")) is not None
    )
    if live_ready:
        return live_block, "live", live_n
    bt_ready = (
        _safe_float(bt_block.get("window_ratio_mean")) is not None
        and _safe_float(bt_block.get("window_hit_rate")) is not None
    )
    if bt_ready:
        return bt_block, "backtest", live_n
    live_partial = (
        live_n > 0
        and (
            _safe_float(live_block.get("window_ratio_mean")) is not None
            or _safe_float(live_block.get("window_hit_rate")) is not None
            or _safe_float(live_block.get("mae_dollars")) is not None
        )
    )
    if live_partial:
        return live_block, "live_low_sample", live_n
    return {
        "n": live_n,
        "mae_dollars": None,
        "window_ratio_mean": None,
        "window_hit_rate": None,
        "window_threshold_mean_dollars": None,
    }, "missing", live_n


def _preferred_pred_price(block: dict) -> float | None:
    if not isinstance(block, dict):
        return None
    for name in ("consensus", "meta"):
        v = block.get(name)
        if isinstance(v, dict):
            pv = _safe_float(v.get("pred_price"))
            if pv is not None:
                return float(pv)
    vals = []
    for v in block.values():
        if not isinstance(v, dict):
            continue
        pv = _safe_float(v.get("pred_price"))
        if pv is not None:
            vals.append(float(pv))
    if not vals:
        return None
    return float(np.mean(vals))


def _model_pred_prices(block: dict) -> list[float]:
    if not isinstance(block, dict):
        return []
    vals = []
    for model_key in ("base", "forest", "boost", "extra_trees"):
        v = block.get(model_key)
        if not isinstance(v, dict):
            continue
        pv = _safe_float(v.get("pred_price"))
        if pv is not None:
            vals.append(float(pv))
    return vals


def _horizon_replay_errors(df: pd.DataFrame, instrument_key: str, max_daily_samples: int = 64, max_week_samples: int = 64) -> dict:
    out = {
        "daily_mae_dollars": None,
        "daily_window_ratio": None,
        "daily_window_hit_rate": None,
        "daily_samples": 0,
        "daily_directional_acc": None,
        "daily_directional_samples": 0,
        "daily_directional_ties": 0,
        "weekend_mae_dollars": None,
        "weekend_window_ratio": None,
        "weekend_window_hit_rate": None,
        "weekend_samples": 0,
        "weekend_directional_acc": None,
        "weekend_directional_samples": 0,
        "weekend_directional_ties": 0,
    }
    if df is None or df.empty:
        return out
    try:
        models = load_models(include_modes=True, instrument_key=instrument_key)
    except Exception:
        return out
    if not isinstance(models, dict) or not models:
        return out

    try:
        ds_daily = build_next_session_dataset(df, instrument_key=instrument_key)
    except Exception:
        ds_daily = pd.DataFrame()
    if ds_daily is not None and not ds_daily.empty:
        ds_daily = ds_daily.sort_index().tail(max(2, int(max_daily_samples)))
        errs = []
        actuals = []
        dir_hits = []
        dir_ties = 0
        for ts, row in ds_daily.iterrows():
            try:
                subset = df[df.index <= ts]
                if subset is None or subset.empty or len(subset) < 300:
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
                pred_h = _preferred_pred_price(res.get("high", {}))
                pred_l = _preferred_pred_price(res.get("low", {}))
                close_v = _safe_float(row.get("close"))
                th = _safe_float(row.get("target_high_ret"))
                tl = _safe_float(row.get("target_low_ret"))
                if close_v is None:
                    continue
                if pred_h is not None and th is not None:
                    act_h = float(close_v * (1.0 + th))
                    errs.append(abs(pred_h - act_h))
                    actuals.append(act_h)
                    vote_h = majority_direction_vote(_model_pred_prices(res.get("high", {})), close_v)
                    dir_ties += int(vote_h.get("ties_excluded", 0) or 0)
                    if vote_h.get("direction") is not None and act_h != close_v:
                        dir_hits.append(1.0 if vote_h["direction"] == ("UP" if act_h > close_v else "DOWN") else 0.0)
                if pred_l is not None and tl is not None:
                    act_l = float(close_v * (1.0 + tl))
                    errs.append(abs(pred_l - act_l))
                    actuals.append(act_l)
                    vote_l = majority_direction_vote(_model_pred_prices(res.get("low", {})), close_v)
                    dir_ties += int(vote_l.get("ties_excluded", 0) or 0)
                    if vote_l.get("direction") is not None and act_l != close_v:
                        dir_hits.append(1.0 if vote_l["direction"] == ("UP" if act_l > close_v else "DOWN") else 0.0)
            except Exception:
                continue
        if errs:
            wm = summarize_price_window_errors(errs, actual_prices=actuals, horizon="daily", instrument_key=instrument_key)
            out["daily_mae_dollars"] = wm.get("mae_dollars")
            out["daily_window_ratio"] = wm.get("window_ratio_mean")
            out["daily_window_hit_rate"] = wm.get("window_hit_rate")
            out["daily_samples"] = int(len(errs))
            out["daily_directional_acc"] = float(np.mean(dir_hits)) if dir_hits else None
            out["daily_directional_samples"] = int(len(dir_hits))
            out["daily_directional_ties"] = int(dir_ties)

    try:
        ds_week = build_weekend_session_dataset(df, instrument_key=instrument_key)
    except Exception:
        ds_week = pd.DataFrame()
    if ds_week is not None and not ds_week.empty:
        ds_week = ds_week.sort_index().tail(max(2, int(max_week_samples)))
        errs_w = []
        actuals_w = []
        dir_hits_w = []
        dir_ties_w = 0
        for ts, row in ds_week.iterrows():
            try:
                subset = df[df.index <= ts]
                if subset is None or subset.empty or len(subset) < 300:
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
                pred_w = _preferred_pred_price(res.get("week", {}))
                close_v = _safe_float(row.get("close"))
                tw = _safe_float(row.get("target_weekend_ret"))
                if pred_w is None or close_v is None or tw is None:
                    continue
                act_w = float(close_v * (1.0 + tw))
                errs_w.append(abs(pred_w - act_w))
                actuals_w.append(act_w)
                vote_w = majority_direction_vote(_model_pred_prices(res.get("week", {})), close_v)
                dir_ties_w += int(vote_w.get("ties_excluded", 0) or 0)
                if vote_w.get("direction") is not None and act_w != close_v:
                    dir_hits_w.append(1.0 if vote_w["direction"] == ("UP" if act_w > close_v else "DOWN") else 0.0)
            except Exception:
                continue
        if errs_w:
            wm = summarize_price_window_errors(errs_w, actual_prices=actuals_w, horizon="weekend", instrument_key=instrument_key)
            out["weekend_mae_dollars"] = wm.get("mae_dollars")
            out["weekend_window_ratio"] = wm.get("window_ratio_mean")
            out["weekend_window_hit_rate"] = wm.get("window_hit_rate")
            out["weekend_samples"] = int(len(errs_w))
            out["weekend_directional_acc"] = float(np.mean(dir_hits_w)) if dir_hits_w else None
            out["weekend_directional_samples"] = int(len(dir_hits_w))
            out["weekend_directional_ties"] = int(dir_ties_w)

    return out


def evaluate_readiness_metrics(instrument_key: str, online: bool = False) -> dict:
    if instrument_key not in INSTRUMENTS:
        return {"instrument": instrument_key, "error": "unknown instrument"}
    df = _load_price_df(instrument_key, online=online)
    if df is None or df.empty:
        return {"instrument": instrument_key, "error": "no price data"}
    wf_splits = max(3, int(READINESS_BACKTEST_SPLITS))
    if online:
        wf = walk_forward_validate(
            df,
            n_splits=wf_splits,
            include_transaction_costs=False,
            instrument_key=instrument_key,
        )
    else:
        wf = _load_cached_walk_forward(instrument_key, df, wf_splits)
        if not wf:
            wf = walk_forward_validate(
                df,
                n_splits=wf_splits,
                include_transaction_costs=False,
                instrument_key=instrument_key,
            )
            _save_cached_walk_forward(instrument_key, df, wf_splits, wf)
    folds = int(wf.get("folds", 0) or 0)
    mean_dir = _safe_mean(wf.get("directional_acc", []))
    mean_majority_dir = _safe_mean(wf.get("majority_direction_acc", []))
    intraday_majority_samples = int(np.nansum(pd.to_numeric(wf.get("majority_direction_samples", []), errors="coerce"))) if isinstance(wf, dict) else 0
    intraday_majority_ties = int(np.nansum(pd.to_numeric(wf.get("majority_direction_ties", []), errors="coerce"))) if isinstance(wf, dict) else 0
    mean_net = _safe_mean(wf.get("net_returns", []))
    mean_sharpe = _safe_mean(wf.get("sharpe_like", []))
    mean_active = _safe_mean(wf.get("active_fraction", []))
    mean_trade_active = _safe_mean(wf.get("trade_active_fraction", []))
    live_h = _horizon_live_errors(instrument_key)
    live_4m = _intraday_live_error(instrument_key)
    bt_h = _horizon_backtest_errors(df, instrument_key, wf=wf)

    live_week = {
        "n": int(live_h.get("weekend_n", 0) or 0),
        "mae_dollars": live_h.get("weekend_mae_dollars"),
        "window_ratio_mean": live_h.get("weekend_window_ratio"),
        "window_hit_rate": live_h.get("weekend_window_hit_rate"),
    }
    bt_week = {
        "n": int(bt_h.get("weekend_samples", 0) or 0),
        "mae_dollars": bt_h.get("weekend_mae_dollars"),
        "window_ratio_mean": bt_h.get("weekend_window_ratio"),
        "window_hit_rate": bt_h.get("weekend_window_hit_rate"),
    }
    live_daily = {
        "n": int(live_h.get("daily_n", 0) or 0),
        "mae_dollars": live_h.get("daily_mae_dollars"),
        "window_ratio_mean": live_h.get("daily_window_ratio"),
        "window_hit_rate": live_h.get("daily_window_hit_rate"),
    }
    bt_daily = {
        "n": int(bt_h.get("daily_samples", 0) or 0),
        "mae_dollars": bt_h.get("daily_mae_dollars"),
        "window_ratio_mean": bt_h.get("daily_window_ratio"),
        "window_hit_rate": bt_h.get("daily_window_hit_rate"),
    }
    live_intr = {
        "n": int(live_4m.get("n", 0) or 0),
        "mae_dollars": live_4m.get("mae_dollars"),
        "window_ratio_mean": live_4m.get("window_ratio_mean"),
        "window_hit_rate": live_4m.get("window_hit_rate"),
        "window_threshold_mean_dollars": live_4m.get("window_threshold_mean_dollars"),
    }
    bt_intr = {
        "n": int(bt_h.get("intraday_folds", 0) or 0),
        "mae_dollars": bt_h.get("intraday_mae_dollars"),
        "window_ratio_mean": bt_h.get("intraday_window_ratio"),
        "window_hit_rate": bt_h.get("intraday_window_hit_rate"),
    }

    week_block, week_src, week_n = _pick_horizon_block(live_week, READINESS_MIN_WEEKEND_SAMPLES, bt_week)
    daily_block, daily_src, daily_n = _pick_horizon_block(live_daily, READINESS_MIN_DAILY_SAMPLES, bt_daily)
    intr_block, intr_src, intr_n = _pick_horizon_block(live_intr, READINESS_MIN_INTRADAY_SAMPLES, bt_intr)

    week_score = window_ratio_score(week_block.get("window_ratio_mean"))
    daily_score = window_ratio_score(daily_block.get("window_ratio_mean"))
    intr_score = window_ratio_score(intr_block.get("window_ratio_mean"))
    try:
        ref_price = float(pd.to_numeric(df.get("close"), errors="coerce").dropna().iloc[-1])
    except Exception:
        ref_price = None
    intr_target_window = price_window_threshold_dollars("intraday", instrument_key=instrument_key, reference_price=ref_price)
    daily_target_window = price_window_threshold_dollars("daily", instrument_key=instrument_key, reference_price=ref_price)
    weekend_target_window = price_window_threshold_dollars("weekend", instrument_key=instrument_key, reference_price=ref_price)
    w_week = max(0.0, float(READINESS_WEIGHT_WEEKEND))
    w_day = max(0.0, float(READINESS_WEIGHT_DAILY))
    w_intra = max(0.0, float(READINESS_WEIGHT_INTRADAY))
    weighted_num = 0.0
    weighted_den = 0.0
    if week_score is not None:
        weighted_num += w_week * week_score
        weighted_den += w_week
    if daily_score is not None:
        weighted_num += w_day * daily_score
        weighted_den += w_day
    if intr_score is not None:
        weighted_num += w_intra * intr_score
        weighted_den += w_intra
    horizon_score = float(weighted_num / weighted_den) if weighted_den > 0 else 0.0

    thresholds = {
        "min_folds": int(READINESS_MIN_FOLDS),
        "min_directional_acc": float(READINESS_MIN_DIRECTIONAL_ACC),
        "min_net_return": float(READINESS_MIN_NET_RETURN),
        "min_sharpe_like": float(READINESS_MIN_SHARPE_LIKE),
        "min_active_fraction": float(READINESS_MIN_ACTIVE_FRACTION),
        "intraday_window_dollars": float(READINESS_INTRADAY_WINDOW_DOLLARS),
        "intraday_window_pct": float(READINESS_INTRADAY_WINDOW_PCT),
        "daily_window_dollars": float(READINESS_DAILY_WINDOW_DOLLARS),
        "daily_window_pct": float(READINESS_DAILY_WINDOW_PCT),
        "weekend_window_dollars": float(READINESS_WEEKEND_WINDOW_DOLLARS),
        "weekend_window_pct": float(READINESS_WEEKEND_WINDOW_PCT),
        "min_intraday_window_hit_rate": float(READINESS_MIN_INTRADAY_WINDOW_HIT_RATE),
        "min_daily_window_hit_rate": float(READINESS_MIN_DAILY_WINDOW_HIT_RATE),
        "min_weekend_window_hit_rate": float(READINESS_MIN_WEEKEND_WINDOW_HIT_RATE),
        "legacy_max_weekend_err_pct": float(READINESS_MAX_WEEKEND_ERR_PCT),
        "legacy_max_daily_err_pct": float(READINESS_MAX_DAILY_ERR_PCT),
        "legacy_max_intraday_err_bps": float(READINESS_MAX_INTRADAY_ERR_BPS),
        "min_horizon_score": float(READINESS_MIN_HORIZON_SCORE),
    }
    req_active = bool(READINESS_REQUIRE_ACTIVE)
    week_hit_req = float(min_window_hit_rate("weekend", instrument_key=instrument_key))
    daily_hit_req = float(min_window_hit_rate("daily", instrument_key=instrument_key))
    intr_hit_req = float(min_window_hit_rate("intraday", instrument_key=instrument_key))
    week_mae = _safe_float(week_block.get("mae_dollars"))
    day_mae = _safe_float(daily_block.get("mae_dollars"))
    intr_mae = _safe_float(intr_block.get("mae_dollars"))
    checks = {
        "folds_ok": bool(folds >= thresholds["min_folds"]),
        "directional_ok": True,
        "net_ok": True,
        "sharpe_ok": True,
        "active_ok": bool((not req_active) or (mean_active is not None and mean_active >= thresholds["min_active_fraction"])),
        "weekend_discrepancy_ok": bool(week_mae is not None and week_mae <= float(weekend_target_window)),
        "daily_discrepancy_ok": bool(day_mae is not None and day_mae <= float(daily_target_window)),
        "intraday_discrepancy_ok": bool(intr_mae is not None and intr_mae <= float(intr_target_window)),
        "weekend_window_ratio_ok": bool(_safe_float(week_block.get("window_ratio_mean")) is not None and float(week_block["window_ratio_mean"]) <= 1.0),
        "weekend_window_hit_ok": bool(_safe_float(week_block.get("window_hit_rate")) is not None and float(week_block["window_hit_rate"]) >= week_hit_req),
        "daily_window_ratio_ok": bool(_safe_float(daily_block.get("window_ratio_mean")) is not None and float(daily_block["window_ratio_mean"]) <= 1.0),
        "daily_window_hit_ok": bool(_safe_float(daily_block.get("window_hit_rate")) is not None and float(daily_block["window_hit_rate"]) >= daily_hit_req),
        "intraday_window_ratio_ok": bool(_safe_float(intr_block.get("window_ratio_mean")) is not None and float(intr_block["window_ratio_mean"]) <= 1.0),
        "intraday_window_hit_ok": bool(_safe_float(intr_block.get("window_hit_rate")) is not None and float(intr_block["window_hit_rate"]) >= intr_hit_req),
        "horizon_score_ok": bool(horizon_score >= thresholds["min_horizon_score"]),
    }
    checks["weekend_window_ok"] = bool(checks["weekend_discrepancy_ok"])
    checks["daily_window_ok"] = bool(checks["daily_discrepancy_ok"])
    checks["intraday_window_ok"] = bool(checks["intraday_discrepancy_ok"])
    legacy_ok = bool(
        checks["folds_ok"]
        and checks["active_ok"]
    )
    if READINESS_USE_HORIZON_PRIORITY:
        horizon_ok = bool(
            checks["weekend_window_ok"]
            and checks["daily_window_ok"]
            and checks["intraday_window_ok"]
        )
        passed = bool(legacy_ok and horizon_ok)
    else:
        passed = bool(legacy_ok)
    return {
        "instrument": instrument_key,
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "passed": passed,
        "checks": checks,
        "metrics": {
            "folds": folds,
            "directional_acc_mean": mean_dir,
            "intraday_updown_acc": mean_majority_dir,
            "intraday_updown_samples": intraday_majority_samples,
            "intraday_updown_ties_excluded": intraday_majority_ties,
            "net_return_mean": mean_net,
            "sharpe_like_mean": mean_sharpe,
            "active_fraction_mean": mean_active,
            "trade_active_fraction_mean": mean_trade_active,
            "horizon_weighted_score": horizon_score,
            "weekend_mae_dollars": week_mae,
            "weekend_avg_discrepancy_dollars": week_mae,
            "weekend_window_ratio": _safe_float(week_block.get("window_ratio_mean")),
            "weekend_window_hit_rate": _safe_float(week_block.get("window_hit_rate")),
            "weekend_target_window_dollars": float(weekend_target_window),
            "weekend_success": bool(checks["weekend_window_ok"]),
            "weekend_updown_acc": _safe_float(week_block.get("directional_acc")),
            "weekend_updown_samples": int(week_block.get("directional_samples", 0) or 0),
            "weekend_updown_ties_excluded": int(week_block.get("directional_ties", 0) or 0),
            "daily_mae_dollars": day_mae,
            "daily_avg_discrepancy_dollars": day_mae,
            "daily_window_ratio": _safe_float(daily_block.get("window_ratio_mean")),
            "daily_window_hit_rate": _safe_float(daily_block.get("window_hit_rate")),
            "daily_target_window_dollars": float(daily_target_window),
            "daily_success": bool(checks["daily_window_ok"]),
            "daily_updown_acc": _safe_float(daily_block.get("directional_acc")),
            "daily_updown_samples": int(daily_block.get("directional_samples", 0) or 0),
            "daily_updown_ties_excluded": int(daily_block.get("directional_ties", 0) or 0),
            "intraday_mae_dollars": intr_mae,
            "intraday_avg_discrepancy_dollars": intr_mae,
            "intraday_window_ratio": _safe_float(intr_block.get("window_ratio_mean")),
            "intraday_window_hit_rate": _safe_float(intr_block.get("window_hit_rate")),
            "intraday_target_window_dollars": float(intr_target_window),
            "intraday_success": bool(checks["intraday_window_ok"]),
            "intraday_price_mae_backtest": _safe_mean(wf.get("price_mae_dollars")),
            "intraday_price_window_hit_rate_backtest": _safe_mean(wf.get("price_window_hit_rate")),
            "intraday_price_window_ratio_backtest": _safe_mean(wf.get("price_window_ratio")),
        },
        "sources": {
            "weekend": {"source": week_src, "live_samples": int(week_n), "live_min_samples": int(READINESS_MIN_WEEKEND_SAMPLES)},
            "daily": {"source": daily_src, "live_samples": int(daily_n), "live_min_samples": int(READINESS_MIN_DAILY_SAMPLES)},
            "intraday": {"source": intr_src, "live_samples": int(intr_n), "live_min_samples": int(READINESS_MIN_INTRADAY_SAMPLES)},
        },
        "horizon_live": live_h,
        "horizon_backtest": bt_h,
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
        "sources": eval_row.get("sources", {}),
        "horizon_live": eval_row.get("horizon_live", {}),
        "horizon_backtest": eval_row.get("horizon_backtest", {}),
        "evaluation": {
            "passed": bool(eval_row.get("passed", False)),
            "checks": eval_row.get("checks", {}),
            "metrics": eval_row.get("metrics", {}),
            "sources": eval_row.get("sources", {}),
            "horizon_live": eval_row.get("horizon_live", {}),
            "horizon_backtest": eval_row.get("horizon_backtest", {}),
            "thresholds": eval_row.get("thresholds", {}),
        },
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
