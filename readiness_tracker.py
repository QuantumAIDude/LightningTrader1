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
    READINESS_MAX_WEEKEND_ERR_PCT,
    READINESS_MAX_DAILY_ERR_PCT,
    READINESS_MAX_INTRADAY_ERR_BPS,
    READINESS_MIN_WEEKEND_SAMPLES,
    READINESS_MIN_DAILY_SAMPLES,
    READINESS_MIN_INTRADAY_SAMPLES,
    TS_SPLIT_GAP_BARS,
)
from data_fetch import update_price_data
from features import build_features_for_training, build_next_session_dataset, build_next_week_dataset, DEFAULT_FEATURES
from model_trainers import walk_forward_validate
from model_utils import load_models, predict_horizon


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


def _ts_split(n_splits: int):
    gap = max(0, int(TS_SPLIT_GAP_BARS))
    try:
        return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    except TypeError:
        return TimeSeriesSplit(n_splits=n_splits)


def _ridge_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])


def _cv_abs_ret_mae(dataset: pd.DataFrame, target_norm_col: str, target_ret_col: str, min_rows: int = 80) -> dict:
    if dataset is None or dataset.empty or len(dataset) < max(6, int(min_rows)):
        return {"folds": 0, "mae_ret": None, "mae_pct": None}
    if target_norm_col not in dataset.columns or target_ret_col not in dataset.columns:
        return {"folds": 0, "mae_ret": None, "mae_pct": None}
    feats = [f for f in DEFAULT_FEATURES if f in dataset.columns]
    if not feats:
        return {"folds": 0, "mae_ret": None, "mae_pct": None}
    X = dataset[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_norm = pd.to_numeric(dataset[target_norm_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_ret = pd.to_numeric(dataset[target_ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    vol = pd.to_numeric(dataset.get("ewm_vol"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.001).clip(lower=1e-6)
    n = len(X)
    scale = max(6, int(max(8, min_rows) // 2))
    n_splits = min(max(2, int(READINESS_BACKTEST_SPLITS)), max(2, n // scale))
    if n_splits < 2:
        return {"folds": 0, "mae_ret": None, "mae_pct": None}
    tss = _ts_split(n_splits)
    maes_ret = []
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
        except Exception:
            continue
    if not maes_ret:
        return {"folds": 0, "mae_ret": None, "mae_pct": None}
    mae_ret = float(np.nanmean(maes_ret))
    return {"folds": int(len(maes_ret)), "mae_ret": mae_ret, "mae_pct": float(mae_ret * 100.0)}


def _horizon_live_errors(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"horizon_accuracy_{instrument_key}.json")
    payload = _load_json(path)
    by_h = payload.get("by_horizon", {}) if isinstance(payload, dict) else {}

    def _agg_h(h_name: str):
        block = by_h.get(h_name, {}) if isinstance(by_h, dict) else {}
        models = block.get("models", {}) if isinstance(block, dict) else {}
        mape_vals = []
        n_vals = []
        for v in models.values():
            if not isinstance(v, dict):
                continue
            mp = _safe_float(v.get("mape_pct"))
            nn = v.get("n", 0)
            try:
                nn = int(nn)
            except Exception:
                nn = 0
            if mp is not None:
                mape_vals.append(float(mp))
            n_vals.append(max(0, int(nn)))
        mape = float(np.nanmedian(mape_vals)) if mape_vals else None
        n_max = int(max(n_vals)) if n_vals else 0
        return mape, n_max

    week_mape, week_n = _agg_h("weekend")
    dh_mape, dh_n = _agg_h("daily_high")
    dl_mape, dl_n = _agg_h("daily_low")
    daily_vals = [v for v in (dh_mape, dl_mape) if v is not None]
    daily_mape = float(np.mean(daily_vals)) if daily_vals else None
    daily_n = int(max(dh_n, dl_n))
    return {
        "weekend_mape_pct": week_mape,
        "weekend_n": week_n,
        "daily_mape_pct": daily_mape,
        "daily_n": daily_n,
    }


def _intraday_live_error(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"calibration_live_{instrument_key}.json")
    payload = _load_json(path)
    errs = payload.get("recent_abs_errors", []) if isinstance(payload, dict) else []
    vals = []
    for e in errs:
        ef = _safe_float(e)
        if ef is not None:
            vals.append(float(ef))
    if vals:
        med_ret = float(np.nanmedian(vals))
        return {
            "n": int(len(vals)),
            "abs_err_bps": float(med_ret * 10000.0),
        }
    # No recent resolved error rows yet: force fallback to fresh backtest estimate.
    return {"n": 0, "abs_err_bps": None}


def _horizon_backtest_errors(df: pd.DataFrame, instrument_key: str) -> dict:
    out = {
        "daily_mape_pct": None,
        "daily_folds": 0,
        "daily_samples": 0,
        "weekend_mape_pct": None,
        "weekend_folds": 0,
        "weekend_samples": 0,
        "intraday_err_bps": None,
        "intraday_folds": 0,
    }
    if df is None or df.empty:
        return out
    # Prefer replay against the actual horizon prediction stack.
    try:
        rep = _horizon_replay_errors(df, instrument_key=instrument_key)
        if rep.get("daily_mape_pct") is not None:
            out["daily_mape_pct"] = float(rep["daily_mape_pct"])
            out["daily_samples"] = int(rep.get("daily_samples", 0) or 0)
            out["daily_folds"] = out["daily_samples"]
        if rep.get("weekend_mape_pct") is not None:
            out["weekend_mape_pct"] = float(rep["weekend_mape_pct"])
            out["weekend_samples"] = int(rep.get("weekend_samples", 0) or 0)
            out["weekend_folds"] = out["weekend_samples"]
    except Exception:
        pass
    try:
        ds_daily = build_next_session_dataset(df, instrument_key=instrument_key)
        dh = _cv_abs_ret_mae(ds_daily, "target_high_ret_norm", "target_high_ret", min_rows=20)
        dl = _cv_abs_ret_mae(ds_daily, "target_low_ret_norm", "target_low_ret", min_rows=20)
        dv = [v for v in (dh.get("mae_pct"), dl.get("mae_pct")) if v is not None]
        if dv and out["daily_mape_pct"] is None:
            out["daily_mape_pct"] = float(np.mean(dv))
            out["daily_folds"] = int(min(dh.get("folds", 0), dl.get("folds", 0)))
    except Exception:
        pass
    try:
        ds_week = build_next_week_dataset(df, instrument_key=instrument_key)
        wk = _cv_abs_ret_mae(ds_week, "target_weekend_ret_norm", "target_weekend_ret", min_rows=8)
        if out["weekend_mape_pct"] is None:
            out["weekend_mape_pct"] = wk.get("mae_pct")
            out["weekend_folds"] = int(wk.get("folds", 0) or 0)
    except Exception:
        pass
    try:
        ds_4m = build_features_for_training(df, predict_minutes=4, instrument_key=instrument_key)
        intr = _cv_abs_ret_mae(ds_4m, "target_ret_norm", "target_ret", min_rows=120)
        if intr.get("mae_ret") is not None:
            out["intraday_err_bps"] = float(intr["mae_ret"] * 10000.0)
            out["intraday_folds"] = int(intr.get("folds", 0) or 0)
    except Exception:
        pass
    return out


def _score_error(err_value, err_limit):
    ev = _safe_float(err_value)
    lim = _safe_float(err_limit)
    if ev is None or lim is None or lim <= 0:
        return None
    return float(max(0.0, 1.0 - (ev / (2.0 * lim))))


def _pick_err(live_val, live_n: int, live_min_n: int, bt_val):
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


def _pred_price_mean(block: dict) -> float | None:
    if not isinstance(block, dict):
        return None
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


def _horizon_replay_errors(df: pd.DataFrame, instrument_key: str, max_daily_samples: int = 10, max_week_samples: int = 6) -> dict:
    out = {"daily_mape_pct": None, "daily_samples": 0, "weekend_mape_pct": None, "weekend_samples": 0}
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
                pred_h = _pred_price_mean(res.get("high", {}))
                pred_l = _pred_price_mean(res.get("low", {}))
                close_v = _safe_float(row.get("close"))
                th = _safe_float(row.get("target_high_ret"))
                tl = _safe_float(row.get("target_low_ret"))
                if close_v is None:
                    continue
                if pred_h is not None and th is not None:
                    act_h = float(close_v * (1.0 + th))
                    if abs(act_h) > 1e-9:
                        errs.append(abs(pred_h - act_h) / abs(act_h) * 100.0)
                if pred_l is not None and tl is not None:
                    act_l = float(close_v * (1.0 + tl))
                    if abs(act_l) > 1e-9:
                        errs.append(abs(pred_l - act_l) / abs(act_l) * 100.0)
            except Exception:
                continue
        if errs:
            out["daily_mape_pct"] = float(np.mean(errs))
            out["daily_samples"] = int(len(errs))

    try:
        ds_week = build_next_week_dataset(df, instrument_key=instrument_key)
    except Exception:
        ds_week = pd.DataFrame()
    if ds_week is not None and not ds_week.empty:
        ds_week = ds_week.sort_index().tail(max(2, int(max_week_samples)))
        errs_w = []
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
                pred_w = _pred_price_mean(res.get("week", {}))
                close_v = _safe_float(row.get("close"))
                tw = _safe_float(row.get("target_weekend_ret"))
                if pred_w is None or close_v is None or tw is None:
                    continue
                act_w = float(close_v * (1.0 + tw))
                if abs(act_w) > 1e-9:
                    errs_w.append(abs(pred_w - act_w) / abs(act_w) * 100.0)
            except Exception:
                continue
        if errs_w:
            out["weekend_mape_pct"] = float(np.mean(errs_w))
            out["weekend_samples"] = int(len(errs_w))

    return out


def evaluate_readiness_metrics(instrument_key: str, online: bool = False) -> dict:
    if instrument_key not in INSTRUMENTS:
        return {"instrument": instrument_key, "error": "unknown instrument"}
    df = _load_price_df(instrument_key, online=online)
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
    live_h = _horizon_live_errors(instrument_key)
    live_4m = _intraday_live_error(instrument_key)
    bt_h = _horizon_backtest_errors(df, instrument_key)

    week_err, week_src, week_n = _pick_err(
        live_h.get("weekend_mape_pct"),
        live_h.get("weekend_n", 0),
        READINESS_MIN_WEEKEND_SAMPLES,
        bt_h.get("weekend_mape_pct"),
    )
    daily_err, daily_src, daily_n = _pick_err(
        live_h.get("daily_mape_pct"),
        live_h.get("daily_n", 0),
        READINESS_MIN_DAILY_SAMPLES,
        bt_h.get("daily_mape_pct"),
    )
    intr_err, intr_src, intr_n = _pick_err(
        live_4m.get("abs_err_bps"),
        live_4m.get("n", 0),
        READINESS_MIN_INTRADAY_SAMPLES,
        bt_h.get("intraday_err_bps"),
    )

    week_score = _score_error(week_err, READINESS_MAX_WEEKEND_ERR_PCT)
    daily_score = _score_error(daily_err, READINESS_MAX_DAILY_ERR_PCT)
    intr_score = _score_error(intr_err, READINESS_MAX_INTRADAY_ERR_BPS)
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
        "max_weekend_err_pct": float(READINESS_MAX_WEEKEND_ERR_PCT),
        "max_daily_err_pct": float(READINESS_MAX_DAILY_ERR_PCT),
        "max_intraday_err_bps": float(READINESS_MAX_INTRADAY_ERR_BPS),
        "min_horizon_score": float(READINESS_MIN_HORIZON_SCORE),
    }
    req_dir = bool(READINESS_REQUIRE_DIRECTIONAL)
    req_profit = bool(READINESS_REQUIRE_PROFITABILITY)
    req_sharpe = bool(READINESS_REQUIRE_SHARPE)
    req_active = bool(READINESS_REQUIRE_ACTIVE)
    checks = {
        "folds_ok": bool(folds >= thresholds["min_folds"]),
        "directional_ok": bool((not req_dir) or (mean_dir is not None and mean_dir >= thresholds["min_directional_acc"])),
        "net_ok": bool((not req_profit) or (mean_net is not None and mean_net >= thresholds["min_net_return"])),
        "sharpe_ok": bool((not req_sharpe) or (mean_sharpe is not None and mean_sharpe >= thresholds["min_sharpe_like"])),
        "active_ok": bool((not req_active) or (mean_active is not None and mean_active >= thresholds["min_active_fraction"])),
        "weekend_window_ok": bool(week_err is not None and week_err <= thresholds["max_weekend_err_pct"]),
        "daily_window_ok": bool(daily_err is not None and daily_err <= thresholds["max_daily_err_pct"]),
        "intraday_window_ok": bool(intr_err is not None and intr_err <= thresholds["max_intraday_err_bps"]),
        "horizon_score_ok": bool(horizon_score >= thresholds["min_horizon_score"]),
    }
    legacy_ok = bool(
        checks["folds_ok"]
        and checks["directional_ok"]
        and checks["net_ok"]
        and checks["sharpe_ok"]
        and checks["active_ok"]
    )
    if READINESS_USE_HORIZON_PRIORITY:
        horizon_ok = bool(
            checks["weekend_window_ok"]
            and checks["daily_window_ok"]
            and checks["intraday_window_ok"]
            and checks["horizon_score_ok"]
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
            "net_return_mean": mean_net,
            "sharpe_like_mean": mean_sharpe,
            "active_fraction_mean": mean_active,
            "horizon_weighted_score": horizon_score,
            "weekend_err_pct": week_err,
            "daily_err_pct": daily_err,
            "intraday_err_bps": intr_err,
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
