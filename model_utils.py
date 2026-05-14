# model_utils.py
import os
import joblib
import math
import json
import threading
import numpy as np
import pandas as pd
import logging
import warnings
from config import (
    MODEL_DIR,
    BASE_MODEL_FILE,
    FOREST_MODEL_FILE,
    BOOST_MODEL_FILE,
    EXTRA_TREES_MODEL_FILE,
    REGIME_MODEL_FILE,
    META_MODEL_FILE,
    INSTRUMENTS,
    INTERVAL,
    SYMBOL,
    DATA_DIR,
    LIVE_CALIBRATION,
    EXTRA_TREES_N_ESTIMATORS,
    EXTRA_TREES_MAX_DEPTH,
    EXTRA_TREES_MIN_SAMPLES_LEAF,
    ALLOW_REMOTE_DAILY,
    DEFAULT_INSTRUMENT,
    MAX_ENSEMBLE_MEMBERS_PREDICT,
    AUTO_FAST_RETRAIN_ON_DRIFT,
    DRIFT_RETRAIN_COOLDOWN_MIN,
    TRANSACTION_COST_PER_TRADE,
    SLIPPAGE_PCT,
    LIVE_COST_MULTIPLIER,
    LIVE_MIN_EDGE_BPS,
    LIVE_UNCERTAINTY_EDGE_MULT,
    DROP_INCOMPLETE_LAST_BAR,
    INCOMPLETE_BAR_GRACE_SEC,
    LIVE_BIAS_CORRECTION,
    LIVE_BIAS_MIN_SAMPLES,
    LIVE_BIAS_WEIGHT_MAX,
    LIVE_BIAS_MAX_ABS_MULT,
    AUTO_HORIZON_RETRAIN_ON_DRIFT,
    HORIZON_DRIFT_RETRAIN_COOLDOWN_MIN,
    HORIZON_ACCURACY_UPDATE_COOLDOWN_SEC,
    HORIZON_ADAPT_MIN_TOTAL_SAMPLES,
    HORIZON_ADAPT_MIN_SAMPLES_PER_HORIZON,
    READINESS_ENFORCE_LIVE_GATE,
    READINESS_MIN_PASS_STREAK_DAYS,
    READINESS_MAX_AGE_HOURS,
    REGIME_LOOK_HOURS_4MIN,
    REGIME_LOOK_HOURS_DAILY,
    REGIME_LOOK_HOURS_WEEKEND,
)
try:
    from config import LOG_PREDICTIONS
except Exception:
    LOG_PREDICTIONS = False
from features import (
    make_feature_matrix,
    build_feature_frame,
    build_features_for_training,
    build_next_session_dataset,
    DEFAULT_FEATURES,
    LIVE_MAX_BARS,
)
from forecast_calibration import apply_intraday_activity_gate, apply_return_calibration
from horizon_overlay import apply_daily_pair_overlay, apply_horizon_overlay
from sklearn.ensemble import ExtraTreesRegressor
from regime_model import predict_regime_probs
from datetime import datetime, timedelta, timezone, time
import yfinance as yf

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore", message="X does not have valid feature names", category=UserWarning)

# Safety caps for live prediction (4min)
MAX_ABS_RET = 0.01  # fallback cap if vol is missing
STALE_MULTIPLIER = 3.5
MIN_STALE_SECONDS = 120
NO_SIGNAL_CONF = 0.40
NO_SIGNAL_DISP = 3.0
TENTATIVE_DIRECTION_CONF_PCT = 20.0
TENTATIVE_DIRECTION_SIZE_MULT = 0.50
STOCK_HORIZON_SUPPORT_CONF_MAX_BONUS = 8.0
STOCK_HORIZON_SUPPORT_SIZE_MAX_MULT = 1.10
STOCK_HORIZON_SUPPORT_SIZE_MIN_MULT = 0.60

_LIVE_CAL_CACHE = {}
_LAST_CAL_UPDATE = {}
_EXTRA_TREES_AUTO_ATTEMPTED = False
_MODEL_CACHE = {}
_CAL_UPDATE_LOCK = threading.Lock()
_CAL_UPDATE_IN_FLIGHT = set()
_DRIFT_RETRAIN_LOCK = threading.Lock()
_DRIFT_RETRAIN_IN_FLIGHT = set()
_LAST_DRIFT_RETRAIN = {}
_HORIZON_ACC_LOCK = threading.Lock()
_HORIZON_ACC_IN_FLIGHT = set()
_LAST_HORIZON_ACC_UPDATE = {}
_HORIZON_RETRAIN_LOCK = threading.Lock()
_HORIZON_RETRAIN_IN_FLIGHT = set()
_LAST_HORIZON_RETRAIN = {}
_READINESS_CACHE = {}

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

BAR_MINUTES = _interval_minutes(INTERVAL)


def _clear_model_cache(instrument_key: str = None):
    instr = str(instrument_key or DEFAULT_INSTRUMENT)
    for include_modes in (True, False):
        try:
            _MODEL_CACHE.pop((include_modes, instr), None)
        except Exception:
            pass


def _apply_summary_calibration(summary: dict | None, calibration: dict | None):
    if not isinstance(summary, dict):
        return summary
    if not isinstance(calibration, dict):
        return summary
    try:
        scale = float(calibration.get("scale", 1.0))
    except Exception:
        scale = 1.0
    try:
        bias = float(calibration.get("bias", 0.0))
    except Exception:
        bias = 0.0
    if (not np.isfinite(scale)) or (not np.isfinite(bias)):
        return summary
    out = dict(summary)
    for key in ("mean", "q10", "q50", "q90"):
        if key in out:
            try:
                out[key] = float((float(out[key]) * scale) + bias)
            except Exception:
                pass
    try:
        std = abs(float(scale)) * float(out.get("std", 0.0))
        out["std"] = float(max(1e-6, std))
    except Exception:
        pass
    try:
        q10 = float(out.get("q10", out.get("mean", 0.0)))
        q90 = float(out.get("q90", out.get("mean", 0.0)))
        if q10 > q90:
            out["q10"], out["q90"] = q90, q10
    except Exception:
        pass
    try:
        mean_val = float(out.get("mean", 0.0))
        std_val = max(1e-6, float(out.get("std", 1e-6)))
        z = mean_val / std_val
        out["p_up"] = float(max(0.001, min(0.999, 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))))
    except Exception:
        pass
    return out


def _restore_summary_target_mean(summary: dict | None, payload: dict | None):
    if not isinstance(summary, dict):
        return summary
    if not isinstance(payload, dict):
        return summary
    meta = dict(payload.get("meta") or {})
    if not bool(meta.get("demeaned")):
        return summary
    try:
        target_mean = float(meta.get("target_mean", 0.0))
    except Exception:
        target_mean = 0.0
    if (not np.isfinite(target_mean)) or abs(target_mean) <= 1e-12:
        return summary
    out = dict(summary)
    for key in ("mean", "q10", "q50", "q90"):
        if key in out:
            try:
                out[key] = float(out[key]) + float(target_mean)
            except Exception:
                pass
    try:
        mean_val = float(out.get("mean", 0.0))
        std_val = max(1e-6, float(out.get("std", 1e-6)))
        z = mean_val / std_val
        out["p_up"] = float(max(0.001, min(0.999, 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))))
    except Exception:
        pass
    return out


def _apply_fallback_return_calibration(summary: dict | None, calibration: dict | None):
    if not isinstance(summary, dict):
        return summary
    if not isinstance(calibration, dict):
        return summary
    try:
        selected = str(calibration.get("selected") or "").strip().lower()
    except Exception:
        selected = ""
    try:
        scale = float(calibration.get("scale", 1.0))
    except Exception:
        scale = 1.0
    try:
        bias = float(calibration.get("bias", 0.0))
    except Exception:
        bias = 0.0
    if selected in {"anchor_only", "insufficient_samples", "no_material_edge", "exception"} and abs(scale) <= 1e-12 and abs(bias) <= 1e-12:
        return summary
    return _apply_summary_calibration(summary, calibration)


def _resolve_predictability_gate(models_dict: dict | None):
    if not isinstance(models_dict, dict):
        return None
    for key in ("base", "forest", "boost", "extra_trees", "meta"):
        payload = models_dict.get(key)
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta", {})
        if not isinstance(meta, dict):
            continue
        gate = meta.get("predictability_gate")
        if isinstance(gate, dict):
            try:
                scale = float(gate.get("scale"))
            except Exception:
                scale = None
            if scale is not None and np.isfinite(scale) and scale > 0.0:
                return gate
    return None

def _live_feature_frame(df_latest: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    """Build a base feature frame for live inference using a capped history window."""
    if df_latest is None or df_latest.empty:
        return pd.DataFrame()
    df_use = df_latest
    try:
        if len(df_use) > LIVE_MAX_BARS:
            df_use = df_use.iloc[-LIVE_MAX_BARS:].copy()
    except Exception:
        pass
    try:
        frame = build_feature_frame(df_use, instrument_key=instrument_key)
    except Exception:
        return pd.DataFrame()
    try:
        frame = frame.replace([np.inf, -np.inf], np.nan)
    except Exception:
        pass
    return frame

def _feature_row_from_frame(frame: pd.DataFrame, feature_list) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    feats = feature_list or DEFAULT_FEATURES
    try:
        feats = list(feats)
    except Exception:
        feats = DEFAULT_FEATURES
    # Ensure all requested features exist
    try:
        missing = [f for f in feats if f not in frame.columns]
        if missing:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    try:
        sub = frame.dropna(subset=feats)
    except Exception:
        sub = frame
    if sub is None or sub.empty:
        return pd.DataFrame()
    try:
        return sub[feats].iloc[-1:].copy()
    except Exception:
        return pd.DataFrame()

def _get_feature_row_cached(df_latest: pd.DataFrame, feature_list, instrument_key: str = None,
                            feature_frame: pd.DataFrame | None = None, cache: dict | None = None) -> pd.DataFrame:
    feats = feature_list or DEFAULT_FEATURES
    try:
        key = tuple(feats)
    except Exception:
        key = None
    if cache is not None and key in cache:
        return cache[key]
    X = pd.DataFrame()
    if feature_frame is not None and not feature_frame.empty:
        X = _feature_row_from_frame(feature_frame, feats)
    if X is None or X.empty:
        try:
            X = make_feature_matrix(df_latest, feature_list=feats, instrument_key=instrument_key)
        except Exception:
            X = pd.DataFrame()
    if cache is not None:
        cache[key] = X
    return X

def _instrument_meta(instrument_key: str = None) -> dict:
    key = instrument_key or SYMBOL
    return INSTRUMENTS.get(key, {})

def _vol_floor_from_meta(instr_meta: dict) -> float:
    try:
        v = float(instr_meta.get("vol_floor", 0.0))
    except Exception:
        v = 0.0
    if v > 0:
        return v
    instr_type = instr_meta.get("type", "equity")
    return 0.00045 if instr_type == "futures" else 0.00025

def _infer_target_norm(meta: dict) -> bool:
    """
    Infer whether a model's outputs are volatility-normalized.
    Defaults to True for modern models; legacy models are inferred by resid_std scale.
    """
    if not isinstance(meta, dict):
        return True
    if "target_norm" in meta:
        try:
            return bool(meta.get("target_norm"))
        except Exception:
            return True
    try:
        resid = float(meta.get("resid_std", float("nan")))
        if np.isfinite(resid):
            # raw 4-min returns are ~1e-4..1e-2; normalized targets are ~1e-1..10
            return resid > 0.05
    except Exception:
        pass
    return True

def _round_to_tick(price: float, instr_meta: dict) -> float:
    try:
        tick = float(instr_meta.get("tick_size", 0.0))
    except Exception:
        tick = 0.0
    if tick and tick > 0:
        return float(round(price / tick) * tick)
    return float(price)

def _weighted_quantile(values, quantile: float, weights=None):
    try:
        vals = np.asarray(values, dtype=float)
        if vals.size == 0:
            return np.nan
        mask = np.isfinite(vals)
        vals = vals[mask]
        if vals.size == 0:
            return np.nan
        if weights is None:
            return float(np.quantile(vals, max(0.0, min(1.0, quantile))))
        w = np.asarray(weights, dtype=float)
        if w.size != mask.size:
            w = np.ones_like(vals, dtype=float)
        else:
            w = w[mask]
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        if float(np.sum(w)) <= 0:
            return float(np.quantile(vals, max(0.0, min(1.0, quantile))))
        order = np.argsort(vals)
        vals_s = vals[order]
        w_s = w[order]
        cum = np.cumsum(w_s)
        target = max(0.0, min(1.0, quantile)) * float(cum[-1])
        idx = int(np.searchsorted(cum, target, side="left"))
        idx = min(max(idx, 0), len(vals_s) - 1)
        return float(vals_s[idx])
    except Exception:
        return np.nan

def _daily_range_priors(df_hist: pd.DataFrame, instrument_key: str = None, lookback_sessions: int = 180):
    """
    Build robust priors for next-session high/low returns from prior close.
    Priors are used to keep daily high/low forecasts realistic when ML outputs collapse.
    """
    try:
        ds = build_next_session_dataset(df_hist, instrument_key=instrument_key)
    except Exception:
        return None
    if ds is None or ds.empty:
        return None
    try:
        use = ds[["target_high_ret", "target_low_ret"]].replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        return None
    if use.empty:
        return None
    if lookback_sessions and len(use) > int(lookback_sessions):
        use = use.tail(int(lookback_sessions))
    if len(use) < 12:
        return None
    high = pd.to_numeric(use["target_high_ret"], errors="coerce").values
    low = pd.to_numeric(use["target_low_ret"], errors="coerce").values
    mask = np.isfinite(high) & np.isfinite(low)
    high = high[mask]
    low = low[mask]
    if len(high) < 12:
        return None
    rng = np.maximum(high - low, 0.0)
    n = len(high)
    # Recent samples receive more weight.
    weights = np.linspace(0.7, 1.3, n, dtype=float)
    high_med = _weighted_quantile(high, 0.50, weights)
    low_med = _weighted_quantile(low, 0.50, weights)
    high_q65 = _weighted_quantile(high, 0.65, weights)
    high_q75 = _weighted_quantile(high, 0.75, weights)
    low_q35 = _weighted_quantile(low, 0.35, weights)
    low_q25 = _weighted_quantile(low, 0.25, weights)
    high_prior = 0.65 * high_med + 0.35 * high_q65
    low_prior = 0.65 * low_med + 0.35 * low_q35
    mid_med = _weighted_quantile(0.5 * (high + low), 0.50, weights)
    range_q45 = _weighted_quantile(rng, 0.45, weights)
    range_q60 = _weighted_quantile(rng, 0.60, weights)
    range_q92 = _weighted_quantile(rng, 0.92, weights)
    # Volatility floor prevents unrealistically narrow high/low spread.
    try:
        close = pd.to_numeric(df_hist.get("close"), errors="coerce")
        ret = close.pct_change().dropna()
        rv = float(ret.tail(max(20, int(240 // max(1, BAR_MINUTES)))).std()) if not ret.empty else 0.0
    except Exception:
        rv = 0.0
    vol_floor = max(0.0015, 2.2 * max(0.0, rv))
    min_range = max(float(range_q45 if np.isfinite(range_q45) else 0.0), vol_floor)
    med_range = max(float(range_q60 if np.isfinite(range_q60) else min_range), min_range)
    max_range = max(float(range_q92 if np.isfinite(range_q92) else med_range * 1.8), med_range)
    try:
        if high_prior < low_prior:
            high_prior, low_prior = low_prior, high_prior
    except Exception:
        pass
    try:
        down_share = abs(float(low_q25)) / (abs(float(low_q25)) + abs(float(high_q75)) + 1e-9)
    except Exception:
        down_share = 0.55
    down_share = float(max(0.50, min(0.72, down_share)))
    return {
        "high_ret_prior": float(high_prior),
        "low_ret_prior": float(low_prior),
        "low_tail_ret_prior": float(low_q25 if np.isfinite(low_q25) else low_prior),
        "mid_ret_prior": float(mid_med if np.isfinite(mid_med) else 0.5 * (high_prior + low_prior)),
        "down_share": float(down_share),
        "min_range_ret": float(min_range),
        "med_range_ret": float(med_range),
        "max_range_ret": float(max_range),
        "n_samples": int(n),
    }

def _enforce_daily_high_low_consistency(high_block: dict, low_block: dict, base_price: float, instr_meta: dict,
                                        priors: dict | None):
    if not isinstance(high_block, dict) or not isinstance(low_block, dict):
        return
    if not isinstance(base_price, (int, float)) or float(base_price) <= 0:
        return
    if not isinstance(priors, dict):
        return
    try:
        min_range = max(0.0005, float(priors.get("min_range_ret", 0.002)))
        med_range = max(min_range, float(priors.get("med_range_ret", min_range)))
        max_range = max(med_range, float(priors.get("max_range_ret", med_range * 2.0)))
        high_prior = float(priors.get("high_ret_prior", med_range))
        low_prior = float(priors.get("low_ret_prior", -med_range))
        low_tail = float(priors.get("low_tail_ret_prior", low_prior))
        mid_prior = float(priors.get("mid_ret_prior", 0.5 * (high_prior + low_prior)))
        down_share = float(priors.get("down_share", 0.55))
        down_share = max(0.50, min(0.72, down_share))
    except Exception:
        return
    try:
        tick = float(instr_meta.get("tick_size", 0.0) or 0.0)
    except Exception:
        tick = 0.0
    if tick <= 0:
        tick = 0.01
    for key in ("base", "forest", "boost", "extra_trees"):
        h = high_block.get(key)
        l = low_block.get(key)
        if not isinstance(h, dict) or not isinstance(l, dict):
            continue
        key_adj = {
            "base": -0.015,
            "forest": -0.005,
            "boost": 0.005,
            "extra_trees": 0.015,
        }.get(key, 0.0)
        hp = h.get("pred_price")
        lp = l.get("pred_price")
        if not isinstance(hp, (int, float)) or not isinstance(lp, (int, float)):
            continue
        try:
            h_conf = float(h.get("confidence", 0.0)) / 100.0
        except Exception:
            h_conf = 0.0
        try:
            l_conf = float(l.get("confidence", 0.0)) / 100.0
        except Exception:
            l_conf = 0.0
        # More confidence -> keep more ML signal, otherwise pull toward historical prior.
        w_h = max(0.25, min(0.90, h_conf + key_adj))
        w_l = max(0.18, min(0.85, (l_conf * 0.88) - (0.8 * key_adj)))
        model_down_share = max(0.48, min(0.75, down_share + (0.5 * key_adj)))
        ret_h_ml = float(hp / base_price) - 1.0
        ret_l_ml = float(lp / base_price) - 1.0
        ret_h = (w_h * ret_h_ml) + ((1.0 - w_h) * high_prior)
        ret_l = (w_l * ret_l_ml) + ((1.0 - w_l) * low_prior)
        # Low-side forecasts are often underestimating downside tails; blend a tail prior when confidence is weak.
        tail_mix = max(0.0, min(0.45, (0.55 - l_conf)))
        ret_l = (1.0 - tail_mix) * ret_l + tail_mix * low_tail
        spread = ret_h - ret_l
        if spread < min_range:
            mid = (0.65 * (0.5 * (ret_h + ret_l))) + (0.35 * mid_prior)
            target = max(min_range, 0.85 * med_range) * (1.0 + (0.15 * key_adj))
            ret_h = mid + (1.0 - model_down_share) * target
            ret_l = mid - model_down_share * target
        spread = ret_h - ret_l
        if spread > max_range:
            mid = 0.5 * (ret_h + ret_l)
            ret_h = mid + (1.0 - model_down_share) * max_range
            ret_l = mid - model_down_share * max_range
        if l_conf < 0.55:
            ret_l = min(ret_l, low_prior)
        if ret_h < ret_l:
            ret_h, ret_l = ret_l, ret_h
        hp_new = _round_to_tick(base_price * (1.0 + ret_h), instr_meta)
        lp_new = _round_to_tick(base_price * (1.0 + ret_l), instr_meta)
        if hp_new < (lp_new + tick):
            mid_p = 0.5 * (hp_new + lp_new)
            hp_new = _round_to_tick(mid_p + tick, instr_meta)
            lp_new = _round_to_tick(mid_p - tick, instr_meta)
        h["pred_price"] = float(round(hp_new, 4))
        l["pred_price"] = float(round(lp_new, 4))
        h["pred_ret"] = float((hp_new / base_price) - 1.0)
        l["pred_ret"] = float((lp_new / base_price) - 1.0)
        h["direction"] = "UP" if hp_new >= base_price else "DOWN"
        l["direction"] = "UP" if lp_new >= base_price else "DOWN"
        note_h = str(h.get("note") or "")
        note_l = str(l.get("note") or "")
        tag = f"Range prior blend (n={int(priors.get('n_samples', 0))})"
        if tag not in note_h:
            h["note"] = f"{note_h} | {tag}" if note_h else tag
        if tag not in note_l:
            l["note"] = f"{note_l} | {tag}" if note_l else tag

def _auto_train_extra_trees(instrument_key: str = None):
    """Lightweight fallback: train ExtraTrees from cached price history if missing on disk."""
    global _EXTRA_TREES_AUTO_ATTEMPTED
    if _EXTRA_TREES_AUTO_ATTEMPTED:
        return None
    _EXTRA_TREES_AUTO_ATTEMPTED = True
    try:
        instr = instrument_key or SYMBOL
        datafile = os.path.join(DATA_DIR, f"prices_{instr}.csv")
        if not os.path.exists(datafile):
            return None
        df = pd.read_csv(datafile, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
        df_full = build_features_for_training(df, predict_minutes=4, instrument_key=instr)
        if df_full is None or len(df_full) < 300:
            return None
        extra_feats = DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3", "ema_5", "ema_20", "trend_strength", "vol_rolling_skew", "vol_rolling_kurt"]
        X = df_full[extra_feats]
        y = df_full["target_ret_norm"]
        n_estimators = min(200, int(EXTRA_TREES_N_ESTIMATORS))
        model = ExtraTreesRegressor(
            n_estimators=n_estimators,
            max_depth=EXTRA_TREES_MAX_DEPTH,
            min_samples_leaf=EXTRA_TREES_MIN_SAMPLES_LEAF,
            random_state=42
        )
        model.fit(X, y)
        preds = model.predict(X)
        resid = float(np.std(y - preds))
        bias = float(np.mean(preds - y))
        try:
            target_mean = float(np.mean(y))
        except Exception:
            target_mean = 0.0
        payload = {
            "model": model,
            "fallback_model": model,
            "features": extra_feats,
            "ensemble_models": {},
            "regime_models": {},
            "performance": {},
            "performance_mae": {},
            "calibration": {},
            "calibration_sig": {"Weak": 0.5, "Medium": 0.5, "Strong": 0.5},
            "meta": {
                "resid_std": max(resid, 1e-6),
                "pred_bias": bias,
                "version": "auto",
                "target_norm": True,
                "demeaned": False,
                "target_mean": float(target_mean)
            }
        }
        try:
            joblib.dump(payload, os.path.join(MODEL_DIR, EXTRA_TREES_MODEL_FILE))
        except Exception:
            pass
        return payload
    except Exception:
        logger.debug("Auto-train ExtraTrees failed", exc_info=True)
        return None

def _session_dates_from_index(local_index, session_open_minute: int, session_close_minute: int):
    local_dates = pd.to_datetime(local_index.date)
    minutes = local_index.hour * 60 + local_index.minute
    if session_open_minute > session_close_minute:
        return pd.to_datetime(np.where(minutes >= session_open_minute, local_dates, local_dates - pd.Timedelta(days=1)))
    return local_dates

def _session_date_for_time(ts_local, session_open_minute: int, session_close_minute: int):
    try:
        minutes = int(ts_local.hour) * 60 + int(ts_local.minute)
        local_date = pd.to_datetime(ts_local.date())
        if session_open_minute > session_close_minute:
            return local_date if minutes >= session_open_minute else (local_date - pd.Timedelta(days=1))
        return local_date
    except Exception:
        return pd.to_datetime(ts_local.date())

def _session_close_datetime(session_date, session_open_minute: int, session_close_minute: int, tz: str):
    try:
        close_hour = int(session_close_minute // 60)
        close_min = int(session_close_minute % 60)
        close_date = session_date
        if session_open_minute > session_close_minute:
            close_date = session_date + pd.Timedelta(days=1)
        close_dt = pd.Timestamp(year=close_date.year, month=close_date.month, day=close_date.day,
                                hour=close_hour, minute=close_min, tz=tz)
        return close_dt
    except Exception:
        return None

def _coerce_now_utc(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _target_session_date(instr_meta: dict, now_utc: pd.Timestamp | None = None):
    tz = instr_meta.get("tz", "America/New_York")
    session_open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    session_close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
    now_ref = _coerce_now_utc(now_utc)
    now_local = now_ref.tz_convert(tz) if now_ref is not None else pd.Timestamp.now(tz=tz)
    session_date = _session_date_for_time(now_local, session_open_minute, session_close_minute)
    close_dt = _session_close_datetime(session_date, session_open_minute, session_close_minute, tz)
    if close_dt is not None and now_local >= close_dt:
        return session_date + pd.Timedelta(days=1)
    return session_date

def _target_week_key(instr_meta: dict, now_utc: pd.Timestamp | None = None):
    tz = instr_meta.get("tz", "America/New_York")
    week_end_day = int(instr_meta.get("week_end_day", 4))
    session_open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    session_close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
    now_ref = _coerce_now_utc(now_utc)
    now_local = now_ref.tz_convert(tz) if now_ref is not None else pd.Timestamp.now(tz=tz)
    session_date = _session_date_for_time(now_local, session_open_minute, session_close_minute)
    close_dt = _session_close_datetime(session_date, session_open_minute, session_close_minute, tz)
    iso = now_local.isocalendar()
    year = int(iso.year)
    week = int(iso.week)
    if int(now_local.weekday()) > week_end_day or (int(now_local.weekday()) == week_end_day and close_dt is not None and now_local >= close_dt):
        week += 1
    return year, week

def _ensure_utc_index(df: pd.DataFrame, tz: str = "America/New_York") -> pd.DataFrame:
    if df is None or df.empty:
        return df
    try:
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            try:
                idx = idx.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
            except Exception:
                idx = idx.tz_localize(tz)
        idx = idx.tz_convert("UTC")
        df.index = idx
    except Exception:
        logger.exception("Failed to localize index in model_utils")
    return df

def _is_point_in_weekly_interval(weekday: int, minute_of_day: int,
                                 start_day: int, start_minute: int,
                                 end_day: int, end_minute: int) -> bool:
    point = int(weekday) * 1440 + int(minute_of_day)
    start = int(start_day) * 1440 + int(start_minute)
    end = int(end_day) * 1440 + int(end_minute)
    if start <= end:
        return start <= point < end
    return point >= start or point < end

def _market_open_now(instr_meta: dict, now_utc: pd.Timestamp) -> bool:
    tz = instr_meta.get("tz", "America/New_York")
    try:
        now_local = pd.Timestamp(now_utc).tz_convert(tz)
    except Exception:
        now_local = pd.Timestamp.now(tz=tz)
    minute = int(now_local.hour) * 60 + int(now_local.minute)
    weekday = int(now_local.weekday())
    open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
    instr_type = str(instr_meta.get("type", "equity")).lower()

    # Day session (typical equities): open and close on the same local day.
    if open_minute <= close_minute:
        if weekday >= 5:
            return False
        return (minute >= open_minute) and (minute < close_minute)

    # Overnight session (typical futures): open in evening, close next day, with weekly closure.
    in_daily_session = (minute >= open_minute) or (minute < close_minute)
    if not in_daily_session:
        return False
    week_start_day = int(instr_meta.get("week_start_day", 6 if instr_type == "futures" else 0))
    week_end_day = int(instr_meta.get("week_end_day", 4))
    in_weekly_closure = _is_point_in_weekly_interval(
        weekday,
        minute,
        week_end_day,
        close_minute,
        week_start_day,
        open_minute
    )
    return not in_weekly_closure

def _staleness_info(df_latest: pd.DataFrame, instrument_key: str = None, now_utc: pd.Timestamp | None = None):
    if df_latest is None or df_latest.empty:
        return {"is_stale": True, "age_sec": None, "threshold_sec": None, "market_open": None, "market_state": "unknown"}
    try:
        last_ts = pd.to_datetime(df_latest.index[-1], utc=True, errors="coerce")
    except Exception:
        last_ts = pd.NaT
    now = _coerce_now_utc(now_utc) or pd.Timestamp.now(tz="UTC")
    try:
        age_sec = max(0.0, float((now - last_ts).total_seconds())) if pd.notna(last_ts) else None
    except Exception:
        age_sec = None
    instr_meta = INSTRUMENTS.get(instrument_key or SYMBOL, {})
    stale_mult = float(instr_meta.get("stale_multiplier", STALE_MULTIPLIER))
    min_stale = float(instr_meta.get("stale_min_seconds", MIN_STALE_SECONDS))
    market_open = _market_open_now(instr_meta, now)
    if market_open:
        threshold_sec = max(min_stale, BAR_MINUTES * 60.0 * stale_mult)
        market_state = "open"
    else:
        instr_type = str(instr_meta.get("type", "equity")).lower()
        default_closed_max_h = 72.0 if instr_type == "equity" else 56.0
        try:
            closed_max_h = float(instr_meta.get("closed_max_stale_hours", default_closed_max_h))
        except Exception:
            closed_max_h = default_closed_max_h
        threshold_sec = max(min_stale, closed_max_h * 3600.0)
        market_state = "closed"
    is_stale = (age_sec is None) or (age_sec > threshold_sec)
    return {
        "is_stale": is_stale,
        "age_sec": age_sec,
        "threshold_sec": threshold_sec,
        "market_open": bool(market_open),
        "market_state": market_state
    }

def _drop_incomplete_last_bar(df_latest: pd.DataFrame, instrument_key: str = None, now_utc: pd.Timestamp | None = None):
    """
    Drop the newest bar if it is likely incomplete (very recent or timestamped in the future).
    This avoids using partial bars that can distort short-horizon signals.
    Returns (df, dropped_flag).
    """
    if not DROP_INCOMPLETE_LAST_BAR:
        return df_latest, False
    if df_latest is None or getattr(df_latest, "empty", True):
        return df_latest, False
    try:
        last_ts = pd.to_datetime(df_latest.index[-1], utc=True, errors="coerce")
    except Exception:
        return df_latest, False
    if pd.isna(last_ts):
        return df_latest, False
    now = _coerce_now_utc(now_utc) or pd.Timestamp.now(tz="UTC")
    try:
        age_sec = float((now - last_ts).total_seconds())
    except Exception:
        return df_latest, False
    # Grace window: either explicit seconds or a fraction of bar length (whichever is larger).
    bar_sec = max(1.0, float(BAR_MINUTES) * 60.0)
    grace = max(1.0, float(INCOMPLETE_BAR_GRACE_SEC), 0.10 * bar_sec)
    # If last timestamp is in the future or too close to now, treat as incomplete.
    if age_sec < -grace or age_sec < grace:
        try:
            if len(df_latest) >= 2:
                return df_latest.iloc[:-1].copy(), True
        except Exception:
            pass
    return df_latest, False

def _stale_conf_factor(stale: dict | None) -> float:
    """Confidence multiplier based on staleness severity."""
    if not stale or not stale.get("is_stale"):
        return 1.0
    try:
        age = float(stale.get("age_sec") or 0.0)
        threshold = float(stale.get("threshold_sec") or MIN_STALE_SECONDS)
    except Exception:
        return 0.7
    if threshold <= 0:
        return 0.7
    if age <= threshold:
        return 1.0
    ratio = max(1.0, age / max(1.0, threshold))
    # sqrt decay: mild penalty near threshold, strong penalty when very stale
    factor = 1.0 / (ratio ** 0.5)
    return max(0.02, min(1.0, factor))

def _is_critical_stale(stale: dict | None) -> bool:
    """Critical stale state should block trading decisions entirely."""
    if not stale or not stale.get("is_stale"):
        return False
    try:
        age = stale.get("age_sec")
        threshold = float(stale.get("threshold_sec") or MIN_STALE_SECONDS)
        market_open = bool(stale.get("market_open", False))
        severe_mult = 10.0 if market_open else 2.0
        if age is None:
            return True
        return float(age) > (threshold * severe_mult)
    except Exception:
        return True

def _prediction_diversity(results_out: dict, model_keys=("base", "forest", "boost", "extra_trees", "meta")) -> dict:
    """
    Compute simple spread diagnostics for model outputs.
    Flags degenerate snapshots where models are near-identical in both price and confidence.
    """
    rows = []
    for k in model_keys:
        v = results_out.get(k, {})
        if not isinstance(v, dict):
            continue
        p = v.get("pred_price")
        c = v.get("confidence")
        r = v.get("pred_ret")
        d = v.get("direction")
        if isinstance(p, (int, float)) and isinstance(c, (int, float)):
            rows.append((k, float(p), float(c), float(r) if isinstance(r, (int, float)) else np.nan, str(d or "-")))
    if len(rows) < 2:
        return {
            "n_valid": int(len(rows)),
            "price_span_bps": None,
            "conf_span_pct": None,
            "degenerate": False
        }
    prices = np.array([r[1] for r in rows], dtype=float)
    confs = np.array([r[2] for r in rows], dtype=float)
    pred_rets = np.array([r[3] for r in rows], dtype=float)
    ref = max(1.0, float(np.nanmean(np.abs(prices))))
    price_span_bps = float((np.nanmax(prices) - np.nanmin(prices)) / ref * 10000.0)
    if np.any(np.isfinite(pred_rets)):
        ret_span_bps = float((np.nanmax(pred_rets) - np.nanmin(pred_rets)) * 10000.0)
    else:
        ret_span_bps = None
    conf_span = float(np.nanmax(confs) - np.nanmin(confs))
    dirs = [r[4] for r in rows]
    unique_dirs = int(len(set(dirs)))

    def _max_bucket(vals):
        buckets = {}
        for v in vals:
            buckets[v] = buckets.get(v, 0) + 1
        return max(buckets.values()) if buckets else 0

    top_price_bucket = _max_bucket([round(r[1], 4) for r in rows])
    top_conf_bucket = _max_bucket([round(r[2], 2) for r in rows])
    valid_ret_rounded = [round(r[3], 7) for r in rows if isinstance(r[3], (int, float)) and np.isfinite(r[3])]
    top_ret_bucket = _max_bucket(valid_ret_rounded) if valid_ret_rounded else 0
    n = len(rows)
    degenerate = False
    if n >= 3:
        ret_near_identical = (ret_span_bps is None) or (ret_span_bps <= 0.5)
        near_identical = (price_span_bps <= 0.5 and conf_span <= 1.0 and ret_near_identical)
        bucket_identical = (top_price_bucket >= (n - 1) and top_conf_bucket >= (n - 1))
        if top_ret_bucket:
            bucket_identical = bucket_identical and (top_ret_bucket >= (n - 1))
        degenerate = bool(near_identical or bucket_identical)

    return {
        "n_valid": int(n),
        "price_span_bps": round(price_span_bps, 4),
        "pred_ret_span_bps": None if ret_span_bps is None else round(ret_span_bps, 4),
        "conf_span_pct": round(conf_span, 4),
        "unique_directions": unique_dirs,
        "top_price_bucket": int(top_price_bucket),
        "top_conf_bucket": int(top_conf_bucket),
        "top_pred_ret_bucket": int(top_ret_bucket),
        "degenerate": bool(degenerate)
    }

def _window_weights(perf: dict):
    wins = ["short", "mid", "long"]
    weights = {}
    total = 0.0
    perf = perf or {}

    def _entry_stats(v):
        if isinstance(v, dict):
            try:
                mae = float(v.get("mae", float("nan")))
            except Exception:
                mae = float("nan")
            try:
                mae_dollars = float(v.get("mae_dollars", float("nan")))
            except Exception:
                mae_dollars = float("nan")
            try:
                window_ratio = float(v.get("window_ratio_mean", float("nan")))
            except Exception:
                window_ratio = float("nan")
            try:
                window_hit = float(v.get("window_hit_rate", float("nan")))
            except Exception:
                window_hit = float("nan")
            try:
                active = float(v.get("active_frac", float("nan")))
            except Exception:
                active = float("nan")
            return mae, mae_dollars, window_ratio, window_hit, active
        try:
            mae = float(v)
        except Exception:
            mae = float("nan")
        return mae, float("nan"), float("nan"), float("nan"), float("nan")

    for w in wins:
        mae, mae_dollars, window_ratio, window_hit, active = _entry_stats(perf.get(w, None))
        if not np.isfinite(mae) or mae <= 0:
            mae = 1.0
        weight = 1.0 / (mae + 1e-6)
        if np.isfinite(mae_dollars) and mae_dollars > 0:
            weight *= max(0.50, min(1.80, 1.0 / max(0.20, mae_dollars)))
        if np.isfinite(window_ratio):
            weight *= max(0.35, min(1.90, 1.0 / max(0.55, window_ratio)))
        if np.isfinite(window_hit):
            weight *= max(0.45, min(1.60, window_hit / 0.60))
        if np.isfinite(active):
            if active < 0.03:
                weight *= 0.75
            elif active > 0.70:
                weight *= 0.85
        weights[w] = weight
        total += weight
    if total <= 0:
        return {w: 1.0 / len(wins) for w in wins}
    return {w: weights[w] / total for w in wins}

def _summarize_preds(preds: np.ndarray):
    if preds is None or preds.size == 0:
        return None
    mean = float(np.mean(preds))
    std = float(np.std(preds))
    q10 = float(np.quantile(preds, 0.1))
    q50 = float(np.quantile(preds, 0.5))
    q90 = float(np.quantile(preds, 0.9))
    p_up = float(np.mean(preds > 0))
    return {"mean": mean, "std": max(std, 1e-6), "q10": q10, "q50": q50, "q90": q90, "p_up": p_up}

def _blend_summaries(a: dict, b: dict, w_a: float):
    if a is None:
        return b
    if b is None:
        return a
    w_b = max(0.0, min(1.0, 1.0 - w_a))
    w_a = max(0.0, min(1.0, w_a))
    mean = w_a * a["mean"] + w_b * b["mean"]
    var = w_a * (a["std"] ** 2 + a["mean"] ** 2) + w_b * (b["std"] ** 2 + b["mean"] ** 2) - mean ** 2
    std = math.sqrt(max(var, 1e-12))
    q10 = w_a * a["q10"] + w_b * b["q10"]
    q50 = w_a * a["q50"] + w_b * b["q50"]
    q90 = w_a * a["q90"] + w_b * b["q90"]
    p_up = w_a * a["p_up"] + w_b * b["p_up"]
    return {"mean": mean, "std": std, "q10": q10, "q50": q50, "q90": q90, "p_up": p_up}

def _regime_prob_for_key(regime_key, comp_probs):
    try:
        if comp_probs is None:
            return None
        if isinstance(comp_probs, (list, tuple)) and str(regime_key).isdigit():
            idx = int(regime_key)
            if 0 <= idx < len(comp_probs):
                return float(comp_probs[idx])
    except Exception:
        return None
    return None

def _select_regime_members(regime_map: dict, win: str, dominant_regime=None, comp_probs=None):
    """Pick the best available regime-specific members for a given window."""
    if not regime_map:
        return [], None, None

    def _members_for(key):
        return regime_map.get(str(key), {}).get(win, []) or []

    # 1) Dominant regime if available
    if dominant_regime is not None:
        members = _members_for(dominant_regime)
        if members:
            return members, str(dominant_regime), _regime_prob_for_key(dominant_regime, comp_probs)

    # 2) Highest-prob available regime
    if isinstance(comp_probs, (list, tuple)):
        scored = []
        for key in regime_map.keys():
            prob = _regime_prob_for_key(key, comp_probs)
            scored.append((prob if prob is not None else -1.0, str(key)))
        scored.sort(reverse=True, key=lambda x: x[0])
        for prob, key in scored:
            members = _members_for(key)
            if members:
                return members, key, prob if prob is not None and prob >= 0 else None

    # 3) Any non-empty regime
    for key in regime_map.keys():
        members = _members_for(key)
        if members:
            return members, str(key), _regime_prob_for_key(key, comp_probs)

    return [], None, None

def _max_abs_ret_from_vol(vol: float, instr_meta: dict = None) -> float:
    try:
        v = float(vol)
    except Exception:
        v = 0.0
    if not np.isfinite(v) or v <= 0:
        return MAX_ABS_RET
    mult = 6.0
    if instr_meta:
        try:
            mult = float(instr_meta.get("max_abs_ret_mult", mult))
        except Exception:
            mult = mult
    return float(max(0.003, min(0.05, mult * v)))

def _estimated_trade_cost_ret(last_price: float, instr_meta: dict = None, spread_frac: float | None = None) -> float:
    """
    Conservative one-way cost proxy in return units:
    commission/slippage base + half-tick spread proxy.
    """
    try:
        lp = float(last_price)
    except Exception:
        lp = 0.0
    try:
        # Use a one-way conservative proxy (half of round-trip assumptions).
        base = 0.5 * (float(TRANSACTION_COST_PER_TRADE) + float(SLIPPAGE_PCT))
    except Exception:
        base = 0.0
    try:
        mult = max(0.1, float(LIVE_COST_MULTIPLIER))
    except Exception:
        mult = 1.0
    tick_spread = 0.0
    try:
        if instr_meta is not None and lp > 0:
            tick = float(instr_meta.get("tick_size", 0.0) or 0.0)
            if tick > 0:
                tick_spread = 0.5 * (tick / lp)
    except Exception:
        tick_spread = 0.0
    spread_cost = 0.0
    try:
        if spread_frac is not None and float(spread_frac) > 0:
            spread_cost = 0.5 * float(spread_frac)
    except Exception:
        spread_cost = 0.0
    spread_term = max(tick_spread, spread_cost)
    return float(max(0.0, mult * (base + spread_term)))


def _live_edge_floor_inputs(
    last_price: float,
    df_latest: pd.DataFrame,
    instr_meta: dict | None = None,
) -> dict:
    spread_frac = None
    try:
        if "ask" in df_latest.columns and "bid" in df_latest.columns:
            spread = float(df_latest["ask"].iloc[-1] - df_latest["bid"].iloc[-1])
            if spread > 0 and float(last_price) > 0:
                spread_frac = spread / float(last_price)
    except Exception:
        spread_frac = None
    try:
        cost_ret_est = _estimated_trade_cost_ret(last_price, instr_meta=instr_meta, spread_frac=spread_frac)
    except Exception:
        cost_ret_est = 0.0
    try:
        min_edge_floor = max(0.0, float(LIVE_MIN_EDGE_BPS) / 10000.0)
    except Exception:
        min_edge_floor = 0.0
    try:
        unc_mult = max(0.0, float(LIVE_UNCERTAINTY_EDGE_MULT))
    except Exception:
        unc_mult = 0.0
    return {
        "spread_frac": spread_frac,
        "cost_ret_est": float(max(0.0, cost_ret_est)),
        "min_edge_floor": float(max(0.0, min_edge_floor)),
        "uncertainty_mult": float(max(0.0, unc_mult)),
    }

def _predict_horizon_norm(models_dict, df_latest, horizon_minutes: int, instrument_key: str = None,
                          feature_frame: pd.DataFrame | None = None, feat_cache: dict | None = None):
    summary = _predict_horizon_summary(
        models_dict,
        df_latest,
        horizon_minutes,
        instrument_key=instrument_key,
        feature_frame=feature_frame,
        feat_cache=feat_cache,
    )
    if not isinstance(summary, dict):
        return None
    try:
        return float(summary.get("mean"))
    except Exception:
        return None


def _predict_horizon_summary(models_dict, df_latest, horizon_minutes: int, instrument_key: str = None,
                             feature_frame: pd.DataFrame | None = None, feat_cache: dict | None = None):
    try:
        h_key = str(int(horizon_minutes))
    except Exception:
        h_key = str(horizon_minutes)
    preds = []
    for key in ["base", "forest", "boost", "extra_trees"]:
        payload = models_dict.get(key)
        if payload is None:
            continue
        h_payload = payload.get("horizon_models", {}).get(h_key)
        if not h_payload:
            continue
        feats = h_payload.get("features", payload.get("features"))
        try:
            X = _get_feature_row_cached(
                df_latest,
                feats,
                instrument_key=instrument_key,
                feature_frame=feature_frame,
                cache=feat_cache
            )
        except Exception:
            X = pd.DataFrame()
        if X is None or X.empty:
            continue
        ensemble_map = h_payload.get("ensemble_models", {}) if h_payload else {}
        win_calibration = h_payload.get("window_calibration", {}) if h_payload else {}
        perf = h_payload.get("performance", h_payload.get("performance_mae", {})) if h_payload else {}
        combined = None
        if isinstance(ensemble_map, dict) and ensemble_map:
            win_weights = _window_weights(perf)
            combined_weight = 0.0
            for win in ["short", "mid", "long"]:
                members = ensemble_map.get(win, []) or []
                if not members:
                    continue
                preds_norm = _ensemble_predict_members(members, X)
                if preds_norm.size == 0:
                    continue
                mean_norm, std_norm, q10_norm, q50_norm, q90_norm = _aggregate_preds(preds_norm)
                win_sum = {
                    "mean": mean_norm,
                    "std": std_norm,
                    "q10": q10_norm,
                    "q50": q50_norm,
                    "q90": q90_norm,
                    "p_up": float(np.mean(preds_norm > 0)),
                }
                win_sum = _apply_summary_calibration(win_sum, win_calibration.get(win))
                w = win_weights.get(win, 0.0)
                if combined is None:
                    combined = win_sum
                    combined_weight = w
                else:
                    total = combined_weight + w
                    w_a = combined_weight / total if total > 0 else 0.5
                    combined = _blend_summaries(combined, win_sum, w_a)
                    combined_weight = total
        if combined is None:
            model = h_payload.get("model") or h_payload.get("fallback_model")
            if model is None:
                continue
            try:
                pred_norm = float(np.asarray(model.predict(X), dtype=float)[0])
            except Exception:
                continue
            meta = dict(h_payload.get("meta") or {})
            try:
                if bool(meta.get("target_norm")) and bool(meta.get("demeaned")):
                    pred_norm += float(meta.get("target_mean", 0.0) or 0.0)
            except Exception:
                pass
            try:
                resid_std = abs(float(meta.get("resid_std", 1e-6) or 1e-6))
            except Exception:
                resid_std = 1e-6
            z = pred_norm / max(1e-6, resid_std)
            combined = {
                "mean": float(pred_norm),
                "std": float(max(1e-6, resid_std)),
                "q10": float(pred_norm - resid_std),
                "q50": float(pred_norm),
                "q90": float(pred_norm + resid_std),
                "p_up": float(max(0.001, min(0.999, 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))))),
            }
        preds.append(dict(combined))
    if not preds:
        return None
    means = np.asarray([float(p.get("mean", 0.0)) for p in preds], dtype=float)
    stds = np.asarray([abs(float(p.get("std", 0.0))) for p in preds], dtype=float)
    p_ups = np.asarray([float(p.get("p_up", 0.5)) for p in preds], dtype=float)
    mean_val = float(np.mean(means))
    spread = float(np.std(means)) if len(means) >= 2 else 0.0
    base_std = float(np.mean(stds)) if len(stds) else 1e-6
    std_val = max(1e-6, base_std, spread)
    direction = "UP" if mean_val > 0.0 else "DOWN" if mean_val < 0.0 else "-"
    strength = float(max(0.0, min(1.0, math.tanh(abs(mean_val) / 1.35))))
    return {
        "mean": mean_val,
        "std": std_val,
        "p_up": float(np.mean(p_ups)) if len(p_ups) else 0.5,
        "direction": direction,
        "strength": strength,
        "models": int(len(preds)),
    }


def _stock_horizon_support(raw_direction: str | None,
                           raw_pred_ret: float | None,
                           current_vol: float | None,
                           h10_summary: dict | None,
                           h30_summary: dict | None) -> dict:
    raw_dir = str(raw_direction or "-").upper()
    try:
        raw_ret = float(raw_pred_ret or 0.0)
    except Exception:
        raw_ret = 0.0
    try:
        vol = max(1e-6, abs(float(current_vol or 0.0)))
    except Exception:
        vol = 1e-6
    raw_norm = raw_ret / vol if vol > 0 else 0.0
    if raw_dir not in {"UP", "DOWN"}:
        raw_dir = "UP" if raw_norm > 0.0 else "DOWN" if raw_norm < 0.0 else "-"
    rows = []
    for horizon, weight, summary in ((10, 0.55, h10_summary), (30, 0.45, h30_summary)):
        if not isinstance(summary, dict):
            continue
        direction = str(summary.get("direction") or "-").upper()
        try:
            mean_norm = float(summary.get("mean", 0.0) or 0.0)
        except Exception:
            mean_norm = 0.0
        try:
            strength = float(summary.get("strength", 0.0) or 0.0)
        except Exception:
            strength = 0.0
        if direction not in {"UP", "DOWN"}:
            direction = "UP" if mean_norm > 0.0 else "DOWN" if mean_norm < 0.0 else "-"
        rows.append(
            {
                "horizon": int(horizon),
                "weight": float(weight),
                "direction": direction,
                "mean_norm": float(mean_norm),
                "strength": float(max(0.0, min(1.0, strength))),
            }
        )
    if raw_dir not in {"UP", "DOWN"} or not rows:
        return {
            "stage": "unavailable",
            "direction": "-",
            "score": 0.0,
            "confidence_bonus": 0.0,
            "size_mult": 1.0,
            "align_score": 1.0,
            "blended_norm": float(raw_norm),
            "rows": rows,
        }
    aligned = 0.0
    opposing = 0.0
    weighted_norm = 0.50 * max(-3.0, min(3.0, raw_norm))
    total_weight = 0.50
    aligned_count = 0
    opposing_count = 0
    for row in rows:
        contribution = float(row["weight"]) * float(row["strength"])
        total_weight += float(row["weight"])
        weighted_norm += float(row["weight"]) * max(-3.0, min(3.0, float(row["mean_norm"])))
        if row["direction"] == raw_dir:
            aligned += contribution
            aligned_count += 1
        elif row["direction"] in {"UP", "DOWN"}:
            opposing += contribution
            opposing_count += 1
    net = aligned - (0.85 * opposing)
    score = max(0.0, min(1.0, net / max(1e-6, total_weight)))
    stage = "mixed"
    if aligned_count >= 1 and score >= 0.18 and opposing <= 0.08:
        stage = "supportive"
    if aligned_count >= 2 and score >= 0.28 and opposing <= 0.04:
        stage = "confirmed"
    if aligned <= 0.0 and opposing > 0.0:
        stage = "contrary"
    conf_bonus = 0.0
    size_mult = 1.0
    if stage == "supportive":
        conf_bonus = 2.5 + (3.5 * score)
        size_mult = 0.85 + (0.20 * score)
    elif stage == "confirmed":
        conf_bonus = 4.0 + (float(STOCK_HORIZON_SUPPORT_CONF_MAX_BONUS) * score * 0.50)
        size_mult = 0.92 + (0.18 * score)
    elif stage == "contrary":
        size_mult = float(STOCK_HORIZON_SUPPORT_SIZE_MIN_MULT)
    align_score = max(
        float(STOCK_HORIZON_SUPPORT_SIZE_MIN_MULT),
        min(float(STOCK_HORIZON_SUPPORT_SIZE_MAX_MULT), float(size_mult)),
    )
    return {
        "stage": stage,
        "direction": raw_dir if stage in {"supportive", "confirmed"} else "-",
        "score": float(score),
        "confidence_bonus": float(max(0.0, conf_bonus)),
        "size_mult": float(size_mult),
        "align_score": float(align_score),
        "blended_norm": float(weighted_norm / max(1e-6, total_weight)),
        "aligned_count": int(aligned_count),
        "opposing_count": int(opposing_count),
        "rows": rows,
    }

def _load_live_calibration(instrument_key: str):
    if not LIVE_CALIBRATION:
        return None
    try:
        instr = str(instrument_key or SYMBOL)
        now = datetime.now(timezone.utc)
        cached = _LIVE_CAL_CACHE.get(instr)
        if isinstance(cached, dict) and cached.get("ts") is not None:
            age = (now - cached["ts"]).total_seconds()
            if age < 60:
                return cached.get("data")
        path = os.path.join(DATA_DIR, f"calibration_live_{instr}.json")
        if not os.path.exists(path):
            return None
        with open(path, "r") as f:
            data = json.load(f)
        # Mark stale calibration so we can avoid trusting old stats
        try:
            last_update = pd.to_datetime(data.get("last_update"), utc=True, errors="coerce")
            if pd.isna(last_update):
                data["_stale"] = True
            else:
                from config import CALIBRATION_MAX_AGE_DAYS
                age_days = (now - last_update).total_seconds() / 86400.0
                data["_stale"] = age_days > float(CALIBRATION_MAX_AGE_DAYS)
        except Exception:
            data["_stale"] = True
        _LIVE_CAL_CACHE[instr] = {"ts": now, "data": data}
        return data
    except Exception:
        logger.exception("Failed to load live calibration")
        return None

def _live_calibration_prob(instrument_key: str, model_key: str, sig_bucket: str):
    data = _load_live_calibration(instrument_key)
    if not data or data.get("_stale"):
        return None, 0
    try:
        bucket = data.get("by_model", {}).get(model_key, {}).get(sig_bucket, None)
        if bucket is None:
            return None, 0
        wins = float(bucket.get("wins", 0))
        total = float(bucket.get("total", 0))
        if total <= 0:
            return None, 0
        # Keep bucket probabilities realistic under low sample sizes.
        smoothed = (wins + 1.0) / (total + 2.0)
        return float(smoothed), int(total)
    except Exception:
        return None, 0

def _live_calibration_bias(instrument_key: str, model_key: str):
    data = _load_live_calibration(instrument_key)
    if not data or data.get("_stale"):
        return 0.0
    try:
        stats = data.get("by_model_stats", {}).get(model_key, None)
        if stats is None:
            return 0.0
        total = float(stats.get("total", 0.0))
        if total <= 0:
            return 0.0
        return float(stats.get("sum_error", 0.0)) / total
    except Exception:
        return 0.0

def _live_bias_adjustment(live_cal: dict | None, model_key: str, pred_ret: float,
                          max_abs_ret: float, drift_flag: bool = False):
    """
    Apply conservative bias correction using recent live errors.
    Returns (adjusted_pred, applied_bias).
    """
    if not LIVE_BIAS_CORRECTION:
        return pred_ret, 0.0
    if not isinstance(live_cal, dict) or live_cal.get("_stale"):
        return pred_ret, 0.0
    try:
        stats = live_cal.get("by_model_stats", {}).get(model_key, {})
        total = float(stats.get("total", 0.0))
        if total < float(LIVE_BIAS_MIN_SAMPLES):
            return pred_ret, 0.0
        bias = float(stats.get("sum_error", 0.0)) / max(1.0, total)
        cap = float(LIVE_BIAS_MAX_ABS_MULT) * float(max_abs_ret)
        if not np.isfinite(cap) or cap <= 0:
            cap = float(max_abs_ret)
        bias = max(-cap, min(cap, bias))
        weight = min(float(LIVE_BIAS_WEIGHT_MAX), total / max(1.0, float(LIVE_BIAS_MIN_SAMPLES) * 4.0))
        if drift_flag:
            weight *= 0.6
        adj = weight * bias
        return float(pred_ret + adj), float(adj)
    except Exception:
        return pred_ret, 0.0

def _live_model_weight_multiplier(live_cal: dict | None, model_key: str) -> float:
    """
    Reliability multiplier from live stats.
    >1 boosts models with stronger recent hit-rate; <1 downweights degrading models.
    """
    if not isinstance(live_cal, dict):
        return 1.0
    if bool(live_cal.get("_stale", False)):
        return 1.0
    try:
        stats = live_cal.get("by_model_stats", {}).get(model_key, {})
        total = float(stats.get("total", 0.0))
        wins = float(stats.get("wins", 0.0))
        if total < 12:
            return 1.0
        acc = wins / max(1.0, total)
        sum_abs = float(stats.get("sum_abs_error", 0.0))
        mae = sum_abs / max(1.0, total)
        acc_mult = max(0.60, min(1.45, acc / 0.52))
        err_mult = max(0.70, min(1.20, 1.0 / (1.0 + 45.0 * max(0.0, mae))))
        return float(acc_mult * err_mult)
    except Exception:
        return 1.0

def _conformal_abs_error_threshold(live_cal: dict | None, quantile: float = 0.80, min_n: int = 40):
    """
    Empirical absolute-error threshold from recent consensus outcomes.
    Used as a light conformal-style uncertainty band for live no-trade gating.
    """
    if not isinstance(live_cal, dict):
        return None
    if bool(live_cal.get("_stale", False)):
        return None
    try:
        vals = live_cal.get("recent_abs_errors", [])
        if not isinstance(vals, list):
            return None
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < int(min_n):
            return None
        q = max(0.55, min(0.95, float(quantile)))
        return float(np.quantile(arr, q))
    except Exception:
        return None

def _schedule_fast_retrain_on_drift(instrument_key: str, live_cal: dict | None):
    """
    Trigger lightweight retraining when live calibration signals drift.
    This keeps models adaptive without blocking inference.
    """
    if not AUTO_FAST_RETRAIN_ON_DRIFT:
        return
    if not isinstance(live_cal, dict):
        return
    drift = bool(live_cal.get("drift_flag", False) or live_cal.get("recent_drift_flag", False))
    if not drift:
        return
    instr = str(instrument_key or SYMBOL)
    now = datetime.now(timezone.utc)
    cooldown_min = max(15, int(DRIFT_RETRAIN_COOLDOWN_MIN))
    with _DRIFT_RETRAIN_LOCK:
        last = _LAST_DRIFT_RETRAIN.get(instr)
        if last is not None and (now - last).total_seconds() < (cooldown_min * 60.0):
            return
        if instr in _DRIFT_RETRAIN_IN_FLIGHT:
            return
        _DRIFT_RETRAIN_IN_FLIGHT.add(instr)
        _LAST_DRIFT_RETRAIN[instr] = now

    def _worker(target_instr: str):
        try:
            from bootstrap_models_fast import bootstrap_fast_models
            logger.warning("Drift-triggered fast retrain started for %s", target_instr)
            bootstrap_fast_models(target_instr)
            # Clear cache to force fresh model reload on next inference.
            _clear_model_cache(target_instr)
            logger.warning("Drift-triggered fast retrain completed for %s", target_instr)
        except Exception:
            logger.exception("Drift-triggered fast retrain failed for %s", target_instr)
        finally:
            with _DRIFT_RETRAIN_LOCK:
                _DRIFT_RETRAIN_IN_FLIGHT.discard(target_instr)

    th = threading.Thread(target=_worker, args=(instr,), daemon=True)
    th.start()

def _schedule_live_calibration_update(instrument_key: str):
    """Schedule calibration refresh in the background to keep prediction latency stable."""
    if not LIVE_CALIBRATION:
        return
    instr = str(instrument_key or SYMBOL)
    now = datetime.now(timezone.utc)
    try:
        last_update = _LAST_CAL_UPDATE.get(instr)
        if last_update is not None and (now - last_update).total_seconds() < 300:
            return
    except Exception:
        pass
    with _CAL_UPDATE_LOCK:
        if instr in _CAL_UPDATE_IN_FLIGHT:
            return
        _CAL_UPDATE_IN_FLIGHT.add(instr)
        _LAST_CAL_UPDATE[instr] = now

    def _job(target_instr: str):
        try:
            from calibration_tracker import update_live_calibration
            update_live_calibration(target_instr, horizon_minutes=4)
        except Exception:
            logger.debug("Live calibration update failed", exc_info=True)
        finally:
            with _CAL_UPDATE_LOCK:
                _CAL_UPDATE_IN_FLIGHT.discard(target_instr)
            # Force a fresh read after background update.
            try:
                _LIVE_CAL_CACHE.pop(target_instr, None)
            except Exception:
                pass

    try:
        th = threading.Thread(target=_job, args=(instr,), daemon=True)
        th.start()
    except Exception:
        with _CAL_UPDATE_LOCK:
            _CAL_UPDATE_IN_FLIGHT.discard(instr)

def _log_prediction(instrument_key, last_price, results_out):
    if not LOG_PREDICTIONS and not LIVE_CALIBRATION:
        return
    try:
        import csv
        ts = datetime.now(timezone.utc).isoformat()
        instr = instrument_key or SYMBOL
        row = {
            "timestamp": ts,
            "instrument": instr,
            "last_price": last_price
        }
        for k in ["base", "forest", "boost", "extra_trees", "meta", "consensus"]:
            v = results_out.get(k, {})
            row[f"{k}_dir"] = v.get("direction")
            row[f"{k}_price"] = v.get("pred_price")
            row[f"{k}_conf"] = v.get("confidence")
            row[f"{k}_pred_ret"] = v.get("pred_ret")
            row[f"{k}_sig_bucket"] = v.get("sig_bucket")
        reg = results_out.get("regime", {})
        row["regime_bullish"] = reg.get("Bullish")
        row["regime_neutral"] = reg.get("Neutral")
        row["regime_bearish"] = reg.get("Bearish")
        row["regime_conf"] = reg.get("regime_conf")
        meta = results_out.get("_meta", {})
        row["data_ts"] = meta.get("data_ts")
        try:
            row["model_versions"] = json.dumps(meta.get("model_versions", {}), sort_keys=True)
        except Exception:
            row["model_versions"] = None

        path = os.path.join(DATA_DIR, f"predictions_{instr}.csv")
        fieldnames = list(row.keys())
        if os.path.exists(path):
            try:
                with open(path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
                if existing_header:
                    # Keep a stable schema per file to avoid malformed rows when fields change.
                    fieldnames = existing_header
            except Exception:
                pass
        write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        row_to_write = {k: row.get(k) for k in fieldnames}
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row_to_write)
    except Exception:
        logger.exception("Failed to log prediction")


def _log_horizon_prediction(instrument_key, last_price, mode, results_out):
    """
    Persist daily/weekend horizon predictions for post-close accuracy accounting.
    This log is used by horizon_accuracy_tracker to resolve realized errors later.
    """
    try:
        import csv

        instr = instrument_key or SYMBOL
        mode_norm = str(mode or "").strip().lower()
        ts = datetime.now(timezone.utc).isoformat()
        row = {
            "timestamp": ts,
            "instrument": instr,
            "mode": mode_norm,
            "last_price": last_price,
            "target_session_date": None,
            "target_week": None,
            "feature_session_date": None,
        }
        meta = results_out.get("meta", {}) if isinstance(results_out, dict) else {}
        if isinstance(meta, dict):
            row["target_session_date"] = meta.get("target_session_date")
            row["target_week"] = meta.get("target_week")
            row["feature_session_date"] = meta.get("feature_session_date")
        reg = results_out.get("regime", {}) if isinstance(results_out, dict) else {}
        row["regime_bullish"] = reg.get("Bullish") if isinstance(reg, dict) else None
        row["regime_neutral"] = reg.get("Neutral") if isinstance(reg, dict) else None
        row["regime_bearish"] = reg.get("Bearish") if isinstance(reg, dict) else None
        row["regime_conf"] = reg.get("regime_conf") if isinstance(reg, dict) else None

        saw_numeric = False
        if "daily" in mode_norm:
            high = results_out.get("high", {}) if isinstance(results_out, dict) else {}
            low = results_out.get("low", {}) if isinstance(results_out, dict) else {}
            for k in ["base", "forest", "boost", "extra_trees", "meta", "consensus"]:
                hv = high.get(k, {}) if isinstance(high, dict) else {}
                lv = low.get(k, {}) if isinstance(low, dict) else {}
                hp = hv.get("pred_price") if isinstance(hv, dict) else None
                lp = lv.get("pred_price") if isinstance(lv, dict) else None
                row[f"{k}_high_pred_price"] = hp
                row[f"{k}_low_pred_price"] = lp
                row[f"{k}_high_conf"] = hv.get("confidence") if isinstance(hv, dict) else None
                row[f"{k}_low_conf"] = lv.get("confidence") if isinstance(lv, dict) else None
                try:
                    if hp is not None and np.isfinite(float(hp)):
                        saw_numeric = True
                except Exception:
                    pass
                try:
                    if lp is not None and np.isfinite(float(lp)):
                        saw_numeric = True
                except Exception:
                    pass
        elif "week" in mode_norm:
            week = results_out.get("week", {}) if isinstance(results_out, dict) else {}
            for k in ["base", "forest", "boost", "extra_trees", "meta", "consensus"]:
                wv = week.get(k, {}) if isinstance(week, dict) else {}
                wp = wv.get("pred_price") if isinstance(wv, dict) else None
                row[f"{k}_week_pred_price"] = wp
                row[f"{k}_week_conf"] = wv.get("confidence") if isinstance(wv, dict) else None
                try:
                    if wp is not None and np.isfinite(float(wp)):
                        saw_numeric = True
                except Exception:
                    pass
        if not saw_numeric:
            return

        path = os.path.join(DATA_DIR, f"predictions_horizon_{instr}.csv")
        fieldnames = list(row.keys())
        if os.path.exists(path):
            try:
                with open(path, "r", newline="") as f:
                    reader = csv.reader(f)
                    existing_header = next(reader, None)
                if existing_header:
                    fieldnames = existing_header
            except Exception:
                pass
        write_header = (not os.path.exists(path)) or (os.path.getsize(path) == 0)
        row_to_write = {k: row.get(k) for k in fieldnames}
        with open(path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row_to_write)
    except Exception:
        logger.exception("Failed to log horizon prediction")


def _schedule_horizon_retrain_if_needed(instrument_key: str, metrics: dict | None):
    if not AUTO_HORIZON_RETRAIN_ON_DRIFT:
        return
    if not isinstance(metrics, dict):
        return
    by_h = metrics.get("by_horizon", {})
    if not isinstance(by_h, dict) or not by_h:
        return
    # Require sufficient realized samples before trusting horizon drift signals.
    try:
        total_samples = 0
        horizon_counts = {}
        for h_name, h_payload in by_h.items():
            models = h_payload.get("models", {}) if isinstance(h_payload, dict) else {}
            n_vals = []
            for m_payload in models.values():
                if not isinstance(m_payload, dict):
                    continue
                try:
                    n_vals.append(int(m_payload.get("n", 0)))
                except Exception:
                    continue
            h_n = max(n_vals) if n_vals else 0
            horizon_counts[h_name] = int(h_n)
            total_samples += int(h_n)
        min_total = max(1, int(HORIZON_ADAPT_MIN_TOTAL_SAMPLES))
        min_per_h = max(1, int(HORIZON_ADAPT_MIN_SAMPLES_PER_HORIZON))
        if total_samples < min_total:
            return
    except Exception:
        horizon_counts = {}
        min_per_h = 1
    daily_drift = bool(by_h.get("daily_high", {}).get("drift_flag", False) or by_h.get("daily_low", {}).get("drift_flag", False))
    week_drift = bool(by_h.get("weekend", {}).get("drift_flag", False))
    if daily_drift:
        if int(horizon_counts.get("daily_high", 0)) < min_per_h and int(horizon_counts.get("daily_low", 0)) < min_per_h:
            daily_drift = False
    if week_drift and int(horizon_counts.get("weekend", 0)) < min_per_h:
        week_drift = False
    if not (daily_drift or week_drift or bool(metrics.get("global_drift_flag", False))):
        return

    instr = str(instrument_key or SYMBOL)
    now = datetime.now(timezone.utc)
    cooldown_min = max(60, int(HORIZON_DRIFT_RETRAIN_COOLDOWN_MIN))
    with _HORIZON_RETRAIN_LOCK:
        last = _LAST_HORIZON_RETRAIN.get(instr)
        if last is not None and (now - last).total_seconds() < (cooldown_min * 60.0):
            return
        if instr in _HORIZON_RETRAIN_IN_FLIGHT:
            return
        _HORIZON_RETRAIN_IN_FLIGHT.add(instr)
        _LAST_HORIZON_RETRAIN[instr] = now

    modes_filter = []
    if daily_drift:
        modes_filter.extend(["daily_high", "daily_low"])
    if week_drift:
        modes_filter.append("weekend")
    if not modes_filter:
        modes_filter = ["daily_high", "daily_low", "weekend"]

    def _worker(target_instr: str, target_modes: list[str]):
        try:
            from data_fetch import update_price_data
            from model_trainers import train_all_models

            logger.warning("Horizon drift retrain started for %s (%s)", target_instr, ",".join(target_modes))
            df = update_price_data(target_instr)
            if df is not None and not df.empty:
                train_all_models(
                    df,
                    instrument_key=target_instr,
                    train_horizons=True,
                    modes_filter=target_modes,
                )
                _clear_model_cache(target_instr)
            logger.warning("Horizon drift retrain completed for %s", target_instr)
        except Exception:
            logger.exception("Horizon drift retrain failed for %s", target_instr)
        finally:
            with _HORIZON_RETRAIN_LOCK:
                _HORIZON_RETRAIN_IN_FLIGHT.discard(target_instr)

    th = threading.Thread(target=_worker, args=(instr, modes_filter), daemon=True)
    th.start()


def _schedule_horizon_accuracy_update(instrument_key: str):
    instr = str(instrument_key or SYMBOL)
    now = datetime.now(timezone.utc)
    cooldown_sec = max(60, int(HORIZON_ACCURACY_UPDATE_COOLDOWN_SEC))
    try:
        last = _LAST_HORIZON_ACC_UPDATE.get(instr)
        if last is not None and (now - last).total_seconds() < cooldown_sec:
            return
    except Exception:
        pass
    with _HORIZON_ACC_LOCK:
        if instr in _HORIZON_ACC_IN_FLIGHT:
            return
        _HORIZON_ACC_IN_FLIGHT.add(instr)
        _LAST_HORIZON_ACC_UPDATE[instr] = now

    def _worker(target_instr: str):
        try:
            from horizon_accuracy_tracker import update_horizon_accuracy

            metrics = update_horizon_accuracy(target_instr)
            _schedule_horizon_retrain_if_needed(target_instr, metrics)
        except Exception:
            logger.debug("Horizon accuracy update failed", exc_info=True)
        finally:
            with _HORIZON_ACC_LOCK:
                _HORIZON_ACC_IN_FLIGHT.discard(target_instr)

    th = threading.Thread(target=_worker, args=(instr,), daemon=True)
    th.start()


def _load_readiness_state_cached(instrument_key: str):
    if not READINESS_ENFORCE_LIVE_GATE:
        return None
    instr = str(instrument_key or SYMBOL)
    now = datetime.now(timezone.utc)
    cached = _READINESS_CACHE.get(instr)
    if isinstance(cached, dict):
        ts = cached.get("ts")
        if isinstance(ts, datetime):
            if (now - ts).total_seconds() < 300:
                return cached.get("state")
    try:
        from readiness_tracker import load_readiness_state

        state = load_readiness_state(instr)
    except Exception:
        state = None
    _READINESS_CACHE[instr] = {"ts": now, "state": state}
    return state


def _readiness_locked_state(instrument_key: str):
    """
    Returns (locked: bool, reason: str, state: dict|None).
    """
    if not READINESS_ENFORCE_LIVE_GATE:
        return False, "", None
    state = _load_readiness_state_cached(instrument_key or SYMBOL)
    if not isinstance(state, dict):
        req = max(1, int(READINESS_MIN_PASS_STREAK_DAYS))
        return True, f"Readiness gate locked (no report; need {req}d pass streak)", None
    fresh = bool(state.get("fresh", False))
    live_ready = bool(state.get("live_ready", False))
    streak = int(state.get("pass_streak_days", 0) or 0)
    req = int(state.get("required_streak_days", max(1, int(READINESS_MIN_PASS_STREAK_DAYS))))
    if not fresh:
        return True, (
            f"Readiness gate locked (report stale > {float(READINESS_MAX_AGE_HOURS):.0f}h; "
            f"streak {streak}/{req})"
        ), state
    if not live_ready:
        return True, f"Readiness gate locked (streak {streak}/{req})", state
    return False, "", state

def _safe_instrument_tag(instr_key: str) -> str:
    return str(instr_key or DEFAULT_INSTRUMENT).replace("/", "_").replace("\\", "_")

def _model_path(prefix: str, mode: str, instr_key: str) -> str:
    tag = _safe_instrument_tag(instr_key)
    return os.path.join(MODEL_DIR, f"{prefix}_{mode}_{tag}.pkl")

def _models_dir_signature() -> tuple:
    try:
        entries = []
        for name in sorted(os.listdir(MODEL_DIR)):
            if not str(name).endswith(".pkl"):
                continue
            p = os.path.join(MODEL_DIR, name)
            try:
                st = os.stat(p)
                entries.append((name, int(st.st_mtime_ns), int(st.st_size)))
            except Exception:
                entries.append((name, None, None))
        return tuple(entries)
    except Exception:
        return tuple()

def _load_mode_models(mode: str, instrument_key: str = None):
    mode_res = {}
    instr_key = instrument_key or DEFAULT_INSTRUMENT
    mapping = {
        "base": "base",
        "forest": "forest",
        "boost": "boost",
        "extra_trees": "extra_trees"
    }
    for key, prefix in mapping.items():
        path = _model_path(prefix, mode, instr_key)
        payload = None
        if os.path.exists(path):
            try:
                payload = joblib.load(path)
            except Exception:
                logger.exception(f"Failed to load model file {path}")
                payload = None
        # fallback to legacy non-instrument file if not found (default instrument only)
        if payload is None and instr_key == DEFAULT_INSTRUMENT:
            legacy = os.path.join(MODEL_DIR, f"{prefix}_{mode}.pkl")
            if os.path.exists(legacy):
                try:
                    payload = joblib.load(legacy)
                except Exception:
                    logger.exception(f"Failed to load legacy model file {legacy}")
                    payload = None
        if payload is None:
            continue
        payload.setdefault("features", payload.get("features", None))
        payload.setdefault("ensemble_models", payload.get("ensemble_models", {}))
        payload.setdefault("regime_models", payload.get("regime_models", {}))
        payload.setdefault("horizon_models", payload.get("horizon_models", {}))
        payload.setdefault("meta", payload.get("meta", {}))
        payload.setdefault("performance", payload.get("performance", payload.get("performance_mae", {})))
        payload.setdefault("performance_mae", payload.get("performance_mae", {}))
        payload.setdefault("window_calibration", payload.get("window_calibration", {}))
        payload.setdefault("calibration", payload.get("calibration", {}))
        payload.setdefault("calibration_sig", payload.get("calibration_sig", {"Weak":0.5,"Medium":0.5,"Strong":0.5}))
        mode_res[key] = payload
    return mode_res

def load_models(include_modes: bool = True, instrument_key: str = None):
    res = {}
    instr_key = instrument_key or DEFAULT_INSTRUMENT
    cache_key = (bool(include_modes), str(instr_key))
    signature = _models_dir_signature()
    cached = _MODEL_CACHE.get(cache_key)
    if cached and cached.get("signature") == signature and isinstance(cached.get("models"), dict):
        return cached["models"]
    mapping = {
        "base": BASE_MODEL_FILE,
        "forest": FOREST_MODEL_FILE,
        "boost": BOOST_MODEL_FILE,
        "extra_trees": EXTRA_TREES_MODEL_FILE
    }
    for key, fname in mapping.items():
        # prefer instrument-scoped 4min models
        path = _model_path(key, "4min", instr_key)
        payload = None
        if os.path.exists(path):
            try:
                payload = joblib.load(path)
            except Exception:
                logger.exception(f"Failed to load model file {path}")
                payload = None
        # fallback to legacy paths if not found (default instrument only)
        if payload is None and instr_key == DEFAULT_INSTRUMENT:
            legacy = os.path.join(MODEL_DIR, fname)
            if os.path.exists(legacy):
                try:
                    payload = joblib.load(legacy)
                except Exception:
                    logger.exception(f"Failed to load legacy model file {legacy}")
                    payload = None
        if payload is None and key == "extra_trees" and instr_key == DEFAULT_INSTRUMENT:
            # backward-compatible fallback for older filename
            legacy = os.path.join(MODEL_DIR, "extra_model.pkl")
            if os.path.exists(legacy):
                try:
                    payload = joblib.load(legacy)
                except Exception:
                    logger.exception(f"Failed to load legacy model file {legacy}")
                    payload = None
        if payload is None:
            continue
        payload.setdefault("features", payload.get("features", None))
        payload.setdefault("ensemble_models", payload.get("ensemble_models", {}))
        payload.setdefault("regime_models", payload.get("regime_models", {}))
        payload.setdefault("horizon_models", payload.get("horizon_models", {}))
        payload.setdefault("meta", payload.get("meta", {}))
        payload.setdefault("performance", payload.get("performance", payload.get("performance_mae", {})))
        payload.setdefault("performance_mae", payload.get("performance_mae", {}))
        payload.setdefault("window_calibration", payload.get("window_calibration", {}))
        payload.setdefault("calibration", payload.get("calibration", {}))
        payload.setdefault("calibration_sig", payload.get("calibration_sig", {"Weak":0.5,"Medium":0.5,"Strong":0.5}))
        res[key] = payload
    # Meta model (stacked)
    meta_path = _model_path("meta", "4min", instr_key)
    if os.path.exists(meta_path):
        try:
            res["meta"] = joblib.load(meta_path)
        except Exception:
            logger.exception("Failed to load meta model")
    else:
        if instr_key == DEFAULT_INSTRUMENT:
            meta_legacy = os.path.join(MODEL_DIR, META_MODEL_FILE)
            if os.path.exists(meta_legacy):
                try:
                    res["meta"] = joblib.load(meta_legacy)
                except Exception:
                    logger.exception("Failed to load legacy meta model")
    regime_path = os.path.join(MODEL_DIR, REGIME_MODEL_FILE)
    # prefer instrument-scoped regime
    inst_regime = _model_path("regime", "4min", instr_key)
    if os.path.exists(inst_regime):
        regime_path = inst_regime
    if os.path.exists(regime_path) and (instr_key == DEFAULT_INSTRUMENT or os.path.exists(inst_regime)):
        try:
            regime_payload = joblib.load(regime_path)
            # guard against legacy (non-GMM) regime payloads
            try:
                from regime_model import extract_regime_pipeline
                scaler, gmm = extract_regime_pipeline(regime_payload)
                if gmm is None:
                    legacy_path = os.path.join(MODEL_DIR, "regime_model.legacy.pkl")
                    try:
                        if os.path.exists(regime_path) and not os.path.exists(legacy_path):
                            os.replace(regime_path, legacy_path)
                            logger.warning("Legacy/invalid regime model detected; moved to regime_model.legacy.pkl")
                        else:
                            logger.warning("Legacy/invalid regime model detected; ignoring payload")
                    except Exception:
                        logger.warning("Legacy/invalid regime model detected; ignoring payload")
                    regime_payload = None
            except Exception:
                pass
            res["regime"] = regime_payload
        except Exception:
            logger.exception("Failed to load regime model")
            res["regime"] = None
    else:
        res["regime"] = None
    if include_modes:
        res["modes"] = {}
        for mode in ("daily_high", "daily_low", "weekend"):
            res["modes"][mode] = _load_mode_models(mode, instrument_key=instr_key)
    _MODEL_CACHE[cache_key] = {"signature": signature, "models": res}
    return res

def _model_tiebreaker_epsilon(model_key: str) -> float:
    h = abs(hash(model_key)) % 1000
    return h / 1_000_000.0

def _dynamic_confidence_by_sigbucket_from_payload(pred_ret, resid_std, payload, model_key=None, instrument_key=None):
    sig_map = payload.get("calibration_sig", {"Weak":0.5,"Medium":0.5,"Strong":0.5})
    if resid_std is None or resid_std <= 0:
        resid_std = float(payload.get("meta", {}).get("resid_std", 1e-6))
    try:
        pred_bias = float(payload.get("meta", {}).get("pred_bias", 0.0))
    except Exception:
        pred_bias = 0.0
    adj_pred = pred_ret - pred_bias
    z = abs(adj_pred) / (resid_std + 1e-12)
    if z < 0.5:
        sig = "Weak"
    elif z < 1.5:
        sig = "Medium"
    else:
        sig = "Strong"
    prob = sig_map.get(sig, None)
    if prob is None:
        try:
            z = adj_pred / (resid_std + 1e-12)
            prob = 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0)))
        except Exception:
            prob = 0.5
    # blend with live calibration (if available)
    if model_key is not None and instrument_key is not None:
        live_prob, n = _live_calibration_prob(instrument_key, model_key, sig)
        if live_prob is not None and n > 5:
            w = min(0.5, n / 200.0)
            prob = (1.0 - w) * prob + w * live_prob
    return max(0.01, min(0.99, float(prob)))

def _aggregate_preds(members_preds):
    if members_preds.size == 0:
        return 0.0, 1e-6, 0.0, 0.0, 0.0
    mean = float(np.mean(members_preds))
    std = float(np.std(members_preds))
    q10 = float(np.quantile(members_preds, 0.1))
    q50 = float(np.quantile(members_preds, 0.5))
    q90 = float(np.quantile(members_preds, 0.9))
    return mean, max(std, 1e-6), q10, q50, q90


def _prediction_band(last_price: float, pred_ret: float, std_ret: float, instr_meta: dict | None = None, z_mult: float = 1.281551565545):
    """
    Return an approximate 80% prediction band around pred_ret in price terms.
    """
    try:
        lp = float(last_price)
        pr = float(pred_ret)
        sr = abs(float(std_ret))
    except Exception:
        return None, None, None
    lo_ret = pr - (float(z_mult) * sr)
    hi_ret = pr + (float(z_mult) * sr)
    lo = _round_to_tick(lp * (1.0 + lo_ret), instr_meta or {})
    hi = _round_to_tick(lp * (1.0 + hi_ret), instr_meta or {})
    exp_abs_err_bps = float(sr * 10000.0)
    return float(round(min(lo, hi), 4)), float(round(max(lo, hi), 4)), exp_abs_err_bps


def _prediction_bands(last_price: float, pred_ret: float, std_ret: float, instr_meta: dict | None = None) -> dict:
    lo70, hi70, _ = _prediction_band(last_price, pred_ret, std_ret, instr_meta=instr_meta, z_mult=1.0364333894937898)
    lo80, hi80, err_bps = _prediction_band(last_price, pred_ret, std_ret, instr_meta=instr_meta, z_mult=1.281551565545)
    lo95, hi95, _ = _prediction_band(last_price, pred_ret, std_ret, instr_meta=instr_meta, z_mult=1.959963984540054)
    return {
        "pred_range_70_low": lo70,
        "pred_range_70_high": hi70,
        "pred_range_80_low": lo80,
        "pred_range_80_high": hi80,
        "pred_range_95_low": lo95,
        "pred_range_95_high": hi95,
        "expected_abs_err_bps": err_bps,
    }


def _direction_prob_fields(direction: str, confidence_pct: float | None) -> dict:
    try:
        conf = max(0.01, min(0.99, float(confidence_pct) / 100.0))
    except Exception:
        conf = 0.5
    direction = str(direction or "-").upper()
    if direction == "UP":
        p_up = conf
    elif direction == "DOWN":
        p_up = 1.0 - conf
    else:
        p_up = 0.5
    p_up = max(0.001, min(0.999, float(p_up)))
    return {
        "p_up": round(float(p_up), 6),
        "p_down": round(float(1.0 - p_up), 6),
    }


def _effective_model_count(weights) -> float:
    try:
        arr = np.asarray(list(weights), dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if arr.size == 0:
            return 0.0
        arr = arr / float(np.sum(arr))
        return float(1.0 / np.sum(arr ** 2))
    except Exception:
        return 0.0


def _weighted_prediction_dispersion(model_preds, per_model_weights: dict | None = None) -> tuple[float, float]:
    vals = []
    weights = []
    for key, pred in (model_preds or []):
        try:
            p = float(pred)
        except Exception:
            continue
        if not np.isfinite(p):
            continue
        vals.append(p)
        try:
            w = float((per_model_weights or {}).get(key, 1.0))
        except Exception:
            w = 1.0
        weights.append(max(1e-9, w))
    if not vals:
        return 0.0, 0.0
    arr = np.asarray(vals, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = w / float(np.sum(w))
    mean = float(np.sum(w * arr))
    var = float(np.sum(w * ((arr - mean) ** 2)))
    return max(0.0, math.sqrt(max(0.0, var))), _effective_model_count(weights)


def _recent_live_abs_error(live_cal: dict | None, min_n: int = 20):
    if not isinstance(live_cal, dict) or bool(live_cal.get("_stale", False)):
        return None
    try:
        vals = [float(v) for v in live_cal.get("recent_abs_errors", [])]
    except Exception:
        vals = []
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr) & (arr >= 0)]
    if arr.size < max(1, int(min_n)):
        return None
    return float(np.median(arr))


def _posterior_shrink_return(pred_ret: float, uncertainty_ret: float) -> tuple[float, float]:
    """
    Empirical-Bayes style shrinkage toward zero.
    Use a linear-in-SNR shrink rather than a quadratic collapse because
    uncertainty is already charged again in the live edge gate and position sizing.
    """
    try:
        mu = float(pred_ret)
        sigma = abs(float(uncertainty_ret))
    except Exception:
        return 0.0, 0.0
    if (not np.isfinite(mu)) or (not np.isfinite(sigma)):
        return 0.0, 0.0
    if sigma <= 1e-12:
        return mu, 1.0
    snr = abs(mu) / sigma
    shrink = snr / (1.0 + snr)
    return float(mu * shrink), float(shrink)


def _uncertainty_scale_from_prob(prob: float | None) -> float:
    """
    Convert calibrated directional reliability into a modest uncertainty adjustment.
    This avoids charging high-quality signal buckets the full global residual std
    while leaving weak buckets essentially unchanged.
    """
    try:
        p = float(prob)
    except Exception:
        return 1.0
    if not np.isfinite(p):
        return 1.0
    p = max(0.50, min(0.80, p))
    edge = (p - 0.50) / 0.30
    return float(max(0.72, min(1.05, 1.05 - (0.25 * edge))))


def _combine_uncertainty_estimates(
    base_uncertainty: float | None,
    live_uncertainty: float | None = None,
    reliability_prob: float | None = None,
) -> float:
    """
    Blend ensemble/disagreement uncertainty with live residual uncertainty.
    These components are correlated views of the same forecast error, so
    summing them in quadrature tends to double-count risk in the recent window.
    Keep the blend conservative, but avoid RSS over-penalization.
    """
    try:
        base = max(0.0, float(base_uncertainty or 0.0))
    except Exception:
        base = 0.0
    if live_uncertainty is None:
        return float(base)
    try:
        live = max(0.0, float(live_uncertainty))
    except Exception:
        return float(base)
    high = max(base, live)
    low = min(base, live)
    try:
        rel = float(reliability_prob)
    except Exception:
        rel = 0.50
    if not np.isfinite(rel):
        rel = 0.50
    rel = max(0.50, min(0.75, rel))
    low_w = 0.20 + (0.08 * ((rel - 0.50) / 0.25))
    low_w = max(0.20, min(0.28, low_w))
    return float(((1.0 - low_w) * high) + (low_w * low))


def _resolve_direction_stage(
    raw_direction: str | None,
    confidence_pct: float | None,
    edge_ret: float | None,
    *,
    conformal_block: bool = False,
    readiness_locked: bool = False,
) -> tuple[str, str]:
    direction = str(raw_direction or "-").upper()
    if direction not in {"UP", "DOWN"}:
        return "-", "blocked"
    try:
        conf_pct = float(confidence_pct or 0.0)
    except Exception:
        conf_pct = 0.0
    try:
        edge_val = float(edge_ret or 0.0)
    except Exception:
        edge_val = 0.0
    if edge_val <= 0.0 or bool(conformal_block) or bool(readiness_locked):
        return "-", "blocked"
    if conf_pct >= (float(NO_SIGNAL_CONF) * 100.0):
        return direction, "finished"
    if conf_pct >= float(TENTATIVE_DIRECTION_CONF_PCT):
        return direction, "tentative"
    return "-", "blocked"


def _ensemble_predict_members(members, X):
    try:
        cap = int(MAX_ENSEMBLE_MEMBERS_PREDICT)
    except Exception:
        cap = 0
    if cap > 0 and members and len(members) > cap:
        try:
            idxs = np.linspace(0, len(members) - 1, cap, dtype=int).tolist()
            members = [members[i] for i in idxs]
        except Exception:
            members = members[:cap]
    preds = []
    for m in members:
        try:
            if m is None:
                continue
            p = m.predict(X)
            if hasattr(p, "__len__"):
                preds.append(float(p[0]))
            else:
                preds.append(float(p))
        except Exception:
            continue
    return np.array(preds) if len(preds) > 0 else np.array([])

def fallback_regime_probs(df: pd.DataFrame) -> dict:
    out = {"Bullish": 0.0, "Neutral": 0.0, "Bearish": 0.0}
    if df is None or df.empty:
        return {"Bullish": 0.0, "Neutral": 100.0, "Bearish": 0.0, "dominant_component": "0", "component_probs": [1.0] + [0.0]*3, "regime_conf": 0.0}
    recent = df["close"].iloc[-24:] if len(df) >= 24 else df["close"]
    if recent.empty:
        return {"Bullish": 0.0, "Neutral": 100.0, "Bearish": 0.0, "dominant_component": "0", "component_probs": [1.0] + [0.0]*3, "regime_conf": 0.0}
    ret_6 = recent.pct_change(6).dropna()
    mean_short = float(ret_6.mean()) if not ret_6.empty else 0.0
    vol_short = float(ret_6.std()) if not ret_6.empty else 0.0
    score = mean_short / (vol_short + 1e-9)
    if score > 0.2:
        out["Bullish"] = 0.7
        out["Neutral"] = 0.25
        out["Bearish"] = 0.05
        dominant = "3"
        comp_probs = [0.05, 0.1, 0.1, 0.75]
    elif score < -0.2:
        out["Bullish"] = 0.05
        out["Neutral"] = 0.25
        out["Bearish"] = 0.7
        dominant = "0"
        comp_probs = [0.75, 0.1, 0.1, 0.05]
    else:
        out["Bullish"] = 0.2
        out["Neutral"] = 0.6
        out["Bearish"] = 0.2
        dominant = "1"
        comp_probs = [0.2, 0.4, 0.3, 0.1]
    for k in out:
        out[k] = round(float(out[k] * 100.0), 1)
        out["dominant_component"] = dominant
        out["component_probs"] = comp_probs
    out["regime_conf"] = 0.3
    return out

def predict_4min(
    models_dict,
    df_latest,
    instrument_key: str = None,
    log_side_effects: bool = True,
    now_utc: pd.Timestamp | None = None,
    use_live_artifacts: bool = True,
    readiness_override: dict | None = None,
    live_calibration_override: dict | None = None,
    feature_frame_override: pd.DataFrame | None = None,
    stock_horizon_blend: bool = True,
):
    results = {}
    if df_latest is None or df_latest.empty:
        return results
    now_ref = _coerce_now_utc(now_utc)
    df_latest, partial_dropped = _drop_incomplete_last_bar(df_latest, instrument_key=instrument_key, now_utc=now_ref)
    if df_latest is None or df_latest.empty:
        return {"error": "No complete bars available"}
    last_price = float(df_latest["close"].iloc[-1])
    instr_meta = _instrument_meta(instrument_key)

    regime_payload = models_dict.get("regime")
    if not regime_payload:
        return {"error": "Regime model missing"}
    try:
        rp = predict_regime_probs(
            regime_payload,
            df_latest,
            recent_look_hours=max(1, int(REGIME_LOOK_HOURS_4MIN)),
            horizon_key="intraday",
        )
    except Exception:
        return {"error": "Regime model invalid"}

    if live_calibration_override is not None:
        live_cal = dict(live_calibration_override)
    elif use_live_artifacts and LIVE_CALIBRATION:
        live_cal = _load_live_calibration(instrument_key)
    else:
        live_cal = None
    drift_flag = bool(live_cal.get("drift_flag", False) or live_cal.get("recent_drift_flag", False)) if live_cal else False
    cal_stale = bool(live_cal.get("_stale", False)) if live_cal else False
    if readiness_override is not None:
        readiness_state = dict(readiness_override)
        readiness_locked = not (bool(readiness_state.get("live_ready")) and bool(readiness_state.get("fresh")))
        readiness_reason = None if not readiness_locked else str(readiness_state.get("reason") or "readiness override blocked trading")
    elif use_live_artifacts:
        readiness_locked, readiness_reason, readiness_state = _readiness_locked_state(instrument_key or SYMBOL)
    else:
        readiness_locked, readiness_reason, readiness_state = False, None, {"live_ready": True, "fresh": True}
    if log_side_effects:
        try:
            _schedule_fast_retrain_on_drift(instrument_key or SYMBOL, live_cal)
        except Exception:
            logger.debug("Drift retrain scheduler failed", exc_info=True)

    dominant_regime = rp.get("dominant_component", None)
    comp_probs = rp.get("component_probs", None)
    dom_prob = 0.0
    if dominant_regime is None:
        dominant_regime = "0"
    try:
        if isinstance(comp_probs, (list, tuple)) and str(dominant_regime).isdigit():
            dom_idx = int(dominant_regime)
            if 0 <= dom_idx < len(comp_probs):
                dom_prob = float(comp_probs[dom_idx])
    except Exception:
        dom_prob = 0.0
    if dom_prob == 0.0 and dominant_regime is not None and comp_probs is None:
        dom_prob = 0.6

    stale = _staleness_info(df_latest, instrument_key=instrument_key, now_utc=now_ref)
    stale_factor = _stale_conf_factor(stale)
    critical_stale = _is_critical_stale(stale)
    stale_note = ""
    if stale.get("is_stale"):
        try:
            mins = stale.get("age_sec", 0.0) / 60.0 if stale.get("age_sec") is not None else None
            stale_note = f"Stale data ({mins:.1f}m old)" if mins is not None else "Stale data"
        except Exception:
            stale_note = "Stale data"

    results_out = {}
    # Always include regime in output (even if models fail later)
    results_out["regime"] = rp
    # Meta info for logging/audit
    try:
        last_ts = pd.to_datetime(df_latest.index[-1], utc=True, errors="coerce")
        data_ts = last_ts.isoformat() if pd.notna(last_ts) else None
    except Exception:
        data_ts = None
    results_out["_meta"] = {
        "data_ts": data_ts,
        "model_versions": {},
        "staleness": stale,
        "partial_bar_dropped": bool(partial_dropped),
        "regime_look_hours": int(REGIME_LOOK_HOURS_4MIN),
        "readiness_locked": bool(readiness_locked),
        "readiness_state": readiness_state if isinstance(readiness_state, dict) else None,
    }
    if critical_stale:
        guard_note = "No-trade guard: critical stale data"
        if stale_note:
            guard_note = f"{guard_note} | {stale_note}"
        frozen_price = _round_to_tick(last_price, instr_meta)
        for key in ["base", "forest", "boost", "extra_trees", "meta"]:
            results_out[key] = {
                "direction": "-",
                "pred_price": float(round(frozen_price, 4)),
                "confidence": 1.0,
                "note": guard_note,
                "pred_ret": 0.0,
                "pred_norm": 0.0,
                "pred_std_norm": 1e-6,
                "sig_bucket": "Weak",
            }
        results_out["consensus"] = {
            "direction": "-",
            "pred_price": float(round(frozen_price, 4)),
            "confidence": 1.0,
            "note": guard_note,
            "position_size": 0.0,
            "pred_ret": 0.0,
            "sig_bucket": "Weak",
        }
        results_out["horizon"] = {
            "h10_norm": None,
            "h30_norm": None,
            "align_score": 0.0,
            "support_stage": "unavailable",
            "support_score": 0.0,
        }
        results_out["_meta"]["critical_stale"] = True
        results_out["_meta"]["diversity"] = _prediction_diversity(results_out)
        if log_side_effects:
            _log_prediction(instrument_key, last_price, results_out)
        return results_out
    model_preds = []
    per_model_weights = {}
    model_uncertainty_rets = []

    # base feature row for meta and sizing (compute once and cache)
    feat_frame = feature_frame_override if isinstance(feature_frame_override, pd.DataFrame) else _live_feature_frame(df_latest, instrument_key=instrument_key)
    try:
        if feat_frame is not None and not feat_frame.empty and len(feat_frame) > LIVE_MAX_BARS:
            feat_frame = feat_frame.iloc[-LIVE_MAX_BARS:].copy()
    except Exception:
        pass
    feat_cache: dict = {}
    try:
        X_base_feat = _get_feature_row_cached(
            df_latest,
            DEFAULT_FEATURES + ["trend_strength"],
            instrument_key=instrument_key,
            feature_frame=feat_frame,
            cache=feat_cache
        )
    except Exception:
        X_base_feat = None
    meta_ewm_vol = float(X_base_feat.get("ewm_vol", 0.001).iloc[0]) if X_base_feat is not None and "ewm_vol" in X_base_feat.columns else 0.001
    meta_ewm_vol = max(meta_ewm_vol, _vol_floor_from_meta(instr_meta))
    meta_trend = float(X_base_feat.get("trend_strength", 0.0).iloc[0]) if X_base_feat is not None and "trend_strength" in X_base_feat.columns else 0.0
    predictability_gate = _resolve_predictability_gate(models_dict)
    predictable_now = True
    if isinstance(predictability_gate, dict):
        try:
            gate_mask = apply_intraday_activity_gate(
                [last_price],
                [meta_ewm_vol],
                predictability_gate,
                instrument_key=instrument_key,
            )
            predictable_now = bool(gate_mask[0]) if len(gate_mask) else True
        except Exception:
            predictable_now = True
    results_out["_meta"]["predictability_gate"] = {
        "enabled": bool(isinstance(predictability_gate, dict)),
        "predictable_now": bool(predictable_now),
        "scale": float(predictability_gate.get("scale")) if isinstance(predictability_gate, dict) and predictability_gate.get("scale") is not None else None,
        "quantile": float(predictability_gate.get("quantile")) if isinstance(predictability_gate, dict) and predictability_gate.get("quantile") is not None else None,
        "role": str(predictability_gate.get("role")) if isinstance(predictability_gate, dict) and predictability_gate.get("role") is not None else None,
    }

    for key in ["base", "forest", "boost", "extra_trees"]:
        payload = models_dict.get(key)
        if payload is None:
            continue
        try:
            ver = payload.get("meta", {}).get("version")
            results_out["_meta"]["model_versions"][key] = ver
        except Exception:
            pass

        feat_list = payload.get("features")
        try:
            X = _get_feature_row_cached(
                df_latest,
                feat_list,
                instrument_key=instrument_key,
                feature_frame=feat_frame,
                cache=feat_cache
            )
        except Exception:
            results_out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Feature error"}
            continue
        if X is None or X.empty:
            results_out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Waiting…"}
            continue

        try:
            current_vol = float(X.get("ewm_vol", 0.001).iloc[0]) if "ewm_vol" in X.columns else 0.001
        except Exception:
            current_vol = 0.001
        # Use additional vol proxies and a floor to avoid near-zero predictions
        vol_candidates = [current_vol]
        for col in ("vol_12", "vol_6", "vol_3", "session_volatility"):
            try:
                if col in X.columns:
                    vol_candidates.append(float(X[col].iloc[0]))
            except Exception:
                continue
        current_vol = max([v for v in vol_candidates if np.isfinite(v)] + [_vol_floor_from_meta(instr_meta)])

        perf = payload.get("performance", payload.get("performance_mae", {})) if payload else {}
        combined_summary = _ensemble_summary_for_payload(payload, X, dominant_regime, dom_prob, comp_probs=comp_probs)
        used_fallback_summary = False

        # Require ensemble summary; fall back to single-model if available
        if combined_summary is None:
            fb = payload.get("fallback_model")
            if fb is None:
                fb = payload.get("model")
            if fb is not None:
                try:
                    pred_norm = float(fb.predict(X)[0])
                    pred_std_norm = float(payload.get("meta", {}).get("resid_std", 1e-6))
                    pred_std_norm = max(pred_std_norm, 1e-6)
                    combined_summary = {"mean": pred_norm, "std": pred_std_norm, "q10": 0.0, "q50": pred_norm, "q90": 0.0, "p_up": float(pred_norm >= 0)}
                    used_fallback_summary = True
                except Exception:
                    combined_summary = None
            if combined_summary is None:
                results_out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Ensemble not trained"}
                continue
        if used_fallback_summary:
            combined_summary = _restore_summary_target_mean(combined_summary, payload)
            combined_summary = _apply_fallback_return_calibration(combined_summary, payload.get("calibration"))

        pred_norm = combined_summary["mean"]
        pred_std_norm = combined_summary["std"]

        # Bayesian shrinkage toward 0 (prior variance scaled by regime confidence)
        try:
            reg_conf = float(rp.get("regime_conf", 0.5))
        except Exception:
            reg_conf = 0.5
        prior_var = max(0.2, min(2.0, 0.6 + reg_conf))
        shrink = prior_var / (prior_var + (pred_std_norm ** 2))
        pred_norm_adj = pred_norm * shrink

        # Regime support adjustment (boost confidence, not direction)
        try:
            support = rp.get("Bullish", 0.0) / 100.0 if pred_norm_adj >= 0 else rp.get("Bearish", 0.0) / 100.0
        except Exception:
            support = 0.0
        try:
            regime_conf = float(rp.get("regime_conf", 1.0))
        except Exception:
            regime_conf = 1.0
        support *= (0.5 + 0.5 * max(0.0, min(1.0, regime_conf)))

        target_normed = _infer_target_norm(payload.get("meta", {}))
        pred_ret_adj = pred_norm_adj * current_vol if target_normed else pred_norm_adj
        max_abs_ret = _max_abs_ret_from_vol(current_vol, instr_meta)
        pred_ret_adj = max(-max_abs_ret, min(max_abs_ret, pred_ret_adj))

        # Optional live bias correction (conservative, based on recent errors).
        pred_ret_adj, bias_adj = _live_bias_adjustment(
            live_cal, key, pred_ret_adj, max_abs_ret, drift_flag=drift_flag
        )
        if bias_adj != 0.0:
            try:
                if target_normed:
                    pred_norm_adj += bias_adj / max(current_vol, 1e-9)
                else:
                    pred_norm_adj += bias_adj
            except Exception:
                pass
            pred_ret_adj = max(-max_abs_ret, min(max_abs_ret, pred_ret_adj))
        else:
            bias_adj = 0.0

        # Calibrate p_up using signal-bucket calibration if available (bias-adjusted signal)
        p_correct_sig = _dynamic_confidence_by_sigbucket_from_payload(
            pred_norm_adj, pred_std_norm, payload, model_key=key, instrument_key=instrument_key
        )
        p_up_sig = p_correct_sig if pred_norm_adj >= 0 else (1.0 - p_correct_sig)
        p_up = 0.7 * combined_summary["p_up"] + 0.3 * p_up_sig
        p_up = max(0.001, min(0.999, p_up))
        uncertainty_scale = _uncertainty_scale_from_prob(p_correct_sig)

        pred_price = _round_to_tick(last_price * (1.0 + pred_ret_adj), instr_meta)

        # Direction should reflect the displayed price move
        if pred_price > last_price:
            direction = "UP"
        elif pred_price < last_price:
            direction = "DOWN"
        else:
            direction = "UP" if pred_ret_adj >= 0 else "DOWN"
        conf = (p_up if direction == "UP" else (1.0 - p_up))
        conf *= (0.7 + 0.3 * support)

        # Penalize high dispersion relative to mean
        dispersion = pred_std_norm / (abs(pred_norm_adj) + 1e-9)
        dispersion = min(dispersion, 3.0)
        conf *= (1.0 / (1.0 + 0.35 * dispersion))

        # Staleness penalty (severity-scaled)
        if stale.get("is_stale"):
            conf *= stale_factor
        if cal_stale:
            conf *= 0.85

        # Drift guard
        if drift_flag:
            conf *= 0.7
        # Keep true low-confidence values visible instead of pinning many models to 20%.
        conf = max(0.01, min(0.99, conf))
        # Additional risk-off for extreme volatility
        try:
            vol_guard = current_vol > 0.01
        except Exception:
            vol_guard = False
        low_conf = (conf < NO_SIGNAL_CONF) or (dispersion > NO_SIGNAL_DISP) or vol_guard or (stale_factor < 0.2)

        # pred_price already computed above

        note_parts = []
        if stale_note:
            note_parts.append(stale_note)
        if partial_dropped:
            note_parts.append("Partial bar dropped")
        if abs(bias_adj) > 1e-6:
            note_parts.append(f"Bias adj {bias_adj * 10000.0:+.2f} bps")
        if low_conf:
            note_parts.append("Low confidence")
        if vol_guard:
            note_parts.append("High volatility")
        if drift_flag:
            note_parts.append("Drift guard")
        if cal_stale:
            note_parts.append("Calibration stale")
        if not predictable_now:
            note_parts.append("Outside 4min predictability gate")
        note = " | ".join(note_parts)
        try:
            std_ret = float(pred_std_norm * current_vol) if target_normed else float(pred_std_norm)
        except Exception:
            std_ret = 0.0
        std_ret *= float(uncertainty_scale)
        band_fields = _prediction_bands(last_price, pred_ret_adj, std_ret, instr_meta=instr_meta)
        try:
            model_uncertainty_rets.append(abs(float(std_ret)))
        except Exception:
            pass
        conf_pct = round(float(conf * 100.0), 2)
        results_out[key] = {
            "direction": direction,
            "pred_price": float(round(pred_price, 4)),
            "confidence": conf_pct,
            "note": note,
            "pred_ret": float(pred_ret_adj),
            "pred_norm": float(pred_norm_adj),
            "pred_std_norm": float(pred_std_norm),
            "reliability_prob": round(float(p_correct_sig), 4),
            "uncertainty_scale": round(float(uncertainty_scale), 4),
            "sig_bucket": "Strong" if (abs(pred_norm_adj) / (pred_std_norm + 1e-12)) >= 1.5 else "Medium" if (abs(pred_norm_adj) / (pred_std_norm + 1e-12)) >= 0.5 else "Weak",
            **band_fields,
            **_direction_prob_fields(direction, conf_pct),
        }
        model_preds.append((key, pred_ret_adj))
        # derive model weight from blended window performance
        try:
            blended_mae = 0.0
            deploy_mult = 1.0
            for w in ["short", "mid", "long"]:
                entry = perf.get(w, 1.0) if isinstance(perf, dict) else 1.0
                if isinstance(entry, dict):
                    try:
                        mae = float(entry.get("mae", 1.0))
                    except Exception:
                        mae = 1.0
                    try:
                        active = float(entry.get("active_frac", float("nan")))
                    except Exception:
                        active = float("nan")
                    try:
                        window_ratio = float(entry.get("window_ratio_mean", float("nan")))
                    except Exception:
                        window_ratio = float("nan")
                    try:
                        window_hit = float(entry.get("window_hit_rate", float("nan")))
                    except Exception:
                        window_hit = float("nan")
                    try:
                        mae_dollars = float(entry.get("mae_dollars", float("nan")))
                    except Exception:
                        mae_dollars = float("nan")
                    if np.isfinite(active):
                        if active < 0.03:
                            deploy_mult *= 0.85
                        elif active > 0.70:
                            deploy_mult *= 0.90
                    if np.isfinite(window_ratio):
                        deploy_mult *= max(0.40, min(1.55, 1.0 / max(0.60, window_ratio)))
                    if np.isfinite(window_hit):
                        deploy_mult *= max(0.55, min(1.45, window_hit / 0.60))
                    if np.isfinite(mae_dollars) and mae_dollars > 0:
                        deploy_mult *= max(0.60, min(1.35, 1.0 / max(0.20, mae_dollars)))
                else:
                    try:
                        mae = float(entry) if entry is not None else 1.0
                    except Exception:
                        mae = 1.0
                if not np.isfinite(mae) or mae <= 0:
                    mae = 1.0
                blended_mae += win_weights.get(w, 1.0/3.0) * mae
            base_w = 1.0 / (blended_mae + 1e-6)
            base_w *= max(0.50, min(2.0, deploy_mult))
            base_w *= _live_model_weight_multiplier(live_cal, key)
            per_model_weights[key] = max(1e-6, float(base_w))
        except Exception:
            per_model_weights[key] = 1.0

    # Meta-model stacking (if available)
    meta_payload = models_dict.get("meta")
    if meta_payload and meta_payload.get("model") is not None:
        try:
            try:
                ver = meta_payload.get("meta", {}).get("version")
                results_out["_meta"]["model_versions"]["meta"] = ver
            except Exception:
                pass
            meta_features = meta_payload.get("features", [])
            meta_row = {
                "base": results_out.get("base", {}).get("pred_norm"),
                "forest": results_out.get("forest", {}).get("pred_norm"),
                "boost": results_out.get("boost", {}).get("pred_norm"),
                "extra_trees": results_out.get("extra_trees", {}).get("pred_norm"),
                "ewm_vol": meta_ewm_vol,
                "trend_strength": meta_trend
            }
            # build feature row with proper ordering
            meta_vec = []
            for f in meta_features:
                meta_vec.append(meta_row.get(f, 0.0))
            meta_X = pd.DataFrame([meta_vec], columns=meta_features)
            pred_meta_norm = float(meta_payload["model"].predict(meta_X)[0])
            meta_resid = float(meta_payload.get("meta", {}).get("resid_std", 1e-6))
            meta_resid = max(meta_resid, 1e-6)
            try:
                if bool(meta_payload.get("meta", {}).get("demeaned")):
                    pred_meta_norm += float(meta_payload.get("meta", {}).get("target_mean", 0.0) or 0.0)
            except Exception:
                pass
            # Bayesian shrinkage for meta
            try:
                reg_conf = float(rp.get("regime_conf", 0.5))
            except Exception:
                reg_conf = 0.5
            prior_var = max(0.2, min(2.0, 0.6 + reg_conf))
            shrink = prior_var / (prior_var + (meta_resid ** 2))
            pred_meta_norm = pred_meta_norm * shrink
            p_up_meta = 0.5 * (1.0 + math.erf((pred_meta_norm / (meta_resid + 1e-12)) / math.sqrt(2.0)))
            conf_meta = max(0.01, min(0.99, p_up_meta if pred_meta_norm >= 0 else (1.0 - p_up_meta)))
            meta_uncertainty_scale = _uncertainty_scale_from_prob(conf_meta)
            if stale.get("is_stale"):
                conf_meta *= 0.5
            if drift_flag:
                conf_meta *= 0.7
            meta_normed = _infer_target_norm(meta_payload.get("meta", {}))
            meta_ret = pred_meta_norm * meta_ewm_vol if meta_normed else pred_meta_norm
            max_abs = _max_abs_ret_from_vol(meta_ewm_vol, instr_meta)
            meta_ret = max(-max_abs, min(max_abs, meta_ret))
            meta_ret, meta_bias_adj = _live_bias_adjustment(
                live_cal, "meta", meta_ret, max_abs, drift_flag=drift_flag
            )
            if meta_bias_adj != 0.0:
                try:
                    if meta_normed:
                        pred_meta_norm += meta_bias_adj / max(meta_ewm_vol, 1e-9)
                    else:
                        pred_meta_norm += meta_bias_adj
                except Exception:
                    pass
                meta_ret = max(-max_abs, min(max_abs, meta_ret))
            else:
                meta_bias_adj = 0.0
            meta_price = _round_to_tick(last_price * (1.0 + meta_ret), instr_meta)
            try:
                meta_std_ret = float(meta_resid * meta_ewm_vol) if meta_normed else float(meta_resid)
            except Exception:
                meta_std_ret = 0.0
            meta_std_ret *= float(meta_uncertainty_scale)
            meta_band_fields = _prediction_bands(last_price, meta_ret, meta_std_ret, instr_meta=instr_meta)
            try:
                model_uncertainty_rets.append(abs(float(meta_std_ret)))
            except Exception:
                pass
            meta_note = "Meta ensemble" + (f" | Bias adj {meta_bias_adj * 10000.0:+.2f} bps" if abs(meta_bias_adj) > 1e-6 else "")
            meta_direction = "UP" if pred_meta_norm >= 0 else "DOWN"
            if not predictable_now:
                meta_note += " | Outside 4min predictability gate"
            meta_conf_pct = round(float(conf_meta * 100.0), 2)
            results_out["meta"] = {
                "direction": meta_direction,
                "pred_price": float(round(meta_price, 4)),
                "confidence": meta_conf_pct,
                "note": meta_note,
                "pred_ret": float(meta_ret),
                "pred_norm": float(pred_meta_norm),
                "pred_std_norm": float(meta_resid),
                "reliability_prob": round(float(conf_meta), 4),
                "uncertainty_scale": round(float(meta_uncertainty_scale), 4),
                "sig_bucket": "Strong" if (abs(pred_meta_norm) / (meta_resid + 1e-12)) >= 1.5 else "Medium" if (abs(pred_meta_norm) / (meta_resid + 1e-12)) >= 0.5 else "Weak",
                **meta_band_fields,
                **_direction_prob_fields(meta_direction, meta_conf_pct),
            }
            model_preds.append(("meta", meta_ret))
            meta_w = 1.0 / (meta_resid + 1e-6)
            meta_w *= _live_model_weight_multiplier(live_cal, "meta")
            per_model_weights["meta"] = max(1e-6, float(meta_w))
        except Exception:
            logger.debug("Meta prediction failed", exc_info=True)

    if not model_preds:
        return results_out

    # Consensus across models
    effective_models = 0.0
    try:
        total_w = sum(per_model_weights.values()) or len(model_preds)
        consensus_ret = 0.0
        pred_vals = []
        for k, pred in model_preds:
            w = per_model_weights.get(k, 1.0)
            consensus_ret += pred * (w / total_w)
            pred_vals.append(float(pred))
        if pred_vals:
            med_ret = float(np.median(pred_vals))
            consensus_ret = (0.75 * float(consensus_ret)) + (0.25 * med_ret)
        disagreement, effective_models = _weighted_prediction_dispersion(model_preds, per_model_weights)
    except Exception:
        consensus_ret = float(np.mean([p for _, p in model_preds]))
        disagreement = float(np.std([p for _, p in model_preds])) if len(model_preds) > 0 else 0.0
        effective_models = float(len(model_preds))
    # Optional bias correction on consensus signal (uses live calibration stats).
    try:
        max_abs_cons = _max_abs_ret_from_vol(meta_ewm_vol, instr_meta)
    except Exception:
        max_abs_cons = MAX_ABS_RET
    consensus_ret, consensus_bias_adj = _live_bias_adjustment(
        live_cal, "consensus", consensus_ret, max_abs_cons, drift_flag=drift_flag
    )
    if consensus_bias_adj != 0.0:
        consensus_ret = max(-max_abs_cons, min(max_abs_cons, consensus_ret))
    else:
        consensus_bias_adj = 0.0
    try:
        max_abs = _max_abs_ret_from_vol(float(np.nanmean([abs(p) for _, p in model_preds]) or 0.0), instr_meta)
        agreement_prob = max(0.01, min(0.99, (1.0 - min(1.0, disagreement / (max_abs + 1e-6)))))
        model_conf_vals = []
        for mk in ["base", "forest", "boost", "extra_trees", "meta"]:
            cv = results_out.get(mk, {}).get("confidence")
            if isinstance(cv, (int, float)):
                model_conf_vals.append(max(0.01, min(0.99, float(cv) / 100.0)))
        model_prob = float(np.mean(model_conf_vals)) if model_conf_vals else agreement_prob
        # Consensus confidence must reflect both model agreement and base model conviction.
        consensus_prob = (0.30 * agreement_prob) + (0.70 * model_prob)
        consensus_conf = max(1.0, min(99.0, 100.0 * consensus_prob))
        if stale.get("is_stale"):
            consensus_conf *= stale_factor
        if drift_flag:
            consensus_conf *= 0.7
        if cal_stale:
            consensus_conf *= 0.85
    except Exception:
        consensus_conf = 50.0
    raw_consensus_conf = float(consensus_conf)

    edge_inputs = _live_edge_floor_inputs(last_price, df_latest, instr_meta=instr_meta)
    spread_frac = edge_inputs.get("spread_frac")
    cost_ret_est = float(edge_inputs.get("cost_ret_est") or 0.0)
    min_edge_floor = float(edge_inputs.get("min_edge_floor") or 0.0)
    unc_mult = float(edge_inputs.get("uncertainty_mult") or 0.0)
    conformal_band = _conformal_abs_error_threshold(live_cal, quantile=0.80, min_n=40)
    live_abs_err_ret = _recent_live_abs_error(live_cal, min_n=20)
    live_uncertainty = None
    live_unc_candidates = [v for v in (conformal_band, live_abs_err_ret) if v is not None]
    if live_unc_candidates:
        live_uncertainty = float(max(live_unc_candidates))
    try:
        member_uncertainty = float(np.median([u for u in model_uncertainty_rets if np.isfinite(u) and u >= 0]))
    except Exception:
        member_uncertainty = 0.0
    try:
        # Tree/boost/ridge members share the same tape and feature space; treat them as highly correlated.
        corr_floor = 0.75
        eff = max(1.0, float(effective_models))
        diversification = math.sqrt(corr_floor + ((1.0 - corr_floor) / eff))
        ensemble_uncertainty = float(member_uncertainty * diversification)
    except Exception:
        ensemble_uncertainty = float(member_uncertainty)
    combined_uncertainty = max(float(disagreement), float(ensemble_uncertainty))
    combined_uncertainty = _combine_uncertainty_estimates(
        combined_uncertainty,
        live_uncertainty,
        reliability_prob=consensus_prob if "consensus_prob" in locals() else None,
    )
    raw_consensus_ret = float(consensus_ret)
    consensus_ret, shrink_factor = _posterior_shrink_return(consensus_ret, combined_uncertainty)
    combined_price = _round_to_tick(last_price * (1.0 + consensus_ret), instr_meta)
    uncertainty_penalty = unc_mult * max(0.0, combined_uncertainty)
    raw_edge_ret = abs(float(raw_consensus_ret)) - (float(cost_ret_est) + float(uncertainty_penalty) + float(min_edge_floor))
    edge_ret = abs(float(consensus_ret)) - (float(cost_ret_est) + float(uncertainty_penalty) + float(min_edge_floor))
    conformal_block = bool(conformal_band is not None and abs(float(consensus_ret)) <= float(conformal_band))
    if edge_ret <= 0.0:
        consensus_conf = min(float(consensus_conf), 35.0)
    else:
        edge_scale = edge_ret / max(1e-6, float(combined_uncertainty))
        conf_mult = max(0.35, min(1.15, 0.55 + (0.25 * edge_scale)))
        consensus_conf *= conf_mult
    # Uncertainty already shrinks the return itself and is charged again in the live edge gate.
    # Keep confidence sensitive to shrink, but avoid collapsing surviving signals a third time.
    consensus_conf *= (0.60 + (0.40 * max(0.0, min(1.0, shrink_factor))))
    if conformal_block:
        consensus_conf = min(float(consensus_conf), 33.0)
        edge_ret = min(float(edge_ret), float(abs(consensus_ret) - float(conformal_band)))
    consensus_conf = max(1.0, min(99.0, float(consensus_conf)))

    vote_dir = 0.0
    vote_strength = 0.0
    try:
        total_w = sum(per_model_weights.values()) or len(model_preds)
        vote_dir = float(sum(per_model_weights.get(k, 1.0) * np.sign(p) for k, p in model_preds))
        vote_strength = abs(vote_dir) / max(float(total_w), 1e-9)
    except Exception:
        vote_dir = 0.0
        vote_strength = 0.0
    if vote_strength >= 0.12 and vote_dir != 0.0:
        consensus_dir = "UP" if vote_dir > 0 else "DOWN"
    else:
        consensus_dir = "UP" if combined_price > last_price else "DOWN" if combined_price < last_price else "-"
    raw_consensus_dir = str(consensus_dir or "-").upper()
    if raw_consensus_dir not in {"UP", "DOWN"} and raw_consensus_ret != 0.0:
        raw_consensus_dir = "UP" if raw_consensus_ret > 0.0 else "DOWN"
    h10_summary = _predict_horizon_summary(
        models_dict,
        df_latest,
        10,
        instrument_key=instrument_key,
        feature_frame=feat_frame,
        feat_cache=feat_cache,
    )
    h30_summary = _predict_horizon_summary(
        models_dict,
        df_latest,
        30,
        instrument_key=instrument_key,
        feature_frame=feat_frame,
        feat_cache=feat_cache,
    )
    try:
        h10 = None if not isinstance(h10_summary, dict) else float(h10_summary.get("mean"))
    except Exception:
        h10 = None
    try:
        h30 = None if not isinstance(h30_summary, dict) else float(h30_summary.get("mean"))
    except Exception:
        h30 = None
    horizon_support = _stock_horizon_support(
        raw_consensus_dir,
        raw_consensus_ret,
        meta_ewm_vol,
        h10_summary,
        h30_summary,
    )
    if bool(stock_horizon_blend) and edge_ret > 0.0:
        try:
            support_stage = str(horizon_support.get("stage") or "")
        except Exception:
            support_stage = ""
        if support_stage in {"supportive", "confirmed"}:
            consensus_conf += float(horizon_support.get("confidence_bonus", 0.0) or 0.0)
    consensus_conf = max(1.0, min(99.0, float(consensus_conf)))
    consensus_dir, direction_stage = _resolve_direction_stage(
        raw_consensus_dir,
        consensus_conf,
        edge_ret,
        conformal_block=conformal_block,
        readiness_locked=readiness_locked,
    )
    try:
        if bool(stock_horizon_blend):
            align_score = float(horizon_support.get("align_score", 1.0) or 1.0)
        else:
            align = 0.0
            align_models = 0
            if h10 is not None:
                align_models += 1
                align += 1.0 if np.sign(h10) == np.sign(consensus_ret) else -1.0
            if h30 is not None:
                align_models += 1
                align += 1.0 if np.sign(h30) == np.sign(consensus_ret) else -1.0
            if align_models <= 0:
                align_score = 1.0
            else:
                align_score = 0.5 + 0.25 * align
                align_score = max(0.1, min(1.0, align_score))
    except Exception:
        align_score = 1.0
    consensus_note = "Weighted consensus"
    if stale_note:
        consensus_note += f" | {stale_note}"
    if partial_dropped:
        consensus_note += " | Partial bar dropped"
    if abs(consensus_bias_adj) > 1e-6:
        consensus_note += f" | Bias adj {consensus_bias_adj * 10000.0:+.2f} bps"
    if shrink_factor < 0.75:
        consensus_note += f" | Uncertainty shrink {shrink_factor:.2f}"
    support_stage = str(horizon_support.get("stage") or "unavailable").lower()
    if support_stage == "confirmed":
        consensus_note += " | 10m/30m stock horizon confirms the lean"
    elif support_stage == "supportive":
        consensus_note += " | 10m/30m stock horizon supports the lean"
    elif support_stage == "contrary":
        consensus_note += " | 10m/30m stock horizon disagrees"
    if direction_stage == "tentative":
        consensus_note += " | Tentative direction only"
    elif consensus_dir == "-":
        consensus_note += " | Low confidence"
    if edge_ret <= 0.0:
        consensus_note += " | Uncertainty-adjusted edge <= 0"
    if conformal_block:
        consensus_note += " | Conformal uncertainty band"
    if drift_flag:
        consensus_note += " | Drift guard"
    if cal_stale:
        consensus_note += " | Calibration stale"
    if readiness_locked and readiness_reason:
        consensus_note += f" | {readiness_reason}"
    if not predictable_now:
        consensus_note += " | Outside 4min predictability gate"

    # Position sizing (vol targeting + regime confidence)
    try:
        from config import TARGET_VOL
        vol_target = float(instr_meta.get("target_vol", TARGET_VOL))
    except Exception:
        vol_target = float(instr_meta.get("target_vol", 0.004))
    try:
        reg_conf = float(rp.get("regime_conf", 0.5))
    except Exception:
        reg_conf = 0.5
    try:
        size_vol = min(1.0, vol_target / max(meta_ewm_vol, 1e-6))
        size_conf = consensus_conf / 100.0
        position_size = max(0.0, min(1.0, size_vol * size_conf * reg_conf * align_score))
        try:
            signal_quality = abs(float(consensus_ret)) / max(1e-9, float(combined_uncertainty))
        except Exception:
            signal_quality = 0.0
        quality_mult = max(0.0, min(1.0, signal_quality / (1.0 + signal_quality)))
        quality_mult = max(0.25, min(1.0, math.sqrt(max(0.0, quality_mult))))
        position_size *= quality_mult
        if drift_flag:
            position_size *= 0.5
        if bool(stock_horizon_blend):
            try:
                position_size *= float(horizon_support.get("size_mult", 1.0) or 1.0)
            except Exception:
                pass
        raw_position_size = float(position_size)
        if direction_stage == "tentative":
            position_size *= float(TENTATIVE_DIRECTION_SIZE_MULT)
        if consensus_dir == "-":
            position_size = 0.0
        if edge_ret <= 0.0:
            position_size = 0.0
        if conformal_block:
            position_size = 0.0
        if readiness_locked:
            position_size = 0.0
    except Exception:
        position_size = 0.0
        raw_position_size = 0.0

    if readiness_locked:
        consensus_conf = min(float(consensus_conf), 33.0)
    try:
        cons_std_ret = float(combined_uncertainty)
    except Exception:
        cons_std_ret = 0.0
    cons_band_fields = _prediction_bands(last_price, consensus_ret, cons_std_ret, instr_meta=instr_meta)
    consensus_conf_pct = round(float(consensus_conf), 2)
    raw_consensus_conf_pct = round(float(raw_consensus_conf), 2)
    raw_prob_fields = _direction_prob_fields(raw_consensus_dir, raw_consensus_conf_pct)

    results_out["consensus"] = {
        "direction": consensus_dir,
        "direction_stage": direction_stage,
        "raw_direction": raw_consensus_dir,
        "pred_price": float(round(combined_price, 4)),
        "confidence": consensus_conf_pct,
        "raw_confidence": raw_consensus_conf_pct,
        "note": consensus_note,
        "position_size": round(float(position_size), 3),
        "raw_position_size": round(float(raw_position_size), 3),
        "pred_ret": float(consensus_ret),
        "raw_pred_ret": float(raw_consensus_ret),
        "edge_bps": round(float(edge_ret * 10000.0), 3),
        "raw_edge_bps": round(float(raw_edge_ret * 10000.0), 3),
        "cost_bps_est": round(float(cost_ret_est * 10000.0), 3),
        "uncertainty_penalty_bps": round(float(uncertainty_penalty * 10000.0), 3),
        "disagreement_bps": round(float(disagreement * 10000.0), 3),
        "model_uncertainty_bps": round(float(ensemble_uncertainty * 10000.0), 3),
        "uncertainty_bps": round(float(combined_uncertainty * 10000.0), 3),
        "effective_models": round(float(effective_models), 3) if effective_models else None,
        "shrink_factor": round(float(shrink_factor), 4),
        "align_score": round(float(align_score), 3),
        "horizon_support_stage": str(horizon_support.get("stage") or "unavailable"),
        "horizon_support_score": round(float(horizon_support.get("score", 0.0) or 0.0), 4),
        "horizon_support_direction": str(horizon_support.get("direction") or "-").upper(),
        "horizon_blended_norm": round(float(horizon_support.get("blended_norm", 0.0) or 0.0), 4),
        "conformal_band_bps": round(float(conformal_band * 10000.0), 3) if conformal_band is not None else None,
        "live_abs_err_bps": round(float(live_abs_err_ret * 10000.0), 3) if live_abs_err_ret is not None else None,
        "sig_bucket": "Strong" if (abs(consensus_ret) / (combined_uncertainty + 1e-9)) >= 1.5 else "Medium" if (abs(consensus_ret) / (combined_uncertainty + 1e-9)) >= 0.5 else "Weak",
        **cons_band_fields,
        **_direction_prob_fields(consensus_dir, consensus_conf_pct),
        "raw_p_up": raw_prob_fields.get("p_up"),
        "raw_p_down": raw_prob_fields.get("p_down"),
    }
    results_out["_meta"]["signal_diagnostics"] = {
        "raw_pred_ret_bps": round(float(raw_consensus_ret * 10000.0), 3),
        "raw_edge_bps": round(float(raw_edge_ret * 10000.0), 3),
        "post_shrink_pred_ret_bps": round(float(consensus_ret * 10000.0), 3),
        "post_shrink_edge_bps": round(float(edge_ret * 10000.0), 3),
        "cost_bps_est": round(float(cost_ret_est * 10000.0), 3),
        "uncertainty_penalty_bps": round(float(uncertainty_penalty * 10000.0), 3),
        "uncertainty_bps": round(float(combined_uncertainty * 10000.0), 3),
        "shrink_factor": round(float(shrink_factor), 4),
        "align_score": round(float(align_score), 3),
        "horizon_support_stage": str(horizon_support.get("stage") or "unavailable"),
        "horizon_support_score": round(float(horizon_support.get("score", 0.0) or 0.0), 4),
    }

    # Degenerate-output guard: near-identical model outputs are usually low-information snapshots.
    diversity = _prediction_diversity(results_out)
    results_out["_meta"]["critical_stale"] = False
    results_out["_meta"]["diversity"] = diversity
    if diversity.get("degenerate"):
        low_conf_count = 0
        for mk in ["base", "forest", "boost", "extra_trees", "meta"]:
            v = results_out.get(mk, {})
            if isinstance(v, dict) and isinstance(v.get("confidence"), (int, float)):
                if float(v["confidence"]) < (NO_SIGNAL_CONF * 100.0):
                    low_conf_count += 1
        if low_conf_count >= 3:
            try:
                results_out["consensus"]["direction"] = "-"
                results_out["consensus"]["position_size"] = 0.0
                results_out["consensus"]["confidence"] = round(
                    min(float(results_out["consensus"].get("confidence", 50.0)), 35.0), 2
                )
            except Exception:
                pass
        try:
            note = str(results_out["consensus"].get("note", "Weighted consensus"))
            if "Degenerate model spread" not in note:
                results_out["consensus"]["note"] = f"{note} | Degenerate model spread"
        except Exception:
            pass

    results_out["regime"] = rp
    results_out["horizon"] = {
        "h10_norm": h10,
        "h30_norm": h30,
        "align_score": round(float(align_score), 3),
        "support_stage": str(horizon_support.get("stage") or "unavailable"),
        "support_direction": str(horizon_support.get("direction") or "-").upper(),
        "support_score": round(float(horizon_support.get("score", 0.0) or 0.0), 4),
        "blended_norm": round(float(horizon_support.get("blended_norm", 0.0) or 0.0), 4),
        "rows": list(horizon_support.get("rows") or []),
    }
    if log_side_effects:
        _log_prediction(instrument_key, last_price, results_out)
        # non-blocking calibration refresh to avoid latency spikes in prediction path
        try:
            _schedule_live_calibration_update(instrument_key or SYMBOL)
        except Exception:
            logger.debug("Live calibration scheduler failed", exc_info=True)
    return results_out

def _fetch_long_daily(yahoo_sym, tz: str = "America/New_York"):
    if not ALLOW_REMOTE_DAILY:
        return None
    try:
        df = yf.download(yahoo_sym, period='5y', interval='1d', progress=False, auto_adjust=False)
        if df.empty:
            logger.info(f"yfinance daily fetch returned empty for {yahoo_sym}")
            return None
        # Handle MultiIndex columns (yfinance sometimes wraps single ticker)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)  # take field names (Open, High, etc.)
        df = _ensure_utc_index(df, tz=tz)
        df.columns = [c.lower().replace(' ', '_') for c in df.columns]
        logger.info(f"Fetched {len(df)} daily bars for {yahoo_sym}")
        return df
    except Exception as e:
        logger.exception(f"Long daily fetch failed for {yahoo_sym}")
        return None


def _apply_horizon_overlay_block(df_hist: pd.DataFrame, instrument_key: str, horizon: str,
                                 reference_price: float, block: dict) -> dict:
    if not isinstance(block, dict):
        return block
    out = block.copy()
    for model_key, payload in out.items():
        if not isinstance(payload, dict):
            continue
        pred_price = payload.get("pred_price")
        if not isinstance(pred_price, (int, float)):
            continue
        overlay = apply_horizon_overlay(
            df_hist,
            instrument_key=instrument_key,
            horizon=horizon,
            model_key=model_key,
            reference_price=reference_price,
            pred_price=pred_price,
        )
        if not isinstance(overlay, dict):
            continue
        adj_price = overlay.get("pred_price")
        if not isinstance(adj_price, (int, float)) or not np.isfinite(float(adj_price)):
            continue
        out[model_key] = payload.copy()
        out[model_key]["pred_price"] = float(round(adj_price, 4))
        overlay_high = overlay.get("pred_high_price")
        overlay_low = overlay.get("pred_low_price")
        if isinstance(overlay_high, (int, float)) and np.isfinite(float(overlay_high)):
            out[model_key]["pred_high_price"] = float(round(float(overlay_high), 4))
        if isinstance(overlay_low, (int, float)) and np.isfinite(float(overlay_low)):
            out[model_key]["pred_low_price"] = float(round(float(overlay_low), 4))
        if adj_price > reference_price:
            out[model_key]["direction"] = "UP"
        elif adj_price < reference_price:
            out[model_key]["direction"] = "DOWN"
        note = str(out[model_key].get("note") or "")
        feat_label = ",".join(str(x) for x in overlay.get("features", [])[:3])
        overlay_note = f"Adaptive overlay w={float(overlay.get('blend_weight', 0.0)):.2f}"
        if feat_label:
            overlay_note = f"{overlay_note} ({feat_label})"
        if overlay.get("structural_note"):
            overlay_note = f"{overlay_note} | {overlay.get('structural_note')}"
        out[model_key]["note"] = f"{note} | {overlay_note}" if note else overlay_note
    return out


def _apply_daily_pair_overlay_block(df_hist: pd.DataFrame, instrument_key: str, high_block: dict, low_block: dict) -> tuple[dict, dict]:
    if not isinstance(high_block, dict) or not isinstance(low_block, dict):
        return high_block, low_block
    out_high = high_block.copy()
    out_low = low_block.copy()
    for model_key in list(out_high.keys()):
        high_payload = out_high.get(model_key)
        low_payload = out_low.get(model_key)
        if not isinstance(high_payload, dict) or not isinstance(low_payload, dict):
            continue
        high_px = high_payload.get("pred_price")
        low_px = low_payload.get("pred_price")
        if not isinstance(high_px, (int, float)) or not isinstance(low_px, (int, float)):
            continue
        overlay = apply_daily_pair_overlay(df_hist, instrument_key, model_key, high_px, low_px)
        if not isinstance(overlay, dict):
            continue
        adj_high = overlay.get("pred_high")
        adj_low = overlay.get("pred_low")
        if not isinstance(adj_high, (int, float)) or not np.isfinite(float(adj_high)):
            continue
        if not isinstance(adj_low, (int, float)) or not np.isfinite(float(adj_low)):
            continue
        out_high[model_key] = high_payload.copy()
        out_low[model_key] = low_payload.copy()
        out_high[model_key]["pred_price"] = float(round(adj_high, 4))
        out_low[model_key]["pred_price"] = float(round(adj_low, 4))
        note_high = str(out_high[model_key].get("note") or "")
        note_low = str(out_low[model_key].get("note") or "")
        overlay_note = "Adaptive center/range overlay"
        out_high[model_key]["note"] = f"{note_high} | {overlay_note}" if note_high else overlay_note
        out_low[model_key]["note"] = f"{note_low} | {overlay_note}" if note_low else overlay_note
    return out_high, out_low

def _historical_daily_high_ratio(
    df: pd.DataFrame,
    minute_of_day,
    lookback_days=90,
    recent_weight=1.0,
    session_open_minute: int = None,
    session_close_minute: int = None
):
    if df is None or df.empty:
        return np.array([]), np.array([])
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    if "high" not in df2.columns or "close" not in df2.columns:
        return np.array([]), np.array([])
    df2 = df2.dropna(subset=["close", "high"])
    try:
        idx = pd.to_datetime(df2.index, utc=True)
        local_index = idx.tz_convert("America/New_York")
    except Exception:
        local_index = pd.to_datetime(df2.index, utc=True)
    df2.index = local_index
    df2["minute_of_day"] = df2.index.hour * 60 + df2.index.minute
    if session_open_minute is not None and session_close_minute is not None:
        df2["session_date"] = _session_dates_from_index(df2.index, session_open_minute, session_close_minute).date
        date_key = "session_date"
    else:
        df2["local_date"] = df2.index.date
        date_key = "local_date"
    ratios = []
    weights = []
    unique_days = sorted(pd.Series(df2[date_key].unique()).tolist(), reverse=True)[:lookback_days]
    tol = max(1, BAR_MINUTES)
    for day in unique_days:
        day_df = df2[df2[date_key] == day]
        if day_df.empty:
            continue
        cand = day_df[day_df["minute_of_day"] <= (minute_of_day + tol)]
        if cand.empty:
            cand = day_df.iloc[:1]
        if cand.empty:
            continue
        row = cand.iloc[-1]
        price_at_time = float(row["close"])
        if pd.isna(price_at_time) or price_at_time <= 0:
            continue
        future_part = day_df[day_df.index > row.name]
        if future_part.empty:
            continue
        future_max = float(future_part["high"].max())
        if pd.isna(future_max):
            continue
        ratio = future_max / price_at_time
        ratios.append(ratio)
        weights.append(1.0 / (1 + len(ratios)*0.05) * recent_weight)
    if len(ratios) == 0:
        logger.debug("No valid daily high ratios found in intraday data")
    return np.array(ratios), np.array(weights)

def _historical_daily_low_ratio(
    df: pd.DataFrame,
    minute_of_day,
    lookback_days=90,
    recent_weight=1.0,
    session_open_minute: int = None,
    session_close_minute: int = None
):
    if df is None or df.empty:
        return np.array([]), np.array([])
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    if "low" not in df2.columns or "close" not in df2.columns:
        return np.array([]), np.array([])
    df2 = df2.dropna(subset=["close", "low"])
    try:
        idx = pd.to_datetime(df2.index, utc=True)
        local_index = idx.tz_convert("America/New_York")
    except Exception:
        local_index = pd.to_datetime(df2.index, utc=True)
    df2.index = local_index
    df2["minute_of_day"] = df2.index.hour * 60 + df2.index.minute
    if session_open_minute is not None and session_close_minute is not None:
        df2["session_date"] = _session_dates_from_index(df2.index, session_open_minute, session_close_minute).date
        date_key = "session_date"
    else:
        df2["local_date"] = df2.index.date
        date_key = "local_date"
    ratios = []
    weights = []
    unique_days = sorted(pd.Series(df2[date_key].unique()).tolist(), reverse=True)[:lookback_days]
    tol = max(1, BAR_MINUTES)
    for day in unique_days:
        day_df = df2[df2[date_key] == day]
        if day_df.empty:
            continue
        cand = day_df[day_df["minute_of_day"] <= (minute_of_day + tol)]
        if cand.empty:
            cand = day_df.iloc[:1]
        if cand.empty:
            continue
        row = cand.iloc[-1]
        price_at_time = float(row["close"])
        if pd.isna(price_at_time) or price_at_time <= 0:
            continue
        future_part = day_df[day_df.index > row.name]
        if future_part.empty:
            continue
        future_min = float(future_part["low"].min())
        if pd.isna(future_min):
            continue
        ratio = future_min / price_at_time
        ratios.append(ratio)
        weights.append(1.0 / (1 + len(ratios)*0.05) * recent_weight)
    if len(ratios) == 0:
        logger.debug("No valid daily low ratios found in intraday data")
    return np.array(ratios), np.array(weights)

def _predict_daily_high_from_history(df, last_ts, last_price, instrument_key=None):
    try:
        last_local = last_ts.tz_convert("America/New_York")
        current_date = last_local.date()
    except Exception:
        current_date = last_ts.date()
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    try:
        idx = pd.to_datetime(df2.index, utc=True)
        local_index = idx.tz_convert("America/New_York")
    except Exception:
        local_index = pd.to_datetime(df2.index, utc=True)
    df2.index = local_index
    df2["local_date"] = df2.index.date
    past_df = df2[df2["local_date"] < current_date]
    instr_meta = INSTRUMENTS.get(instrument_key, {})
    open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
    ratios, weights = _historical_daily_high_ratio(
        past_df,
        open_minute,
        lookback_days=120,
        session_open_minute=open_minute,
        session_close_minute=close_minute
    )
    yahoo_sym = instr_meta.get('yahoo', instrument_key or SYMBOL)
    tz = instr_meta.get("tz", "America/New_York")
    if ratios.size < 8:
        daily_df = _fetch_long_daily(yahoo_sym, tz=tz)
        if daily_df is not None:
            daily_df['local_date'] = daily_df.index.tz_convert("America/New_York").date
            past_daily = daily_df[daily_df['local_date'] < current_date]
            past_daily = past_daily.dropna(subset=['open', 'high'])
            if not past_daily.empty:
                past_daily['ratio'] = past_daily['high'] / past_daily['open']
                ratios_daily = past_daily['ratio'].values[-120:]
                weights_daily = np.linspace(1.3, 0.7, len(ratios_daily))
                if ratios.size > 0:
                    ratios = np.concatenate((ratios, ratios_daily))
                    weights = np.concatenate((weights, weights_daily * 0.7))
                else:
                    ratios = ratios_daily
                    weights = weights_daily
                logger.info(f"DailyHigh using {len(ratios_daily)} daily open-to-high ratios")
    valid_mask = np.isfinite(ratios)
    ratios = ratios[valid_mask]
    weights = weights[valid_mask] if len(weights) == len(ratios) else np.ones_like(ratios)
    today_df = df2[df2["local_date"] == current_date]
    if today_df.empty:
        recent_vol = df["close"].pct_change().rolling(20).std().iloc[-1] if len(df) >= 20 else 0.0
        k = 4.0
        ratio_est = 1.0 + k * (recent_vol or 0.005)
        ratio_est = max(0.98, min(1.25, ratio_est))
        conf = 25.0 if recent_vol > 0 else 15.0
        pred_high = last_price * ratio_est
        return float(round(pred_high, 4)), float(round(conf, 2)), f"Pre-open fallback (vol ratio {ratio_est:.3f})"
    open_price = float(today_df["open"].iloc[0]) if "open" in today_df.columns and not pd.isna(today_df["open"].iloc[0]) else float(today_df["close"].iloc[0])
    high_so_far = float(today_df["high"].max()) if "high" in today_df.columns else open_price
    if ratios.size == 0:
        recent_vol = past_df["close"].pct_change().rolling(20).std().iloc[-1] if len(past_df) >= 20 else 0.005
        k = 4.0
        ratio_est = 1.0 + k * recent_vol
        ratio_est = max(0.98, min(1.25, ratio_est))
        conf = 25.0 if recent_vol > 0 else 15.0
        pred_high = open_price * ratio_est
        pred_high = max(pred_high, high_so_far)
        return float(round(pred_high, 4)), float(round(conf, 2)), f"Daily-high fallback (vol ratio {ratio_est:.3f})"
    try:
        order = np.argsort(ratios)
        ratios_s = ratios[order]
        weights_s = weights[order] if len(weights) == len(ratios) else np.ones_like(ratios_s)
        cumw = np.cumsum(weights_s)
        cutoff = cumw[-1] / 2.0
        idx = np.searchsorted(cumw, cutoff)
        median_ratio = float(ratios_s[min(max(0, idx), len(ratios_s)-1)])
        mean_ratio = float(np.average(ratios, weights=weights))
        ratio_est = 0.75 * median_ratio + 0.25 * mean_ratio
        disp = float(np.std(ratios))
        conf = 30.0 if mean_ratio <= 0 else max(10.0, min(95.0, 85.0 * (1 - disp / (mean_ratio + 1e-6))))
        conf *= min(1.0, len(ratios) / 8.0)
    except Exception:
        ratio_est = 1.02
        conf = 35.0
    pred_high = open_price * ratio_est
    pred_high = max(pred_high, high_so_far)
    note = f"Daily-high from open (ratio {ratio_est:.3f}, {len(ratios)} analogs)"
    return float(round(pred_high, 4)), float(round(conf, 2)), note

def _predict_daily_low_from_history(df, last_ts, last_price, instrument_key=None):
    try:
        last_local = last_ts.tz_convert("America/New_York")
        current_date = last_local.date()
    except Exception:
        current_date = last_ts.date()
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    try:
        idx = pd.to_datetime(df2.index, utc=True)
        local_index = idx.tz_convert("America/New_York")
    except Exception:
        local_index = pd.to_datetime(df2.index, utc=True)
    df2.index = local_index
    df2["local_date"] = df2.index.date
    past_df = df2[df2["local_date"] < current_date]
    instr_meta = INSTRUMENTS.get(instrument_key, {})
    open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
    ratios, weights = _historical_daily_low_ratio(
        past_df,
        open_minute,
        lookback_days=120,
        session_open_minute=open_minute,
        session_close_minute=close_minute
    )
    yahoo_sym = instr_meta.get('yahoo', instrument_key or SYMBOL)
    tz = instr_meta.get("tz", "America/New_York")
    if ratios.size < 8:
        daily_df = _fetch_long_daily(yahoo_sym, tz=tz)
        if daily_df is not None:
            daily_df['local_date'] = daily_df.index.tz_convert("America/New_York").date
            past_daily = daily_df[daily_df['local_date'] < current_date]
            past_daily = past_daily.dropna(subset=['open', 'low'])
            if not past_daily.empty:
                past_daily['ratio'] = past_daily['low'] / past_daily['open']
                ratios_daily = past_daily['ratio'].values[-120:]
                weights_daily = np.linspace(1.3, 0.7, len(ratios_daily))
                if ratios.size > 0:
                    ratios = np.concatenate((ratios, ratios_daily))
                    weights = np.concatenate((weights, weights_daily * 0.7))
                else:
                    ratios = ratios_daily
                    weights = weights_daily
                logger.info(f"DailyLow using {len(ratios_daily)} daily open-to-low ratios")
    valid_mask = np.isfinite(ratios)
    ratios = ratios[valid_mask]
    weights = weights[valid_mask] if len(weights) == len(ratios) else np.ones_like(ratios)
    today_df = df2[df2["local_date"] == current_date]
    if today_df.empty:
        recent_vol = df["close"].pct_change().rolling(20).std().iloc[-1] if len(df) >= 20 else 0.0
        k = 4.0
        ratio_est = 1.0 - k * (recent_vol or 0.005)
        ratio_est = max(0.7, min(1.02, ratio_est))
        conf = 25.0 if recent_vol > 0 else 15.0
        pred_low = last_price * ratio_est
        return float(round(pred_low, 4)), float(round(conf, 2)), f"Pre-open fallback (vol ratio {ratio_est:.3f})"
    open_price = float(today_df["open"].iloc[0]) if "open" in today_df.columns and not pd.isna(today_df["open"].iloc[0]) else float(today_df["close"].iloc[0])
    low_so_far = float(today_df["low"].min()) if "low" in today_df.columns else open_price
    if ratios.size == 0:
        recent_vol = past_df["close"].pct_change().rolling(20).std().iloc[-1] if len(past_df) >= 20 else 0.005
        k = 4.0
        ratio_est = 1.0 - k * recent_vol
        ratio_est = max(0.7, min(1.02, ratio_est))
        conf = 25.0 if recent_vol > 0 else 15.0
        pred_low = open_price * ratio_est
        pred_low = min(pred_low, low_so_far)
        return float(round(pred_low, 4)), float(round(conf, 2)), f"Daily-low fallback (vol ratio {ratio_est:.3f})"
    try:
        order = np.argsort(ratios)
        ratios_s = ratios[order]
        weights_s = weights[order] if len(weights) == len(ratios) else np.ones_like(ratios_s)
        cumw = np.cumsum(weights_s)
        cutoff = cumw[-1] / 2.0
        idx = np.searchsorted(cumw, cutoff)
        median_ratio = float(ratios_s[min(max(0, idx), len(ratios_s)-1)])
        mean_ratio = float(np.average(ratios, weights=weights))
        ratio_est = 0.75 * median_ratio + 0.25 * mean_ratio
        disp = float(np.std(ratios))
        conf = 30.0 if mean_ratio <= 0 else max(10.0, min(95.0, 85.0 * (1 - disp / (abs(mean_ratio) + 1e-6))))
        conf *= min(1.0, len(ratios) / 8.0)
    except Exception:
        ratio_est = 0.98
        conf = 35.0
    pred_low = open_price * ratio_est
    pred_low = min(pred_low, low_so_far)
    note = f"Daily-low from open (ratio {ratio_est:.3f}, {len(ratios)} analogs)"
    return float(round(pred_low, 4)), float(round(conf, 2)), note

def _historical_weekend_ratio(df, current_dt, lookback_weeks=60):
    if df is None or df.empty:
        return np.array([])
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    if "close" not in df2.columns:
        return np.array([])
    df2 = df2.dropna(subset=["close"])
    try:
        idx = pd.to_datetime(df2.index, utc=True)
        local_index = idx.tz_convert("America/New_York")
    except Exception:
        local_index = pd.to_datetime(df2.index, utc=True)
    df2.index = local_index
    df2["dow"] = df2.index.weekday
    df2["minute_of_day"] = df2.index.hour * 60 + df2.index.minute
    try:
        isocal = df2.index.isocalendar()
        df2["iso_year"] = isocal.year
        df2["iso_week"] = isocal.week
    except Exception:
        df2["iso_year"] = df2.index.year
        df2["iso_week"] = df2.index.isocalendar().week
    ratios = []
    weeks = sorted(df2[["iso_year","iso_week"]].drop_duplicates().values.tolist(), reverse=True)[:lookback_weeks]
    count = 0
    cur_local = current_dt.tz_convert("America/New_York") if current_dt.tzinfo else pd.to_datetime(current_dt).tz_localize("America/New_York")
    cur_dow = cur_local.weekday()
    cur_mod = cur_local.hour * 60 + cur_local.minute
    tol = max(1, BAR_MINUTES)
    for y, w in weeks:
        mask = (df2["iso_year"] == y) & (df2["iso_week"] == w)
        week_df = df2[mask]
        if week_df.empty:
            continue
        day_df = week_df[week_df["dow"] == cur_dow]
        if day_df.empty:
            continue
        cand = day_df[day_df["minute_of_day"] <= (cur_mod + tol)]
        if cand.empty:
            continue
        row = cand.iloc[-1]
        price_at_time = float(row["close"])
        if pd.isna(price_at_time) or price_at_time <= 0:
            continue
        fri_df = week_df[week_df["dow"] == 4]
        if fri_df.empty:
            continue
        fri_close = float(fri_df["close"].iloc[-1])
        if pd.isna(fri_close):
            continue
        ratio = fri_close / price_at_time
        ratios.append(ratio)
        count += 1
    if len(ratios) == 0:
        logger.debug("No valid weekend ratios found in intraday data")
    return np.array(ratios)

def _predict_weekend_close_from_history(df, last_ts, last_price, instrument_key=None):
    try:
        last_local = last_ts.tz_convert("America/New_York")
        current_date = last_local.date()
        current_dow = last_local.weekday()
    except Exception:
        current_date = last_ts.date()
        current_dow = last_ts.weekday()
    df2 = df.copy()
    df2.columns = [c.lower() for c in df2.columns]
    try:
        idx = pd.to_datetime(df2.index, utc=True)
        local_index = idx.tz_convert("America/New_York")
    except Exception:
        local_index = pd.to_datetime(df2.index, utc=True)
    df2.index = local_index
    isocal = df2.index.isocalendar()
    df2["iso_year"] = isocal.year
    df2["iso_week"] = isocal.week
    current_iso_year = isocal['year'].iloc[-1] if not df2.empty else current_date.year
    current_iso_week = isocal['week'].iloc[-1] if not df2.empty else current_date.isocalendar().week
    past_mask = ((df2["iso_year"] < current_iso_year) |
                 ((df2["iso_year"] == current_iso_year) & (df2["iso_week"] < current_iso_week)))
    past_df = df2[past_mask]
    instr_meta = INSTRUMENTS.get(instrument_key, {})
    week_start_day = int(instr_meta.get("week_start_day", 0))
    open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    days_back = (current_dow - week_start_day) % 7
    week_start_date = current_date - timedelta(days=days_back)
    open_hour = int(open_minute // 60)
    open_min = int(open_minute % 60)
    week_start_time = datetime.combine(week_start_date, time(open_hour, open_min))
    week_start_ts = pd.to_datetime(week_start_time).tz_localize("America/New_York")
    ratios = _historical_weekend_ratio(past_df, week_start_ts, lookback_weeks=80)
    yahoo_sym = instr_meta.get('yahoo', instrument_key or SYMBOL)
    tz = instr_meta.get("tz", "America/New_York")
    if ratios.size < 6:
        daily_df = _fetch_long_daily(yahoo_sym, tz=tz)
        if daily_df is not None:
            daily_df['dow'] = daily_df.index.tz_convert("America/New_York").weekday
            isocal_d = daily_df.index.isocalendar()
            daily_df["iso_year"] = isocal_d.year
            daily_df["iso_week"] = isocal_d.week
            past_daily_mask = ((daily_df["iso_year"] < current_iso_year) |
                               ((daily_df["iso_year"] == current_iso_year) & (daily_df["iso_week"] < current_iso_week)))
            past_daily = daily_df[past_daily_mask]
            groups = past_daily.groupby(['iso_year', 'iso_week'])
            ratios_daily = []
            for name, g in groups:
                monday = g[g['dow'] == 0]
                friday = g[g['dow'] == 4]
                if not monday.empty and not friday.empty:
                    monday_open = monday['open'].iloc[0]
                    friday_close = friday['close'].iloc[-1]
                    ratio = friday_close / monday_open
                    ratios_daily.append(ratio)
            if ratios_daily:
                ratios_d = np.array(ratios_daily)
                weights_d = np.linspace(1.3, 0.7, len(ratios_d))
                if ratios.size > 0:
                    ratios = np.concatenate((ratios, ratios_d))
                    weights = np.concatenate((np.ones(len(ratios)), weights_d * 0.7))
                else:
                    ratios = ratios_d
                    weights = weights_d
                logger.info(f"WeekEnd using {len(ratios_d)} daily Monday-to-Friday ratios")
    ratios = ratios[np.isfinite(ratios)]
    this_week_mask = ((df2["iso_year"] == current_iso_year) & (df2["iso_week"] == current_iso_week))
    this_week_df = df2[this_week_mask]
    if this_week_df.empty:
        recent_vol = df["close"].pct_change().rolling(20).std().iloc[-1] if len(df) >= 20 else 0.005
        k = 6.0
        ratio_est = 1.0 + k * (recent_vol or 0.005)
        ratio_est = max(0.95, min(1.4, ratio_est))
        conf = 20.0 if recent_vol > 0 else 10.0
        pred_price = last_price * ratio_est
        return float(round(pred_price, 4)), float(round(conf, 2)), f"Pre-week fallback (vol ratio {ratio_est:.3f})"
    monday_df = this_week_df[this_week_df.index.weekday == 0]
    if monday_df.empty:
        return last_price, 25.0, "Weekend fallback - no Monday data yet"
    monday_open = float(monday_df["open"].iloc[0]) if "open" in monday_df.columns and not pd.isna(monday_df["open"].iloc[0]) else float(monday_df["close"].iloc[0])
    if ratios.size == 0:
        recent_vol = past_df["close"].pct_change().rolling(20).std().iloc[-1] if len(past_df) >= 20 else 0.005
        k = 6.0
        ratio_est = 1.0 + k * recent_vol
        ratio_est = max(0.95, min(1.4, ratio_est))
        conf = 20.0 if recent_vol > 0 else 10.0
        pred_price = monday_open * ratio_est
        return float(round(pred_price, 4)), float(round(conf, 2)), f"Week-end fallback (vol ratio {ratio_est:.3f})"
    weights = np.linspace(1.5, 0.5, len(ratios)) if len(ratios) > 1 else np.ones(len(ratios))
    try:
        median_ratio = float(np.median(ratios))
        mean_ratio = float(np.average(ratios, weights=weights))
        ratio_est = 0.7 * median_ratio + 0.3 * mean_ratio
        rel_std = float(np.std(ratios)) / (abs(mean_ratio) + 1e-12)
        conf = max(10.0, min(95.0, 85.0 * (1 - rel_std)))
        conf *= min(1.0, len(ratios) / 6.0)
    except Exception:
        ratio_est = 1.03
        conf = 30.0
    pred_price = monday_open * ratio_est
    note = f"Week-end from Monday open (ratio {ratio_est:.3f}, {len(ratios)} analogs)"
    return float(round(pred_price, 4)), float(round(conf, 2)), note

def _ensemble_summary_for_payload(payload, X, dominant_regime=None, dom_prob=0.0, comp_probs=None):
    ensemble_map = payload.get("ensemble_models", {}) if payload else {}
    regime_map = payload.get("regime_models", {}) if payload else {}
    win_calibration = payload.get("window_calibration", {}) if payload else {}
    perf = payload.get("performance", payload.get("performance_mae", {})) if payload else {}
    win_weights = _window_weights(perf)
    combined_summary = None
    combined_weight = 0.0
    for win in ["short", "mid", "long"]:
        members = ensemble_map.get(win, []) or []
        if not members:
            continue
        global_sum = _summarize_preds(_ensemble_predict_members(members, X))
        reg_sum = None
        if regime_map and dominant_regime is not None:
            reg_members, reg_key, reg_prob = _select_regime_members(
                regime_map,
                win,
                dominant_regime=dominant_regime,
                comp_probs=comp_probs
            )
            if reg_members:
                reg_sum = _summarize_preds(_ensemble_predict_members(reg_members, X))
        if reg_sum is not None and global_sum is not None:
            reg_weight = reg_prob if reg_prob is not None else dom_prob
            w_reg = max(0.0, min(0.8, reg_weight))
            win_sum = _blend_summaries(reg_sum, global_sum, w_reg)
        else:
            win_sum = reg_sum or global_sum
        if win_sum is None:
            continue
        win_sum = _apply_summary_calibration(win_sum, win_calibration.get(win))
        w = win_weights.get(win, 0.0)
        if combined_summary is None:
            combined_summary = win_sum
            combined_weight = w
        else:
            total = combined_weight + w
            w_a = combined_weight / total if total > 0 else 0.5
            combined_summary = _blend_summaries(combined_summary, win_sum, w_a)
            combined_weight = total
    return combined_summary

def _predict_mode_group(models_group, df_latest, instrument_key, rp, dominant_regime, dom_prob, stale, mode_label,
                        clamp_min=None, clamp_max=None, horizon_scale=1.0, max_cap=0.08,
                        feature_row: pd.DataFrame | None = None, last_price_override: float | None = None,
                        comp_probs=None):
    if df_latest is None or df_latest.empty:
        return {}
    if not models_group:
        return {
            k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": f"Missing ML models ({mode_label})"}
            for k in ["base", "forest", "boost", "extra_trees"]
        }
    instr_meta = _instrument_meta(instrument_key)
    last_price = float(last_price_override) if last_price_override is not None else float(df_latest["close"].iloc[-1])
    stale_note = ""
    stale_factor = _stale_conf_factor(stale)
    if stale.get("is_stale"):
        try:
            mins = stale.get("age_sec", 0.0) / 60.0 if stale.get("age_sec") is not None else None
            stale_note = f"Stale data ({mins:.1f}m old)" if mins is not None else "Stale data"
        except Exception:
            stale_note = "Stale data"
    out = {}
    for key in ["base", "forest", "boost", "extra_trees"]:
        payload = models_group.get(key)
        if payload is None:
            out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "No model"}
            continue
        feat_list = payload.get("features")
        try:
            if feature_row is not None:
                X = feature_row[feat_list] if feat_list else feature_row
            else:
                X = make_feature_matrix(df_latest, feature_list=feat_list, instrument_key=instrument_key) if feat_list else make_feature_matrix(df_latest, instrument_key=instrument_key)
        except Exception:
            out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Feature error"}
            continue
        if X is None or X.empty:
            out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Waiting…"}
            continue
        try:
            current_vol = float(X.get("ewm_vol", 0.001).iloc[0]) if "ewm_vol" in X.columns else 0.001
        except Exception:
            current_vol = 0.001
        vol_candidates = [current_vol]
        for col in ("vol_12", "vol_6", "vol_3", "session_volatility"):
            try:
                if col in X.columns:
                    vol_candidates.append(float(X[col].iloc[0]))
            except Exception:
                continue
        current_vol = max([v for v in vol_candidates if np.isfinite(v)] + [_vol_floor_from_meta(instr_meta)])
        combined_summary = _ensemble_summary_for_payload(payload, X, dominant_regime, dom_prob, comp_probs=comp_probs)
        used_fallback_summary = False
        if combined_summary is None:
            fb = payload.get("fallback_model")
            if fb is None:
                fb = payload.get("model")
            if fb is not None:
                try:
                    pred_norm = float(fb.predict(X)[0])
                    pred_std_norm = float(payload.get("meta", {}).get("resid_std", 1e-6))
                    pred_std_norm = max(pred_std_norm, 1e-6)
                    combined_summary = {"mean": pred_norm, "std": pred_std_norm, "q10": 0.0, "q50": pred_norm, "q90": 0.0, "p_up": float(pred_norm >= 0)}
                    used_fallback_summary = True
                except Exception:
                    combined_summary = None
            if combined_summary is None:
                out[key] = {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Ensemble not trained"}
                continue
        if used_fallback_summary:
            combined_summary = _restore_summary_target_mean(combined_summary, payload)
            combined_summary = _apply_fallback_return_calibration(combined_summary, payload.get("calibration"))
        pred_norm = combined_summary["mean"]
        pred_std_norm = combined_summary["std"]
        try:
            reg_conf = float(rp.get("regime_conf", 0.5))
        except Exception:
            reg_conf = 0.5
        prior_var = max(0.2, min(2.0, 0.6 + reg_conf))
        shrink = prior_var / (prior_var + (pred_std_norm ** 2))
        pred_norm = pred_norm * shrink
        p_correct_sig = _dynamic_confidence_by_sigbucket_from_payload(
            pred_norm, pred_std_norm, payload, model_key=key, instrument_key=instrument_key
        )
        p_up_sig = p_correct_sig if pred_norm >= 0 else (1.0 - p_correct_sig)
        p_up = 0.7 * combined_summary["p_up"] + 0.3 * p_up_sig
        p_up = max(0.001, min(0.999, p_up))
        target_normed = _infer_target_norm(payload.get("meta", {}))
        pred_ret = pred_norm * current_vol if target_normed else pred_norm
        max_abs_ret = _max_abs_ret_from_vol(current_vol, instr_meta) * max(1.0, horizon_scale)
        max_abs_ret = min(max_cap, max_abs_ret)
        pred_ret = max(-max_abs_ret, min(max_abs_ret, pred_ret))
        pred_price = last_price * (1.0 + pred_ret)
        clamp_note = None
        if clamp_min is not None and pred_price < clamp_min:
            pred_price = clamp_min
            clamp_note = "Clamped to high so far"
        if clamp_max is not None and pred_price > clamp_max:
            pred_price = clamp_max
            clamp_note = "Clamped to low so far"
        pred_price = _round_to_tick(pred_price, instr_meta)
        direction = "UP" if pred_price >= last_price else "DOWN"
        conf = p_up if direction == "UP" else (1.0 - p_up)
        try:
            support = rp.get("Bullish", 0.0) / 100.0 if direction == "UP" else rp.get("Bearish", 0.0) / 100.0
        except Exception:
            support = 0.0
        support *= (0.5 + 0.5 * max(0.0, min(1.0, reg_conf)))
        conf *= (0.7 + 0.3 * support)
        dispersion = pred_std_norm / (abs(pred_norm) + 1e-9)
        dispersion = min(dispersion, 3.0)
        conf *= (1.0 / (1.0 + 0.35 * dispersion))
        if stale.get("is_stale"):
            conf *= stale_factor
        # Keep true low-confidence values visible instead of pinning many models to 20%.
        conf = max(0.01, min(0.99, conf))
        note_parts = ["ML ensemble"]
        if clamp_note:
            note_parts.append(clamp_note)
        if stale_note:
            note_parts.append(stale_note)
        try:
            std_ret = float(pred_std_norm * current_vol) if target_normed else float(pred_std_norm)
        except Exception:
            std_ret = 0.0
        band_fields = _prediction_bands(last_price, pred_ret, std_ret, instr_meta=instr_meta)
        conf_pct = round(float(conf * 100.0), 2)
        out[key] = {
            "direction": direction,
            "pred_price": float(round(pred_price, 4)),
            "confidence": conf_pct,
            "note": " | ".join(note_parts),
            "pred_ret": float(pred_ret),
            "pred_norm": float(pred_norm),
            "pred_std_norm": float(pred_std_norm),
            "sig_bucket": "Strong" if (abs(pred_norm) / (pred_std_norm + 1e-12)) >= 1.5 else "Medium" if (abs(pred_norm) / (pred_std_norm + 1e-12)) >= 0.5 else "Weak",
            **band_fields,
            "p_up": round(float(p_up), 6),
            "p_down": round(float(1.0 - p_up), 6),
        }
    return out

def predict_horizon(
    models_dict,
    df_latest,
    instrument_key: str = None,
    mode: str = "DailyHigh",
    log_side_effects: bool = True,
):
    if df_latest is None or df_latest.empty:
        return {}
    df_latest, _ = _drop_incomplete_last_bar(df_latest, instrument_key=instrument_key)
    if df_latest is None or df_latest.empty:
        return {"error": "No complete bars available"}
    stale = _staleness_info(df_latest, instrument_key=instrument_key)
    mode_norm = (mode or "DailyHigh").strip().lower()
    regime_payload = models_dict.get("regime")
    if not regime_payload:
        return {"error": "Regime model missing"}
    try:
        if mode_norm in ("dailyhigh", "daily_high", "daily-high", "dailylow", "daily_low", "daily-low"):
            look_hours = max(1, int(REGIME_LOOK_HOURS_DAILY))
        elif mode_norm in ("weekend", "week_end", "week-end", "weekend_close", "week_end_close"):
            look_hours = max(1, int(REGIME_LOOK_HOURS_WEEKEND))
        else:
            look_hours = max(1, int(REGIME_LOOK_HOURS_4MIN))
        rp = predict_regime_probs(
            regime_payload,
            df_latest,
            recent_look_hours=look_hours,
            horizon_key="weekend" if mode_norm in ("weekend", "week_end", "week-end", "weekend_close", "week_end_close") else "daily",
        )
    except Exception:
        return {"error": "Regime model invalid"}
    # Determine dominant regime for regime-aware ensembles
    dominant_regime = rp.get("dominant_component", None)
    comp_probs = rp.get("component_probs", None)
    dom_prob = 0.0
    if dominant_regime is None:
        dominant_regime = "0"
    try:
        if isinstance(comp_probs, (list, tuple)) and str(dominant_regime).isdigit():
            dom_idx = int(dominant_regime)
            if 0 <= dom_idx < len(comp_probs):
                dom_prob = float(comp_probs[dom_idx])
    except Exception:
        dom_prob = 0.0
    if dom_prob == 0.0 and dominant_regime is not None and comp_probs is None:
        dom_prob = 0.6

    modes = models_dict.get("modes", {}) if isinstance(models_dict, dict) else {}
    last_price = float(df_latest["close"].iloc[-1])
    if mode_norm in ("dailyhigh", "daily_high", "daily-high", "dailylow", "daily_low", "daily-low"):
        # Guard: require real OHLC for daily high/low
        try:
            if not all(c in df_latest.columns for c in ("open", "high", "low")):
                raise RuntimeError("Missing real OHLC data")
        except Exception:
            placeholder = {
                k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Missing real OHLC data"}
                for k in ["base", "forest", "boost", "extra_trees"]
            }
            return {"high": placeholder, "low": placeholder, "regime": rp}

        instr_meta = _instrument_meta(instrument_key)
        tz = instr_meta.get("tz", "America/New_York")
        session_open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
        session_close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
        try:
            idx_local = pd.to_datetime(df_latest.index, utc=True).tz_convert(tz)
        except Exception:
            idx_local = pd.to_datetime(df_latest.index)
        session_dates = _session_dates_from_index(idx_local, session_open_minute, session_close_minute)
        target_session = _target_session_date(instr_meta)
        # Normalize to date keys to avoid dtype mismatches (datetime vs date)
        try:
            session_dates_key = pd.to_datetime(session_dates).date
        except Exception:
            session_dates_key = session_dates
        try:
            target_session_key = pd.to_datetime(target_session).date()
        except Exception:
            target_session_key = target_session
        try:
            prev_dates = session_dates_key[session_dates_key < target_session_key]
        except Exception:
            prev_dates = []
        # Strict cutoff: require a prior session
        if prev_dates is None or len(prev_dates) == 0:
            placeholder = {
                k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Insufficient prior session history"}
                for k in ["base", "forest", "boost", "extra_trees"]
            }
            return {"high": placeholder, "low": placeholder, "regime": rp}
        try:
            feature_session = prev_dates.max()
        except Exception:
            feature_session = prev_dates[-1]

        # Build a historical slice up to the prior session (no current-session data)
        try:
            session_series = pd.Series(session_dates_key, index=df_latest.index)
            df_hist = df_latest.loc[session_series <= feature_session].copy()
        except Exception:
            df_hist = df_latest.copy()
        if feature_session is None or (isinstance(feature_session, float) and np.isnan(feature_session)):
            placeholder = {
                k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Insufficient session history"}
                for k in ["base", "forest", "boost", "extra_trees"]
            }
            return {"high": placeholder, "low": placeholder, "regime": rp}

        feat_df = build_feature_frame(df_hist, instrument_key=instrument_key)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        if "session_date" not in feat_df.columns:
            try:
                idx_local_hist = pd.to_datetime(df_hist.index, utc=True).tz_convert(tz)
            except Exception:
                idx_local_hist = pd.to_datetime(df_hist.index)
            session_dates_hist = _session_dates_from_index(idx_local_hist, session_open_minute, session_close_minute)
            try:
                session_dates_hist = pd.to_datetime(session_dates_hist).date
            except Exception:
                pass
            feat_df["session_date"] = session_dates_hist
        feat_df = feat_df.dropna(subset=["close"])
        try:
            session_col = feat_df["session_date"]
            if pd.api.types.is_datetime64_any_dtype(session_col):
                session_key = session_col.dt.date
            else:
                session_key = session_col
        except Exception:
            session_key = feat_df["session_date"]
        row_candidates = feat_df[session_key == feature_session]
        if row_candidates.empty:
            placeholder = {
                k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Missing session features"}
                for k in ["base", "forest", "boost", "extra_trees"]
            }
            return {"high": placeholder, "low": placeholder, "regime": rp}
        feature_row = row_candidates.tail(1).copy()
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        try:
            last_price_feature = float(feature_row["close"].iloc[-1])
        except Exception:
            last_price_feature = last_price

        high_models = modes.get("daily_high", {})
        low_models = modes.get("daily_low", {})
        # Daily forecasts are intentionally based on prior session close; avoid stale penalty
        stale_daily = {"is_stale": False, "age_sec": None, "threshold_sec": None}
        high = _predict_mode_group(
            high_models,
            df_hist,
            instrument_key,
            rp,
            dominant_regime,
            dom_prob,
            stale_daily,
            "daily_high",
            horizon_scale=1.0,
            max_cap=0.25,
            feature_row=feature_row,
            last_price_override=last_price_feature,
            comp_probs=comp_probs
        )
        low = _predict_mode_group(
            low_models,
            df_hist,
            instrument_key,
            rp,
            dominant_regime,
            dom_prob,
            stale_daily,
            "daily_low",
            horizon_scale=1.0,
            max_cap=0.25,
            feature_row=feature_row,
            last_price_override=last_price_feature,
            comp_probs=comp_probs
        )
        high = _apply_horizon_overlay_block(df_hist, instrument_key, "daily_high", last_price_feature, high)
        low = _apply_horizon_overlay_block(df_hist, instrument_key, "daily_low", last_price_feature, low)
        high, low = _apply_daily_pair_overlay_block(df_hist, instrument_key, high, low)
        # Keep high/low forecasts realistic by blending with historical session priors
        # and enforcing a minimum daily range based on observed distributions.
        try:
            priors = _daily_range_priors(df_hist, instrument_key=instrument_key)
            _enforce_daily_high_low_consistency(
                high,
                low,
                base_price=last_price_feature,
                instr_meta=instr_meta,
                priors=priors,
            )
        except Exception:
            logger.debug("Daily high/low consistency adjustment failed", exc_info=True)
        # Annotate notes to reflect cutoff
        for block in (high, low):
            try:
                for k, v in block.items():
                    if not isinstance(v, dict):
                        continue
                    note = v.get("note") or ""
                    suffix = "As of prior session close"
                    v["note"] = f"{note} | {suffix}" if note else suffix
            except Exception:
                pass
        try:
            target_meta = str(pd.to_datetime(target_session).date())
        except Exception:
            target_meta = str(target_session)
        try:
            feature_meta = str(pd.to_datetime(feature_session).date())
        except Exception:
            feature_meta = str(feature_session)
        meta = {
            "target_session_date": target_meta,
            "feature_session_date": feature_meta,
            "regime_look_hours": int(look_hours),
        }
        out_daily = {"high": high, "low": low, "regime": rp, "meta": meta}
        if log_side_effects:
            try:
                _log_horizon_prediction(
                    instrument_key=instrument_key,
                    last_price=last_price_feature,
                    mode="daily_high_low",
                    results_out=out_daily,
                )
                _schedule_horizon_accuracy_update(instrument_key or SYMBOL)
            except Exception:
                logger.debug("Daily horizon side-effects failed", exc_info=True)
        return out_daily

    if mode_norm in ("weekend", "week_end", "week-end", "weekend_close", "week_end_close"):
        instr_meta = _instrument_meta(instrument_key)
        tz = instr_meta.get("tz", "America/New_York")
        session_open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
        session_close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
        try:
            idx_local = pd.to_datetime(df_latest.index, utc=True).tz_convert(tz)
        except Exception:
            idx_local = pd.to_datetime(df_latest.index)
        session_dates = _session_dates_from_index(idx_local, session_open_minute, session_close_minute)
        target_session = _target_session_date(instr_meta)
        try:
            session_dates_key = pd.to_datetime(session_dates).date
        except Exception:
            session_dates_key = session_dates
        try:
            target_session_key = pd.to_datetime(target_session).date()
        except Exception:
            target_session_key = target_session
        try:
            prev_dates = session_dates_key[session_dates_key < target_session_key]
        except Exception:
            prev_dates = []
        if prev_dates is None or len(prev_dates) == 0:
            placeholder = {
                k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Insufficient prior session history"}
                for k in ["base", "forest", "boost", "extra_trees"]
            }
            return {"week": placeholder, "regime": rp}
        try:
            feature_session = prev_dates.max()
        except Exception:
            feature_session = prev_dates[-1]

        # Historical slice up to prior session close (no current-session data)
        try:
            session_series = pd.Series(session_dates_key, index=df_latest.index)
            df_hist = df_latest.loc[session_series <= feature_session].copy()
        except Exception:
            df_hist = df_latest.copy()

        feat_df = build_feature_frame(df_hist, instrument_key=instrument_key)
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = feat_df.dropna(subset=["close"])
        if feat_df.empty:
            placeholder = {
                k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": "Missing session features"}
                for k in ["base", "forest", "boost", "extra_trees"]
            }
            return {"week": placeholder, "regime": rp}
        feature_row = feat_df.tail(1).copy()
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        try:
            last_price_feature = float(feature_row["close"].iloc[-1])
        except Exception:
            last_price_feature = last_price

        week_models = modes.get("weekend", {})
        # Detect legacy weekend models (trained on week-end bars)
        legacy_weekend = False
        try:
            ok = False
            for mk in ["base", "forest", "boost", "extra_trees"]:
                payload = week_models.get(mk)
                if not isinstance(payload, dict):
                    continue
                meta = payload.get("meta", {})
                if meta.get("weekend_horizon") == "per_session":
                    ok = True
                    break
            legacy_weekend = not ok
        except Exception:
            legacy_weekend = False
        stale_week = {"is_stale": False, "age_sec": None, "threshold_sec": None}
        week = _predict_mode_group(
            week_models,
            df_hist,
            instrument_key,
            rp,
            dominant_regime,
            dom_prob,
            stale_week,
            "weekend",
            horizon_scale=1.0,
            max_cap=0.3,
            feature_row=feature_row,
            last_price_override=last_price_feature,
            comp_probs=comp_probs
        )
        week = _apply_horizon_overlay_block(df_hist, instrument_key, "weekend", last_price_feature, week)
        try:
            for k, v in week.items():
                if not isinstance(v, dict):
                    continue
                note = v.get("note") or ""
                suffix = "As of prior session close"
                v["note"] = f"{note} | {suffix}" if note else suffix
                if legacy_weekend:
                    v["note"] = f"{v['note']} | Legacy weekend model (retrain recommended)"
        except Exception:
            pass
        try:
            target_week = _target_week_key(instr_meta)
            target_week_str = f"{target_week[0]}-W{target_week[1]}"
        except Exception:
            target_week_str = None
        meta = {
            "target_week": target_week_str,
            "feature_session_date": str(feature_session),
            "regime_look_hours": int(look_hours),
        }
        out_week = {"week": week, "regime": rp, "meta": meta}
        if log_side_effects:
            try:
                _log_horizon_prediction(
                    instrument_key=instrument_key,
                    last_price=last_price_feature,
                    mode="weekend",
                    results_out=out_week,
                )
                _schedule_horizon_accuracy_update(instrument_key or SYMBOL)
            except Exception:
                logger.debug("Weekend horizon side-effects failed", exc_info=True)
        return out_week

    # Unknown mode: return placeholders
    placeholder = {
        k: {"direction": "-", "pred_price": "-", "confidence": "-", "note": f"Unknown mode {mode}"}
        for k in ["base", "forest", "boost", "extra_trees"]
    }
    return {"week": placeholder, "regime": rp}
