from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from config import (
    ALLOW_REMOTE_CONTEXT_DAILY,
    CONTEXT_PIPELINE_ENABLED,
    CONTEXT_REFRESH_MAX_AGE_HOURS,
    DATA_DIR,
    DEFAULT_INSTRUMENT,
    INSTRUMENTS,
)

try:
    import yfinance as yf
except Exception:
    yf = None

_ROLE_ORDER = ("market", "sector", "vol", "lead_fut", "rates")
_CACHE: dict[tuple[str, str], pd.DataFrame] = {}
_MISSING_CACHE: dict[tuple[str, str], float] = {}

CONTEXT_FEATURE_COLUMNS = (
    "ctx_market_ret_1",
    "ctx_market_ret_5",
    "ctx_market_vol_20",
    "ctx_market_gap_1",
    "ctx_sector_ret_1",
    "ctx_sector_ret_5",
    "ctx_vol_ret_1",
    "ctx_vol_ret_5",
    "ctx_vol_level_z",
    "ctx_lead_fut_ret_1",
    "ctx_lead_fut_ret_5",
    "ctx_lead_fut_gap_1",
    "ctx_rates_ret_1",
    "ctx_rates_ret_5",
    "ctx_sector_rel_1",
    "ctx_lead_lag_spread_1",
    "ctx_market_vol_spread_1",
    "ctx_rates_risk_spread_1",
)


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _safe_tag(value: str) -> str:
    out = str(value or "").upper()
    for ch in ("\\", "/", "^", "=", ":", " "):
        out = out.replace(ch, "_")
    return out


def _context_symbols(instrument_key: str = None) -> dict[str, str]:
    meta = INSTRUMENTS.get(str(instrument_key or DEFAULT_INSTRUMENT or "").upper(), {})
    raw = meta.get("context_daily_symbols", {})
    if not isinstance(raw, dict):
        return {}
    out = {}
    for role in _ROLE_ORDER:
        sym = raw.get(role)
        if not sym:
            continue
        out[str(role)] = str(sym).strip()
    return out


def _context_history_path(instrument_key: str, role: str) -> str:
    return os.path.join(DATA_DIR, f"context_{_safe_tag(instrument_key)}_{_safe_tag(role)}.csv")


def _normalize_daily_history(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    try:
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
    except Exception:
        pass
    try:
        out.columns = [str(c).lower().replace(" ", "_") for c in out.columns]
    except Exception:
        pass
    try:
        idx = pd.to_datetime(out.index, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC")
        out.index = idx
    except Exception:
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()].sort_index().copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = np.nan if col != "volume" else 0.0
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["volume"] = out["volume"].fillna(0.0)
    try:
        out = out[~out.index.duplicated(keep="last")].copy()
    except Exception:
        pass
    return out


def _daily_trading_dates(index_like) -> pd.DatetimeIndex:
    idx = pd.to_datetime(index_like, errors="coerce")
    idx = idx[~idx.isna()]
    if len(idx) == 0:
        return pd.DatetimeIndex([], tz="America/New_York")
    if getattr(idx, "tz", None) is not None:
        try:
            idx_utc = idx.tz_convert("UTC")
            if bool(((idx_utc.hour == 0) & (idx_utc.minute == 0) & (idx_utc.second == 0)).all()):
                return pd.DatetimeIndex(idx_utc.tz_localize(None).normalize()).tz_localize("America/New_York")
            return idx.tz_convert("America/New_York").normalize()
        except Exception:
            pass
    # Daily context bars keyed by plain dates should map to the same local trading date,
    # not shift backward because of an assumed UTC midnight.
    return pd.DatetimeIndex(idx.normalize()).tz_localize("America/New_York")


def _fetch_remote_daily(symbol: str) -> pd.DataFrame:
    if yf is None or not ALLOW_REMOTE_CONTEXT_DAILY:
        return pd.DataFrame()
    try:
        df = yf.download(symbol, period="5y", interval="1d", progress=False, auto_adjust=False)
    except Exception:
        return pd.DataFrame()
    return _normalize_daily_history(df)


def load_context_daily_history(instrument_key: str, role: str, allow_remote: bool = True) -> pd.DataFrame:
    if not CONTEXT_PIPELINE_ENABLED:
        return pd.DataFrame()
    instr = str(instrument_key or DEFAULT_INSTRUMENT or "").upper()
    role_key = str(role or "").lower()
    cache_key = (instr, role_key)
    cached = _CACHE.get(cache_key)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    path = _context_history_path(instr, role_key)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = _normalize_daily_history(df)
            _CACHE[cache_key] = df.copy()
            return df
        except Exception:
            pass
    if not allow_remote:
        return pd.DataFrame()
    miss_ts = _MISSING_CACHE.get(cache_key)
    if miss_ts is not None and (time.time() - miss_ts) < 900.0:
        return pd.DataFrame()
    symbol = _context_symbols(instr).get(role_key)
    if not symbol:
        return pd.DataFrame()
    df = _fetch_remote_daily(symbol)
    if df.empty:
        _MISSING_CACHE[cache_key] = time.time()
        return pd.DataFrame()
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        df.to_csv(path)
    except Exception:
        pass
    _CACHE[cache_key] = df.copy()
    return df


def maybe_refresh_context_data(instrument_key: str):
    if not CONTEXT_PIPELINE_ENABLED:
        return
    instr = str(instrument_key or DEFAULT_INSTRUMENT or "").upper()
    for role in _context_symbols(instr):
        path = _context_history_path(instr, role)
        try:
            age_hours = (time.time() - os.path.getmtime(path)) / 3600.0 if os.path.exists(path) else None
        except Exception:
            age_hours = None
        if age_hours is not None and age_hours < float(CONTEXT_REFRESH_MAX_AGE_HOURS):
            continue
        df = load_context_daily_history(instr, role, allow_remote=True)
        if df.empty:
            continue


def _derived_role_frame(df: pd.DataFrame, role: str, shift_days: int = 1) -> pd.DataFrame:
    hist = _normalize_daily_history(df)
    if hist.empty:
        return pd.DataFrame()
    out = hist.copy()
    out["date_key"] = pd.Index(_daily_trading_dates(out.index))
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["gap_1"] = (out["open"] / out["close"].shift(1)) - 1.0
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=10).std()
    roll_mean = out["close"].rolling(20, min_periods=10).mean()
    roll_std = out["close"].rolling(20, min_periods=10).std().replace(0.0, np.nan)
    out["level_z"] = ((out["close"] - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan)
    feat_cols = ["ret_1", "ret_5", "gap_1", "vol_20", "level_z"]
    if int(shift_days) > 0:
        out[feat_cols] = out[feat_cols].shift(int(shift_days))
    out = out[["date_key"] + feat_cols].dropna(subset=["date_key"]).copy()
    out = out.drop_duplicates(subset=["date_key"], keep="last")
    out = out.set_index("date_key").sort_index()
    if out.empty:
        return out
    full_idx = pd.date_range(out.index.min(), out.index.max() + pd.Timedelta(days=7), freq="D", tz=out.index.tz)
    out = out.reindex(full_idx).ffill()
    rename_map = {}
    if role == "vol":
        rename_map = {
            "ret_1": "ctx_vol_ret_1",
            "ret_5": "ctx_vol_ret_5",
            "level_z": "ctx_vol_level_z",
        }
    elif role == "market":
        rename_map = {
            "ret_1": "ctx_market_ret_1",
            "ret_5": "ctx_market_ret_5",
            "vol_20": "ctx_market_vol_20",
            "gap_1": "ctx_market_gap_1",
        }
    elif role == "sector":
        rename_map = {
            "ret_1": "ctx_sector_ret_1",
            "ret_5": "ctx_sector_ret_5",
        }
    elif role == "lead_fut":
        rename_map = {
            "ret_1": "ctx_lead_fut_ret_1",
            "ret_5": "ctx_lead_fut_ret_5",
            "gap_1": "ctx_lead_fut_gap_1",
        }
    elif role == "rates":
        rename_map = {
            "ret_1": "ctx_rates_ret_1",
            "ret_5": "ctx_rates_ret_5",
        }
    keep = [c for c in rename_map if c in out.columns]
    out = out[keep].rename(columns={k: v for k, v in rename_map.items() if k in keep})
    return out


def load_context_features_for_index(index_utc, instrument_key: str = None, shift_days: int = 1) -> pd.DataFrame:
    idx = pd.to_datetime(index_utc, utc=True, errors="coerce")
    out = pd.DataFrame(index=idx)
    for col in CONTEXT_FEATURE_COLUMNS:
        out[col] = 0.0
    if not CONTEXT_PIPELINE_ENABLED or len(idx) == 0:
        return out
    symbols = _context_symbols(instrument_key)
    if not symbols:
        return out
    try:
        date_keys = pd.Index(idx.tz_convert("America/New_York").normalize())
    except Exception:
        date_keys = pd.Index(idx.normalize())
    for role in _ROLE_ORDER:
        if role not in symbols:
            continue
        hist = load_context_daily_history(instrument_key or DEFAULT_INSTRUMENT, role, allow_remote=True)
        if hist.empty:
            continue
        derived = _derived_role_frame(hist, role=role, shift_days=shift_days)
        if derived.empty:
            continue
        aligned = derived.reindex(date_keys).ffill()
        aligned.index = out.index
        for col in aligned.columns:
            out[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).values
    # low-capacity cross-context features
    try:
        out["ctx_sector_rel_1"] = out["ctx_sector_ret_1"] - out["ctx_market_ret_1"]
    except Exception:
        out["ctx_sector_rel_1"] = 0.0
    try:
        out["ctx_lead_lag_spread_1"] = out["ctx_lead_fut_ret_1"] - out["ctx_market_ret_1"]
    except Exception:
        out["ctx_lead_lag_spread_1"] = 0.0
    try:
        out["ctx_market_vol_spread_1"] = out["ctx_market_ret_1"] - out["ctx_vol_ret_1"]
    except Exception:
        out["ctx_market_vol_spread_1"] = 0.0
    try:
        out["ctx_rates_risk_spread_1"] = out["ctx_market_ret_1"] - out["ctx_rates_ret_1"]
    except Exception:
        out["ctx_rates_risk_spread_1"] = 0.0
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_context_features(df: pd.DataFrame, instrument_key: str = None, shift_days: int = 1) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in CONTEXT_FEATURE_COLUMNS:
        if col not in out.columns:
            out[col] = 0.0
    try:
        aligned = load_context_features_for_index(out.index, instrument_key=instrument_key, shift_days=shift_days)
        if aligned is not None and not aligned.empty:
            for col in CONTEXT_FEATURE_COLUMNS:
                if col in aligned.columns:
                    out[col] = pd.to_numeric(aligned[col], errors="coerce").reindex(out.index).fillna(0.0)
    except Exception:
        pass
    return out
