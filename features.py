# features.py — improved, leakage-safe feature builder with microstructure and higher-timeframe context
import pandas as pd
import numpy as np

# Try to import INTERVAL from config; if not available, fall back to "1m"
try:
    from config import (
        INTERVAL,
        TRANSACTION_COST_PER_TRADE,
        SLIPPAGE_PCT,
        INSTRUMENTS,
        DEFAULT_INSTRUMENT,
        NEWS_PIPELINE_ENABLED,
        NEWS_PIPELINE_SYMBOL,
    )
except Exception:
    INTERVAL = "1m"
    TRANSACTION_COST_PER_TRADE = 0.0003
    SLIPPAGE_PCT = 0.0005
    INSTRUMENTS = {}
    DEFAULT_INSTRUMENT = None
    NEWS_PIPELINE_ENABLED = False
    NEWS_PIPELINE_SYMBOL = "QQQ"

try:
    from news_pipeline import load_news_features_for_index
except Exception:
    load_news_features_for_index = None

try:
    from market_context import add_context_features
except Exception:
    add_context_features = None

# ===== CONFIG =====
# We'll target a 4-minute prediction horizon by default; compute bars from INTERVAL
PREDICT_MINUTES = 4  # target horizon in minutes (explicit)
DEFAULT_FEATURES = [
    "ret_1", "ret_3", "ret_6", "vol_6", "vol_12", "mom_6", "mom_12", "vol_z",
    # intraday session features
    "minute_of_day", "mins_since_open", "mins_to_close", "tod_sin", "tod_cos",
    "phase_open_ramp", "phase_lunch_lull", "phase_close_ramp", "phase_us_open",
    "tod_ret_mean", "tod_ret_std", "wday_tod_ret_mean", "tod_vol_ratio",
    # trend features (helpful for futures)
    "kf_slope", "kf_level", "ema_spread", "ewm_vol",
    # microstructure and higher timeframe
    "spread", "bid_ask_imbalance", "vwap_1h", "sma_1h", "sma_4h", "vwap_daily",
    # new features (added/extended)
    "vol_norm_ret", "atr_6", "vol_rolling_skew", "vol_rolling_kurt", "volume_imbalance", "hurst",
    # robustness/realized-volatility
    "rv_20", "robust_vol_20", "semivar_imbalance_12", "fracdiff_log_close_d04",
    # market structure + event intensity
    "range_1", "body_1", "dollar_vol_z", "roll_max_dist", "roll_min_dist", "cusum_intensity", "event_time_since",
    # normalization and session/futures-specific
    "ret_1_norm", "overnight_gap", "session_volatility", "session_ret_from_open",
    # Per-stock news sentiment context (lag-safe daily aggregates)
    "news_sent_mean_d1", "news_sent_sum_d3", "news_headline_count_d1", "news_pos_ratio_d3", "news_neg_ratio_d3",
    "news_sent_ewm_d7", "news_intensity_ewm_d7", "news_entropy_d3", "news_relevance_mean_d3", "news_sent_surprise_d1",
    "news_political_count_d3", "news_political_sent_d3", "news_political_relevance_d3",
    "news_adaptive_weight_d7", "news_weighted_sent_d7", "news_weighted_political_d7",
    # low-capacity cross-asset context
    "ctx_market_ret_1", "ctx_sector_rel_1", "ctx_vol_level_z", "ctx_lead_lag_spread_1", "ctx_rates_risk_spread_1",
]
DAILY_HORIZON_FEATURES = [
    "ret_1", "ret_6", "mom_6", "mom_12",
    "overnight_gap", "session_volatility", "session_ret_from_open",
    "ewm_vol", "trend_strength", "range_1",
    "roll_max_dist", "roll_min_dist",
    "news_sent_mean_d1", "news_sent_ewm_d7", "news_intensity_ewm_d7",
    "news_adaptive_weight_d7", "news_weighted_sent_d7", "news_weighted_political_d7",
    "ctx_market_ret_1", "ctx_market_ret_5", "ctx_vol_level_z", "ctx_sector_rel_1",
]
WEEKEND_HORIZON_FEATURES = [
    "ret_1", "ret_6", "mom_6", "mom_12",
    "overnight_gap", "session_volatility", "session_ret_from_open",
    "ewm_vol", "trend_strength", "range_1",
    "roll_max_dist", "roll_min_dist",
    "wday", "days_to_week_end",
    "news_sent_mean_d1", "news_sent_ewm_d7", "news_intensity_ewm_d7",
    "news_adaptive_weight_d7", "news_weighted_sent_d7", "news_weighted_political_d7",
    "ctx_market_ret_1", "ctx_market_ret_5", "ctx_vol_level_z", "ctx_lead_lag_spread_1",
]
LIVE_MAX_BARS = 2000  # cap live feature generation to keep UI responsive
_FRACDIFF_WEIGHT_CACHE = {}

def _kalman_trend(series, q=1e-5, r=1e-2):
    import numpy as _np
    idx = series.index
    n = len(series)
    if n == 0:
        return pd.DataFrame(index=idx, columns=["kf_level", "kf_slope"])
    vals = series.values.astype(float)
    x = _np.array([vals[0], 0.0], dtype=float)
    P = _np.eye(2) * 0.1
    F = _np.array([[1.0, 1.0], [0.0, 1.0]])
    H = _np.array([1.0, 0.0]).reshape((1, 2))
    Q = _np.eye(2) * q
    R = _np.array([[r]])
    levels = _np.empty(n, dtype=float)
    slopes = _np.empty(n, dtype=float)
    for i, y in enumerate(vals):
        x = F.dot(x)
        P = F.dot(P).dot(F.T) + Q
        y_pred = float(H.dot(x)[0])
        S = float((H.dot(P).dot(H.T) + R)[0, 0])
        if S <= 0:
            K = _np.zeros((2,1))
        else:
            K = P.dot(H.T) / S
        x = x + (K.flatten() * (y - y_pred))
        P = P - K.dot(H).dot(P)
        levels[i] = x[0]
        slopes[i] = x[1]
    return pd.DataFrame({"kf_level": levels, "kf_slope": slopes}, index=idx)

def _hurst_exponent(ts, min_window=10, max_window=100):
    # Simple Hurst exponent estimator (rescaled range) — small and robust
    try:
        N = len(ts)
        if N < min_window:
            return 0.5
        max_w = min(max_window, N//2)
        ws = np.unique(np.floor(np.logspace(np.log10(min_window), np.log10(max_w), num=8)).astype(int))
        rs = []
        for w in ws:
            if w < 2:
                continue
            nblocks = N // w
            if nblocks < 2:
                continue
            block_rs = []
            for i in range(nblocks):
                block = ts[i*w:(i+1)*w]
                if len(block) < 2:
                    continue
                mean = np.mean(block)
                dev = block - mean
                Z = np.cumsum(dev)
                R = np.max(Z) - np.min(Z)
                S = np.std(block, ddof=1)
                if S > 0:
                    block_rs.append(R / S)
            if block_rs:
                rs.append(np.mean(block_rs))
        if len(rs) < 2:
            return 0.5
        # linear fit of log(R/S) vs log(window)
        x = np.log(ws[:len(rs)])
        y = np.log(rs)
        slope = np.polyfit(x, y, 1)[0]
        return float(slope)
    except Exception:
        return 0.5


def _fractional_diff_weights(d: float, max_terms: int = 48, tol: float = 1e-4):
    key = (round(float(d), 4), int(max_terms), float(tol))
    cached = _FRACDIFF_WEIGHT_CACHE.get(key)
    if cached is not None:
        return cached
    weights = [1.0]
    try:
        for k in range(1, max_terms):
            nxt = -weights[-1] * ((float(d) - float(k) + 1.0) / float(k))
            weights.append(float(nxt))
            if abs(nxt) < float(tol) and k >= 8:
                break
    except Exception:
        weights = [1.0]
    out = np.asarray(list(reversed(weights)), dtype=float)
    _FRACDIFF_WEIGHT_CACHE[key] = out
    return out

def _compute_vwap(series_price, series_vol):
    try:
        pv = series_price * series_vol
        vwap = pv.sum() / (series_vol.sum() + 1e-12)
        return vwap
    except Exception:
        return np.nan

# Helper: convert interval string like "2m" or "1h" -> minutes-per-bar
def _interval_minutes(interval: str) -> int:
    interval = str(interval).strip().lower()
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    # fallback: try parse as minutes
    try:
        return int(interval)
    except Exception:
        return 1

BAR_MINUTES = _interval_minutes(INTERVAL)
# number of bars per hour / day assuming typical US session 6.5h = 390 minutes
HOUR_BARS = max(1, 60 // BAR_MINUTES)
DAY_BARS = max(1, 390 // BAR_MINUTES)

def _filter_extreme_moves(df: pd.DataFrame, instr_meta: dict | None = None) -> pd.DataFrame:
    """
    Drop obviously bad bars (feed glitches) to stabilize training targets.
    This filter is only applied in training builders, not live inference.
    """
    if df is None or df.empty or "close" not in df.columns:
        return df
    series = pd.to_numeric(df["close"], errors="coerce")
    ret = series.pct_change()
    vol = ret.rolling(120, min_periods=30).std()
    try:
        vol_floor = float((instr_meta or {}).get("vol_floor", 0.00025))
    except Exception:
        vol_floor = 0.00025
    vol = vol.clip(lower=vol_floor)
    spike_mask = (ret.abs() > 0.18) | (ret.abs() > 12.0 * vol)
    if spike_mask.any():
        df = df.loc[~spike_mask].copy()
    return df

def _predict_bars_from_minutes(predict_minutes: int) -> int:
    try:
        pm = int(predict_minutes)
    except Exception:
        pm = PREDICT_MINUTES
    return max(1, int(max(1, pm) // max(1, BAR_MINUTES)))

# compute PREDICT_BARS so PREDICT_MINUTES is precisely the horizon (default 4 minutes)
PREDICT_BARS = _predict_bars_from_minutes(PREDICT_MINUTES)

def _is_session_start(idx_series):
    """Given a tz-aware index (NY), return boolean mask marking first bar of local trading day (9:30)."""
    try:
        local = idx_series.tz_convert("America/New_York")
    except Exception:
        try:
            local = pd.to_datetime(idx_series).tz_localize("UTC").tz_convert("America/New_York")
        except Exception:
            local = pd.to_datetime(idx_series)
    minutes = local.hour * 60 + local.minute
    # trading session open minute (9:30 = 570)
    return minutes == (9 * 60 + 30)

def _get_instrument_meta(instrument_key: str = None) -> dict:
    key = instrument_key or DEFAULT_INSTRUMENT
    if not key:
        return {}
    return INSTRUMENTS.get(key, {})

def _session_minutes(idx_ny, session_open_minute: int, session_close_minute: int):
    minutes = idx_ny.hour * 60 + idx_ny.minute
    # If session spans midnight (open > close), use modular arithmetic.
    if session_open_minute > session_close_minute:
        mins_since_open = (minutes - session_open_minute) % 1440
        mins_to_close = (session_close_minute - minutes) % 1440
    else:
        mins_since_open = minutes - session_open_minute
        mins_to_close = session_close_minute - minutes
    return minutes, mins_since_open, mins_to_close

def _session_dates(idx_ny, session_open_minute: int, session_close_minute: int):
    # session date used for grouping overnight gaps and intraday volatility
    local_dates = pd.to_datetime(idx_ny.date)
    minutes = idx_ny.hour * 60 + idx_ny.minute
    if session_open_minute > session_close_minute:
        # overnight session: if time is before open, it belongs to previous session date
        return pd.to_datetime(np.where(minutes >= session_open_minute, local_dates, local_dates - pd.Timedelta(days=1)))
    return local_dates

def _window_mask(minute_series, start_minute: int, end_minute: int):
    # inclusive clock-time window; supports midnight wraparound windows
    if start_minute <= end_minute:
        return (minute_series >= start_minute) & (minute_series <= end_minute)
    return (minute_series >= start_minute) | (minute_series <= end_minute)


def _attach_news_features(df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    cols = [
        "news_sent_mean_d1",
        "news_sent_sum_d3",
        "news_headline_count_d1",
        "news_pos_ratio_d3",
        "news_neg_ratio_d3",
        "news_sent_ewm_d7",
        "news_intensity_ewm_d7",
        "news_entropy_d3",
        "news_relevance_mean_d3",
        "news_sent_surprise_d1",
        "news_political_count_d3",
        "news_political_sent_d3",
        "news_political_relevance_d3",
        "news_adaptive_weight_d7",
        "news_weighted_sent_d7",
        "news_weighted_political_d7",
    ]
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    if not NEWS_PIPELINE_ENABLED:
        df.attrs["news_feature_status"] = "disabled"
        return df
    symbol = str(instrument_key or DEFAULT_INSTRUMENT or "").upper()
    meta = _get_instrument_meta(instrument_key)
    if not symbol or not bool(meta.get("news_enabled", True)):
        df.attrs["news_feature_status"] = "disabled"
        return df
    if load_news_features_for_index is None:
        df.attrs["news_feature_status"] = "unavailable"
        return df
    try:
        instr_meta = _get_instrument_meta(instrument_key)
        tz = instr_meta.get("tz", "America/New_York")
        aligned = load_news_features_for_index(symbol=symbol, index_utc=df.index, tz=tz)
        if aligned is None or aligned.empty:
            df.attrs["news_feature_status"] = "missing"
            return df
        for c in cols:
            if c in aligned.columns:
                df[c] = pd.to_numeric(aligned[c], errors="coerce").reindex(df.index).fillna(0.0)
        df.attrs["news_feature_status"] = "ok"
    except Exception:
        # Keep feature contract stable even when news ingestion fails.
        df.attrs["news_feature_status"] = "error"
    return df

def _build_base_features(df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    df = df.copy()
    # Normalize columns to avoid duplicate/ambiguous OHLC labels
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
    except Exception:
        pass
    try:
        df.columns = [str(c).lower() for c in df.columns]
    except Exception:
        pass
    try:
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
    except Exception:
        pass
    df = df.sort_index()
    try:
        if not df.index.is_unique:
            df = df[~df.index.duplicated(keep="last")].copy()
    except Exception:
        pass
    # Drop NaN in close early to avoid issues downstream
    df = df.dropna(subset=["close"])
    # Ensure basic OHLCV columns exist
    if "volume" not in df.columns:
        df["volume"] = 0.0
    if "open" not in df.columns:
        df["open"] = df["close"]
    if "high" not in df.columns:
        df["high"] = df["close"]
    if "low" not in df.columns:
        df["low"] = df["close"]
    # Coerce OHLCV to numeric to avoid object downcasting warnings
    for col in ("open", "high", "low", "close", "volume"):
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            pass
    # Track whether we have real OHLC (before filling)
    try:
        df["_has_ohlc"] = (~df["open"].isna()) & (~df["high"].isna()) & (~df["low"].isna())
    except Exception:
        df["_has_ohlc"] = False
    # Fill missing OHLC with close to avoid NaNs downstream (feature stability)
    try:
        df["open"] = df["open"].fillna(df["close"])
        df["high"] = df["high"].fillna(df["close"])
        df["low"] = df["low"].fillna(df["close"])
    except Exception:
        pass
    try:
        df["volume"] = df["volume"].fillna(0.0)
    except Exception:
        pass
    # Ensure we operate with a tz-aware index; if naive assume UTC
    try:
        idx = pd.to_datetime(df.index)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        idx_ny = idx.tz_convert("America/New_York")
    except Exception:
        try:
            idx_ny = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
        except Exception:
            idx_ny = df.index
    df.index = pd.to_datetime(df.index, utc=True)
    # minute_of_day: 0..1439
    try:
        df["minute_of_day"] = idx_ny.hour * 60 + idx_ny.minute
    except Exception:
        df["minute_of_day"] = 0
    # Detect extended-hours trading (futures/24h) to avoid misleading session features
    try:
        minutes = idx_ny.hour * 60 + idx_ny.minute
        in_rth = (minutes >= (9 * 60 + 30)) & (minutes <= (16 * 60))
        extended_hours = (np.mean(~in_rth) > 0.05)
    except Exception:
        extended_hours = False
    # Instrument-aware session timing
    instr_meta = _get_instrument_meta(instrument_key)
    session_open_minute = int(instr_meta.get("session_open_minute", 9 * 60 + 30))
    session_close_minute = int(instr_meta.get("session_close_minute", 16 * 60))
    instr_type = instr_meta.get("type", "equity")
    if instr_type == "futures" or extended_hours:
        _, mins_since_open, mins_to_close = _session_minutes(idx_ny, session_open_minute, session_close_minute)
        df["mins_since_open"] = mins_since_open
        df["mins_to_close"] = mins_to_close
    else:
        df["mins_since_open"] = df["minute_of_day"] - session_open_minute
        df["mins_to_close"] = session_close_minute - df["minute_of_day"]
    # cyclical time-of-day features (helpful for 24h or extended-hours markets)
    try:
        angle = 2.0 * np.pi * (df["minute_of_day"] / 1440.0)
        df["tod_sin"] = np.sin(angle)
        df["tod_cos"] = np.cos(angle)
    except Exception:
        df["tod_sin"] = 0.0
        df["tod_cos"] = 1.0
    # Deterministic intraday phase features differ for equities vs futures.
    try:
        minute_series = pd.Series(pd.to_numeric(df["minute_of_day"], errors="coerce").fillna(0).astype(int), index=df.index)
        if instr_type == "futures" or extended_hours:
            mins_from_open = (minute_series - session_open_minute) % 1440
            mins_to_close = (session_close_minute - minute_series) % 1440
            phase_open = (mins_from_open >= 0) & (mins_from_open < 45)
            phase_close = (mins_to_close >= 0) & (mins_to_close <= 60)
            phase_lunch = _window_mask(minute_series, 11 * 60 + 30, 13 * 60 + 30)
            # Overnight-traded instruments often react to US cash open.
            phase_us_open = _window_mask(minute_series, 9 * 60 + 20, 10 * 60 + 30)
        else:
            rth_mask = _window_mask(minute_series, session_open_minute, session_close_minute)
            open_end = (session_open_minute + 45) % 1440
            close_start = (session_close_minute - 60) % 1440
            phase_open = _window_mask(minute_series, session_open_minute, open_end) & rth_mask
            phase_lunch = _window_mask(minute_series, 11 * 60 + 30, 13 * 60 + 30) & rth_mask
            phase_close = _window_mask(minute_series, close_start, session_close_minute) & rth_mask
            phase_us_open = _window_mask(minute_series, 9 * 60 + 30, 10 * 60 + 30) & rth_mask
        df["phase_open_ramp"] = phase_open.astype(float)
        df["phase_lunch_lull"] = phase_lunch.astype(float)
        df["phase_close_ramp"] = phase_close.astype(float)
        df["phase_us_open"] = phase_us_open.astype(float)
    except Exception:
        df["phase_open_ramp"] = 0.0
        df["phase_lunch_lull"] = 0.0
        df["phase_close_ramp"] = 0.0
        df["phase_us_open"] = 0.0
    # We'll compute returns using the raw close returns and then shift them so features at t use data up to t-1 (no leakage).
    rets = df["close"].pct_change()
    df["ret_1"] = rets.shift(1)
    df["ret_2"] = rets.shift(2)
    df["ret_3"] = rets.shift(3)
    df["ret_4"] = rets.shift(4)
    df["ret_6"] = rets.shift(6)
    # Volatility (rolling std) — computed on lagged returns to avoid peeking
    df["vol_3"] = df["ret_1"].rolling(3, min_periods=1).std()
    df["vol_6"] = df["ret_1"].rolling(6, min_periods=1).std()
    df["vol_12"] = df["ret_1"].rolling(12, min_periods=1).std()
    # Momentum / displacement on shifted closes
    df["close_shift_1"] = df["close"].shift(1)
    df["open_shift_1"] = df["open"].shift(1) if "open" in df.columns else df["close_shift_1"]
    df["high_shift_1"] = df["high"].shift(1) if "high" in df.columns else df["close_shift_1"]
    df["low_shift_1"] = df["low"].shift(1) if "low" in df.columns else df["close_shift_1"]
    df["mom_3"] = df["close_shift_1"] - df["close_shift_1"].shift(3)
    df["mom_6"] = df["close_shift_1"] - df["close_shift_1"].shift(6)
    df["mom_12"] = df["close_shift_1"] - df["close_shift_1"].shift(12)
    # Volume z-score (robust)
    vol_series = pd.to_numeric(df["volume"], errors="coerce")
    vol_mean = vol_series.shift(1).rolling(20, min_periods=1).mean()
    vol_std = vol_series.shift(1).rolling(20, min_periods=1).std().replace(0, pd.NA)
    df["vol_z"] = (vol_series - vol_mean) / vol_std
    df["vol_z"] = pd.to_numeric(df["vol_z"], errors="coerce").fillna(0.0)
    # Intraday deterministic baselines: minute-of-day and weekday-minute seasonal priors.
    try:
        minute_key = pd.Series(pd.to_numeric(df["minute_of_day"], errors="coerce").fillna(0).astype(int), index=df.index)
        try:
            dow_key = pd.Series(idx_ny.weekday, index=df.index).astype(int)
        except Exception:
            dow_key = pd.Series(0, index=df.index, dtype=int)
        lag_ret = pd.to_numeric(df["ret_1"], errors="coerce").fillna(0.0)
        lag_ret_sq = lag_ret * lag_ret

        ret_prior_sum = lag_ret.groupby(minute_key).cumsum() - lag_ret
        ret_prior_count = lag_ret.groupby(minute_key).cumcount()
        ret_prior_mean = ret_prior_sum / ret_prior_count.replace(0, np.nan)
        ret2_prior_sum = lag_ret_sq.groupby(minute_key).cumsum() - lag_ret_sq
        ret_prior_var = (ret2_prior_sum / ret_prior_count.replace(0, np.nan)) - (ret_prior_mean * ret_prior_mean)
        df["tod_ret_mean"] = ret_prior_mean.fillna(0.0)
        df["tod_ret_std"] = np.sqrt(ret_prior_var.clip(lower=0.0)).fillna(df["vol_6"]).fillna(0.0)

        wday_minute_key = (dow_key * 1440 + minute_key).astype(int)
        wday_prior_sum = lag_ret.groupby(wday_minute_key).cumsum() - lag_ret
        wday_prior_count = lag_ret.groupby(wday_minute_key).cumcount()
        df["wday_tod_ret_mean"] = (wday_prior_sum / wday_prior_count.replace(0, np.nan)).fillna(df["tod_ret_mean"]).fillna(0.0)

        lag_volume = pd.to_numeric(df["volume"].shift(1), errors="coerce").fillna(0.0)
        vol_prior_sum = lag_volume.groupby(minute_key).cumsum() - lag_volume
        vol_prior_count = lag_volume.groupby(minute_key).cumcount()
        vol_prior_mean = vol_prior_sum / vol_prior_count.replace(0, np.nan)
        vol_ratio = lag_volume / vol_prior_mean.replace(0, np.nan)
        df["tod_vol_ratio"] = vol_ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 10.0)
    except Exception:
        df["tod_ret_mean"] = 0.0
        df["tod_ret_std"] = 0.0
        df["wday_tod_ret_mean"] = 0.0
        df["tod_vol_ratio"] = 1.0
    # Dollar-volume z-score (liquidity proxy)
    try:
        dv = (df["close_shift_1"].astype(float) * df["volume"].shift(1).astype(float))
        dv = pd.to_numeric(dv, errors="coerce")
        dv_mean = dv.rolling(20, min_periods=1).mean()
        dv_std = dv.rolling(20, min_periods=1).std().replace(0, pd.NA)
        df["dollar_vol_z"] = (dv - dv_mean) / dv_std
        df["dollar_vol_z"] = pd.to_numeric(df["dollar_vol_z"], errors="coerce").fillna(0.0)
    except Exception:
        df["dollar_vol_z"] = 0.0
    # --- Trend features (Kalman + EMAs + EWMA vol) ---
    try:
        # Use close_shift_1 for trend estimation to avoid label leakage
        series = df["close_shift_1"].ffill().fillna(0.0)
        kf = _kalman_trend(series)
        df["kf_level"] = kf["kf_level"]
        df["kf_slope"] = kf["kf_slope"]
    except Exception:
        df["kf_level"] = 0.0
        df["kf_slope"] = 0.0
    try:
        df["ema_5"] = df["close_shift_1"].ewm(span=5, adjust=False).mean()
        df["ema_20"] = df["close_shift_1"].ewm(span=20, adjust=False).mean()
        df["ema_spread"] = df["ema_5"] - df["ema_20"]
    except Exception:
        df["ema_5"] = df["close"]
        df["ema_20"] = df["close"]
        df["ema_spread"] = 0.0
    try:
        # EWM vol on lagged returns (avoid leakage)
        df["ewm_vol"] = df["ret_1"].ewm(span=20, adjust=False).std().fillna(0.0)
    except Exception:
        df["ewm_vol"] = 0.0
    try:
        abs_ret = df["ret_1"].abs()
        df["rv_20"] = np.sqrt(np.pi / 2.0) * abs_ret.rolling(20, min_periods=5).mean()
        df["robust_vol_20"] = abs_ret.rolling(20, min_periods=5).median()
    except Exception:
        df["rv_20"] = 0.0
        df["robust_vol_20"] = 0.0
    try:
        pos_ret = df["ret_1"].clip(lower=0.0)
        neg_ret = (-df["ret_1"].clip(upper=0.0)).clip(lower=0.0)
        up_semivar = (pos_ret ** 2).rolling(12, min_periods=3).mean()
        down_semivar = (neg_ret ** 2).rolling(12, min_periods=3).mean()
        den = (up_semivar + down_semivar).replace(0.0, np.nan)
        df["semivar_imbalance_12"] = ((up_semivar - down_semivar) / den).fillna(0.0)
    except Exception:
        df["semivar_imbalance_12"] = 0.0
    try:
        df["trend_strength"] = (df["ema_spread"].abs() / (df["ewm_vol"] + 1e-9)).fillna(0.0)
    except Exception:
        df["trend_strength"] = 0.0
    # --- Microstructure features (best-effort) ---
    try:
        if "ask" in df.columns and "bid" in df.columns:
            df["spread"] = df["ask"] - df["bid"]
            df["spread"] = df["spread"] / (df["close"].replace(0, 1e-9))
        else:
            df["spread"] = np.nan
    except Exception:
        df["spread"] = np.nan
    try:
        if "bid_size" in df.columns and "ask_size" in df.columns:
            df["bid_ask_imbalance"] = (df["bid_size"] - df["ask_size"]) / (df["bid_size"] + df["ask_size"] + 1e-9)
        else:
            df["bid_ask_imbalance"] = 0.0
    except Exception:
        df["bid_ask_imbalance"] = 0.0
    # --- Higher-timeframe context (1h, 4h, daily VWAP) ---
    try:
        df["pv"] = df["close"] * df["volume"]
        df["vwap_1h"] = df["pv"].rolling(window=HOUR_BARS, min_periods=1).sum() / (df["volume"].rolling(window=HOUR_BARS, min_periods=1).sum() + 1e-12)
        df["vwap_daily"] = df["pv"].rolling(window=DAY_BARS, min_periods=1).sum() / (df["volume"].rolling(window=DAY_BARS, min_periods=1).sum() + 1e-12)
        df["sma_1h"] = df["close"].rolling(window=HOUR_BARS, min_periods=1).mean()
        df["sma_4h"] = df["close"].rolling(window=HOUR_BARS * 4, min_periods=1).mean()
    except Exception:
        df["vwap_1h"] = np.nan
        df["vwap_daily"] = np.nan
        df["sma_1h"] = np.nan
        df["sma_4h"] = np.nan
    # new features
    try:
        df["vol_norm_ret"] = df["ret_1"] / (df["vol_6"].replace(0, 1e-9))
    except Exception:
        df["vol_norm_ret"] = 0.0
    try:
        rng = (df["close_shift_1"] - df["close_shift_1"].shift(1)).abs()
        df["atr_6"] = rng.shift(0).rolling(6, min_periods=1).mean()
    except Exception:
        df["atr_6"] = 0.0
    try:
        df["vol_rolling_skew"] = df["ret_1"].rolling(24, min_periods=1).skew().fillna(0.0)
        df["vol_rolling_kurt"] = df["ret_1"].rolling(24, min_periods=1).kurt().fillna(0.0)
    except Exception:
        df["vol_rolling_skew"] = 0.0
        df["vol_rolling_kurt"] = 0.0
    try:
        price_diff = df["close_shift_1"].diff()
        vol_mean = df["volume"].shift(1).rolling(20, min_periods=1).mean()
        df["volume_imbalance"] = np.sign(price_diff.fillna(0.0)) * (df["volume"].shift(1) / (vol_mean + 1e-9))
        df["volume_imbalance"] = df["volume_imbalance"].fillna(0.0).clip(-5.0, 5.0)
    except Exception:
        df["volume_imbalance"] = 0.0
    # Candle structure + proximity to recent extrema
    try:
        df["range_1"] = (df["high_shift_1"] - df["low_shift_1"]) / (df["close_shift_1"].replace(0, 1e-9))
    except Exception:
        df["range_1"] = 0.0
    try:
        df["body_1"] = (df["close_shift_1"] - df["open_shift_1"]) / (df["open_shift_1"].replace(0, 1e-9))
    except Exception:
        df["body_1"] = 0.0
    try:
        roll_window = max(12, HOUR_BARS * 2)
        roll_max = df["close_shift_1"].rolling(roll_window, min_periods=1).max()
        roll_min = df["close_shift_1"].rolling(roll_window, min_periods=1).min()
        df["roll_max_dist"] = (df["close_shift_1"] - roll_max) / (roll_max.replace(0, 1e-9))
        df["roll_min_dist"] = (df["close_shift_1"] - roll_min) / (roll_min.replace(0, 1e-9))
    except Exception:
        df["roll_max_dist"] = 0.0
        df["roll_min_dist"] = 0.0
    try:
        window_len = max(32, BAR_MINUTES)
        df["hurst"] = df["close_shift_1"].rolling(window=window_len, min_periods=32).apply(lambda x: _hurst_exponent(x.values), raw=False).fillna(0.5)
    except Exception:
        df["hurst"] = 0.5
    try:
        weights = _fractional_diff_weights(0.40, max_terms=max(24, HOUR_BARS * 6), tol=1e-4)
        log_px = np.log(df["close_shift_1"].replace(0, np.nan)).ffill()
        fd = log_px.rolling(len(weights), min_periods=len(weights)).apply(lambda x: float(np.dot(weights, x)), raw=True)
        fd_mean = fd.rolling(32, min_periods=8).mean()
        fd_std = fd.rolling(32, min_periods=8).std().replace(0.0, np.nan)
        df["fracdiff_log_close_d04"] = ((fd - fd_mean) / fd_std).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-6.0, 6.0)
    except Exception:
        df["fracdiff_log_close_d04"] = 0.0

    # ----- NEW: volatility-normalized 1-bar return (helps short-horizon models avoid absolute-bias) -----
    try:
        # Use ewm_vol (lagged) to normalize ret_1
        df["ret_1_norm"] = df["ret_1"] / (df["ewm_vol"].replace(0, 1e-9))
        # Clip extremely large values that would dominate training
        df["ret_1_norm"] = df["ret_1_norm"].clip(lower=-10.0, upper=10.0)
    except Exception:
        df["ret_1_norm"] = 0.0

    # ----- NEW: session/overnight features (useful for futures and gap-aware assets) -----
    try:
        # create NY-local session grouping to detect session boundaries
        idx = pd.to_datetime(df.index, utc=True)
        local_idx = idx.tz_convert("America/New_York")
        session_dates = _session_dates(local_idx, session_open_minute, session_close_minute)
        df["session_date"] = pd.to_datetime(session_dates).date
        # First/last bars by session.
        session_first_close = df.groupby("session_date")["close"].transform("first")
        session_first_open = df.groupby("session_date")["open"].transform("first")
        session_last_close_map = df.groupby("session_date")["close"].last().shift(1)
        prev_session_close = pd.Series(df["session_date"], index=df.index).map(session_last_close_map)
        # mark first bar of session
        first_idx = df.groupby("session_date").head(1).index
        df["_is_session_start"] = df.index.isin(first_idx)
        # overnight gap: (session open - previous session close) / previous session close
        # Keep the same value across the whole session as a session-context feature.
        gap_base = prev_session_close.replace(0, np.nan)
        df["overnight_gap"] = (session_first_open - gap_base) / gap_base
        df["overnight_gap"] = df["overnight_gap"].fillna(0.0)
        # session volatility: rolling std within same session to capture intraday bursts
        df["session_volatility"] = df.groupby("session_date")["ret_1"].transform(lambda x: x.rolling(max(3, int(HOUR_BARS//2)), min_periods=1).std()).fillna(0.0)
        # session return so far from session open (useful for futures overnight drift)
        open_base = session_first_open.replace(0, np.nan)
        df["session_ret_from_open"] = ((df["close"] / open_base) - 1.0).fillna(0.0)
    except Exception:
        df["overnight_gap"] = 0.0
        df["session_volatility"] = 0.0
        df["session_ret_from_open"] = 0.0
        df["_is_session_start"] = False

    # CUSUM event intensity (information-driven sampling proxy)
    try:
        ret = df["ret_1"].fillna(0.0).values
        th = (df["ewm_vol"].fillna(0.0) * 4.0).values
        pos = 0.0
        neg = 0.0
        events = []
        time_since = []
        since = 0
        for i in range(len(ret)):
            r = float(ret[i])
            pos = max(0.0, pos + r)
            neg = min(0.0, neg + r)
            hit = int(pos > th[i] or neg < -th[i])
            if hit:
                pos = 0.0
                neg = 0.0
                since = 0
            else:
                since += 1
            events.append(hit)
            time_since.append(since)
        df["cusum_event"] = events
        df["cusum_intensity"] = pd.Series(events, index=df.index).rolling(24, min_periods=1).mean().fillna(0.0)
        df["event_time_since"] = pd.Series(time_since, index=df.index).fillna(0.0)
    except Exception:
        df["cusum_intensity"] = 0.0
        df["event_time_since"] = 0.0

    # News-driven context per workspace. Values remain zero when the news layer is unavailable.
    df = _attach_news_features(df, instrument_key=instrument_key)
    if add_context_features is not None:
        # Intraday forecasts only get prior daily context, never same-day close information.
        df = add_context_features(df, instrument_key=instrument_key, shift_days=1)

    # Clip extreme feature values to stabilize training
    try:
        df["dollar_vol_z"] = df["dollar_vol_z"].clip(-5.0, 5.0)
        df["range_1"] = df["range_1"].clip(0.0, 0.1)
        df["body_1"] = df["body_1"].clip(-0.1, 0.1)
        df["roll_max_dist"] = df["roll_max_dist"].clip(-0.2, 0.0)
        df["roll_min_dist"] = df["roll_min_dist"].clip(0.0, 0.2)
        df["cusum_intensity"] = df["cusum_intensity"].clip(0.0, 1.0)
        df["event_time_since"] = np.log1p(df["event_time_since"].clip(0.0, 500.0))
        df["semivar_imbalance_12"] = df["semivar_imbalance_12"].clip(-1.0, 1.0)
        df["news_headline_count_d1"] = np.log1p(df["news_headline_count_d1"].clip(lower=0.0))
        df["news_intensity_ewm_d7"] = np.log1p(df["news_intensity_ewm_d7"].clip(lower=0.0))
        df["news_relevance_mean_d3"] = df["news_relevance_mean_d3"].clip(0.0, 1.0)
        df["news_entropy_d3"] = df["news_entropy_d3"].clip(0.0, 1.0)
        df["news_political_count_d3"] = np.log1p(df["news_political_count_d3"].clip(lower=0.0))
        df["news_political_relevance_d3"] = df["news_political_relevance_d3"].clip(0.0, 1.0)
        df["news_adaptive_weight_d7"] = df["news_adaptive_weight_d7"].clip(0.0, 2.0)
    except Exception:
        pass

    # Fill NaNs conservatively (no backfill to avoid leakage)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().fillna(0.0)
    # Drop internal helper column before returning
    if "_is_session_start" in df.columns:
        try:
            df = df.drop(columns=["_is_session_start"])
        except Exception:
            pass
    # Keep session_date if helpful downstream (it won't hurt), but if you prefer remove it uncomment next line:
    # df = df.drop(columns=["session_date"])
    return df

def build_features_for_training(
        df: pd.DataFrame,
        triple_barrier=True,
        tb_profit_k=2.0,
        tb_stop_k=1.0,
        tb_max_bars=60,
        predict_minutes: int = None,
        predict_bars: int = None,
        instrument_key: str = None,
        daily_horizon: bool = False,  # NEW: param for daily mode
        weekend_horizon: bool = False  # NEW: param for weekend mode
) -> pd.DataFrame:
    """ Build features and training targets. Uses lagged/shifted computations to avoid lookahead leakage.
    Compute a triple-barrier-ish label by default. Also compute target price for regression. """
    instr_meta = _get_instrument_meta(instrument_key)
    df = _filter_extreme_moves(df, instr_meta=instr_meta)
    df = _build_base_features(df, instrument_key=instrument_key)
    if (daily_horizon or weekend_horizon) and add_context_features is not None:
        df = add_context_features(df, instrument_key=instrument_key, shift_days=0)
    if daily_horizon:
        try:
            has_ohlc_mask = df.get("_has_ohlc", pd.Series([False] * len(df), index=df.index)).astype(bool)
            has_ohlc_rate = float(has_ohlc_mask.mean()) if len(has_ohlc_mask) else 0.0
        except Exception:
            has_ohlc_mask = pd.Series([False] * len(df), index=df.index)
            has_ohlc_rate = 0.0
        # If OHLC coverage is low, keep only rows with real OHLC instead of bailing.
        if has_ohlc_rate < 0.9:
            df = df.loc[has_ohlc_mask].copy()
        if df is None or df.empty or len(df) < 200:
            raise RuntimeError("Daily horizon requires sufficient real OHLC data.")
    try:
        vol_floor = float(instr_meta.get("vol_floor", 0.00025)) if instr_meta else 0.00025
    except Exception:
        vol_floor = 0.00025
    vol_norm = df["ewm_vol"].clip(lower=max(1e-6, vol_floor))
    # Targets predict_bars ahead (default PREDICT_BARS)
    if predict_bars is None:
        predict_bars = _predict_bars_from_minutes(predict_minutes or PREDICT_MINUTES)
    df["target_price"] = df["close"].shift(-predict_bars)
    # NOTE: Removed cost subtraction to avoid negative bias in targets; apply costs in backtest instead
    # cost_sub = TRANSACTION_COST_PER_TRADE + SLIPPAGE_PCT
    df["target_ret"] = (df["target_price"] / df["close"]) - 1.0  # - cost_sub  # commented out
    # --- NEW: normalized target (for regression stability) ---
    try:
        df["target_ret_norm"] = df["target_ret"] / vol_norm
        df["target_ret_norm"] = df["target_ret_norm"].clip(lower=-50.0, upper=50.0)
    except Exception:
        df["target_ret_norm"] = df["target_ret"]
    df["target_dir"] = (df["target_ret"] > 0).astype(int)
    instr_meta = _get_instrument_meta(instrument_key)
    if daily_horizon:  # NEW: compute daily high/low targets
        target_high = []
        target_low = []
        for date, group in df.groupby("session_date"):
            group = group.sort_index()
            for i in range(len(group)):
                future_high = group["high"].iloc[i:].max() if "high" in group.columns else group["close"].iloc[i:].max()
                future_low = group["low"].iloc[i:].min() if "low" in group.columns else group["close"].iloc[i:].min()
                close = group["close"].iloc[i]
                high_ret = (future_high - close) / close if close > 0 else 0.0
                low_ret = (future_low - close) / close if close > 0 else 0.0
                target_high.append(high_ret)
                target_low.append(low_ret)
        df["target_high_ret"] = target_high
        df["target_low_ret"] = target_low
        df["target_high_ret_norm"] = df["target_high_ret"] / vol_norm
        df["target_low_ret_norm"] = df["target_low_ret"] / vol_norm
    if weekend_horizon:
        try:
            idx_local = df.index
            try:
                idx_local = pd.to_datetime(df.index, utc=True).tz_convert("America/New_York")
            except Exception:
                idx_local = pd.to_datetime(df.index)
            df["iso_year"] = idx_local.isocalendar().year
            df["iso_week"] = idx_local.isocalendar().week
            df["dow"] = idx_local.weekday
            week_end_day = int(instr_meta.get("week_end_day", 4))
            last_key = None
            if len(idx_local) > 0:
                last_iso = idx_local[-1].isocalendar()
                last_key = (int(last_iso.year), int(last_iso.week))
            def _week_close(g):
                # drop the trailing (possibly incomplete) week to avoid partial targets
                if last_key and tuple(g.name) == last_key:
                    return np.nan
                g_end = g[g["dow"] == week_end_day]
                if not g_end.empty:
                    return float(g_end["close"].iloc[-1])
                return float(g["close"].iloc[-1])
            week_close = df.groupby(["iso_year", "iso_week"], sort=False)[["dow", "close"]].apply(_week_close)
            df = df.join(week_close.rename("week_end_close"), on=["iso_year", "iso_week"])
            df["target_weekend_ret"] = (df["week_end_close"] / df["close"]) - 1.0
            df["target_weekend_ret_norm"] = df["target_weekend_ret"] / vol_norm
        except Exception:
            df["target_weekend_ret"] = 0.0
            df["target_weekend_ret_norm"] = 0.0
    # Optional triple-barrier-like label: uses volatility-normalized thresholds
    if triple_barrier:
        # extend max bars for longer horizons
        try:
            tb_max_bars = max(int(tb_max_bars), int(predict_bars) * 6)
        except Exception:
            pass
        vols = df["ewm_vol"].replace(0, 1e-9)
        profit_th = tb_profit_k * vols
        stop_th = -tb_stop_k * vols
        closes = df["close"]
        if len(closes) >= 2:
            future_max = closes.iloc[::-1].rolling(window=tb_max_bars, min_periods=1).max().iloc[::-1].shift(-1)
            future_min = closes.iloc[::-1].rolling(window=tb_max_bars, min_periods=1).min().iloc[::-1].shift(-1)
        else:
            future_max = pd.Series(index=closes.index, dtype=float)
            future_min = pd.Series(index=closes.index, dtype=float)
        base = closes
        fut_ret_max = (future_max / base) - 1.0
        fut_ret_min = (future_min / base) - 1.0
        labels = []
        for i in range(len(df)):
            try:
                p_th = profit_th.iat[i]
                s_th = stop_th.iat[i]
                fm = fut_ret_max.iat[i] if not pd.isna(fut_ret_max.iat[i]) else None
                fn = fut_ret_min.iat[i] if not pd.isna(fut_ret_min.iat[i]) else None
                lab = None
                if fm is not None and fm >= p_th:
                    lab = "profit"
                elif fn is not None and fn <= s_th:
                    lab = "stop"
                else:
                    lab = "timeout"
            except Exception:
                lab = "timeout"
            labels.append(lab)
        df["tb_label"] = labels
    df = df.dropna()
    return df

def make_feature_matrix(df: pd.DataFrame, feature_list=None, instrument_key: str = None) -> pd.DataFrame:
    """ Builds features for live prediction. Returns the most recent row as a DataFrame (1 row) or an empty DataFrame.
    All features are computed using lagged/shifted inputs, so this is safe for live use. """
    if feature_list is None:
        feature_list = DEFAULT_FEATURES
    try:
        if df is not None and len(df) > LIVE_MAX_BARS:
            df = df.iloc[-LIVE_MAX_BARS:].copy()
    except Exception:
        pass
    df_feat = _build_base_features(df, instrument_key=instrument_key)
    # Keep rows that have no NaNs for the requested features
    df_feat = df_feat.dropna(subset=feature_list)
    if df_feat.empty:
        return pd.DataFrame()
    # Return one-row DataFrame (most recent)
    row = df_feat[feature_list].iloc[-1:].copy()
    return row

def build_feature_frame(df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    """Build full feature frame for inference/training selection (no target columns)."""
    return _build_base_features(df, instrument_key=instrument_key)

def build_next_session_dataset(df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    """Build one-row-per-session dataset to predict next session high/low from prior session features."""
    df_feat = _build_base_features(df, instrument_key=instrument_key)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    if df_feat.empty or "session_date" not in df_feat.columns:
        return pd.DataFrame()
    groups = df_feat.groupby("session_date", sort=True)
    if len(groups) < 2:
        return pd.DataFrame()
    session_stats = groups.agg({"high": "max", "low": "min"})
    session_stats["next_high"] = session_stats["high"].shift(-1)
    session_stats["next_low"] = session_stats["low"].shift(-1)
    session_last = groups.tail(1).copy()
    session_last = session_last.join(session_stats[["next_high", "next_low"]], on="session_date")
    session_last = session_last.dropna(subset=["next_high", "next_low", "close"])
    if session_last.empty:
        return pd.DataFrame()
    if add_context_features is not None:
        session_last = add_context_features(session_last, instrument_key=instrument_key, shift_days=0)
    session_last["target_high_ret"] = (session_last["next_high"] / session_last["close"]) - 1.0
    session_last["target_low_ret"] = (session_last["next_low"] / session_last["close"]) - 1.0
    instr_meta = _get_instrument_meta(instrument_key)
    try:
        vol_floor = float(instr_meta.get("vol_floor", 0.00025)) if instr_meta else 0.00025
    except Exception:
        vol_floor = 0.00025
    vol_norm = session_last["ewm_vol"].clip(lower=max(1e-6, vol_floor))
    session_last["target_high_ret_norm"] = session_last["target_high_ret"] / vol_norm
    session_last["target_low_ret_norm"] = session_last["target_low_ret"] / vol_norm
    return session_last.dropna(subset=["target_high_ret_norm", "target_low_ret_norm"])

def build_weekend_session_dataset(df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    """Build one-row-per-session dataset to predict the current week's close from each prior session close."""
    df_feat = _build_base_features(df, instrument_key=instrument_key)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    if df_feat.empty:
        return pd.DataFrame()
    try:
        idx_local = pd.to_datetime(df_feat.index, utc=True).tz_convert("America/New_York")
    except Exception:
        idx_local = pd.to_datetime(df_feat.index)
    df_feat = df_feat.copy()
    df_feat["iso_year"] = idx_local.isocalendar().year
    df_feat["iso_week"] = idx_local.isocalendar().week
    df_feat["dow"] = idx_local.weekday
    if "session_date" not in df_feat.columns:
        df_feat["session_date"] = pd.to_datetime(idx_local.date)
    instr_meta = _get_instrument_meta(instrument_key)
    try:
        week_end_day = int(instr_meta.get("week_end_day", 4))
    except Exception:
        week_end_day = 4
    session_groups = df_feat.groupby("session_date", sort=True)
    if len(session_groups) < 2:
        return pd.DataFrame()
    session_last = session_groups.tail(1).copy()
    session_last = session_last.sort_index()
    if session_last.empty:
        return pd.DataFrame()
    try:
        last_key = (
            int(pd.to_numeric(session_last["iso_year"], errors="coerce").iloc[-1]),
            int(pd.to_numeric(session_last["iso_week"], errors="coerce").iloc[-1]),
        )
    except Exception:
        last_key = None

    def _week_close(g):
        if last_key and tuple(g.name) == last_key:
            return np.nan
        g_end = g[g["dow"] == week_end_day]
        if not g_end.empty:
            return float(g_end["close"].iloc[-1])
        return float(g["close"].iloc[-1])

    week_close = session_last.groupby(["iso_year", "iso_week"], sort=True)[["dow", "close"]].apply(_week_close)
    session_last = session_last.join(week_close.rename("week_end_close"), on=["iso_year", "iso_week"])
    session_last = session_last.dropna(subset=["week_end_close", "close"])
    if session_last.empty:
        return pd.DataFrame()
    if add_context_features is not None:
        session_last = add_context_features(session_last, instrument_key=instrument_key, shift_days=0)
    session_last["target_weekend_ret"] = (session_last["week_end_close"] / session_last["close"]) - 1.0
    session_last["wday"] = pd.to_numeric(session_last["dow"], errors="coerce").fillna(0.0).astype(float)
    session_last["days_to_week_end"] = (float(week_end_day) - session_last["wday"]).clip(lower=0.0)
    try:
        vol_floor = float(instr_meta.get("vol_floor", 0.00025)) if instr_meta else 0.00025
    except Exception:
        vol_floor = 0.00025
    vol_norm = session_last["ewm_vol"].clip(lower=max(1e-6, vol_floor))
    session_last["target_weekend_ret_norm"] = session_last["target_weekend_ret"] / vol_norm
    return session_last.dropna(subset=["target_weekend_ret_norm"])

def build_next_week_dataset(df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    """Build one-row-per-week dataset to predict next week close from prior week features."""
    df_feat = _build_base_features(df, instrument_key=instrument_key)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    if df_feat.empty:
        return pd.DataFrame()
    try:
        idx_local = pd.to_datetime(df_feat.index, utc=True).tz_convert("America/New_York")
    except Exception:
        idx_local = pd.to_datetime(df_feat.index)
    df_feat["iso_year"] = idx_local.isocalendar().year
    df_feat["iso_week"] = idx_local.isocalendar().week
    df_feat["dow"] = idx_local.weekday
    instr_meta = _get_instrument_meta(instrument_key)
    try:
        week_end_day = int(instr_meta.get("week_end_day", 4))
    except Exception:
        week_end_day = 4
    week_groups = df_feat.groupby(["iso_year", "iso_week"], sort=True)
    if len(week_groups) < 2:
        return pd.DataFrame()

    def _week_close(g):
        g_end = g[g["dow"] == week_end_day]
        if not g_end.empty:
            return float(g_end["close"].iloc[-1])
        return float(g["close"].iloc[-1])

    week_close = week_groups[["dow", "close"]].apply(_week_close)
    next_week_close = week_close.shift(-1)
    week_last = week_groups.tail(1).copy()
    week_last = week_last.join(next_week_close.rename("next_week_close"), on=["iso_year", "iso_week"])
    week_last = week_last.dropna(subset=["next_week_close", "close"])
    if week_last.empty:
        return pd.DataFrame()
    if add_context_features is not None:
        week_last = add_context_features(week_last, instrument_key=instrument_key, shift_days=0)
    week_last["target_weekend_ret"] = (week_last["next_week_close"] / week_last["close"]) - 1.0
    try:
        vol_floor = float(instr_meta.get("vol_floor", 0.00025)) if instr_meta else 0.00025
    except Exception:
        vol_floor = 0.00025
    vol_norm = week_last["ewm_vol"].clip(lower=max(1e-6, vol_floor))
    week_last["target_weekend_ret_norm"] = week_last["target_weekend_ret"] / vol_norm
    return week_last.dropna(subset=["target_weekend_ret_norm"])
