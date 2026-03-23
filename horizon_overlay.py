from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_DIR
from features import _build_base_features, build_next_session_dataset, build_weekend_session_dataset
from accuracy_windows import min_window_hit_rate, summarize_price_window_errors
from forecast_calibration import apply_return_calibration, fit_return_calibration, score_return_forecast
try:
    from market_context import add_context_features
except Exception:
    add_context_features = None
try:
    from news_pipeline import load_news_features_for_index, news_archive_coverage
except Exception:
    load_news_features_for_index = None
    news_archive_coverage = None

try:
    import yfinance as yf
except Exception:
    yf = None

_MODEL_KEYS = ("base", "forest", "boost", "extra_trees")
_CACHE: dict[str, dict] = {}
_DAILY_HISTORY_CACHE: dict[str, pd.DataFrame] = {}
_CURRENT_HEURISTIC_CACHE: dict[tuple, float | None] = {}
_SNAPSHOT_CACHE: dict[str, pd.DataFrame] = {}
_OUTCOME_CACHE: dict[str, pd.DataFrame] = {}

_STRUCTURAL_DAILY_CENTER_FEATURES = (
    "ret_1",
    "ema_spread",
    "vol_20",
    "close_loc",
    "wday",
    "ctx_market_ret_1",
    "ctx_vol_level_z",
)
_STRUCTURAL_DAILY_RANGE_FEATURES = (
    "range_1",
    "atr_5",
    "vol_20",
    "wday",
    "ctx_vol_level_z",
    "ctx_market_vol_20",
)
_STRUCTURAL_WEEKEND_FEATURES = (
    "days_to_week_end",
    "week_ret_sofar",
    "week_range_sofar",
    "prev_week_ret",
    "prev_week_range",
    "gap_1",
    "ctx_market_ret_5",
    "ctx_vol_level_z",
)
_STRUCTURAL_WEEKEND_SESSION_FEATURES = (
    "days_to_week_end",
    "week_ret_sofar",
    "week_range_sofar",
    "week_close_loc",
    "session_body",
    "session_range",
    "prev_week_ret",
    "prev_week_range",
    "ctx_market_ret_1",
    "ctx_vol_level_z",
)
_LONG_HORIZON_NEWS_FEATURES = (
    "news_sent_ewm_d7",
    "news_intensity_ewm_d7",
    "news_relevance_mean_d3",
)


def _clean_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        out.loc[:, num_cols] = out.loc[:, num_cols].replace([np.inf, -np.inf], np.nan)
    return out.infer_objects(copy=False)

_HORIZON_SPECS = {
    "daily_high": {
        "horizon": "daily",
        "target_col": "target_high_ret",
        "history_limit": 32,
        "default_min_train": 20,
        "candidates": (
            {"features": ("overnight_gap", "mom_12", "wday"), "k": 3, "min_train": 20},
            {"features": ("overnight_gap", "mom_12", "trend_strength"), "k": 5, "min_train": 20},
            {"features": ("session_volatility", "mom_12", "trend_strength"), "k": 5, "min_train": 20},
            {"features": ("ret_1", "trend_strength", "wday"), "k": 5, "min_train": 20},
        ),
    },
    "daily_low": {
        "horizon": "daily",
        "target_col": "target_low_ret",
        "history_limit": 32,
        "default_min_train": 20,
        "candidates": (
            {"features": ("ewm_vol", "overnight_gap", "mom_12"), "k": 7, "min_train": 20},
            {"features": ("mom_12", "wday"), "k": 7, "min_train": 20},
            {"features": ("ret_1", "mom_12", "wday"), "k": 7, "min_train": 20},
            {"features": ("overnight_gap", "ret_6", "trend_strength"), "k": 7, "min_train": 20},
        ),
    },
    "weekend": {
        "horizon": "weekend",
        "target_col": "target_weekend_ret",
        "history_limit": 36,
        "default_min_train": 12,
        "candidates": (
            {"features": ("days_to_week_end", "session_ret_from_open", "trend_strength"), "k": 5, "min_train": 12},
            {"features": ("days_to_week_end", "overnight_gap", "trend_strength"), "k": 5, "min_train": 12},
            {"features": ("days_to_week_end", "ret_6", "trend_strength", "wday"), "k": 5, "min_train": 12},
            {"features": ("days_to_week_end", "session_volatility", "mom_12", "wday"), "k": 5, "min_train": 12},
        ),
    },
}


def _overlay_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_overlay_{instrument_key}.json")


def _daily_long_history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{instrument_key}_daily_long.csv")


def _resolved_paths(instrument_key: str) -> tuple[str, str]:
    return (
        os.path.join(DATA_DIR, f"horizon_accuracy_backfill_resolved_{instrument_key}.csv"),
        os.path.join(DATA_DIR, f"horizon_accuracy_resolved_{instrument_key}.csv"),
    )


def _structural_snapshot_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_structural_snapshots_{instrument_key}.csv")


def _structural_outcome_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"horizon_structural_outcomes_{instrument_key}.csv")


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


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
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    except Exception:
        out.index = pd.to_datetime(out.index, errors="coerce")
    out = out.sort_index()
    out = out[~out.index.isna()].copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in out.columns:
            out[col] = np.nan if col != "volume" else 0.0
        try:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        except Exception:
            pass
    out = out.dropna(subset=["close"])
    out["open"] = out["open"].fillna(out["close"])
    out["high"] = out["high"].fillna(out["close"])
    out["low"] = out["low"].fillna(out["close"])
    out["volume"] = pd.to_numeric(out["volume"], errors="coerce").fillna(0.0)
    try:
        out = out[~out.index.duplicated(keep="last")].copy()
    except Exception:
        pass
    return out


def _load_long_daily_history(instrument_key: str) -> pd.DataFrame:
    instrument = str(instrument_key or "").upper()
    cached = _DAILY_HISTORY_CACHE.get(instrument)
    if isinstance(cached, pd.DataFrame) and not cached.empty:
        return cached.copy()
    path = _daily_long_history_path(instrument)
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = _normalize_daily_history(df)
            if not df.empty:
                _DAILY_HISTORY_CACHE[instrument] = df.copy()
                return df
        except Exception:
            pass
    if yf is None:
        return pd.DataFrame()
    try:
        fetched = yf.download(instrument, period="5y", interval="1d", progress=False, auto_adjust=False)
        fetched = _normalize_daily_history(fetched)
        if fetched.empty:
            return pd.DataFrame()
        os.makedirs(DATA_DIR, exist_ok=True)
        fetched.to_csv(path)
        _DAILY_HISTORY_CACHE[instrument] = fetched.copy()
        return fetched
    except Exception:
        return pd.DataFrame()


def _to_iso_key(ts) -> str | None:
    try:
        value = pd.to_datetime(ts, utc=True, errors="coerce")
    except Exception:
        value = pd.NaT
    if pd.isna(value):
        return None
    return value.isoformat()


def _write_json_if_changed(path: str, payload: dict) -> bool:
    try:
        incoming = json.dumps(payload, sort_keys=True)
    except Exception:
        incoming = None
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                existing_obj = json.load(f)
            existing = json.dumps(existing_obj, sort_keys=True)
            if incoming is not None and existing == incoming:
                return False
        except Exception:
            pass
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
        return True
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _load_prices(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
        df.columns = [str(c).lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _load_resolved_history(instrument_key: str) -> pd.DataFrame:
    frames = []
    for path in _resolved_paths(instrument_key):
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if df.empty:
            continue
        if "resolved_ts" in df.columns:
            df["resolved_ts"] = pd.to_datetime(df["resolved_ts"], utc=True, errors="coerce")
        if "pred_price" in df.columns:
            df["pred_price"] = pd.to_numeric(df["pred_price"], errors="coerce")
        if "actual_price" in df.columns:
            df["actual_price"] = pd.to_numeric(df["actual_price"], errors="coerce")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out = out[out.get("model", "").isin(_MODEL_KEYS)].copy()
    if "target_key" in out.columns:
        out["target_key"] = out["target_key"].astype(str)
    return out.dropna(subset=["pred_price", "actual_price"])


def _load_structural_snapshots(instrument_key: str) -> pd.DataFrame:
    instrument = str(instrument_key or "").upper()
    cached = _SNAPSHOT_CACHE.get(instrument)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    path = _structural_snapshot_path(instrument)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "target_key" in df.columns:
        df["target_key"] = df["target_key"].astype(str)
    if "date_key" in df.columns:
        df["date_key"] = df["date_key"].astype(str)
    for col in df.columns:
        if col in ("snapshot_type", "target_key", "date_key"):
            continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
        except Exception:
            pass
    _SNAPSHOT_CACHE[instrument] = df.copy()
    return df


def _save_structural_snapshots(instrument_key: str, df: pd.DataFrame):
    instrument = str(instrument_key or "").upper()
    path = _structural_snapshot_path(instrument)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out.to_csv(path, index=False)
    _SNAPSHOT_CACHE[instrument] = out.copy()


def _load_structural_outcomes(instrument_key: str) -> pd.DataFrame:
    instrument = str(instrument_key or "").upper()
    cached = _OUTCOME_CACHE.get(instrument)
    if isinstance(cached, pd.DataFrame):
        return cached.copy()
    path = _structural_outcome_path(instrument)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    for col in ("snapshot_type", "candidate_key", "target_key", "bucket_key", "variant"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    for col in df.columns:
        if col in ("snapshot_type", "candidate_key", "target_key", "bucket_key", "variant"):
            continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().any():
                df[col] = converted
        except Exception:
            pass
    _OUTCOME_CACHE[instrument] = df.copy()
    return df


def _save_structural_outcomes(instrument_key: str, df: pd.DataFrame):
    instrument = str(instrument_key or "").upper()
    path = _structural_outcome_path(instrument)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    out.to_csv(path, index=False)
    _OUTCOME_CACHE[instrument] = out.copy()


def _date_key_from_target_key(target_key) -> str | None:
    target = str(target_key or "").strip()
    if len(target) >= 10 and target[4:5] == "-" and target[7:8] == "-" and "T" not in target and " " not in target:
        return target[:10]
    parsed = pd.to_datetime(target_key, utc=True, errors="coerce")
    if pd.notna(parsed):
        try:
            return str(parsed.tz_convert("America/New_York").date())
        except Exception:
            return str(parsed.date())
    if len(target) >= 10 and target[4:5] == "-" and target[7:8] == "-":
        return target[:10]
    return None


def _local_wday(index_like) -> np.ndarray:
    try:
        idx = pd.to_datetime(index_like, utc=True, errors="coerce").tz_convert("America/New_York")
    except Exception:
        idx = pd.to_datetime(index_like, errors="coerce")
    return np.asarray(idx.weekday, dtype=float)


def _window_objective(metrics: dict | None, horizon: str, instrument_key: str) -> float | None:
    if not isinstance(metrics, dict):
        return None
    ratio = _safe_float(metrics.get("window_ratio_mean"))
    mae = _safe_float(metrics.get("mae_dollars"))
    hit = _safe_float(metrics.get("window_hit_rate"))
    if ratio is None or mae is None or hit is None:
        return None
    hit_req = float(min_window_hit_rate("daily" if "daily" in horizon else horizon, instrument_key=instrument_key))
    return float(ratio + (1.25 * max(0.0, hit_req - hit)) + (0.05 * mae))


def _search_price_blend(base_prices, direct_prices, actual_prices, horizon: str, instrument_key: str) -> dict:
    try:
        base = np.asarray(base_prices, dtype=float)
        direct = np.asarray(direct_prices, dtype=float)
        actual = np.asarray(actual_prices, dtype=float)
    except Exception:
        return {"enabled": False, "weight": 0.0, "selected_metrics": {}, "baseline_metrics": {}, "objective_improvement": None, "n": 0}
    valid = np.isfinite(base) & np.isfinite(direct) & np.isfinite(actual) & (actual > 0.0)
    if int(np.sum(valid)) < 4:
        return {"enabled": False, "weight": 0.0, "selected_metrics": {}, "baseline_metrics": {}, "objective_improvement": None, "n": int(np.sum(valid))}
    base = base[valid]
    direct = direct[valid]
    actual = actual[valid]
    base_metrics = summarize_price_window_errors(np.abs(base - actual), actual_prices=actual, horizon=horizon, instrument_key=instrument_key)
    base_obj = _window_objective(base_metrics, horizon=horizon, instrument_key=instrument_key)
    best_weight = 0.0
    best_metrics = base_metrics
    best_obj = base_obj
    for weight in np.linspace(0.0, 1.0, 21):
        pred = ((1.0 - float(weight)) * base) + (float(weight) * direct)
        metrics = summarize_price_window_errors(np.abs(pred - actual), actual_prices=actual, horizon=horizon, instrument_key=instrument_key)
        obj = _window_objective(metrics, horizon=horizon, instrument_key=instrument_key)
        if obj is None:
            continue
        if best_obj is None or float(obj) + 1e-9 < float(best_obj):
            best_weight = float(weight)
            best_metrics = metrics
            best_obj = obj
    improvement = None if base_obj is None or best_obj is None else float(base_obj - best_obj)
    return {
        "enabled": bool(improvement is not None and improvement >= 0.02 and best_weight > 0.0),
        "weight": float(best_weight),
        "selected_metrics": best_metrics,
        "baseline_metrics": base_metrics,
        "objective_improvement": improvement,
        "n": int(len(actual)),
    }


def _daily_long_feature_frame(daily_df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    df = _normalize_daily_history(daily_df)
    if df.empty:
        return pd.DataFrame()
    out = df.copy()
    try:
        idx_local = pd.to_datetime(out.index, utc=True, errors="coerce").tz_convert("America/New_York")
    except Exception:
        idx_local = pd.to_datetime(out.index, errors="coerce")
    out["date_key"] = pd.Index(idx_local.date).astype(str)
    out["wday"] = np.asarray(idx_local.weekday, dtype=float)
    out["ret_1"] = out["close"].pct_change()
    out["ret_5"] = out["close"].pct_change(5)
    out["vol_5"] = out["ret_1"].rolling(5, min_periods=3).std()
    out["vol_20"] = out["ret_1"].rolling(20, min_periods=10).std()
    out["range_1"] = (out["high"] / out["low"]) - 1.0
    out["body_1"] = (out["close"] / out["open"]) - 1.0
    out["gap_1"] = (out["open"] / out["close"].shift(1)) - 1.0
    out["atr_5"] = ((out["high"] - out["low"]) / out["close"]).rolling(5, min_periods=3).mean()
    out["ema_5"] = out["close"].ewm(span=5, adjust=False).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()
    out["ema_spread"] = (out["ema_5"] / out["ema_20"]) - 1.0
    out["close_loc"] = ((out["close"] - out["low"]) / (out["high"] - out["low"]).replace(0.0, np.nan)).clip(0.0, 1.0)
    if add_context_features is not None:
        out = add_context_features(out, instrument_key=instrument_key, shift_days=0)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _build_structural_daily_pair_dataset(daily_df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    feat = _daily_long_feature_frame(daily_df, instrument_key=instrument_key)
    if feat.empty:
        return pd.DataFrame()
    out = feat.copy()
    out["target_high_ret"] = (out["high"].shift(-1) / out["close"]) - 1.0
    out["target_low_ret"] = (out["low"].shift(-1) / out["close"]) - 1.0
    out["target_center_ret"] = 0.5 * (out["target_high_ret"] + out["target_low_ret"])
    out["target_range_ret"] = (out["target_high_ret"] - out["target_low_ret"]).clip(lower=1e-4)
    return out.dropna(subset=["target_center_ret", "target_range_ret"] + list(_STRUCTURAL_DAILY_CENTER_FEATURES) + list(_STRUCTURAL_DAILY_RANGE_FEATURES))


def _build_structural_weekend_dataset(daily_df: pd.DataFrame, instrument_key: str = None) -> pd.DataFrame:
    feat = _daily_long_feature_frame(daily_df, instrument_key=instrument_key)
    if feat.empty:
        return pd.DataFrame()
    out = feat.copy()
    try:
        idx_local = pd.to_datetime(out.index, utc=True, errors="coerce").tz_convert("America/New_York")
    except Exception:
        idx_local = pd.to_datetime(out.index, errors="coerce")
    iso = idx_local.isocalendar()
    out["iso_year"] = iso.year.astype(int)
    out["iso_week"] = iso.week.astype(int)
    week_keys = ["iso_year", "iso_week"]
    out["week_close"] = out.groupby(week_keys)["close"].transform("last")
    out["week_open"] = out.groupby(week_keys)["open"].transform("first")
    out["week_high_sofar"] = out.groupby(week_keys)["high"].cummax()
    out["week_low_sofar"] = out.groupby(week_keys)["low"].cummin()
    out["days_to_week_end"] = pd.Series(4.0 - np.asarray(idx_local.weekday, dtype=float), index=out.index).clip(lower=0.0)
    out["week_ret_sofar"] = (out["close"] / out["week_open"]) - 1.0
    out["week_range_sofar"] = (out["week_high_sofar"] / out["week_low_sofar"]) - 1.0
    out["week_close_loc"] = ((out["close"] - out["week_low_sofar"]) / (out["week_high_sofar"] - out["week_low_sofar"]).replace(0.0, np.nan)).clip(0.0, 1.0)
    week_stats = out.groupby(week_keys).agg(
        week_close=("close", "last"),
        week_open=("open", "first"),
        week_high=("high", "max"),
        week_low=("low", "min"),
    )
    week_stats["week_ret"] = (week_stats["week_close"] / week_stats["week_open"]) - 1.0
    week_stats["week_range"] = (week_stats["week_high"] / week_stats["week_low"]) - 1.0
    for col, src in (("prev_week_ret", "week_ret"), ("prev_week_range", "week_range")):
        shifted = week_stats[src].shift(1)
        out[col] = out.set_index(week_keys).index.map(shifted)
    last_key = (int(out["iso_year"].iloc[-1]), int(out["iso_week"].iloc[-1]))
    out = out[[(y, w) != last_key for y, w in zip(out["iso_year"], out["iso_week"])]].copy()
    out["target_weekend_ret"] = (out["week_close"] / out["close"]) - 1.0
    keep_cols = ["target_weekend_ret"] + list(_STRUCTURAL_WEEKEND_FEATURES)
    return out.dropna(subset=keep_cols)


def _bucket_edges(values, bucket_count: int) -> list[float]:
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return []
    arr = arr[np.isfinite(arr)]
    if arr.size < max(8, int(bucket_count) * 4):
        return []
    qs = np.linspace(0.0, 1.0, int(bucket_count) + 1)[1:-1]
    try:
        edges = np.quantile(arr, qs)
    except Exception:
        return []
    edges = [float(x) for x in np.ravel(edges) if np.isfinite(x)]
    uniq = []
    for edge in sorted(edges):
        if not uniq or abs(edge - uniq[-1]) > 1e-9:
            uniq.append(edge)
    return uniq


def _bucket_label(value, edges: list[float]) -> str:
    try:
        val = float(value)
    except Exception:
        val = np.nan
    if not np.isfinite(val):
        return "bucket_0"
    for idx, edge in enumerate(edges):
        if val <= float(edge):
            return f"bucket_{idx}"
    return f"bucket_{len(edges)}"


def _grid_bucket_label(vol_value, vol_edges: list[float], close_loc_value, close_loc_edges: list[float]) -> str:
    return f"v{_bucket_label(vol_value, vol_edges)}|c{_bucket_label(close_loc_value, close_loc_edges)}"


def _week_bucket_label(wday_value, coarse: bool = False) -> str:
    try:
        wday = int(float(wday_value))
    except Exception:
        wday = -1
    if not coarse:
        return str(wday)
    if wday in (0, 1):
        return "mon_tue"
    if wday == 2:
        return "wed"
    return "thu_fri"


def _attach_snapshot_news(frame: pd.DataFrame, instrument_key: str) -> tuple[pd.DataFrame, dict]:
    out = frame.copy()
    for col in _LONG_HORIZON_NEWS_FEATURES:
        if col not in out.columns:
            out[col] = 0.0
    coverage = {"sufficient_long_horizon": False}
    if load_news_features_for_index is None or out.empty or "target_key" not in out.columns:
        return out, coverage
    try:
        target_idx = pd.DatetimeIndex(pd.to_datetime(out["target_key"], utc=True, errors="coerce"))
    except Exception:
        target_idx = pd.DatetimeIndex([])
    if len(target_idx) == 0:
        return out, coverage
    try:
        news_feats = load_news_features_for_index(str(instrument_key or "").upper(), target_idx, tz="America/New_York")
    except Exception:
        return out, coverage
    if news_feats is not None and not news_feats.empty:
        aligned = news_feats.reset_index(drop=True)
        for col in _LONG_HORIZON_NEWS_FEATURES:
            if col in aligned.columns:
                out[col] = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if news_archive_coverage is not None:
        try:
            coverage = news_archive_coverage(str(instrument_key or "").upper(), trading_dates=target_idx, tz="America/New_York")
        except Exception:
            coverage = {"sufficient_long_horizon": False}
    if not bool(coverage.get("sufficient_long_horizon", False)):
        for col in _LONG_HORIZON_NEWS_FEATURES:
            out[col] = 0.0
    return out, coverage


def _build_weekly_session_close_dataset(prices: pd.DataFrame, instrument_key: str,
                                        include_incomplete_last_week: bool = False) -> pd.DataFrame:
    feat = _build_base_features(prices, instrument_key=instrument_key)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    if feat.empty:
        return pd.DataFrame()
    if "session_date" not in feat.columns:
        return pd.DataFrame()
    session_groups = feat.groupby("session_date", sort=True)
    if len(session_groups) < 2:
        return pd.DataFrame()
    session_last = session_groups.tail(1).copy()
    if session_last.empty:
        return pd.DataFrame()
    session_stats = session_groups.agg(
        session_open=("open", "first"),
        session_close=("close", "last"),
        session_high=("high", "max"),
        session_low=("low", "min"),
        session_volume=("volume", "sum"),
    )
    session_last = session_last.join(session_stats, on="session_date")
    try:
        idx_local = pd.to_datetime(session_last.index, utc=True, errors="coerce").tz_convert("America/New_York")
    except Exception:
        idx_local = pd.to_datetime(session_last.index, errors="coerce")
    iso = idx_local.isocalendar()
    session_last["iso_year"] = iso.year.astype(int)
    session_last["iso_week"] = iso.week.astype(int)
    session_last["wday"] = np.asarray(idx_local.weekday, dtype=float)
    session_last["date_key"] = pd.Index(idx_local.date).astype(str)
    session_last["target_key"] = [_to_iso_key(ts) for ts in session_last.index]
    week_keys = ["iso_year", "iso_week"]
    session_last["week_open"] = session_last.groupby(week_keys)["session_open"].transform("first")
    session_last["week_high_sofar"] = session_last.groupby(week_keys)["session_high"].cummax()
    session_last["week_low_sofar"] = session_last.groupby(week_keys)["session_low"].cummin()
    session_last["days_to_week_end"] = pd.Series(4.0 - session_last["wday"], index=session_last.index).clip(lower=0.0)
    session_last["week_ret_sofar"] = (session_last["session_close"] / session_last["week_open"]) - 1.0
    session_last["week_range_sofar"] = (session_last["week_high_sofar"] / session_last["week_low_sofar"]) - 1.0
    session_last["week_close_loc"] = (
        (session_last["session_close"] - session_last["week_low_sofar"])
        / (session_last["week_high_sofar"] - session_last["week_low_sofar"]).replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    session_last["session_body"] = (session_last["session_close"] / session_last["session_open"]) - 1.0
    session_last["session_range"] = (session_last["session_high"] / session_last["session_low"]) - 1.0
    week_stats = session_last.groupby(week_keys).agg(
        week_open=("session_open", "first"),
        week_close=("session_close", "last"),
        week_high=("session_high", "max"),
        week_low=("session_low", "min"),
    )
    week_stats["week_ret"] = (week_stats["week_close"] / week_stats["week_open"]) - 1.0
    week_stats["week_range"] = (week_stats["week_high"] / week_stats["week_low"]) - 1.0
    shifted_ret = week_stats["week_ret"].shift(1)
    shifted_rng = week_stats["week_range"].shift(1)
    session_last["prev_week_ret"] = session_last.set_index(week_keys).index.map(shifted_ret)
    session_last["prev_week_range"] = session_last.set_index(week_keys).index.map(shifted_rng)
    session_last["week_end_close"] = session_last.set_index(week_keys).index.map(week_stats["week_close"])
    session_last["week_end_high"] = session_last.set_index(week_keys).index.map(week_stats["week_high"])
    session_last["week_end_low"] = session_last.set_index(week_keys).index.map(week_stats["week_low"])
    if not include_incomplete_last_week and len(session_last) > 0:
        last_key = (int(session_last["iso_year"].iloc[-1]), int(session_last["iso_week"].iloc[-1]))
        session_last = session_last[[(y, w) != last_key for y, w in zip(session_last["iso_year"], session_last["iso_week"])]].copy()
    if "week_end_close" not in session_last.columns:
        return pd.DataFrame()
    session_last["target_weekend_ret"] = (session_last["week_end_close"] / session_last["session_close"]) - 1.0
    session_last["target_week_high_ret"] = (session_last["week_end_high"] / session_last["session_close"]) - 1.0
    session_last["target_week_low_ret"] = (session_last["week_end_low"] / session_last["session_close"]) - 1.0
    if add_context_features is not None:
        session_last = add_context_features(session_last, instrument_key=instrument_key, shift_days=0)
    keep_cols = [
        "target_key",
        "date_key",
        "close",
        "wday",
        "days_to_week_end",
        "week_high_sofar",
        "week_low_sofar",
        "week_ret_sofar",
        "week_range_sofar",
        "week_close_loc",
        "session_body",
        "session_range",
        "session_volatility",
        "session_ret_from_open",
        "trend_strength",
        "overnight_gap",
        "ret_6",
        "mom_12",
        "ctx_market_ret_1",
        "ctx_market_ret_5",
        "ctx_market_vol_20",
        "ctx_vol_level_z",
        "prev_week_ret",
        "prev_week_range",
        "target_weekend_ret",
        "target_week_high_ret",
        "target_week_low_ret",
    ]
    keep_cols = [c for c in keep_cols if c in session_last.columns]
    out = session_last[keep_cols].replace([np.inf, -np.inf], np.nan).dropna(subset=["target_key", "close", "target_weekend_ret"])
    return out


def refresh_structural_snapshots(instrument_key: str, prices: pd.DataFrame | None = None,
                                 max_daily_targets: int = 32, max_week_targets: int = 28) -> pd.DataFrame:
    instrument = str(instrument_key or "").upper()
    price_df = prices if isinstance(prices, pd.DataFrame) else _load_prices(os.path.join(DATA_DIR, f"prices_{instrument}.csv"))
    if price_df is None or price_df.empty:
        return pd.DataFrame()
    daily_long = _load_long_daily_history(instrument)
    daily_struct = _build_structural_daily_pair_dataset(daily_long, instrument_key=instrument)
    if not daily_struct.empty:
        daily_struct = daily_struct[
            [
                "date_key",
                "close",
                "target_high_ret",
                "target_low_ret",
                "target_center_ret",
                "target_range_ret",
                "ret_1",
                "ret_5",
                "vol_5",
                "vol_20",
                "range_1",
                "body_1",
                "gap_1",
                "atr_5",
                "ema_spread",
                "close_loc",
                "ctx_market_ret_1",
                "ctx_market_ret_5",
                "ctx_market_vol_20",
                "ctx_vol_level_z",
            ]
        ].drop_duplicates(subset=["date_key"], keep="last")
    daily_targets = _daily_training_frame(price_df, instrument).tail(max(2, int(max_daily_targets))).copy()
    daily_rows = []
    if not daily_targets.empty:
        daily_targets["date_key"] = daily_targets["target_key"].map(_date_key_from_target_key)
        if not daily_struct.empty:
            daily_targets = daily_targets.merge(daily_struct, on="date_key", how="left", suffixes=("", "_daily"))
        daily_targets["snapshot_type"] = "daily_pair"
        daily_targets, coverage = _attach_snapshot_news(daily_targets, instrument)
        daily_targets["news_long_horizon_ok"] = int(1 if coverage.get("sufficient_long_horizon") else 0)
        daily_rows.append(daily_targets)
    weekly_targets = _build_weekly_session_close_dataset(price_df, instrument, include_incomplete_last_week=False).tail(max(2, int(max_week_targets))).copy()
    if not weekly_targets.empty:
        if not daily_struct.empty:
            merge_cols = [
                "date_key",
                "vol_20",
                "atr_5",
                "ema_spread",
                "close_loc",
                "gap_1",
                "ctx_market_ret_1",
                "ctx_market_ret_5",
                "ctx_market_vol_20",
                "ctx_vol_level_z",
            ]
            weekly_targets = weekly_targets.merge(daily_struct[merge_cols], on="date_key", how="left")
        weekly_targets["snapshot_type"] = "weekend_session"
        weekly_targets, coverage = _attach_snapshot_news(weekly_targets, instrument)
        weekly_targets["news_long_horizon_ok"] = int(1 if coverage.get("sufficient_long_horizon") else 0)
        daily_rows.append(weekly_targets)
    out = pd.concat(daily_rows, ignore_index=True, sort=False) if daily_rows else pd.DataFrame()
    if not out.empty:
        out = out.drop_duplicates(subset=["snapshot_type", "target_key"], keep="last").sort_values(["snapshot_type", "target_key"])
    _save_structural_snapshots(instrument, out)
    try:
        refresh_structural_candidate_outcomes(
            instrument,
            prices=price_df,
            snapshots=out,
        )
    except Exception:
        pass
    return out


def _daily_outcome_specs(dataset: pd.DataFrame) -> list[dict]:
    specs = [
        {
            "candidate_key": "daily_long_prior",
            "variant": "global",
            "center_features": tuple(_STRUCTURAL_DAILY_CENTER_FEATURES),
            "range_features": tuple(_STRUCTURAL_DAILY_RANGE_FEATURES),
            "bucket_mode": None,
        }
    ]
    try:
        vol_edges = _bucket_edges(pd.to_numeric(dataset["vol_20"], errors="coerce").to_numpy(dtype=float), 2)
        close_edges = _bucket_edges(pd.to_numeric(dataset["close_loc"], errors="coerce").to_numpy(dtype=float), 2)
    except Exception:
        vol_edges = []
        close_edges = []
    if vol_edges and close_edges:
        labels = [
            _grid_bucket_label(v, vol_edges, c, close_edges)
            for v, c in zip(
                pd.to_numeric(dataset["vol_20"], errors="coerce").fillna(0.0),
                pd.to_numeric(dataset["close_loc"], errors="coerce").fillna(0.0),
            )
        ]
        counts = pd.Series(labels).value_counts()
        if len(counts) == 4 and int(counts.min()) >= 4:
            specs.append(
                {
                    "candidate_key": "daily_long_regime_2x2",
                    "variant": "regime_grid_2x2",
                    "center_features": tuple(_STRUCTURAL_DAILY_CENTER_FEATURES),
                    "range_features": tuple(_STRUCTURAL_DAILY_RANGE_FEATURES),
                    "bucket_mode": {"kind": "vol_close_2x2", "vol_edges": list(vol_edges), "close_edges": list(close_edges)},
                }
            )
    if "news_long_horizon_ok" in dataset.columns and int(pd.to_numeric(dataset["news_long_horizon_ok"], errors="coerce").fillna(0).max()) > 0:
        specs.append(
            {
                "candidate_key": "daily_long_news_prior",
                "variant": "global_news",
                "center_features": tuple(list(_STRUCTURAL_DAILY_CENTER_FEATURES) + ["news_sent_ewm_d7", "news_intensity_ewm_d7"]),
                "range_features": tuple(list(_STRUCTURAL_DAILY_RANGE_FEATURES) + ["news_intensity_ewm_d7"]),
                "bucket_mode": None,
            }
        )
    return specs


def _weekend_outcome_specs(dataset: pd.DataFrame) -> list[dict]:
    specs = [
        {
            "candidate_key": "weekend_daily_long",
            "variant": "daily_long",
            "features": tuple(_STRUCTURAL_WEEKEND_FEATURES),
            "bucket_mode": None,
        }
    ]
    raw_counts = pd.Series([_week_bucket_label(v, coarse=False) for v in pd.to_numeric(dataset.get("wday"), errors="coerce").fillna(-1)]).value_counts()
    raw_min = int(raw_counts.min()) if not raw_counts.empty else 0
    coarse_counts = pd.Series([_week_bucket_label(v, coarse=True) for v in pd.to_numeric(dataset.get("wday"), errors="coerce").fillna(-1)]).value_counts()
    coarse_min = int(coarse_counts.min()) if not coarse_counts.empty else 0
    if len(raw_counts) >= 4 and raw_min >= 8:
        specs.append(
            {
                "candidate_key": "weekend_session_wday",
                "variant": "session_close_by_wday",
                "features": tuple(_STRUCTURAL_WEEKEND_SESSION_FEATURES),
                "bucket_mode": "wday",
            }
        )
    elif len(coarse_counts) >= 3 and coarse_min >= 4:
        specs.append(
            {
                "candidate_key": "weekend_session_coarse3",
                "variant": "session_close_coarse3",
                "features": tuple(_STRUCTURAL_WEEKEND_SESSION_FEATURES),
                "bucket_mode": "coarse3",
            }
        )
    if "news_long_horizon_ok" in dataset.columns and int(pd.to_numeric(dataset["news_long_horizon_ok"], errors="coerce").fillna(0).max()) > 0:
        specs.append(
            {
                "candidate_key": "weekend_session_coarse3_news",
                "variant": "session_close_coarse3_news",
                "features": tuple(list(_STRUCTURAL_WEEKEND_SESSION_FEATURES) + ["news_sent_ewm_d7", "news_intensity_ewm_d7"]),
                "bucket_mode": "coarse3",
            }
        )
    return specs


def _run_daily_candidate_outcomes(dataset: pd.DataFrame, spec: dict) -> list[dict]:
    frame = dataset.copy()
    needed = ["target_key", "close", "target_high_ret", "target_low_ret"] + list(spec.get("center_features", ())) + list(spec.get("range_features", ()))
    if any(col not in frame.columns for col in needed):
        return []
    frame["target_center_ret"] = 0.5 * (
        pd.to_numeric(frame["target_high_ret"], errors="coerce") + pd.to_numeric(frame["target_low_ret"], errors="coerce")
    )
    frame["target_range_ret"] = (
        pd.to_numeric(frame["target_high_ret"], errors="coerce") - pd.to_numeric(frame["target_low_ret"], errors="coerce")
    ).clip(lower=1e-4)
    if spec.get("bucket_mode"):
        mode = spec["bucket_mode"]
        frame["bucket_key"] = [
            _grid_bucket_label(v, mode["vol_edges"], c, mode["close_edges"])
            for v, c in zip(
                pd.to_numeric(frame["vol_20"], errors="coerce").fillna(0.0),
                pd.to_numeric(frame["close_loc"], errors="coerce").fillna(0.0),
            )
        ]
    else:
        frame["bucket_key"] = "global"
    frame = _clean_numeric_frame(frame).sort_values("target_key")
    frame = frame.dropna(subset=["target_key", "close", "target_center_ret", "target_range_ret"] + list(spec.get("center_features", ())) + list(spec.get("range_features", ())))
    rows = []
    min_train = max(8, len(spec.get("center_features", ())) + 3, len(spec.get("range_features", ())) + 3)
    for i in range(len(frame)):
        tr_all = frame.iloc[:i].copy()
        if len(tr_all) < min_train:
            continue
        te = frame.iloc[i]
        if spec.get("bucket_mode"):
            tr = tr_all[tr_all["bucket_key"] == te["bucket_key"]].copy()
            if len(tr) < min_train:
                continue
        else:
            tr = tr_all
        center_model = _fit_linear_map(tr, tr["target_center_ret"].to_numpy(dtype=float), tuple(spec["center_features"]), ridge_alpha=3.0)
        range_model = _fit_linear_map(tr, np.log(tr["target_range_ret"].to_numpy(dtype=float)), tuple(spec["range_features"]), ridge_alpha=3.0)
        if center_model is None or range_model is None:
            continue
        ctx = te.to_dict()
        center_val = _predict_linear_map(center_model, ctx)
        log_range_val = _predict_linear_map(range_model, ctx)
        close_val = _safe_float(te.get("close"))
        if center_val is None or log_range_val is None or close_val is None or close_val <= 0.0:
            continue
        rng = float(max(1e-4, np.exp(log_range_val)))
        rows.append(
            {
                "snapshot_type": "daily_pair",
                "candidate_key": str(spec["candidate_key"]),
                "variant": str(spec.get("variant") or "global"),
                "target_key": str(te["target_key"]),
                "bucket_key": str(te["bucket_key"]),
                "pred_high_price": float(close_val * (1.0 + center_val + (0.5 * rng))),
                "pred_low_price": float(close_val * (1.0 + center_val - (0.5 * rng))),
                "train_n": int(len(tr)),
            }
        )
    return rows


def _run_weekend_candidate_outcomes(dataset: pd.DataFrame, spec: dict) -> list[dict]:
    frame = dataset.copy()
    needed = ["target_key", "close", "target_weekend_ret"] + list(spec.get("features", ()))
    if any(col not in frame.columns for col in needed):
        return []
    mode = spec.get("bucket_mode")
    if mode == "wday":
        frame["bucket_key"] = [_week_bucket_label(v, coarse=False) for v in pd.to_numeric(frame.get("wday"), errors="coerce").fillna(-1)]
    elif mode == "coarse3":
        frame["bucket_key"] = [_week_bucket_label(v, coarse=True) for v in pd.to_numeric(frame.get("wday"), errors="coerce").fillna(-1)]
    else:
        frame["bucket_key"] = "global"
    frame = _clean_numeric_frame(frame).sort_values("target_key")
    frame = frame.dropna(subset=["target_key", "close", "target_weekend_ret"] + list(spec.get("features", ())))
    rows = []
    min_train = max(5, len(spec.get("features", ())) + 2)
    has_range_targets = all(col in frame.columns for col in ("target_week_high_ret", "target_week_low_ret", "week_high_sofar", "week_low_sofar"))
    for i in range(len(frame)):
        tr_all = frame.iloc[:i].copy()
        if len(tr_all) < min_train:
            continue
        te = frame.iloc[i]
        if mode in ("wday", "coarse3"):
            tr = tr_all[tr_all["bucket_key"] == te["bucket_key"]].copy()
            if len(tr) < min_train:
                continue
        else:
            tr = tr_all
        close_val = _safe_float(te.get("close"))
        if close_val is None or close_val <= 0.0:
            continue
        pred_ret = None
        pred_high_price = None
        pred_low_price = None
        if float(te.get("days_to_week_end", 1.0) or 0.0) <= 0.0:
            pred_ret = 0.0
            pred_high_price = _safe_float(te.get("week_high_sofar")) or close_val
            pred_low_price = _safe_float(te.get("week_low_sofar")) or close_val
        else:
            ret_model = _fit_linear_map(tr, tr["target_weekend_ret"].to_numpy(dtype=float), tuple(spec["features"]), ridge_alpha=4.0 if mode else 3.0)
            if ret_model is None:
                continue
            pred_ret = _predict_linear_map(ret_model, te.to_dict())
            if pred_ret is None:
                continue
            if has_range_targets:
                high_model = _fit_linear_map(tr, tr["target_week_high_ret"].to_numpy(dtype=float), tuple(spec["features"]), ridge_alpha=4.0 if mode else 3.0)
                low_model = _fit_linear_map(tr, tr["target_week_low_ret"].to_numpy(dtype=float), tuple(spec["features"]), ridge_alpha=4.0 if mode else 3.0)
                pred_high_ret = _predict_linear_map(high_model, te.to_dict()) if high_model is not None else None
                pred_low_ret = _predict_linear_map(low_model, te.to_dict()) if low_model is not None else None
                high_sofar = _safe_float(te.get("week_high_sofar"))
                low_sofar = _safe_float(te.get("week_low_sofar"))
                if pred_high_ret is not None:
                    pred_high_price = max(high_sofar or close_val, float(close_val * (1.0 + max(0.0, pred_high_ret))))
                if pred_low_ret is not None:
                    pred_low_price = min(low_sofar or close_val, float(close_val * (1.0 + min(0.0, pred_low_ret))))
        rows.append(
            {
                "snapshot_type": "weekend_session",
                "candidate_key": str(spec["candidate_key"]),
                "variant": str(spec.get("variant") or "global"),
                "target_key": str(te["target_key"]),
                "bucket_key": str(te["bucket_key"]),
                "pred_price": float(close_val * (1.0 + float(pred_ret))),
                "pred_high_price": float(pred_high_price) if pred_high_price is not None else np.nan,
                "pred_low_price": float(pred_low_price) if pred_low_price is not None else np.nan,
                "train_n": int(len(tr)),
            }
        )
    return rows


def refresh_structural_candidate_outcomes(instrument_key: str, prices: pd.DataFrame | None = None,
                                          snapshots: pd.DataFrame | None = None) -> pd.DataFrame:
    instrument = str(instrument_key or "").upper()
    frame = snapshots.copy() if isinstance(snapshots, pd.DataFrame) else _load_structural_snapshots(instrument)
    if frame.empty:
        return pd.DataFrame()
    rows = []
    daily_frame = frame[frame.get("snapshot_type", "") == "daily_pair"].copy()
    if not daily_frame.empty:
        for spec in _daily_outcome_specs(daily_frame):
            rows.extend(_run_daily_candidate_outcomes(daily_frame, spec))
    week_frame = frame[frame.get("snapshot_type", "") == "weekend_session"].copy()
    if not week_frame.empty:
        for spec in _weekend_outcome_specs(week_frame):
            rows.extend(_run_weekend_candidate_outcomes(week_frame, spec))
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.drop_duplicates(subset=["snapshot_type", "candidate_key", "target_key"], keep="last").sort_values(["snapshot_type", "candidate_key", "target_key"])
    _save_structural_outcomes(instrument, out)
    return out


def _attach_current_news_context(index_utc, ctx: dict, instrument_key: str) -> dict:
    out = dict(ctx or {})
    for col in _LONG_HORIZON_NEWS_FEATURES:
        out.setdefault(col, 0.0)
    if load_news_features_for_index is None:
        return out
    if news_archive_coverage is not None:
        try:
            coverage = news_archive_coverage(str(instrument_key or "").upper(), trading_dates=index_utc, tz="America/New_York")
            if not bool(coverage.get("sufficient_long_horizon", False)):
                return out
        except Exception:
            return out
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(index_utc, utc=True, errors="coerce"))
    except Exception:
        return out
    if len(idx) == 0:
        return out
    try:
        feats = load_news_features_for_index(str(instrument_key or "").upper(), idx, tz="America/New_York")
    except Exception:
        return out
    if feats is None or feats.empty:
        return out
    row = feats.iloc[-1].to_dict()
    for col in _LONG_HORIZON_NEWS_FEATURES:
        out[col] = _safe_float(row.get(col)) or 0.0
    return out


def _current_weekly_session_row(prices: pd.DataFrame, instrument_key: str) -> pd.Series | None:
    dataset = _build_weekly_session_close_dataset(prices, instrument_key, include_incomplete_last_week=True)
    if dataset.empty:
        return None
    return dataset.iloc[-1]


def _latest_structural_row(daily_df: pd.DataFrame, prices: pd.DataFrame, builder) -> pd.Series | None:
    if daily_df is None or daily_df.empty or prices is None or prices.empty:
        return None
    frame = builder(daily_df)
    if frame.empty:
        return None
    try:
        last_ts = pd.to_datetime(prices.index.max(), utc=True, errors="coerce")
        target_date = str(last_ts.tz_convert("America/New_York").date()) if pd.notna(last_ts) else None
    except Exception:
        target_date = None
    if target_date:
        candidates = frame[frame["date_key"] <= target_date]
        if not candidates.empty:
            return candidates.iloc[-1]
    return frame.iloc[-1] if not frame.empty else None


def _knn_predict(x_train: np.ndarray, y_train: np.ndarray, x_now: np.ndarray, k: int) -> float | None:
    if x_train.size == 0 or y_train.size == 0 or x_now.size == 0:
        return None
    scale = np.nanstd(x_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    try:
        dist = np.sqrt(np.sum(np.square((x_train - x_now) / scale), axis=1))
    except Exception:
        return None
    if dist.size == 0:
        return None
    take = np.argsort(dist)[: max(1, min(int(k), dist.size))]
    dd = np.maximum(dist[take], 1e-6)
    weights = 1.0 / dd
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return None
    return float(np.sum(weights * y_train[take]) / denom)


def _daily_training_frame(prices: pd.DataFrame, instrument_key: str) -> pd.DataFrame:
    ds = build_next_session_dataset(prices, instrument_key=instrument_key).sort_index()
    if ds.empty:
        return pd.DataFrame()
    out = ds.copy()
    out["wday"] = _local_wday(out.index)
    out["target_key"] = [_to_iso_key(ts) for ts in out.index]
    return out


def _week_training_frame(prices: pd.DataFrame, instrument_key: str) -> pd.DataFrame:
    ds = build_weekend_session_dataset(prices, instrument_key=instrument_key).sort_index()
    if ds.empty:
        return pd.DataFrame()
    out = ds.copy()
    out["wday"] = _local_wday(out.index)
    if "days_to_week_end" not in out.columns:
        out["days_to_week_end"] = (4.0 - out["wday"]).clip(lower=0.0)
    out["target_key"] = [_to_iso_key(ts) for ts in out.index]
    return out


def _daily_current_row(prices: pd.DataFrame, instrument_key: str) -> pd.DataFrame:
    feat = _build_base_features(prices, instrument_key=instrument_key)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    if feat.empty or "session_date" not in feat.columns:
        return pd.DataFrame()
    current = feat.groupby("session_date", sort=True).tail(1).tail(1).copy()
    if current.empty:
        return pd.DataFrame()
    if add_context_features is not None:
        current = add_context_features(current, instrument_key=instrument_key, shift_days=0)
    current["wday"] = _local_wday(current.index)
    return current


def _week_current_row(prices: pd.DataFrame, instrument_key: str) -> pd.DataFrame:
    feat = _build_base_features(prices, instrument_key=instrument_key)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna(subset=["close"])
    if feat.empty:
        return pd.DataFrame()
    try:
        idx_local = pd.to_datetime(feat.index, utc=True, errors="coerce").tz_convert("America/New_York")
    except Exception:
        idx_local = pd.to_datetime(feat.index, errors="coerce")
    feat = feat.copy()
    feat["iso_year"] = idx_local.isocalendar().year.astype(int)
    feat["iso_week"] = idx_local.isocalendar().week.astype(int)
    feat["wday"] = np.asarray(idx_local.weekday, dtype=float)
    feat["days_to_week_end"] = (4.0 - feat["wday"]).clip(lower=0.0)
    current = feat.groupby(["iso_year", "iso_week"], sort=True).tail(1).tail(1).copy()
    if add_context_features is not None and not current.empty:
        current = add_context_features(current, instrument_key=instrument_key, shift_days=0)
    return current if not current.empty else pd.DataFrame()


def _fit_linear_map(frame: pd.DataFrame, target, features: tuple[str, ...], ridge_alpha: float = 1.0) -> dict | None:
    if frame is None or frame.empty or not features:
        return None
    cols = [c for c in features if c in frame.columns]
    if len(cols) != len(features):
        return None
    try:
        X = frame[list(cols)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        y = np.asarray(target, dtype=float)
    except Exception:
        return None
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    if int(np.sum(mask)) < max(6, len(cols) + 2):
        return None
    X = X[mask]
    y = y[mask]
    x_mean = np.mean(X, axis=0)
    x_std = np.std(X, axis=0)
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-9), x_std, 1.0)
    Xs = (X - x_mean) / x_std
    y_mean = float(np.mean(y))
    yc = y - y_mean
    eye = np.eye(Xs.shape[1], dtype=float) * float(max(1e-9, ridge_alpha))
    try:
        beta = np.linalg.solve(Xs.T @ Xs + eye, Xs.T @ yc)
    except Exception:
        beta = np.linalg.lstsq(Xs.T @ Xs + eye, Xs.T @ yc, rcond=None)[0]
    return {
        "features": list(cols),
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "beta": beta.tolist(),
        "y_mean": y_mean,
    }


def _predict_linear_map(payload: dict | None, feature_values: dict) -> float | None:
    if not isinstance(payload, dict):
        return None
    features = [str(x) for x in payload.get("features", [])]
    if not features:
        return None
    try:
        x_mean = np.asarray(payload.get("x_mean", []), dtype=float)
        x_std = np.asarray(payload.get("x_std", []), dtype=float)
        beta = np.asarray(payload.get("beta", []), dtype=float)
        y_mean = float(payload.get("y_mean", 0.0))
    except Exception:
        return None
    if x_mean.size != len(features) or x_std.size != len(features) or beta.size != len(features):
        return None
    try:
        x = np.asarray([feature_values.get(name, 0.0) for name in features], dtype=float)
    except Exception:
        return None
    if not np.isfinite(x).all():
        return None
    x_std = np.where(np.isfinite(x_std) & (x_std > 1e-9), x_std, 1.0)
    return float(y_mean + np.dot((x - x_mean) / x_std, beta))


def _daily_candidate_features(candidate_key: str) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    key = str(candidate_key or "")
    if key == "daily_long_news_prior":
        return (
            tuple(list(_STRUCTURAL_DAILY_CENTER_FEATURES) + ["news_sent_ewm_d7", "news_intensity_ewm_d7"]),
            tuple(list(_STRUCTURAL_DAILY_RANGE_FEATURES) + ["news_intensity_ewm_d7"]),
            None,
        )
    if key == "daily_long_regime_2x2":
        return tuple(_STRUCTURAL_DAILY_CENTER_FEATURES), tuple(_STRUCTURAL_DAILY_RANGE_FEATURES), "vol_close_2x2"
    return tuple(_STRUCTURAL_DAILY_CENTER_FEATURES), tuple(_STRUCTURAL_DAILY_RANGE_FEATURES), None


def _weekend_candidate_features(candidate_key: str) -> tuple[tuple[str, ...], str | None]:
    key = str(candidate_key or "")
    if key == "weekend_session_wday":
        return tuple(_STRUCTURAL_WEEKEND_SESSION_FEATURES), "wday"
    if key in ("weekend_session_coarse3", "weekend_session_coarse3_news"):
        feats = list(_STRUCTURAL_WEEKEND_SESSION_FEATURES)
        if key.endswith("_news"):
            feats.extend(["news_sent_ewm_d7", "news_intensity_ewm_d7"])
        return tuple(feats), "coarse3"
    return tuple(_STRUCTURAL_WEEKEND_FEATURES), None


def _evaluate_daily_outcomes(resolved: pd.DataFrame, outcomes: pd.DataFrame, instrument_key: str) -> list[dict]:
    if outcomes is None or outcomes.empty:
        return []
    high_hist = resolved[resolved["horizon"] == "daily_high"].copy()
    low_hist = resolved[resolved["horizon"] == "daily_low"].copy()
    rows = []
    for candidate_key, block in outcomes.groupby("candidate_key"):
        hist_high = {str(k): float(v) for k, v in zip(block["target_key"], pd.to_numeric(block["pred_high_price"], errors="coerce")) if np.isfinite(v)}
        hist_low = {str(k): float(v) for k, v in zip(block["target_key"], pd.to_numeric(block["pred_low_price"], errors="coerce")) if np.isfinite(v)}
        if not hist_high or not hist_low:
            continue
        high_weights = {}
        low_weights = {}
        high_metrics_by_model = {}
        low_metrics_by_model = {}
        for model_key in _MODEL_KEYS:
            h_block = high_hist[high_hist["model"] == model_key].copy()
            if not h_block.empty:
                h_block = h_block[h_block["target_key"].astype(str).isin(hist_high)].copy()
                if not h_block.empty:
                    blend = _search_price_blend(
                        h_block["pred_price"].values,
                        np.asarray([hist_high[str(k)] for k in h_block["target_key"].astype(str)], dtype=float),
                        h_block["actual_price"].values,
                        horizon="daily",
                        instrument_key=instrument_key,
                    )
                    high_weights[model_key] = float(blend.get("weight", 0.0) or 0.0)
                    high_metrics_by_model[model_key] = blend
            l_block = low_hist[low_hist["model"] == model_key].copy()
            if not l_block.empty:
                l_block = l_block[l_block["target_key"].astype(str).isin(hist_low)].copy()
                if not l_block.empty:
                    blend = _search_price_blend(
                        l_block["pred_price"].values,
                        np.asarray([hist_low[str(k)] for k in l_block["target_key"].astype(str)], dtype=float),
                        l_block["actual_price"].values,
                        horizon="daily",
                        instrument_key=instrument_key,
                    )
                    low_weights[model_key] = float(blend.get("weight", 0.0) or 0.0)
                    low_metrics_by_model[model_key] = blend
        high_selected = [v.get("selected_metrics", {}) for v in high_metrics_by_model.values() if isinstance(v, dict) and v.get("selected_metrics")]
        low_selected = [v.get("selected_metrics", {}) for v in low_metrics_by_model.values() if isinstance(v, dict) and v.get("selected_metrics")]
        hi_mae = [_safe_float(v.get("mae_dollars")) for v in high_selected]
        lo_mae = [_safe_float(v.get("mae_dollars")) for v in low_selected]
        hi_ratio = [_safe_float(v.get("window_ratio_mean")) for v in high_selected]
        lo_ratio = [_safe_float(v.get("window_ratio_mean")) for v in low_selected]
        hi_hit = [_safe_float(v.get("window_hit_rate")) for v in high_selected]
        lo_hit = [_safe_float(v.get("window_hit_rate")) for v in low_selected]
        hi_mae = [v for v in hi_mae if v is not None]
        lo_mae = [v for v in lo_mae if v is not None]
        hi_ratio = [v for v in hi_ratio if v is not None]
        lo_ratio = [v for v in lo_ratio if v is not None]
        hi_hit = [v for v in hi_hit if v is not None]
        lo_hit = [v for v in lo_hit if v is not None]
        aggregate = {
            "mae_dollars": float(0.5 * (np.mean(hi_mae) + np.mean(lo_mae))) if hi_mae and lo_mae else None,
            "window_ratio_mean": float(0.5 * (np.mean(hi_ratio) + np.mean(lo_ratio))) if hi_ratio and lo_ratio else None,
            "window_hit_rate": float(0.5 * (np.mean(hi_hit) + np.mean(lo_hit))) if hi_hit and lo_hit else None,
        }
        rows.append(
            {
                "candidate_key": str(candidate_key),
                "aggregate_metrics": aggregate,
                "objective_score": _window_objective(aggregate, horizon="daily", instrument_key=instrument_key),
                "high_blend_weights": high_weights,
                "low_blend_weights": low_weights,
                "high_metrics_by_model": high_metrics_by_model,
                "low_metrics_by_model": low_metrics_by_model,
                "hist_n": int(block["target_key"].nunique()),
            }
        )
    return rows


def _fit_daily_live_models(dataset: pd.DataFrame, candidate_key: str) -> dict:
    center_features, range_features, bucket_mode = _daily_candidate_features(candidate_key)
    needed = ["target_key", "close", "target_high_ret", "target_low_ret"] + list(center_features) + list(range_features)
    frame = dataset.dropna(subset=[c for c in needed if c in dataset.columns]).copy()
    if frame.empty:
        return {}
    frame["target_center_ret"] = 0.5 * (
        pd.to_numeric(frame["target_high_ret"], errors="coerce") + pd.to_numeric(frame["target_low_ret"], errors="coerce")
    )
    frame["target_range_ret"] = (
        pd.to_numeric(frame["target_high_ret"], errors="coerce") - pd.to_numeric(frame["target_low_ret"], errors="coerce")
    ).clip(lower=1e-4)
    center_model = _fit_linear_map(frame, frame["target_center_ret"].to_numpy(dtype=float), center_features, ridge_alpha=3.0)
    range_model = _fit_linear_map(frame, np.log(frame["target_range_ret"].to_numpy(dtype=float)), range_features, ridge_alpha=3.0)
    if center_model is None or range_model is None:
        return {}
    out = {
        "center_model": center_model,
        "range_model": range_model,
        "center_features": list(center_features),
        "range_features": list(range_features),
        "vol_bucket_edges": [],
        "close_loc_bucket_edges": [],
        "bucket_models": {},
        "variant": "global",
    }
    if bucket_mode == "vol_close_2x2":
        vol_edges = _bucket_edges(pd.to_numeric(frame["vol_20"], errors="coerce").to_numpy(dtype=float), 2)
        close_edges = _bucket_edges(pd.to_numeric(frame["close_loc"], errors="coerce").to_numpy(dtype=float), 2)
        if vol_edges and close_edges:
            frame["bucket_key"] = [
                _grid_bucket_label(v, vol_edges, c, close_edges)
                for v, c in zip(
                    pd.to_numeric(frame["vol_20"], errors="coerce").fillna(0.0),
                    pd.to_numeric(frame["close_loc"], errors="coerce").fillna(0.0),
                )
            ]
            counts = frame["bucket_key"].value_counts()
            if len(counts) == 4 and int(counts.min()) >= 4:
                out["variant"] = "regime_grid_2x2"
                out["vol_bucket_edges"] = list(vol_edges)
                out["close_loc_bucket_edges"] = list(close_edges)
                for bucket_key, block in frame.groupby("bucket_key"):
                    if len(block) < 4:
                        continue
                    b_center = _fit_linear_map(block, block["target_center_ret"].to_numpy(dtype=float), center_features, ridge_alpha=3.0)
                    b_range = _fit_linear_map(block, np.log(block["target_range_ret"].to_numpy(dtype=float)), range_features, ridge_alpha=3.0)
                    if b_center is not None and b_range is not None:
                        out["bucket_models"][str(bucket_key)] = {"center_model": b_center, "range_model": b_range, "n": int(len(block))}
    return out


def _evaluate_weekend_outcomes(resolved: pd.DataFrame, outcomes: pd.DataFrame, instrument_key: str) -> list[dict]:
    if outcomes is None or outcomes.empty:
        return []
    week_hist = resolved[resolved["horizon"] == "weekend"].copy()
    rows = []
    for candidate_key, block in outcomes.groupby("candidate_key"):
        hist_prices = {str(k): float(v) for k, v in zip(block["target_key"], pd.to_numeric(block["pred_price"], errors="coerce")) if np.isfinite(v)}
        if not hist_prices:
            continue
        blend_weights = {}
        metrics_by_model = {}
        for model_key in _MODEL_KEYS:
            m_block = week_hist[week_hist["model"] == model_key].copy()
            if m_block.empty:
                continue
            m_block = m_block[m_block["target_key"].astype(str).isin(hist_prices)].copy()
            if m_block.empty:
                continue
            blend = _search_price_blend(
                m_block["pred_price"].values,
                np.asarray([hist_prices[str(k)] for k in m_block["target_key"].astype(str)], dtype=float),
                m_block["actual_price"].values,
                horizon="weekend",
                instrument_key=instrument_key,
            )
            blend_weights[model_key] = float(blend.get("weight", 0.0) or 0.0)
            metrics_by_model[model_key] = blend
        selected = [v.get("selected_metrics", {}) for v in metrics_by_model.values() if isinstance(v, dict) and v.get("selected_metrics")]
        maes = [_safe_float(v.get("mae_dollars")) for v in selected]
        ratios = [_safe_float(v.get("window_ratio_mean")) for v in selected]
        hits = [_safe_float(v.get("window_hit_rate")) for v in selected]
        maes = [v for v in maes if v is not None]
        ratios = [v for v in ratios if v is not None]
        hits = [v for v in hits if v is not None]
        aggregate = {
            "mae_dollars": float(np.mean(maes)) if maes else None,
            "window_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "window_hit_rate": float(np.mean(hits)) if hits else None,
        }
        rows.append(
            {
                "candidate_key": str(candidate_key),
                "aggregate_metrics": aggregate,
                "objective_score": _window_objective(aggregate, horizon="weekend", instrument_key=instrument_key),
                "blend_weights": blend_weights,
                "metrics_by_model": metrics_by_model,
                "hist_n": int(block["target_key"].nunique()),
            }
        )
    return rows


def _fit_weekend_live_models(dataset: pd.DataFrame, candidate_key: str) -> dict:
    features, bucket_mode = _weekend_candidate_features(candidate_key)
    needed = ["target_key", "close", "target_weekend_ret"] + list(features)
    frame = dataset.dropna(subset=[c for c in needed if c in dataset.columns]).copy()
    if frame.empty:
        return {}
    out = {
        "features": list(features),
        "variant": "daily_long" if bucket_mode is None else ("session_close_coarse3" if bucket_mode == "coarse3" else "session_close_by_wday"),
    }
    has_range_targets = all(col in frame.columns for col in ("target_week_high_ret", "target_week_low_ret"))
    if bucket_mode is None:
        ret_model = _fit_linear_map(frame, frame["target_weekend_ret"].to_numpy(dtype=float), features, ridge_alpha=3.0)
        if ret_model is None:
            return {}
        out["return_model"] = ret_model
        if has_range_targets:
            out["high_return_model"] = _fit_linear_map(frame, frame["target_week_high_ret"].to_numpy(dtype=float), features, ridge_alpha=3.0)
            out["low_return_model"] = _fit_linear_map(frame, frame["target_week_low_ret"].to_numpy(dtype=float), features, ridge_alpha=3.0)
        return out
    frame["bucket_key"] = [
        _week_bucket_label(v, coarse=(bucket_mode == "coarse3"))
        for v in pd.to_numeric(frame.get("wday"), errors="coerce").fillna(-1)
    ]
    bucket_models = {}
    min_bucket_n = 4
    for bucket_key, block in frame.groupby("bucket_key"):
        if len(block) < min_bucket_n:
            continue
        ret_model = _fit_linear_map(block, block["target_weekend_ret"].to_numpy(dtype=float), features, ridge_alpha=4.0)
        if ret_model is None:
            continue
        entry = {"return_model": ret_model, "n": int(len(block))}
        if has_range_targets:
            hi_model = _fit_linear_map(block, block["target_week_high_ret"].to_numpy(dtype=float), features, ridge_alpha=4.0)
            lo_model = _fit_linear_map(block, block["target_week_low_ret"].to_numpy(dtype=float), features, ridge_alpha=4.0)
            if hi_model is not None:
                entry["high_return_model"] = hi_model
            if lo_model is not None:
                entry["low_return_model"] = lo_model
        bucket_models[str(bucket_key)] = entry
    out["bucket_models"] = bucket_models
    return out


def _fit_structural_daily_pair_overlay(resolved: pd.DataFrame, daily_df: pd.DataFrame,
                                       instrument_key: str, snapshots: pd.DataFrame | None = None) -> dict:
    if snapshots is None or snapshots.empty:
        return {}
    dataset = snapshots[snapshots.get("snapshot_type", "") == "daily_pair"].copy()
    if dataset.empty:
        return {}
    needed = ["target_key", "close", "target_high_ret", "target_low_ret", "vol_20"]
    for col in needed:
        if col not in dataset.columns:
            return {}
    dataset["target_center_ret"] = 0.5 * (
        pd.to_numeric(dataset["target_high_ret"], errors="coerce") + pd.to_numeric(dataset["target_low_ret"], errors="coerce")
    )
    dataset["target_range_ret"] = (
        pd.to_numeric(dataset["target_high_ret"], errors="coerce") - pd.to_numeric(dataset["target_low_ret"], errors="coerce")
    ).clip(lower=1e-4)
    dataset = dataset.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).sort_values("target_key")
    dataset = dataset.dropna(subset=["target_key", "close", "target_center_ret", "target_range_ret"] + list(_STRUCTURAL_DAILY_CENTER_FEATURES) + list(_STRUCTURAL_DAILY_RANGE_FEATURES))
    if len(dataset) < 12:
        return {}
    try:
        outcome_block = _load_structural_outcomes(instrument_key)
        outcome_block = outcome_block[outcome_block.get("snapshot_type", "") == "daily_pair"].copy()
    except Exception:
        outcome_block = pd.DataFrame()
    if outcome_block is not None and not outcome_block.empty:
        candidates = [c for c in _evaluate_daily_outcomes(resolved, outcome_block, instrument_key) if c.get("objective_score") is not None]
        if candidates:
            best = min(candidates, key=lambda c: float(c["objective_score"]))
            live = _fit_daily_live_models(dataset, best["candidate_key"])
            if live:
                enabled = bool(
                    any(float(v or 0.0) > 0.0 for v in best.get("high_blend_weights", {}).values())
                    or any(float(v or 0.0) > 0.0 for v in best.get("low_blend_weights", {}).values())
                )
                return {
                    "enabled": enabled,
                    "selected": str(best["candidate_key"]) if enabled else "model_only",
                    "variant": str(live.get("variant") or "global"),
                    "n": int(max([int((v.get("n", 0) or 0)) for v in list(best.get("high_metrics_by_model", {}).values()) + list(best.get("low_metrics_by_model", {}).values()) if isinstance(v, dict)], default=0)),
                    "hist_n": int(best.get("hist_n", 0) or 0),
                    "center_model": live.get("center_model"),
                    "range_model": live.get("range_model"),
                    "center_features": list(live.get("center_features", [])),
                    "range_features": list(live.get("range_features", [])),
                    "vol_bucket_edges": list(live.get("vol_bucket_edges", [])),
                    "close_loc_bucket_edges": list(live.get("close_loc_bucket_edges", [])),
                    "bucket_models": live.get("bucket_models", {}),
                    "high_blend_weights": best.get("high_blend_weights", {}),
                    "low_blend_weights": best.get("low_blend_weights", {}),
                    "high_metrics_by_model": best.get("high_metrics_by_model", {}),
                    "low_metrics_by_model": best.get("low_metrics_by_model", {}),
                    "aggregate_metrics": best.get("aggregate_metrics", {}),
                }

    def _train_subset(frame: pd.DataFrame, row: pd.Series, edges: list[float] | None, min_bucket_n: int) -> pd.DataFrame:
        if not edges:
            return frame
        bucket = _bucket_label(row.get("vol_20"), edges)
        subset = frame[frame["bucket"] == bucket].copy()
        return subset if len(subset) >= int(min_bucket_n) else pd.DataFrame()

    def _evaluate(center_features: tuple[str, ...], range_features: tuple[str, ...], edges: list[float] | None = None) -> dict | None:
        frame = dataset.copy()
        frame["bucket"] = [_bucket_label(v, edges or []) for v in frame.get("vol_20", [])]
        min_train = max(8, len(center_features) + 3, len(range_features) + 3)
        min_bucket_n = max(8, min_train)
        hist_high = {}
        hist_low = {}
        for i in range(len(frame)):
            tr_all = frame.iloc[:i].copy()
            if len(tr_all) < min_train:
                continue
            te = frame.iloc[i]
            tr = _train_subset(tr_all, te, edges, min_bucket_n)
            if tr.empty:
                if edges:
                    continue
                tr = tr_all
            center_model = _fit_linear_map(tr, tr["target_center_ret"].to_numpy(dtype=float), center_features, ridge_alpha=3.0)
            range_model = _fit_linear_map(tr, np.log(tr["target_range_ret"].to_numpy(dtype=float)), range_features, ridge_alpha=3.0)
            if center_model is None or range_model is None:
                continue
            ctx = te.to_dict()
            center_val = _predict_linear_map(center_model, ctx)
            log_range_val = _predict_linear_map(range_model, ctx)
            close_val = _safe_float(te.get("close"))
            if center_val is None or log_range_val is None or close_val is None or close_val <= 0.0:
                continue
            rng = float(max(1e-4, np.exp(log_range_val)))
            hist_high[str(te["target_key"])] = float(close_val * (1.0 + center_val + (0.5 * rng)))
            hist_low[str(te["target_key"])] = float(close_val * (1.0 + center_val - (0.5 * rng)))
        min_hist_n = max(8, int(np.ceil(0.4 * float(len(frame)))))
        if len(hist_high) < min_hist_n or len(hist_low) < min_hist_n:
            return None
        high_hist = resolved[resolved["horizon"] == "daily_high"].copy()
        low_hist = resolved[resolved["horizon"] == "daily_low"].copy()
        high_weights = {}
        low_weights = {}
        high_metrics_by_model = {}
        low_metrics_by_model = {}
        for model_key in _MODEL_KEYS:
            h_block = high_hist[high_hist["model"] == model_key].copy()
            if not h_block.empty:
                h_block = h_block[h_block["target_key"].astype(str).isin(hist_high)].copy()
                if not h_block.empty:
                    blend = _search_price_blend(
                        h_block["pred_price"].values,
                        np.asarray([hist_high[str(k)] for k in h_block["target_key"].astype(str)], dtype=float),
                        h_block["actual_price"].values,
                        horizon="daily",
                        instrument_key=instrument_key,
                    )
                    high_weights[model_key] = float(blend.get("weight", 0.0) or 0.0)
                    high_metrics_by_model[model_key] = blend
            l_block = low_hist[low_hist["model"] == model_key].copy()
            if not l_block.empty:
                l_block = l_block[l_block["target_key"].astype(str).isin(hist_low)].copy()
                if not l_block.empty:
                    blend = _search_price_blend(
                        l_block["pred_price"].values,
                        np.asarray([hist_low[str(k)] for k in l_block["target_key"].astype(str)], dtype=float),
                        l_block["actual_price"].values,
                        horizon="daily",
                        instrument_key=instrument_key,
                    )
                    low_weights[model_key] = float(blend.get("weight", 0.0) or 0.0)
                    low_metrics_by_model[model_key] = blend
        high_selected = [v.get("selected_metrics", {}) for v in high_metrics_by_model.values() if isinstance(v, dict) and v.get("selected_metrics")]
        low_selected = [v.get("selected_metrics", {}) for v in low_metrics_by_model.values() if isinstance(v, dict) and v.get("selected_metrics")]
        hi_mae = [_safe_float(v.get("mae_dollars")) for v in high_selected]
        lo_mae = [_safe_float(v.get("mae_dollars")) for v in low_selected]
        hi_ratio = [_safe_float(v.get("window_ratio_mean")) for v in high_selected]
        lo_ratio = [_safe_float(v.get("window_ratio_mean")) for v in low_selected]
        hi_hit = [_safe_float(v.get("window_hit_rate")) for v in high_selected]
        lo_hit = [_safe_float(v.get("window_hit_rate")) for v in low_selected]
        hi_mae = [v for v in hi_mae if v is not None]
        lo_mae = [v for v in lo_mae if v is not None]
        hi_ratio = [v for v in hi_ratio if v is not None]
        lo_ratio = [v for v in lo_ratio if v is not None]
        hi_hit = [v for v in hi_hit if v is not None]
        lo_hit = [v for v in lo_hit if v is not None]
        aggregate = {
            "mae_dollars": float(0.5 * (np.mean(hi_mae) + np.mean(lo_mae))) if hi_mae and lo_mae else None,
            "window_ratio_mean": float(0.5 * (np.mean(hi_ratio) + np.mean(lo_ratio))) if hi_ratio and lo_ratio else None,
            "window_hit_rate": float(0.5 * (np.mean(hi_hit) + np.mean(lo_hit))) if hi_hit and lo_hit else None,
        }
        return {
            "aggregate_metrics": aggregate,
            "high_blend_weights": high_weights,
            "low_blend_weights": low_weights,
            "high_metrics_by_model": high_metrics_by_model,
            "low_metrics_by_model": low_metrics_by_model,
            "objective_score": _window_objective(aggregate, horizon="daily", instrument_key=instrument_key),
            "bucket_edges": list(edges or []),
            "resolved_n": max(
                [int((v.get("n", 0) or 0)) for v in list(high_metrics_by_model.values()) + list(low_metrics_by_model.values()) if isinstance(v, dict)],
                default=0,
            ),
            "hist_n": int(min(len(hist_high), len(hist_low))),
        }

    candidates = [
        {
            "label": "daily_long_prior",
            "center_features": tuple(_STRUCTURAL_DAILY_CENTER_FEATURES),
            "range_features": tuple(_STRUCTURAL_DAILY_RANGE_FEATURES),
            "bucket_edges": [],
        }
    ]
    valid_edges = []
    for bucket_count in (2, 3):
        edges = _bucket_edges(pd.to_numeric(dataset["vol_20"], errors="coerce").to_numpy(dtype=float), bucket_count)
        if not edges:
            continue
        buckets = pd.Series([_bucket_label(v, edges) for v in pd.to_numeric(dataset["vol_20"], errors="coerce").fillna(0.0)], index=dataset.index)
        counts = buckets.value_counts()
        if not counts.empty and int(counts.min()) >= 8:
            valid_edges.append(edges)
            candidates.append(
                {
                    "label": f"daily_long_regime_{bucket_count}",
                    "center_features": tuple(_STRUCTURAL_DAILY_CENTER_FEATURES),
                    "range_features": tuple(_STRUCTURAL_DAILY_RANGE_FEATURES),
                    "bucket_edges": edges,
                }
            )
    news_ok = all(col in dataset.columns for col in _LONG_HORIZON_NEWS_FEATURES) and bool(
        float(pd.to_numeric(dataset["news_intensity_ewm_d7"], errors="coerce").abs().sum()) > 0.0
    )
    if news_ok:
        candidates.append(
            {
                "label": "daily_long_news_prior",
                "center_features": tuple(list(_STRUCTURAL_DAILY_CENTER_FEATURES) + ["news_sent_ewm_d7", "news_intensity_ewm_d7"]),
                "range_features": tuple(list(_STRUCTURAL_DAILY_RANGE_FEATURES) + ["news_intensity_ewm_d7"]),
                "bucket_edges": [],
            }
        )
    best = None
    for cand in candidates:
        fit = _evaluate(cand["center_features"], cand["range_features"], cand["bucket_edges"])
        if fit is None or fit.get("objective_score") is None:
            continue
        if best is None or float(fit["objective_score"]) + 1e-9 < float(best["objective_score"]):
            best = {**cand, **fit}
    if best is None:
        return {}
    center_model = _fit_linear_map(dataset, dataset["target_center_ret"].to_numpy(dtype=float), tuple(best["center_features"]), ridge_alpha=3.0)
    range_model = _fit_linear_map(dataset, np.log(dataset["target_range_ret"].to_numpy(dtype=float)), tuple(best["range_features"]), ridge_alpha=3.0)
    if center_model is None or range_model is None:
        return {}
    bucket_models = {}
    edges = list(best.get("bucket_edges", []) or [])
    if edges:
        dataset["bucket"] = [_bucket_label(v, edges) for v in dataset["vol_20"]]
        for bucket_name, block in dataset.groupby("bucket"):
            if len(block) < 8:
                continue
            b_center = _fit_linear_map(block, block["target_center_ret"].to_numpy(dtype=float), tuple(best["center_features"]), ridge_alpha=3.0)
            b_range = _fit_linear_map(block, np.log(block["target_range_ret"].to_numpy(dtype=float)), tuple(best["range_features"]), ridge_alpha=3.0)
            if b_center is None or b_range is None:
                continue
            bucket_models[str(bucket_name)] = {
                "center_model": b_center,
                "range_model": b_range,
                "n": int(len(block)),
            }
    enabled = bool(any(float(v or 0.0) > 0.0 for v in best["high_blend_weights"].values()) or any(float(v or 0.0) > 0.0 for v in best["low_blend_weights"].values()))
    return {
        "enabled": enabled,
        "selected": str(best["label"]) if enabled else "model_only",
        "variant": "regime_bucketed" if edges else "global",
        "n": int(best.get("resolved_n", 0) or 0),
        "hist_n": int(best.get("hist_n", 0) or 0),
        "center_model": center_model,
        "range_model": range_model,
        "center_features": list(best["center_features"]),
        "range_features": list(best["range_features"]),
        "vol_bucket_edges": edges,
        "bucket_models": bucket_models,
        "high_blend_weights": best["high_blend_weights"],
        "low_blend_weights": best["low_blend_weights"],
        "high_metrics_by_model": best["high_metrics_by_model"],
        "low_metrics_by_model": best["low_metrics_by_model"],
        "aggregate_metrics": best["aggregate_metrics"],
    }


def _fit_structural_weekend_overlay(resolved: pd.DataFrame, daily_df: pd.DataFrame,
                                    instrument_key: str, snapshots: pd.DataFrame | None = None) -> dict:
    if snapshots is None or snapshots.empty:
        return {}
    dataset = snapshots[snapshots.get("snapshot_type", "") == "weekend_session"].copy()
    if dataset.empty or "target_weekend_ret" not in dataset.columns:
        return {}
    dataset = dataset.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False).sort_values("target_key")
    base_required = ["target_key", "close"] + list(_STRUCTURAL_WEEKEND_FEATURES)
    session_required = ["target_key", "close"] + list(_STRUCTURAL_WEEKEND_SESSION_FEATURES)
    if not set(base_required).issubset(dataset.columns):
        return {}
    dataset = dataset.dropna(subset=base_required)
    if len(dataset) < 10:
        return {}
    try:
        outcome_block = _load_structural_outcomes(instrument_key)
        outcome_block = outcome_block[outcome_block.get("snapshot_type", "") == "weekend_session"].copy()
    except Exception:
        outcome_block = pd.DataFrame()
    if outcome_block is not None and not outcome_block.empty:
        candidates = [c for c in _evaluate_weekend_outcomes(resolved, outcome_block, instrument_key) if c.get("objective_score") is not None]
        if candidates:
            best = min(candidates, key=lambda c: float(c["objective_score"]))
            live = _fit_weekend_live_models(dataset, best["candidate_key"])
            if live:
                enabled = bool(any(float(v or 0.0) > 0.0 for v in best.get("blend_weights", {}).values()))
                return {
                    "enabled": enabled,
                    "selected": str(best["candidate_key"]) if enabled else "model_only",
                    "variant": str(live.get("variant") or "daily_long"),
                    "n": int(max([int((v.get("n", 0) or 0)) for v in best.get("metrics_by_model", {}).values() if isinstance(v, dict)], default=0)),
                    "hist_n": int(best.get("hist_n", 0) or 0),
                    "features": list(live.get("features", [])),
                    "blend_weights": best.get("blend_weights", {}),
                    "metrics_by_model": best.get("metrics_by_model", {}),
                    "aggregate_metrics": best.get("aggregate_metrics", {}),
                    "return_model": live.get("return_model"),
                    "high_return_model": live.get("high_return_model"),
                    "low_return_model": live.get("low_return_model"),
                    "bucket_models": live.get("bucket_models", {}),
                }

    def _blend_payload(hist_prices: dict[str, float]) -> dict:
        week_hist = resolved[resolved["horizon"] == "weekend"].copy()
        blend_weights = {}
        metrics_by_model = {}
        for model_key in _MODEL_KEYS:
            block = week_hist[week_hist["model"] == model_key].copy()
            if block.empty:
                continue
            block = block[block["target_key"].astype(str).isin(hist_prices)].copy()
            if block.empty:
                continue
            blend = _search_price_blend(
                block["pred_price"].values,
                np.asarray([hist_prices[str(k)] for k in block["target_key"].astype(str)], dtype=float),
                block["actual_price"].values,
                horizon="weekend",
                instrument_key=instrument_key,
            )
            blend_weights[model_key] = float(blend.get("weight", 0.0) or 0.0)
            metrics_by_model[model_key] = blend
        selected = [v.get("selected_metrics", {}) for v in metrics_by_model.values() if isinstance(v, dict) and v.get("selected_metrics")]
        maes = [_safe_float(v.get("mae_dollars")) for v in selected]
        ratios = [_safe_float(v.get("window_ratio_mean")) for v in selected]
        hits = [_safe_float(v.get("window_hit_rate")) for v in selected]
        maes = [v for v in maes if v is not None]
        ratios = [v for v in ratios if v is not None]
        hits = [v for v in hits if v is not None]
        aggregate = {
            "mae_dollars": float(np.mean(maes)) if maes else None,
            "window_ratio_mean": float(np.mean(ratios)) if ratios else None,
            "window_hit_rate": float(np.mean(hits)) if hits else None,
        }
        return {
            "blend_weights": blend_weights,
            "metrics_by_model": metrics_by_model,
            "aggregate_metrics": aggregate,
            "resolved_n": max([int(v.get("n", 0) or 0) for v in metrics_by_model.values() if isinstance(v, dict)], default=0),
            "objective_score": _window_objective(aggregate, horizon="weekend", instrument_key=instrument_key),
        }

    def _evaluate_global(features: tuple[str, ...]) -> dict | None:
        frame = dataset.dropna(subset=list(features) + ["target_weekend_ret"]).copy()
        if len(frame) < 10:
            return None
        min_train = max(8, len(features) + 3)
        hist_prices = {}
        for i in range(len(frame)):
            tr = frame.iloc[:i].copy()
            te = frame.iloc[i]
            if len(tr) < min_train:
                continue
            model = _fit_linear_map(tr, tr["target_weekend_ret"].to_numpy(dtype=float), features, ridge_alpha=3.0)
            if model is None:
                continue
            pred_ret = _predict_linear_map(model, te.to_dict())
            close_val = _safe_float(te.get("close"))
            if pred_ret is None or close_val is None or close_val <= 0.0:
                continue
            hist_prices[str(te["target_key"])] = float(close_val * (1.0 + pred_ret))
        min_hist_n = max(8, int(np.ceil(0.5 * float(len(frame)))))
        if len(hist_prices) < min_hist_n:
            return None
        return {"variant": "daily_long", "features": list(features), "hist_prices": hist_prices, "hist_n": int(len(hist_prices)), **_blend_payload(hist_prices)}

    def _evaluate_session_bucketed(features: tuple[str, ...]) -> dict | None:
        frame = dataset.dropna(subset=list(features) + ["target_weekend_ret", "wday"]).copy()
        if len(frame) < 10:
            return None
        frame["wday_bucket"] = pd.to_numeric(frame["wday"], errors="coerce").fillna(-1).astype(int).astype(str)
        min_train = max(5, len(features) + 2)
        hist_prices = {}
        for i in range(len(frame)):
            tr_all = frame.iloc[:i].copy()
            te = frame.iloc[i]
            if te.get("days_to_week_end", 1.0) <= 0.0:
                close_val = _safe_float(te.get("close"))
                if close_val is not None and close_val > 0.0:
                    hist_prices[str(te["target_key"])] = float(close_val)
                continue
            bucket = str(te["wday_bucket"])
            tr = tr_all[tr_all["wday_bucket"] == bucket].copy()
            if len(tr) < min_train:
                continue
            model = _fit_linear_map(tr, tr["target_weekend_ret"].to_numpy(dtype=float), features, ridge_alpha=4.0)
            if model is None:
                continue
            pred_ret = _predict_linear_map(model, te.to_dict())
            close_val = _safe_float(te.get("close"))
            if pred_ret is None or close_val is None or close_val <= 0.0:
                continue
            hist_prices[str(te["target_key"])] = float(close_val * (1.0 + pred_ret))
        min_hist_n = max(8, int(np.ceil(0.5 * float(len(frame)))))
        if len(hist_prices) < min_hist_n:
            return None
        return {"variant": "session_close_by_wday", "features": list(features), "hist_prices": hist_prices, "hist_n": int(len(hist_prices)), **_blend_payload(hist_prices)}

    candidates = []
    global_fit = _evaluate_global(tuple(_STRUCTURAL_WEEKEND_FEATURES))
    if global_fit is not None:
        candidates.append(global_fit)
    if set(session_required).issubset(dataset.columns):
        session_fit = _evaluate_session_bucketed(tuple(_STRUCTURAL_WEEKEND_SESSION_FEATURES))
        if session_fit is not None:
            candidates.append(session_fit)
        news_ok = all(col in dataset.columns for col in _LONG_HORIZON_NEWS_FEATURES) and bool(
            float(pd.to_numeric(dataset["news_intensity_ewm_d7"], errors="coerce").abs().sum()) > 0.0
        )
        if news_ok:
            session_news_fit = _evaluate_session_bucketed(tuple(list(_STRUCTURAL_WEEKEND_SESSION_FEATURES) + ["news_sent_ewm_d7", "news_intensity_ewm_d7"]))
            if session_news_fit is not None:
                candidates.append(session_news_fit)
    if not candidates:
        return {}
    best = min(
        [c for c in candidates if c.get("objective_score") is not None],
        key=lambda c: float(c["objective_score"]),
        default=None,
    )
    if best is None:
        return {}
    enabled = bool(any(float(v or 0.0) > 0.0 for v in best.get("blend_weights", {}).values()))
    payload = {
        "enabled": enabled,
        "selected": str(best["variant"]) if enabled else "model_only",
        "variant": str(best["variant"]),
        "n": int(best.get("resolved_n", 0) or 0),
        "hist_n": int(best.get("hist_n", 0) or 0),
        "features": list(best["features"]),
        "blend_weights": best["blend_weights"],
        "metrics_by_model": best["metrics_by_model"],
        "aggregate_metrics": best["aggregate_metrics"],
    }
    if best["variant"] == "session_close_by_wday":
        frame = dataset.dropna(subset=list(best["features"]) + ["target_weekend_ret", "wday"]).copy()
        frame["wday_bucket"] = pd.to_numeric(frame["wday"], errors="coerce").fillna(-1).astype(int).astype(str)
        bucket_models = {}
        for bucket, block in frame.groupby("wday_bucket"):
            if bucket == "4":
                bucket_models[bucket] = {"identity_close": True, "n": int(len(block))}
                continue
            if len(block) < max(5, len(best["features"]) + 2):
                continue
            model = _fit_linear_map(block, block["target_weekend_ret"].to_numpy(dtype=float), tuple(best["features"]), ridge_alpha=4.0)
            if model is not None:
                bucket_models[bucket] = {"return_model": model, "n": int(len(block))}
        payload["bucket_models"] = bucket_models
    else:
        model = _fit_linear_map(dataset, dataset["target_weekend_ret"].to_numpy(dtype=float), tuple(best["features"]), ridge_alpha=3.0)
        if model is None:
            return {}
        payload["return_model"] = model
    return payload


def _fit_daily_pair_overlay(merged_high: pd.DataFrame, merged_low: pd.DataFrame, instrument_key: str) -> dict:
    joined = merged_high[["target_key", "pred_price", "actual_price", "wday"]].rename(
        columns={"pred_price": "pred_high", "actual_price": "act_high"}
    ).merge(
        merged_low[["target_key", "pred_price", "actual_price"]].rename(
            columns={"pred_price": "pred_low", "actual_price": "act_low"}
        ),
        on="target_key",
        how="inner",
    )
    joined = joined.sort_values("target_key").dropna()
    if len(joined) < 10:
        return {}
    joined["pred_center"] = 0.5 * (pd.to_numeric(joined["pred_high"], errors="coerce") + pd.to_numeric(joined["pred_low"], errors="coerce"))
    joined["act_center"] = 0.5 * (pd.to_numeric(joined["act_high"], errors="coerce") + pd.to_numeric(joined["act_low"], errors="coerce"))
    joined["pred_range"] = (pd.to_numeric(joined["pred_high"], errors="coerce") - pd.to_numeric(joined["pred_low"], errors="coerce")).clip(lower=0.01)
    joined["act_range"] = (pd.to_numeric(joined["act_high"], errors="coerce") - pd.to_numeric(joined["act_low"], errors="coerce")).clip(lower=0.01)
    joined["wday"] = pd.to_numeric(joined["wday"], errors="coerce").fillna(0.0)
    joined = joined.replace([np.inf, -np.inf], np.nan).dropna()
    if len(joined) < 10:
        return {}

    center_candidates = (
        ("pred_center",),
        ("pred_center", "wday"),
        ("pred_center", "pred_range", "wday"),
    )
    range_candidates = (
        ("pred_range",),
        ("pred_range", "wday"),
        ("pred_range", "pred_center", "wday"),
    )

    base_hi = summarize_price_window_errors(
        np.abs(joined["pred_high"].to_numpy(dtype=float) - joined["act_high"].to_numpy(dtype=float)),
        actual_prices=joined["act_high"].to_numpy(dtype=float),
        horizon="daily",
        instrument_key=instrument_key,
    )
    base_lo = summarize_price_window_errors(
        np.abs(joined["pred_low"].to_numpy(dtype=float) - joined["act_low"].to_numpy(dtype=float)),
        actual_prices=joined["act_low"].to_numpy(dtype=float),
        horizon="daily",
        instrument_key=instrument_key,
    )
    try:
        base_score = (
            0.5 * (float(base_hi["window_ratio_mean"]) + float(base_lo["window_ratio_mean"]))
            + 1.25 * max(0.0, 0.58 - (0.5 * (float(base_hi["window_hit_rate"]) + float(base_lo["window_hit_rate"]))))
            + 0.05 * (float(base_hi["mae_dollars"]) + float(base_lo["mae_dollars"]))
        )
    except Exception:
        base_score = None

    start = max(8, len(joined) // 3)
    best = None
    for center_feats in center_candidates:
        for range_feats in range_candidates:
            pred_hi = []
            pred_lo = []
            act_hi = []
            act_lo = []
            for i in range(start, len(joined)):
                tr = joined.iloc[:i].copy()
                te = joined.iloc[i:i + 1].copy()
                center_model = _fit_linear_map(tr, tr["act_center"].to_numpy(dtype=float), center_feats)
                range_model = _fit_linear_map(tr, np.log(tr["act_range"].to_numpy(dtype=float)), range_feats)
                if center_model is None or range_model is None:
                    continue
                ctx = te.iloc[0].to_dict()
                center_val = _predict_linear_map(center_model, ctx)
                log_range_val = _predict_linear_map(range_model, ctx)
                if center_val is None or log_range_val is None:
                    continue
                rng = float(max(0.01, np.exp(log_range_val)))
                pred_hi.append(float(center_val + (0.5 * rng)))
                pred_lo.append(float(center_val - (0.5 * rng)))
                act_hi.append(float(te["act_high"].iloc[0]))
                act_lo.append(float(te["act_low"].iloc[0]))
            if len(pred_hi) < max(6, start // 2):
                continue
            hi_metrics = summarize_price_window_errors(
                np.abs(np.asarray(pred_hi, dtype=float) - np.asarray(act_hi, dtype=float)),
                actual_prices=np.asarray(act_hi, dtype=float),
                horizon="daily",
                instrument_key=instrument_key,
            )
            lo_metrics = summarize_price_window_errors(
                np.abs(np.asarray(pred_lo, dtype=float) - np.asarray(act_lo, dtype=float)),
                actual_prices=np.asarray(act_lo, dtype=float),
                horizon="daily",
                instrument_key=instrument_key,
            )
            try:
                hi_ratio = float(hi_metrics.get("window_ratio_mean"))
                lo_ratio = float(lo_metrics.get("window_ratio_mean"))
                hi_hit = float(hi_metrics.get("window_hit_rate"))
                lo_hit = float(lo_metrics.get("window_hit_rate"))
                hi_mae = float(hi_metrics.get("mae_dollars"))
                lo_mae = float(lo_metrics.get("mae_dollars"))
                obj = (
                    0.5 * (hi_ratio + lo_ratio)
                    + 1.25 * max(0.0, 0.58 - (0.5 * (hi_hit + lo_hit)))
                    + 0.05 * (hi_mae + lo_mae)
                )
            except Exception:
                obj = None
            if obj is None:
                continue
            if best is None or float(obj) + 1e-9 < float(best["objective_score"]):
                best = {
                    "objective_score": float(obj),
                    "high_metrics": hi_metrics,
                    "low_metrics": lo_metrics,
                    "center_features": list(center_feats),
                    "range_features": list(range_feats),
                }
    if best is None:
        return {}
    improvement = None if base_score is None else float(base_score - best["objective_score"])
    center_model = _fit_linear_map(joined, joined["act_center"].to_numpy(dtype=float), tuple(best["center_features"]))
    range_model = _fit_linear_map(joined, np.log(joined["act_range"].to_numpy(dtype=float)), tuple(best["range_features"]))
    if center_model is None or range_model is None:
        return {}
    return {
        "enabled": bool(improvement is not None and improvement >= 0.05),
        "selected": "center_range" if improvement is not None and improvement >= 0.05 else "model_only",
        "objective_improvement": improvement,
        "n": int(len(joined)),
        "center_model": center_model,
        "range_model": range_model,
        "high_metrics": best["high_metrics"],
        "low_metrics": best["low_metrics"],
    }


def _heuristic_series(frame: pd.DataFrame, target_col: str, features: tuple[str, ...], k: int, min_train: int) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    needed = [c for c in features if c in frame.columns]
    if len(needed) != len(features) or target_col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    X = frame[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    y = pd.to_numeric(frame[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    out = np.full(len(frame), np.nan, dtype=float)
    for i in range(len(frame)):
        if i < int(min_train):
            continue
        pred = _knn_predict(X[:i], y[:i], X[i], int(k))
        if pred is not None and np.isfinite(pred):
            out[i] = float(pred)
    return pd.Series(out, index=frame.index, dtype=float)


def _current_heuristic_return(prices: pd.DataFrame, instrument_key: str, horizon: str,
                              features: tuple[str, ...], k: int, min_train: int) -> float | None:
    spec = _HORIZON_SPECS.get(horizon)
    if spec is None or prices is None or prices.empty:
        return None
    try:
        cache_key = (
            str(instrument_key or "").upper(),
            str(horizon),
            tuple(str(x) for x in features),
            int(k),
            int(min_train),
            int(len(prices)),
            _to_iso_key(prices.index[-1]),
        )
    except Exception:
        cache_key = None
    if cache_key is not None and cache_key in _CURRENT_HEURISTIC_CACHE:
        return _CURRENT_HEURISTIC_CACHE.get(cache_key)
    if horizon in ("daily_high", "daily_low"):
        hist = _daily_training_frame(prices, instrument_key)
        cur = _daily_current_row(prices, instrument_key)
    else:
        hist = _week_training_frame(prices, instrument_key)
        cur = _week_current_row(prices, instrument_key)
    if hist.empty or cur.empty or len(hist) < int(min_train):
        if cache_key is not None:
            _CURRENT_HEURISTIC_CACHE[cache_key] = None
        return None
    if any(col not in hist.columns for col in features) or any(col not in cur.columns for col in features):
        if cache_key is not None:
            _CURRENT_HEURISTIC_CACHE[cache_key] = None
        return None
    try:
        x_train = hist[list(features)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        y_train = pd.to_numeric(hist[spec["target_col"]], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        x_now = pd.to_numeric(cur.iloc[0][list(features)], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    except Exception:
        if cache_key is not None:
            _CURRENT_HEURISTIC_CACHE[cache_key] = None
        return None
    out = _knn_predict(x_train, y_train, x_now, int(k))
    if cache_key is not None:
        _CURRENT_HEURISTIC_CACHE[cache_key] = out
    return out


def _blend_metrics(model_ret: np.ndarray, heur_ret: np.ndarray, actual_ret: np.ndarray, close: np.ndarray,
                   horizon: str, instrument_key: str):
    best = None
    valid = np.isfinite(model_ret) & np.isfinite(heur_ret) & np.isfinite(actual_ret) & np.isfinite(close) & (close > 0.0)
    if int(np.sum(valid)) < 4:
        return None
    mr = model_ret[valid]
    hr = heur_ret[valid]
    ar = actual_ret[valid]
    cl = close[valid]
    for weight in np.linspace(0.0, 1.0, 21):
        blend = ((1.0 - weight) * mr) + (weight * hr)
        metrics = score_return_forecast(blend, ar, cl, horizon=horizon, instrument_key=instrument_key)
        if metrics.get("objective_score") is None:
            continue
        if best is None or float(metrics["objective_score"]) < float(best["metrics"]["objective_score"]):
            best = {
                "blend_weight": float(weight),
                "metrics": metrics,
                "blend_ret": blend,
                "actual_ret": ar,
                "close": cl,
            }
    return best


def _fit_model_overlay(merged: pd.DataFrame, instrument_key: str, horizon_key: str) -> dict:
    spec = _HORIZON_SPECS[horizon_key]
    horizon = spec["horizon"]
    model_ret = ((pd.to_numeric(merged["pred_price"], errors="coerce") / pd.to_numeric(merged["close"], errors="coerce")) - 1.0).to_numpy(dtype=float)
    actual_ret = pd.to_numeric(merged[spec["target_col"]], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(merged["close"], errors="coerce").to_numpy(dtype=float)
    base_metrics = score_return_forecast(model_ret, actual_ret, close, horizon=horizon, instrument_key=instrument_key)
    best = {
        "enabled": False,
        "selected": "model_only",
        "baseline_metrics": base_metrics,
        "selected_metrics": base_metrics,
        "blend_weight": 0.0,
        "calibration": {"scale": 1.0, "bias": 0.0, "selected": "identity"},
        "features": [],
        "k": None,
        "min_train": spec["default_min_train"],
        "n": int(len(merged)),
    }
    if horizon_key == "weekend":
        try:
            d2we = pd.to_numeric(merged.get("days_to_week_end"), errors="coerce").dropna()
            if d2we.empty or int(d2we.nunique()) < 2:
                best["selected"] = "insufficient_week_context"
                return best
        except Exception:
            best["selected"] = "insufficient_week_context"
            return best
    for cand in spec["candidates"]:
        heur_ret = pd.to_numeric(merged[cand["heur_col"]], errors="coerce").to_numpy(dtype=float)
        fitted = _blend_metrics(model_ret, heur_ret, actual_ret, close, horizon, instrument_key)
        if fitted is None:
            continue
        chosen_metrics = fitted["metrics"]
        chosen_cal = {"scale": 1.0, "bias": 0.0, "selected": "identity"}
        if int(len(fitted["blend_ret"])) >= max(6, int(cand["min_train"] // 2)):
            calib = fit_return_calibration(
                fitted["blend_ret"],
                fitted["actual_ret"],
                fitted["close"],
                horizon=horizon,
                instrument_key=instrument_key,
                allow_flip=False,
                min_samples=max(6, int(cand["min_train"] // 2)),
            )
            calib_ret = apply_return_calibration(fitted["blend_ret"], calib)
            calib_metrics = score_return_forecast(
                calib_ret,
                fitted["actual_ret"],
                fitted["close"],
                horizon=horizon,
                instrument_key=instrument_key,
            )
            if calib_metrics.get("objective_score") is not None and chosen_metrics.get("objective_score") is not None:
                if float(calib_metrics["objective_score"]) + 1e-9 < float(chosen_metrics["objective_score"]):
                    chosen_metrics = calib_metrics
                    chosen_cal = calib
        improvement = None
        if base_metrics.get("objective_score") is not None and chosen_metrics.get("objective_score") is not None:
            improvement = float(base_metrics["objective_score"]) - float(chosen_metrics["objective_score"])
        if best["selected_metrics"].get("objective_score") is None or (
            chosen_metrics.get("objective_score") is not None
            and float(chosen_metrics["objective_score"]) + 1e-9 < float(best["selected_metrics"]["objective_score"])
        ):
            best = {
                "enabled": bool(improvement is not None and improvement >= 0.02),
                "selected": "heuristic_blend" if improvement is not None and improvement >= 0.02 else "model_only",
                "baseline_metrics": base_metrics,
                "selected_metrics": chosen_metrics,
                "blend_weight": float(fitted["blend_weight"]),
                "calibration": chosen_cal,
                "features": list(cand["features"]),
                "k": int(cand["k"]),
                "min_train": int(cand["min_train"]),
                "n": int(len(merged)),
                "objective_improvement": improvement,
            }
    return best


def fit_horizon_overlay(instrument_key: str, prices: pd.DataFrame | None = None) -> dict:
    instrument = str(instrument_key or "").upper()
    prices_path = os.path.join(DATA_DIR, f"prices_{instrument}.csv")
    price_df = prices if isinstance(prices, pd.DataFrame) else _load_prices(prices_path)
    resolved = _load_resolved_history(instrument)
    payload = {
        "instrument": instrument,
        "last_update": datetime.now(timezone.utc).isoformat(),
        "horizons": {},
        "daily_pair": {},
        "daily_pair_structural": {},
        "weekend_structural": {},
        "sample_summary": {},
    }
    if price_df is None or price_df.empty or resolved.empty:
        _write_json_if_changed(_overlay_path(instrument), payload)
        _CACHE[instrument] = {"ts": datetime.now(timezone.utc), "data": payload}
        return payload
    snapshots = _load_structural_snapshots(instrument)
    if snapshots.empty:
        try:
            snapshots = refresh_structural_snapshots(instrument, prices=price_df)
        except Exception:
            snapshots = pd.DataFrame()
    outcomes = _load_structural_outcomes(instrument)
    if snapshots is not None and not snapshots.empty and outcomes.empty:
        try:
            outcomes = refresh_structural_candidate_outcomes(instrument, prices=price_df, snapshots=snapshots)
        except Exception:
            outcomes = pd.DataFrame()

    daily_frame = _daily_training_frame(price_df, instrument)
    week_frame = _week_training_frame(price_df, instrument)
    frame_map = {
        "daily_high": daily_frame,
        "daily_low": daily_frame,
        "weekend": week_frame,
    }
    for horizon_key, spec in _HORIZON_SPECS.items():
        frame = frame_map[horizon_key]
        if frame.empty:
            payload["horizons"][horizon_key] = {}
            continue
        for idx, cand in enumerate(spec["candidates"]):
            frame[f"heur_{horizon_key}_{idx}"] = _heuristic_series(
                frame,
                spec["target_col"],
                cand["features"],
                cand["k"],
                cand["min_train"],
            )
            spec["candidates"][idx]["heur_col"] = f"heur_{horizon_key}_{idx}"

        models_out = {}
        horizon_hist = resolved[resolved["horizon"] == horizon_key].copy()
        merged_map = {}
        for model_key in _MODEL_KEYS:
            block = horizon_hist[horizon_hist["model"] == model_key].copy()
            if block.empty:
                continue
            merged = block.merge(
                frame[["target_key", "close", "wday", spec["target_col"]] + [cand["heur_col"] for cand in spec["candidates"]] + ([c for c in ["days_to_week_end"] if c in frame.columns])],
                on="target_key",
                how="inner",
            )
            if merged.empty:
                continue
            merged = merged.sort_values("resolved_ts").tail(int(spec["history_limit"]))
            merged_map[model_key] = merged.copy()
            models_out[model_key] = _fit_model_overlay(merged, instrument, horizon_key)
        payload["horizons"][horizon_key] = models_out
        payload["sample_summary"][horizon_key] = {
            "resolved_rows": int(len(horizon_hist)),
            "training_rows": int(len(frame)),
        }
        if horizon_key == "daily_high":
            low_hist = resolved[resolved["horizon"] == "daily_low"].copy()
            low_map = {}
            for model_key in _MODEL_KEYS:
                block = low_hist[low_hist["model"] == model_key].copy()
                if block.empty:
                    continue
                merged_low = block.merge(
                    daily_frame[["target_key", "close", "wday", "target_low_ret"]],
                    on="target_key",
                    how="inner",
                )
                if merged_low.empty:
                    continue
                low_map[model_key] = merged_low.sort_values("resolved_ts").tail(int(_HORIZON_SPECS["daily_low"]["history_limit"]))
            for model_key, merged_high in merged_map.items():
                merged_low = low_map.get(model_key)
                if merged_low is None or merged_low.empty:
                    continue
                pair_payload = _fit_daily_pair_overlay(merged_high, merged_low, instrument)
                if pair_payload:
                    payload["daily_pair"][model_key] = pair_payload

    daily_long = _load_long_daily_history(instrument)
    if daily_long is not None and not daily_long.empty:
        try:
            structural_pair = _fit_structural_daily_pair_overlay(resolved, daily_long, instrument, snapshots=snapshots)
        except Exception:
            structural_pair = {}
        try:
            structural_weekend = _fit_structural_weekend_overlay(resolved, daily_long, instrument, snapshots=snapshots)
        except Exception:
            structural_weekend = {}
        current_daily_rows = []
        for entry in payload.get("daily_pair", {}).values():
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
            current_daily_rows.append(
                {
                    "mae_dollars": 0.5 * (hi_mae + lo_mae),
                    "window_ratio_mean": 0.5 * (hi_ratio + lo_ratio) if hi_ratio is not None and lo_ratio is not None else None,
                    "window_hit_rate": 0.5 * (hi_hit + lo_hit) if hi_hit is not None and lo_hit is not None else None,
                }
            )
        if current_daily_rows:
            current_daily = {
                "mae_dollars": float(np.mean([r["mae_dollars"] for r in current_daily_rows if r["mae_dollars"] is not None])),
                "window_ratio_mean": float(np.mean([r["window_ratio_mean"] for r in current_daily_rows if r["window_ratio_mean"] is not None])),
                "window_hit_rate": float(np.mean([r["window_hit_rate"] for r in current_daily_rows if r["window_hit_rate"] is not None])),
            }
        else:
            current_daily = {}
        current_week_rows = []
        for entry in payload.get("horizons", {}).get("weekend", {}).values():
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("selected_metrics", {})
            if _safe_float(metrics.get("mae_dollars")) is None:
                continue
            current_week_rows.append(
                {
                    "mae_dollars": _safe_float(metrics.get("mae_dollars")),
                    "window_ratio_mean": _safe_float(metrics.get("window_ratio_mean")),
                    "window_hit_rate": _safe_float(metrics.get("window_hit_rate")),
                }
            )
        if current_week_rows:
            current_week = {
                "mae_dollars": float(np.mean([r["mae_dollars"] for r in current_week_rows if r["mae_dollars"] is not None])),
                "window_ratio_mean": float(np.mean([r["window_ratio_mean"] for r in current_week_rows if r["window_ratio_mean"] is not None])),
                "window_hit_rate": float(np.mean([r["window_hit_rate"] for r in current_week_rows if r["window_hit_rate"] is not None])),
            }
        else:
            current_week = {}
        if structural_pair:
            cand_obj = _window_objective(structural_pair.get("aggregate_metrics", {}), horizon="daily", instrument_key=instrument)
            base_obj = _window_objective(current_daily, horizon="daily", instrument_key=instrument)
            if cand_obj is None or (base_obj is not None and float(cand_obj) + 1e-9 >= float(base_obj)):
                structural_pair["enabled"] = False
                structural_pair["selected"] = "model_only"
                structural_pair["high_blend_weights"] = {k: 0.0 for k in structural_pair.get("high_blend_weights", {}).keys()}
                structural_pair["low_blend_weights"] = {k: 0.0 for k in structural_pair.get("low_blend_weights", {}).keys()}
        if structural_weekend:
            cand_obj = _window_objective(structural_weekend.get("aggregate_metrics", {}), horizon="weekend", instrument_key=instrument)
            base_obj = _window_objective(current_week, horizon="weekend", instrument_key=instrument)
            if cand_obj is None or (base_obj is not None and float(cand_obj) + 1e-9 >= float(base_obj)):
                structural_weekend["enabled"] = False
                structural_weekend["selected"] = "model_only"
                structural_weekend["blend_weights"] = {k: 0.0 for k in structural_weekend.get("blend_weights", {}).keys()}
        if structural_pair:
            payload["daily_pair_structural"] = structural_pair
        if structural_weekend:
            payload["weekend_structural"] = structural_weekend
        payload["sample_summary"]["daily_long_history_rows"] = int(len(daily_long))
    if snapshots is not None and not snapshots.empty:
        payload["sample_summary"]["snapshot_rows"] = int(len(snapshots))
        payload["sample_summary"]["snapshot_daily_rows"] = int((snapshots.get("snapshot_type") == "daily_pair").sum()) if "snapshot_type" in snapshots.columns else 0
        payload["sample_summary"]["snapshot_weekend_rows"] = int((snapshots.get("snapshot_type") == "weekend_session").sum()) if "snapshot_type" in snapshots.columns else 0
    if outcomes is not None and not outcomes.empty:
        payload["sample_summary"]["outcome_rows"] = int(len(outcomes))
        payload["sample_summary"]["outcome_daily_rows"] = int((outcomes.get("snapshot_type") == "daily_pair").sum()) if "snapshot_type" in outcomes.columns else 0
        payload["sample_summary"]["outcome_weekend_rows"] = int((outcomes.get("snapshot_type") == "weekend_session").sum()) if "snapshot_type" in outcomes.columns else 0
    if news_archive_coverage is not None:
        try:
            payload["sample_summary"]["news_coverage"] = news_archive_coverage(instrument)
        except Exception:
            pass

    _write_json_if_changed(_overlay_path(instrument), payload)
    _CACHE[instrument] = {"ts": datetime.now(timezone.utc), "data": payload}
    return payload


def load_horizon_overlay(instrument_key: str) -> dict:
    instrument = str(instrument_key or "").upper()
    now = datetime.now(timezone.utc)
    cached = _CACHE.get(instrument)
    if isinstance(cached, dict):
        ts = cached.get("ts")
        if ts is not None and (now - ts).total_seconds() < 60.0:
            return cached.get("data", {})
    payload = _load_json(_overlay_path(instrument))
    _CACHE[instrument] = {"ts": now, "data": payload}
    return payload


def apply_daily_pair_overlay(prices: pd.DataFrame, instrument_key: str, model_key: str,
                             pred_high, pred_low) -> dict | None:
    overlay = load_horizon_overlay(instrument_key)
    pair_map = overlay.get("daily_pair", {}) if isinstance(overlay, dict) else {}
    payload = pair_map.get(model_key, {})
    high_px = _safe_float(pred_high)
    low_px = _safe_float(pred_low)
    if high_px is None or low_px is None or high_px <= low_px:
        return None
    applied = False
    out_high = float(high_px)
    out_low = float(low_px)
    if isinstance(payload, dict) and bool(payload.get("enabled", False)):
        cur = _daily_current_row(prices, instrument_key)
        if cur is not None and not cur.empty:
            try:
                ctx = cur.iloc[-1].to_dict()
            except Exception:
                ctx = {}
            ctx["pred_center"] = 0.5 * (float(out_high) + float(out_low))
            ctx["pred_range"] = max(0.01, float(out_high) - float(out_low))
            center_val = _predict_linear_map(payload.get("center_model"), ctx)
            log_range_val = _predict_linear_map(payload.get("range_model"), ctx)
            if center_val is not None and log_range_val is not None:
                adj_range = max(0.01, float(np.exp(log_range_val)))
                out_high = float(center_val + (0.5 * adj_range))
                out_low = float(center_val - (0.5 * adj_range))
                applied = True

    structural = overlay.get("daily_pair_structural", {}) if isinstance(overlay, dict) else {}
    if isinstance(structural, dict) and structural:
        daily_hist = _load_long_daily_history(instrument_key)
        cur = _latest_structural_row(daily_hist, prices, _build_structural_daily_pair_dataset)
        if cur is not None:
            try:
                ctx = cur.to_dict()
            except Exception:
                ctx = {}
            ctx = _attach_current_news_context(pd.DatetimeIndex([prices.index.max()]), ctx, instrument_key)
            center_model = structural.get("center_model")
            range_model = structural.get("range_model")
            vol_edges = structural.get("vol_bucket_edges", []) if isinstance(structural, dict) else []
            close_edges = structural.get("close_loc_bucket_edges", []) if isinstance(structural, dict) else []
            if vol_edges and close_edges:
                bucket = _grid_bucket_label(ctx.get("vol_20"), list(vol_edges), ctx.get("close_loc"), list(close_edges))
                bucket_payload = (structural.get("bucket_models", {}) or {}).get(bucket, {})
                if isinstance(bucket_payload, dict):
                    center_model = bucket_payload.get("center_model", center_model)
                    range_model = bucket_payload.get("range_model", range_model)
            elif vol_edges:
                bucket = _bucket_label(ctx.get("vol_20"), list(vol_edges))
                bucket_payload = (structural.get("bucket_models", {}) or {}).get(bucket, {})
                if isinstance(bucket_payload, dict):
                    center_model = bucket_payload.get("center_model", center_model)
                    range_model = bucket_payload.get("range_model", range_model)
            center_val = _predict_linear_map(center_model, ctx)
            log_range_val = _predict_linear_map(range_model, ctx)
            close_val = _safe_float(ctx.get("close"))
            if center_val is not None and log_range_val is not None and close_val is not None and close_val > 0.0:
                rng = float(max(1e-4, np.exp(log_range_val)))
                direct_high = float(close_val * (1.0 + center_val + (0.5 * rng)))
                direct_low = float(close_val * (1.0 + center_val - (0.5 * rng)))
                try:
                    high_weight = float((structural.get("high_blend_weights", {}) or {}).get(model_key, 0.0) or 0.0)
                except Exception:
                    high_weight = 0.0
                try:
                    low_weight = float((structural.get("low_blend_weights", {}) or {}).get(model_key, 0.0) or 0.0)
                except Exception:
                    low_weight = 0.0
                if high_weight > 0.0:
                    out_high = float(((1.0 - high_weight) * out_high) + (high_weight * direct_high))
                    applied = True
                if low_weight > 0.0:
                    out_low = float(((1.0 - low_weight) * out_low) + (low_weight * direct_low))
                    applied = True

    if not applied:
        return None
    return {"pred_high": out_high, "pred_low": out_low}


def apply_horizon_overlay(prices: pd.DataFrame, instrument_key: str, horizon: str, model_key: str,
                          reference_price, pred_price) -> dict | None:
    overlay = load_horizon_overlay(instrument_key)
    horizon_map = overlay.get("horizons", {}) if isinstance(overlay, dict) else {}
    model_overlay = horizon_map.get(horizon, {}).get(model_key, {})
    ref_price = _safe_float(reference_price)
    pred_px = _safe_float(pred_price)
    if ref_price is None or ref_price <= 0.0 or pred_px is None:
        return None
    adjusted_price = float(pred_px)
    features = ()
    weight = 0.0
    metrics = {}
    applied = False
    pred_high_price = None
    pred_low_price = None
    if isinstance(model_overlay, dict) and bool(model_overlay.get("enabled", False)):
        features = tuple(str(x) for x in model_overlay.get("features", []))
        k = int(model_overlay.get("k", 0) or 0)
        min_train = int(model_overlay.get("min_train", 0) or 0)
        heur_ret = _current_heuristic_return(prices, instrument_key, horizon, features, k, min_train)
        if heur_ret is not None and np.isfinite(float(heur_ret)):
            weight = float(model_overlay.get("blend_weight", 0.0) or 0.0)
            model_ret = (adjusted_price / ref_price) - 1.0
            heur_price = ref_price * (1.0 + float(heur_ret))
            heur_ret_ref = (heur_price / ref_price) - 1.0
            blend_ret = ((1.0 - weight) * model_ret) + (weight * heur_ret_ref)
            calib = model_overlay.get("calibration", {})
            try:
                calibrated_ret = float(apply_return_calibration([blend_ret], calib)[0])
            except Exception:
                calibrated_ret = float(blend_ret)
            adjusted_price = float(ref_price * (1.0 + calibrated_ret))
            metrics = model_overlay.get("selected_metrics", {})
            applied = True
            weight = max(weight, float(model_overlay.get("blend_weight", 0.0) or 0.0))

    structural_note = None
    if str(horizon).lower() == "weekend":
        structural = overlay.get("weekend_structural", {}) if isinstance(overlay, dict) else {}
        if isinstance(structural, dict) and structural:
            if str(structural.get("variant")) in ("session_close_by_wday", "session_close_coarse3"):
                cur = _current_weekly_session_row(prices, instrument_key)
            else:
                daily_hist = _load_long_daily_history(instrument_key)
                cur = _latest_structural_row(daily_hist, prices, _build_structural_weekend_dataset)
            if cur is not None:
                try:
                    ctx = cur.to_dict()
                except Exception:
                    ctx = {}
                ctx = _attach_current_news_context(pd.DatetimeIndex([prices.index.max()]), ctx, instrument_key)
                pred_ret = None
                pred_high_ret = None
                pred_low_ret = None
                variant = str(structural.get("variant") or "daily_long")
                if variant in ("session_close_by_wday", "session_close_coarse3"):
                    try:
                        bucket = _week_bucket_label(ctx.get("wday", -1), coarse=(variant == "session_close_coarse3"))
                    except Exception:
                        bucket = "-1"
                    bucket_payload = (structural.get("bucket_models", {}) or {}).get(bucket, {})
                    if float(ctx.get("days_to_week_end", 1.0) or 0.0) <= 0.0:
                        pred_ret = 0.0
                    elif isinstance(bucket_payload, dict) and bucket_payload.get("identity_close"):
                        pred_ret = 0.0
                    else:
                        pred_ret = _predict_linear_map(bucket_payload.get("return_model") if isinstance(bucket_payload, dict) else None, ctx)
                        pred_high_ret = _predict_linear_map(bucket_payload.get("high_return_model") if isinstance(bucket_payload, dict) else None, ctx)
                        pred_low_ret = _predict_linear_map(bucket_payload.get("low_return_model") if isinstance(bucket_payload, dict) else None, ctx)
                else:
                    if float(ctx.get("days_to_week_end", 1.0) or 0.0) <= 0.0:
                        pred_ret = 0.0
                    else:
                        pred_ret = _predict_linear_map(structural.get("return_model"), ctx)
                        pred_high_ret = _predict_linear_map(structural.get("high_return_model"), ctx)
                        pred_low_ret = _predict_linear_map(structural.get("low_return_model"), ctx)
                close_val = _safe_float(ctx.get("close"))
                try:
                    structural_weight = float((structural.get("blend_weights", {}) or {}).get(model_key, 0.0) or 0.0)
                except Exception:
                    structural_weight = 0.0
                high_sofar = _safe_float(ctx.get("week_high_sofar"))
                low_sofar = _safe_float(ctx.get("week_low_sofar"))
                if close_val is not None and close_val > 0.0:
                    if float(ctx.get("days_to_week_end", 1.0) or 0.0) <= 0.0:
                        pred_high_price = float(high_sofar or close_val)
                        pred_low_price = float(low_sofar or close_val)
                        applied = True
                    else:
                        if pred_high_ret is not None:
                            pred_high_price = float(max(high_sofar or close_val, close_val * (1.0 + max(0.0, float(pred_high_ret)))))
                            applied = True
                        if pred_low_ret is not None:
                            pred_low_price = float(min(low_sofar or close_val, close_val * (1.0 + min(0.0, float(pred_low_ret)))))
                            applied = True
                if pred_ret is not None and close_val is not None and close_val > 0.0 and structural_weight > 0.0:
                    direct_price = float(close_val * (1.0 + pred_ret))
                    adjusted_price = float(((1.0 - structural_weight) * adjusted_price) + (structural_weight * direct_price))
                    metrics = structural.get("aggregate_metrics", {}) or metrics
                    structural_note = variant or "daily-long prior"
                    applied = True
                    weight = max(weight, structural_weight)

    if not applied:
        return None
    out = {
        "pred_price": float(adjusted_price),
        "blend_weight": float(weight),
        "window_ratio_mean": _safe_float(metrics.get("window_ratio_mean")),
        "window_hit_rate": _safe_float(metrics.get("window_hit_rate")),
        "objective_improvement": _safe_float(model_overlay.get("objective_improvement")) if isinstance(model_overlay, dict) else None,
        "features": list(features),
    }
    if pred_high_price is not None:
        out["pred_high_price"] = float(pred_high_price)
    if pred_low_price is not None:
        out["pred_low_price"] = float(pred_low_price)
    if structural_note:
        out["structural_note"] = structural_note
    return out
