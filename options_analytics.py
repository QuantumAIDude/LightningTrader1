from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import (
    ALLOW_REMOTE_OPTIONS,
    DATA_DIR,
    OPTIONS_IV_HISTORY_LOOKBACK,
    OPTIONS_PANEL_ENABLED,
    OPTIONS_REFRESH_COOLDOWN_SEC,
)

try:
    import yfinance as yf
except Exception:
    yf = None


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _cache_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"options_snapshot_{str(symbol or '').upper()}.json")


def _history_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"options_snapshot_history_{str(symbol or '').upper()}.csv")


def _json_default(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type {type(obj)!r}")


def _load_cached_snapshot(symbol: str) -> dict | None:
    path = _cache_path(symbol)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        return None
    return None


def _save_cached_snapshot(symbol: str, payload: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_cache_path(symbol), "w") as f:
            json.dump(payload, f, default=_json_default, indent=2, sort_keys=True)
    except Exception:
        pass


def _load_history(symbol: str) -> pd.DataFrame:
    path = _history_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty:
            return df
        if "fetched_at" in df.columns:
            df["fetched_at"] = pd.to_datetime(df["fetched_at"], utc=True, errors="coerce")
        return df.sort_values("fetched_at")
    except Exception:
        return pd.DataFrame()


def _append_history(symbol: str, payload: dict):
    current_iv = _safe_float(payload.get("atm_iv_mid"))
    exp_pct = _safe_float(payload.get("expected_move_pct"))
    fetched_at = payload.get("fetched_at")
    if current_iv is None or fetched_at is None:
        return
    row = pd.DataFrame(
        [
            {
                "fetched_at": pd.Timestamp(fetched_at).isoformat(),
                "atm_iv_mid": current_iv,
                "expected_move_pct": exp_pct,
                "source": payload.get("source"),
            }
        ]
    )
    hist = _load_history(symbol)
    try:
        if not hist.empty:
            if "fetched_at" in hist.columns:
                hist["fetched_at"] = pd.to_datetime(hist["fetched_at"], utc=True, errors="coerce")
            row_ts = pd.Timestamp(fetched_at)
            if row_ts.tzinfo is None:
                row_ts = row_ts.tz_localize("UTC")
            hist = hist[hist["fetched_at"] != row_ts]
            hist = pd.concat([hist, row], ignore_index=True)
        else:
            hist = row
        hist = hist.sort_values("fetched_at").tail(max(30, int(OPTIONS_IV_HISTORY_LOOKBACK)))
        hist.to_csv(_history_path(symbol), index=False)
    except Exception:
        pass


def _load_long_daily_history(symbol: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, f"prices_{str(symbol or '').upper()}_daily_long.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.columns = [str(c).lower() for c in df.columns]
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def _realized_vol20(symbol: str) -> float | None:
    df = _load_long_daily_history(symbol)
    if df.empty or "close" not in df.columns:
        return None
    close = pd.to_numeric(df["close"], errors="coerce").dropna()
    ret = close.pct_change().dropna()
    if len(ret) < 10:
        return None
    return _safe_float(ret.tail(20).std(ddof=0) * np.sqrt(252.0))


def _enrich_history_metrics(symbol: str, payload: dict) -> dict:
    out = dict(payload or {})
    call_iv = _safe_float(out.get("atm_call_iv"))
    put_iv = _safe_float(out.get("atm_put_iv"))
    atm_iv_mid = None
    if call_iv is not None and put_iv is not None:
        atm_iv_mid = float(0.5 * (call_iv + put_iv))
    elif call_iv is not None:
        atm_iv_mid = call_iv
    elif put_iv is not None:
        atm_iv_mid = put_iv
    out["atm_iv_mid"] = atm_iv_mid
    if out.get("status") == "ok":
        _append_history(symbol, out)
    hist = _load_history(symbol)
    if not hist.empty and "atm_iv_mid" in hist.columns and atm_iv_mid is not None:
        vals = pd.to_numeric(hist["atm_iv_mid"], errors="coerce").dropna().tail(max(30, int(OPTIONS_IV_HISTORY_LOOKBACK)))
        if len(vals) >= 5:
            vmin = float(vals.min())
            vmax = float(vals.max())
            vrange = max(1e-9, vmax - vmin)
            out["iv_rank"] = float(max(0.0, min(1.0, (atm_iv_mid - vmin) / vrange)))
            out["iv_percentile"] = float((vals <= atm_iv_mid).mean())
            out["history_points"] = int(len(vals))
    rv20 = _realized_vol20(symbol)
    out["rv20"] = rv20
    if rv20 not in (None, 0.0) and atm_iv_mid is not None:
        out["iv_vs_rv20"] = float(atm_iv_mid / rv20)
    source = str(out.get("source") or out.get("status") or "unavailable").lower()
    if source == "remote":
        out["source_kind"] = "live"
    elif source in {"cache", "stale_cache"}:
        out["source_kind"] = "cache"
    elif source == "disabled":
        out["source_kind"] = "placeholder"
    else:
        out["source_kind"] = source if source else "unavailable"
    return out


def _chain_frame(df: pd.DataFrame, side: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    for col in (
        "strike",
        "bid",
        "ask",
        "lastPrice",
        "volume",
        "openInterest",
        "impliedVolatility",
    ):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["side"] = str(side)
    mid = 0.5 * (out["bid"].fillna(0.0) + out["ask"].fillna(0.0))
    out["mid"] = np.where(mid > 0.0, mid, out["lastPrice"])
    out["notional_oi"] = out["strike"].fillna(0.0) * out["openInterest"].fillna(0.0)
    out["notional_vol"] = out["strike"].fillna(0.0) * out["volume"].fillna(0.0)
    return out


def _atm_row(frame: pd.DataFrame, spot_price: float | None) -> dict | None:
    if frame is None or frame.empty or spot_price is None:
        return None
    try:
        tmp = frame.copy()
        tmp["dist"] = (tmp["strike"] - float(spot_price)).abs()
        tmp = tmp.sort_values(["dist", "openInterest", "volume"], ascending=[True, False, False])
        if tmp.empty:
            return None
        row = tmp.iloc[0].to_dict()
        return row
    except Exception:
        return None


def _max_pain(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    if calls.empty and puts.empty:
        return None
    strikes = pd.concat([calls.get("strike", pd.Series(dtype=float)), puts.get("strike", pd.Series(dtype=float))], axis=0)
    strikes = pd.to_numeric(strikes, errors="coerce").dropna().sort_values().unique()
    if len(strikes) == 0:
        return None
    best_strike = None
    best_loss = None
    call_strike = pd.to_numeric(calls.get("strike"), errors="coerce").fillna(0.0).to_numpy(dtype=float) if not calls.empty else np.asarray([], dtype=float)
    call_oi = pd.to_numeric(calls.get("openInterest"), errors="coerce").fillna(0.0).to_numpy(dtype=float) if not calls.empty else np.asarray([], dtype=float)
    put_strike = pd.to_numeric(puts.get("strike"), errors="coerce").fillna(0.0).to_numpy(dtype=float) if not puts.empty else np.asarray([], dtype=float)
    put_oi = pd.to_numeric(puts.get("openInterest"), errors="coerce").fillna(0.0).to_numpy(dtype=float) if not puts.empty else np.asarray([], dtype=float)
    for strike in strikes:
        call_loss = np.maximum(0.0, strike - call_strike) * call_oi
        put_loss = np.maximum(0.0, put_strike - strike) * put_oi
        total = float(np.sum(call_loss) + np.sum(put_loss))
        if best_loss is None or total < best_loss:
            best_loss = total
            best_strike = float(strike)
    return best_strike


def _top_wall(frame: pd.DataFrame, field: str) -> float | None:
    if frame is None or frame.empty or field not in frame.columns:
        return None
    try:
        tmp = frame.dropna(subset=["strike"]).copy()
        if tmp.empty:
            return None
        grouped = tmp.groupby("strike")[field].sum().sort_values(ascending=False)
        if grouped.empty:
            return None
        return float(grouped.index[0])
    except Exception:
        return None


def _fetch_remote_snapshot(symbol: str, spot_price: float | None = None) -> dict:
    if not OPTIONS_PANEL_ENABLED:
        return {"status": "disabled"}
    if yf is None or not ALLOW_REMOTE_OPTIONS:
        return {"status": "unavailable", "reason": "options_remote_disabled"}
    ticker = yf.Ticker(str(symbol).upper())
    expiries = list(getattr(ticker, "options", []) or [])
    if not expiries:
        return {"status": "unavailable", "reason": "no_expiries"}
    expiry = expiries[0]
    chain = ticker.option_chain(expiry)
    calls = _chain_frame(chain.calls, "call")
    puts = _chain_frame(chain.puts, "put")
    if calls.empty and puts.empty:
        return {"status": "unavailable", "reason": "empty_chain", "expiry": expiry}
    if spot_price is None:
        for frame in (calls, puts):
            if not frame.empty:
                strike_median = _safe_float(frame["strike"].median())
                if strike_median is not None:
                    spot_price = strike_median
                    break
    atm_call = _atm_row(calls, spot_price)
    atm_put = _atm_row(puts, spot_price)
    call_mid = _safe_float(atm_call.get("mid")) if atm_call else None
    put_mid = _safe_float(atm_put.get("mid")) if atm_put else None
    atm_straddle = None
    if call_mid is not None and put_mid is not None:
        atm_straddle = float(max(0.0, call_mid) + max(0.0, put_mid))
    total_call_vol = _safe_float(calls["volume"].fillna(0.0).sum()) if not calls.empty else 0.0
    total_put_vol = _safe_float(puts["volume"].fillna(0.0).sum()) if not puts.empty else 0.0
    total_call_oi = _safe_float(calls["openInterest"].fillna(0.0).sum()) if not calls.empty else 0.0
    total_put_oi = _safe_float(puts["openInterest"].fillna(0.0).sum()) if not puts.empty else 0.0
    try:
        exp_ts = pd.Timestamp(expiry).tz_localize("UTC")
        dte = max(0, int(np.ceil((exp_ts - pd.Timestamp.now(tz="UTC")).total_seconds() / 86400.0)))
    except Exception:
        dte = None
    out = {
        "status": "ok",
        "symbol": str(symbol).upper(),
        "expiry": expiry,
        "days_to_expiry": dte,
        "spot_price": _safe_float(spot_price),
        "atm_call_strike": _safe_float(atm_call.get("strike")) if atm_call else None,
        "atm_put_strike": _safe_float(atm_put.get("strike")) if atm_put else None,
        "atm_call_mid": call_mid,
        "atm_put_mid": put_mid,
        "atm_straddle_mid": atm_straddle,
        "expected_move_dollars": atm_straddle,
        "expected_move_pct": float(atm_straddle / float(spot_price)) if atm_straddle is not None and _safe_float(spot_price) not in (None, 0.0) else None,
        "atm_call_iv": _safe_float(atm_call.get("impliedVolatility")) if atm_call else None,
        "atm_put_iv": _safe_float(atm_put.get("impliedVolatility")) if atm_put else None,
        "iv_skew_put_minus_call": (
            float(_safe_float(atm_put.get("impliedVolatility")) - _safe_float(atm_call.get("impliedVolatility")))
            if atm_call and atm_put and _safe_float(atm_call.get("impliedVolatility")) is not None and _safe_float(atm_put.get("impliedVolatility")) is not None
            else None
        ),
        "put_call_volume_ratio": float(total_put_vol / total_call_vol) if total_call_vol not in (None, 0.0) else None,
        "put_call_oi_ratio": float(total_put_oi / total_call_oi) if total_call_oi not in (None, 0.0) else None,
        "call_wall_oi": _top_wall(calls, "openInterest"),
        "put_wall_oi": _top_wall(puts, "openInterest"),
        "call_wall_vol": _top_wall(calls, "volume"),
        "put_wall_vol": _top_wall(puts, "volume"),
        "max_pain": _max_pain(calls, puts),
        "total_call_volume": total_call_vol,
        "total_put_volume": total_put_vol,
        "total_call_oi": total_call_oi,
        "total_put_oi": total_put_oi,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }
    return out


def get_options_snapshot(symbol: str, spot_price: float | None = None, max_age_sec: int | None = None) -> dict:
    if not OPTIONS_PANEL_ENABLED:
        return {"status": "disabled"}
    cached = _load_cached_snapshot(symbol)
    age_limit = int(max_age_sec if max_age_sec is not None else OPTIONS_REFRESH_COOLDOWN_SEC)
    now_ts = time.time()
    if isinstance(cached, dict):
        try:
            fetched_at = pd.Timestamp(cached.get("fetched_at"))
            if fetched_at.tzinfo is None:
                fetched_at = fetched_at.tz_localize("UTC")
            age_sec = max(0.0, now_ts - fetched_at.timestamp())
        except Exception:
            age_sec = None
        if age_sec is not None and age_sec <= float(age_limit):
            cached = dict(cached)
            cached["source"] = "cache" if cached.get("status") == "ok" else "cache_failure"
            return _enrich_history_metrics(symbol, cached)
    try:
        remote = _fetch_remote_snapshot(symbol, spot_price=spot_price)
    except Exception as exc:
        remote = {"status": "error", "reason": str(exc)}
    if isinstance(remote, dict) and remote.get("status") == "ok":
        remote["source"] = "remote"
        _save_cached_snapshot(symbol, remote)
        return _enrich_history_metrics(symbol, remote)
    if isinstance(remote, dict):
        remote = dict(remote)
        remote.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
        _save_cached_snapshot(symbol, remote)
    if isinstance(cached, dict):
        cached = dict(cached)
        cached["source"] = "stale_cache"
        if isinstance(remote, dict) and remote.get("reason"):
            cached["refresh_error"] = remote.get("reason")
        return _enrich_history_metrics(symbol, cached)
    return _enrich_history_metrics(symbol, remote if isinstance(remote, dict) else {"status": "unavailable"})
