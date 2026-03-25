from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA_DIR, DEFAULT_INSTRUMENT


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _long_daily_path(instrument_key: str) -> str:
    return f"{DATA_DIR}/prices_{str(instrument_key or DEFAULT_INSTRUMENT).upper()}_daily_long.csv"


def _load_long_daily_history(instrument_key: str) -> pd.DataFrame:
    path = _long_daily_path(instrument_key)
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.columns = [str(c).lower() for c in df.columns]
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def get_liquidity_snapshot(df: pd.DataFrame, instrument_key: str) -> dict:
    current_price = _safe_float(pd.to_numeric(df.get("close"), errors="coerce").iloc[-1]) if df is not None and not df.empty and "close" in df.columns else None
    spread_dollars = None
    spread_bps = None
    if df is not None and not df.empty and current_price not in (None, 0.0):
        try:
            if "ask" in df.columns and "bid" in df.columns:
                ask = _safe_float(df["ask"].iloc[-1])
                bid = _safe_float(df["bid"].iloc[-1])
                if ask is not None and bid is not None and ask > bid:
                    spread_dollars = float(ask - bid)
                    spread_bps = float((spread_dollars / current_price) * 10000.0)
        except Exception:
            pass

    daily_long = _load_long_daily_history(instrument_key)
    adv = None
    dollar_vol = None
    atr14 = None
    if daily_long is not None and not daily_long.empty and "close" in daily_long.columns:
        close = pd.to_numeric(daily_long["close"], errors="coerce")
        volume = pd.to_numeric(daily_long.get("volume"), errors="coerce").fillna(0.0)
        adv = _safe_float(volume.tail(20).mean())
        dollar_vol = _safe_float((close * volume).tail(20).mean())
        high = pd.to_numeric(daily_long.get("high"), errors="coerce")
        low = pd.to_numeric(daily_long.get("low"), errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr14 = _safe_float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) else None

    rel_vol = None
    if df is not None and not df.empty:
        try:
            session = df.copy()
            session["date"] = pd.to_datetime(session.index, utc=True, errors="coerce").date
            today = session["date"].iloc[-1]
            today_vol = _safe_float(pd.to_numeric(session.loc[session["date"] == today, "volume"], errors="coerce").sum())
            session_vol = session.groupby("date")["volume"].sum().tail(21)
            avg_session_vol = _safe_float(session_vol.iloc[:-1].mean()) if len(session_vol) >= 2 else None
            if today_vol is not None and avg_session_vol not in (None, 0.0):
                rel_vol = float(today_vol / avg_session_vol)
        except Exception:
            pass

    slippage_bps = None
    if spread_bps is not None:
        impact = 0.0
        if dollar_vol is not None:
            if dollar_vol < 10_000_000:
                impact += 12.0
            elif dollar_vol < 50_000_000:
                impact += 6.0
            elif dollar_vol < 150_000_000:
                impact += 3.0
            else:
                impact += 1.0
        else:
            impact += 5.0
        if rel_vol is not None and rel_vol < 0.8:
            impact += 2.0
        slippage_bps = float((0.5 * spread_bps) + impact)

    score_parts = []
    if spread_bps is not None:
        score_parts.append(max(0.0, min(1.0, 1.0 - (spread_bps / 18.0))))
    if dollar_vol is not None:
        score_parts.append(max(0.0, min(1.0, np.log10(max(1.0, dollar_vol)) / 9.0)))
    if rel_vol is not None:
        score_parts.append(max(0.0, min(1.0, rel_vol / 1.4)))
    tradability_score = float(np.mean(score_parts)) if score_parts else 0.45
    if tradability_score >= 0.72:
        label = "Good"
    elif tradability_score >= 0.55:
        label = "Fair"
    else:
        label = "Fragile"

    warnings = []
    if spread_bps is None:
        warnings.append("live spread unavailable")
    elif spread_bps >= 12.0:
        warnings.append("wide spread")
    if dollar_vol is not None and dollar_vol < 15_000_000:
        warnings.append("light dollar volume")
    if rel_vol is not None and rel_vol < 0.75:
        warnings.append("subpar participation")

    rows = [
        ("Spread", "-" if spread_dollars is None else f"${spread_dollars:.2f}", "-" if spread_bps is None else f"{spread_bps:.1f} bps"),
        ("Slippage Est", "-" if slippage_bps is None else f"{slippage_bps:.1f} bps", "Heuristic; no book depth"),
        ("ADV / $Vol", "-" if adv is None else f"{adv:,.0f}", "-" if dollar_vol is None else f"${dollar_vol/1_000_000:.1f}M"),
        ("Rel Volume", "-" if rel_vol is None else f"{rel_vol:.2f}x", f"ATR {_safe_float(atr14) or 0.0:.2f}"),
        ("Tradability", f"{int(round(100.0 * tradability_score))}/100", label),
        ("Depth Feed", "Not wired", "Real order book required"),
    ]
    return {
        "rows": rows,
        "tradability_score": tradability_score,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "warnings": warnings,
        "depth_supported": False,
        "orderflow_supported": False,
    }
