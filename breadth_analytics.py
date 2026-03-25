from __future__ import annotations

import contextlib
import io
import json
import os
import time

import numpy as np
import pandas as pd

from config import (
    ALLOW_REMOTE_BREADTH,
    BREADTH_REFRESH_COOLDOWN_SEC,
    DATA_DIR,
    DEFAULT_INSTRUMENT,
    INSTRUMENTS,
)
from market_context import load_context_daily_history

try:
    import yfinance as yf
except Exception:
    yf = None

_INDEX_TICKERS = ["SPY", "QQQ", "DIA", "IWM"]
_SECTOR_TICKERS = ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]
_VOL_TICKER = "^VIX"


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _instrument_meta(instrument_key: str | None) -> dict:
    key = str(instrument_key or DEFAULT_INSTRUMENT or "").upper()
    return INSTRUMENTS.get(key, {})


def _breadth_path() -> str:
    return os.path.join(DATA_DIR, "market_breadth_snapshot.json")


def _direct_cache_path() -> str:
    return os.path.join(DATA_DIR, "market_breadth_direct_cache.json")


def _cache_path() -> str:
    return os.path.join(DATA_DIR, "market_breadth_live.json")


def _load_json(path: str) -> dict | None:
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


def _save_json(path: str, payload: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _load_direct_breadth_snapshot() -> dict | None:
    return _load_json(_breadth_path())


def save_direct_breadth_snapshot(payload: dict, source: str = "external") -> bool:
    if not isinstance(payload, dict):
        return False
    out = dict(payload)
    out.setdefault("source", source)
    out.setdefault("fetched_at", pd.Timestamp.now(tz="UTC").isoformat())
    _save_json(_direct_cache_path(), out)
    return True


def _pct_text(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{100.0 * val:+.2f}%"


def _ratio_text(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{val:.2f}x"


def _time_text(value) -> str:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        age_h = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0)
        if age_h < 1.0:
            return f"{int(round(age_h * 60.0))}m ago"
        if age_h < 24.0:
            return f"{age_h:.1f}h ago"
        return f"{age_h / 24.0:.1f}d ago"
    except Exception:
        return "-"


def _context_metrics(instrument_key: str) -> dict:
    rows = {}
    meta = _instrument_meta(instrument_key)
    mapping = meta.get("context_daily_symbols", {}) if isinstance(meta, dict) else {}
    for role in ("market", "sector", "vol", "lead_fut", "rates"):
        sym = mapping.get(role)
        if not sym:
            continue
        hist = load_context_daily_history(str(instrument_key).upper(), role, allow_remote=False)
        if hist is None or hist.empty or "close" not in hist.columns:
            continue
        close = pd.to_numeric(hist["close"], errors="coerce").dropna()
        if len(close) < 2:
            continue
        rows[role] = {
            "symbol": str(sym),
            "ret_1": _safe_float(close.pct_change().iloc[-1]),
            "ret_5": _safe_float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else None,
            "level": _safe_float(close.iloc[-1]),
        }
    return rows


def _extract_last_returns(frame: pd.DataFrame, tickers: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    if frame is None or frame.empty:
        return out
    if isinstance(frame.columns, pd.MultiIndex):
        for ticker in tickers:
            try:
                close = pd.to_numeric(frame[(ticker, "Close")], errors="coerce").dropna()
            except Exception:
                close = pd.Series(dtype=float)
            out[ticker] = _safe_float(close.pct_change().iloc[-1]) if len(close) >= 2 else None
        return out
    for ticker in tickers:
        out[ticker] = None
    return out


def _fetch_live_partial_breadth() -> dict:
    if not ALLOW_REMOTE_BREADTH or yf is None:
        return {"status": "unavailable", "reason": "breadth_remote_disabled"}
    tickers = _INDEX_TICKERS + _SECTOR_TICKERS + [_VOL_TICKER]
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        frame = yf.download(
            tickers=" ".join(tickers),
            period="7d",
            interval="1d",
            progress=False,
            auto_adjust=False,
            threads=False,
            timeout=4,
            group_by="ticker",
        )
    rets = _extract_last_returns(frame, tickers)
    sector_vals = [v for k, v in rets.items() if k in _SECTOR_TICKERS and v is not None]
    index_vals = [v for k, v in rets.items() if k in _INDEX_TICKERS and v is not None]
    if not sector_vals and not index_vals:
        return {"status": "unavailable", "reason": "breadth_no_live_data"}
    sectors_up = sum(1 for v in sector_vals if v > 0)
    sectors_dn = sum(1 for v in sector_vals if v < 0)
    sector_participation = float(sectors_up / max(1, sectors_up + sectors_dn)) if (sectors_up + sectors_dn) else None
    indices_up = sum(1 for v in index_vals if v > 0)
    index_participation = float(indices_up / max(1, len(index_vals))) if index_vals else None
    vol_ret = rets.get(_VOL_TICKER)
    vol_support = None if vol_ret is None else (1.0 if vol_ret < 0 else 0.0)
    score_parts = [v for v in (sector_participation, index_participation, vol_support) if v is not None]
    risk_on_score = float(np.mean(score_parts)) if score_parts else 0.5
    leaders = sorted([(k, v) for k, v in rets.items() if k in _INDEX_TICKERS and v is not None], key=lambda x: x[1], reverse=True)
    leader_text = ", ".join(f"{sym} {_pct_text(val)}" for sym, val in leaders[:2]) if leaders else "-"
    return {
        "status": "ok",
        "fetched_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "risk_on_score": risk_on_score,
        "sector_up": sectors_up,
        "sector_down": sectors_dn,
        "sector_participation": sector_participation,
        "index_participation": index_participation,
        "vol_ret_1": vol_ret,
        "leader_text": leader_text,
        "index_returns": {k: v for k, v in rets.items() if k in _INDEX_TICKERS},
        "sector_returns": {k: v for k, v in rets.items() if k in _SECTOR_TICKERS},
        "provider_note": "Live market breadth proxy from major indices and sector ETFs",
    }


def _direct_snapshot_rows(payload: dict, source_kind: str) -> dict:
    adv = _safe_float(payload.get("advancers"))
    dec = _safe_float(payload.get("decliners"))
    unchanged = _safe_float(payload.get("unchanged"))
    up_vol = _safe_float(payload.get("up_volume"))
    down_vol = _safe_float(payload.get("down_volume"))
    total = None if adv is None or dec is None else adv + dec
    breadth_ratio = None if total in (None, 0.0) else adv / total
    updown_ratio = None if up_vol is None or down_vol in (None, 0.0) else up_vol / down_vol
    updown_share = None if up_vol is None or down_vol is None or (up_vol + down_vol) <= 0 else float(up_vol / (up_vol + down_vol))
    breadth_thrust = _safe_float(payload.get("breadth_thrust"))
    if breadth_thrust is None:
        breadth_thrust = breadth_ratio
    thrust_state = "Neutral"
    if breadth_thrust is not None:
        if breadth_thrust >= 0.62:
            thrust_state = "Strong"
        elif breadth_thrust <= 0.38:
            thrust_state = "Weak"
    risk_parts = [v for v in (breadth_ratio, updown_share, breadth_thrust) if v is not None]
    risk_on_score = float(np.mean(risk_parts)) if risk_parts else 0.5
    source = str(payload.get("source") or "external")
    as_of = payload.get("as_of") or payload.get("fetched_at")
    feed_value = "Direct" if source_kind == "direct" else "Direct cache"
    detail = f"{source} | {_time_text(as_of)}"
    rows = [
        ("Feed", feed_value, detail),
        ("A/D Issues", "-" if breadth_ratio is None else f"{100.0 * breadth_ratio:.0f}% up", f"{int(adv or 0)} adv / {int(dec or 0)} dec / {int(unchanged or 0)} unch"),
        ("Up/Down Volume", _ratio_text(updown_ratio), f"Up {up_vol or 0:.0f} / Down {down_vol or 0:.0f}"),
        ("Breadth Thrust", "-" if breadth_thrust is None else f"{100.0 * breadth_thrust:.0f}/100", f"{thrust_state} breadth impulse"),
        ("Risk-On Score", f"{int(round(100.0 * max(0.0, min(1.0, risk_on_score))))}/100", "Direct internals composite"),
    ]
    warnings = []
    if source_kind == "cache":
        warnings.append("using cached direct internals")
    return {
        "status": "direct",
        "rows": rows,
        "risk_on_score": risk_on_score,
        "direct_feed": True,
        "kind": "direct breadth",
        "source_kind": source_kind,
        "provider_note": f"Direct breadth internals ({source})",
        "as_of": as_of,
        "warnings": warnings,
        "breadth_thrust": breadth_thrust,
    }


def _live_rows(snapshot: dict, source_kind: str, refresh_error: str | None = None) -> dict:
    sector_up = int(snapshot.get("sector_up") or 0)
    sector_down = int(snapshot.get("sector_down") or 0)
    sector_participation = _safe_float(snapshot.get("sector_participation"))
    index_participation = _safe_float(snapshot.get("index_participation"))
    vol_ret = _safe_float(snapshot.get("vol_ret_1"))
    risk_on_score = max(0.0, min(1.0, _safe_float(snapshot.get("risk_on_score")) or 0.5))
    feed_label = "Live partial" if source_kind == "live" else "Cached live partial"
    feed_detail = "Real market data; not exchange A/D"
    if refresh_error:
        feed_detail = f"{feed_detail} | refresh failed"
    rows = [
        ("Feed", feed_label, feed_detail),
        ("Sector Participation", "-" if sector_participation is None else f"{100.0 * sector_participation:.0f}% up", f"{sector_up} up / {sector_down} down sector ETFs"),
        ("Major Index RS", "-" if index_participation is None else f"{100.0 * index_participation:.0f}% up", str(snapshot.get("leader_text") or "-")),
        ("Vol Pressure", _pct_text(vol_ret), "VIX change; lower usually supports risk-on"),
        ("Risk-On Score", f"{int(round(100.0 * risk_on_score))}/100", "Live partial breadth composite"),
    ]
    return {
        "status": "live_partial",
        "rows": rows,
        "risk_on_score": risk_on_score,
        "direct_feed": False,
        "kind": "live_partial breadth",
        "source_kind": source_kind,
        "provider_note": str(snapshot.get("provider_note") or feed_detail),
        "as_of": snapshot.get("fetched_at"),
        "warnings": [] if source_kind == "live" else [str(refresh_error or "live refresh unavailable")],
    }


def _proxy_snapshot(instrument_key: str) -> dict:
    ctx = _context_metrics(instrument_key)
    market = ctx.get("market", {})
    sector = ctx.get("sector", {})
    vol = ctx.get("vol", {})
    lead = ctx.get("lead_fut", {})
    rates = ctx.get("rates", {})

    signals = []
    if _safe_float(market.get("ret_1")) is not None:
        signals.append(1.0 if market["ret_1"] > 0 else 0.0)
    if _safe_float(market.get("ret_5")) is not None:
        signals.append(1.0 if market["ret_5"] > 0 else 0.0)
    sector_rel = None
    if _safe_float(sector.get("ret_1")) is not None and _safe_float(market.get("ret_1")) is not None:
        sector_rel = float(sector["ret_1"] - market["ret_1"])
        signals.append(1.0 if sector_rel > 0 else 0.0)
    if _safe_float(lead.get("ret_1")) is not None:
        signals.append(1.0 if lead["ret_1"] > 0 else 0.0)
    if _safe_float(vol.get("ret_1")) is not None:
        signals.append(1.0 if vol["ret_1"] < 0 else 0.0)
    if _safe_float(rates.get("ret_1")) is not None:
        signals.append(1.0 if rates["ret_1"] < 0 else 0.0)
    score = float(np.mean(signals)) if signals else 0.5
    if score >= 0.66:
        state = "Risk-on"
    elif score <= 0.34:
        state = "Risk-off"
    else:
        state = "Mixed"

    rows = [
        ("Feed", "Proxy only", "Context-derived; not direct A/D internals"),
        ("Proxy Breadth", f"{int(round(100.0 * score))}/100", f"{state} | context-derived, not A/D data"),
        ("Market / Sector", f"{_pct_text(market.get('ret_1'))} / {_pct_text(sector.get('ret_1'))}", f"Sector rel {_pct_text(sector_rel)}"),
        ("Lead / Vol", f"{_pct_text(lead.get('ret_1'))} / {_pct_text(vol.get('ret_1'))}", "Futures lead / vol pressure"),
        ("Rates Tilt", _pct_text(rates.get('ret_1')), "Rates ETF used as risk proxy"),
    ]
    return {
        "status": "proxy_only" if ctx else "unavailable",
        "rows": rows,
        "risk_on_score": score,
        "direct_feed": False,
        "kind": "proxy breadth",
        "source_kind": "proxy",
        "provider_note": "Context-derived breadth proxy",
        "warnings": [] if ctx else ["No breadth or context feed available"],
    }


def get_breadth_snapshot(instrument_key: str) -> dict:
    instrument_key = str(instrument_key or DEFAULT_INSTRUMENT).upper()
    direct = _load_direct_breadth_snapshot()
    if isinstance(direct, dict):
        return _direct_snapshot_rows(direct, source_kind="direct")

    direct_cache = _load_json(_direct_cache_path())
    if isinstance(direct_cache, dict):
        return _direct_snapshot_rows(direct_cache, source_kind="cache")

    cached = _load_json(_cache_path())
    cached_payload = cached if isinstance(cached, dict) else {}
    cached_age = None
    try:
        ts = pd.Timestamp(cached_payload.get("fetched_at"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        cached_age = max(0.0, time.time() - ts.timestamp())
    except Exception:
        cached_age = None
    cached_live = bool(cached_payload.get("status") == "ok" and isinstance(cached_payload.get("sector_returns"), dict))
    if cached_live and isinstance(cached_age, float) and cached_age <= float(BREADTH_REFRESH_COOLDOWN_SEC):
        return _live_rows(cached_payload, source_kind="cache")

    refresh_error = None
    try:
        remote = _fetch_live_partial_breadth()
    except Exception as exc:
        remote = {"status": "error", "reason": str(exc)}
    if isinstance(remote, dict) and remote.get("status") == "ok":
        _save_json(_cache_path(), remote)
        return _live_rows(remote, source_kind="live")
    if isinstance(remote, dict):
        refresh_error = str(remote.get("reason") or remote.get("status") or "breadth_remote_failed")

    if cached_live:
        return _live_rows(cached_payload, source_kind="cache", refresh_error=refresh_error)
    return _proxy_snapshot(instrument_key)
