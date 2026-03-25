from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone

import pandas as pd

from config import ALLOW_REMOTE_EVENTS, DATA_DIR, EVENTS_REFRESH_COOLDOWN_SEC

try:
    import yfinance as yf
except Exception:
    yf = None


def _cache_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"event_snapshot_{str(symbol or '').upper()}.json")


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
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _normalize_ts(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts
    except Exception:
        return None


def _extract_earnings_date(ticker) -> pd.Timestamp | None:
    try:
        cal = getattr(ticker, "calendar", None)
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for col in cal.columns:
                if "earn" in str(col).lower():
                    val = cal[col].iloc[0]
                    ts = _normalize_ts(val)
                    if ts is not None:
                        return ts
        if isinstance(cal, dict):
            for key, val in cal.items():
                if "earn" in str(key).lower():
                    ts = _normalize_ts(val)
                    if ts is not None:
                        return ts
    except Exception:
        pass
    try:
        frame = ticker.get_earnings_dates(limit=1)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            ts = _normalize_ts(frame.index[0])
            if ts is not None:
                return ts
    except Exception:
        pass
    return None


def _fetch_remote_event_snapshot(symbol: str) -> dict:
    if yf is None or not ALLOW_REMOTE_EVENTS:
        return {"status": "unavailable", "reason": "events_remote_disabled"}
    ticker = yf.Ticker(str(symbol).upper())
    earnings_ts = _extract_earnings_date(ticker)
    if earnings_ts is None:
        return {"status": "unavailable", "reason": "no_earnings_date"}
    now = pd.Timestamp.now(tz="UTC")
    dte = max(0.0, (earnings_ts - now).total_seconds() / 86400.0)
    return {
        "status": "ok",
        "symbol": str(symbol).upper(),
        "next_earnings": earnings_ts.isoformat(),
        "days_to_earnings": float(dte),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def get_event_snapshot(symbol: str, max_age_sec: int | None = None) -> dict:
    cached = _load_cached_snapshot(symbol)
    age_limit = int(max_age_sec if max_age_sec is not None else EVENTS_REFRESH_COOLDOWN_SEC)
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
            return cached
    try:
        remote = _fetch_remote_event_snapshot(symbol)
    except Exception as exc:
        remote = {"status": "error", "reason": str(exc)}
    if isinstance(remote, dict):
        remote = dict(remote)
        remote.setdefault("fetched_at", datetime.now(timezone.utc).isoformat())
        if remote.get("status") == "ok":
            remote["source"] = "remote"
        _save_cached_snapshot(symbol, remote)
    if isinstance(remote, dict) and remote.get("status") == "ok":
        return remote
    if isinstance(cached, dict):
        cached = dict(cached)
        cached["source"] = "stale_cache"
        if isinstance(remote, dict) and remote.get("reason"):
            cached["refresh_error"] = remote.get("reason")
        return cached
    return remote if isinstance(remote, dict) else {"status": "unavailable"}
