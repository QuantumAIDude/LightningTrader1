from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd

from config import (
    ALLOW_REMOTE_MACRO_EVENTS,
    DATA_DIR,
    MACRO_EVENTS_REFRESH_COOLDOWN_SEC,
    TRADING_ECONOMICS_CALENDAR_COUNTRY,
    TRADING_ECONOMICS_CALENDAR_KEY,
)

_HIGH = {"fed", "fomc", "cpi", "ppi", "nfp", "jobs", "payrolls", "gdp"}
_MED = {"treasury", "auction", "pce", "ism", "retail_sales", "consumer_confidence"}
_TARGET = _HIGH | _MED


def _calendar_path() -> str:
    return os.path.join(DATA_DIR, "macro_calendar.json")


def _cache_path() -> str:
    return os.path.join(DATA_DIR, "macro_calendar_cache.json")


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


def _severity_from_text(category: str, title: str, fallback: str | None = None) -> str:
    sev = str(fallback or "").lower().strip()
    if sev in {"high", "medium", "low"}:
        return sev
    text = f"{category} {title}".lower()
    if any(tok in text for tok in _HIGH):
        return "high"
    if any(tok in text for tok in _MED):
        return "medium"
    return "low"


def _is_target_event(title: str, category: str) -> bool:
    text = f"{title} {category}".lower()
    return any(tok in text for tok in _TARGET)


def _normalize_event(item: dict, source: str = "local_file") -> dict | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("title") or item.get("name") or "").strip()
    when_raw = item.get("scheduled_at") or item.get("timestamp") or item.get("when")
    if not title or when_raw is None:
        return None
    ts = _normalize_ts(when_raw)
    if ts is None:
        return None
    category = str(item.get("category") or item.get("event_code") or title.split()[0]).lower().replace(" ", "_")
    sev = _severity_from_text(category, title, fallback=item.get("severity") or item.get("importance"))
    if not _is_target_event(title, category):
        return None
    return {
        "id": str(item.get("id") or f"{category}:{ts.isoformat()}"),
        "event_type": "macro",
        "category": category,
        "title": title,
        "scheduled_at": ts.isoformat(),
        "severity": sev,
        "scope": str(item.get("scope") or "market"),
        "source": str(item.get("source") or source),
        "status": "scheduled",
        "heuristic": False,
    }


def _normalize_te_event(item: dict) -> dict | None:
    if not isinstance(item, dict):
        return None
    title = str(item.get("Event") or item.get("Title") or "").strip()
    category = str(item.get("Category") or item.get("Event") or "").lower().replace(" ", "_")
    ts = _normalize_ts(item.get("Date") or item.get("CalendarDate") or item.get("scheduled_at"))
    if not title or ts is None or not _is_target_event(title, category):
        return None
    importance = item.get("Importance")
    sev = None
    try:
        importance = int(importance)
        sev = "high" if importance >= 3 else "medium" if importance >= 2 else "low"
    except Exception:
        sev = None
    sev = _severity_from_text(category, title, fallback=sev)
    return {
        "id": str(item.get("CalendarId") or item.get("Ticker") or f"{category}:{ts.isoformat()}"),
        "event_type": "macro",
        "category": category,
        "title": title,
        "scheduled_at": ts.isoformat(),
        "severity": sev,
        "scope": str(item.get("Country") or "market"),
        "source": "tradingeconomics_live",
        "status": "scheduled",
        "heuristic": False,
    }


def load_macro_events() -> list[dict]:
    path = _calendar_path()
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            payload = json.load(f)
    except Exception:
        return []
    items = payload.get("events") if isinstance(payload, dict) else payload
    out = []
    if isinstance(items, list):
        for item in items:
            norm = _normalize_event(item)
            if norm is not None:
                out.append(norm)
    out.sort(key=lambda x: x["scheduled_at"])
    return out


def _load_cached_snapshot() -> dict | None:
    path = _cache_path()
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


def _save_cached_snapshot(payload: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_cache_path(), "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _fetch_remote_macro_events(lookahead_hours: int = 96) -> dict:
    if not ALLOW_REMOTE_MACRO_EVENTS:
        return {"status": "unavailable", "reason": "macro_remote_disabled"}
    now = pd.Timestamp.now(tz="UTC")
    end = now + pd.Timedelta(hours=max(int(lookahead_hours), 24 * 7))
    country = str(TRADING_ECONOMICS_CALENDAR_COUNTRY or "united states").strip()
    key = str(TRADING_ECONOMICS_CALENDAR_KEY or "guest:guest").strip()
    url = (
        "https://api.tradingeconomics.com/calendar/country/"
        f"{quote(country)}/{now.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"
        f"?c={quote(key, safe=':')}&f=json"
    )
    req = Request(url, headers={"User-Agent": "LightningTrader4000/1.0 macro-calendar"})
    with urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    items = payload if isinstance(payload, list) else []
    events = []
    for item in items:
        norm = _normalize_te_event(item)
        if norm is not None:
            events.append(norm)
    deduped = {}
    for evt in events:
        deduped[evt["id"]] = evt
    events = sorted(deduped.values(), key=lambda x: x["scheduled_at"])
    if not events:
        return {"status": "unavailable", "reason": "no_macro_events"}
    return {
        "status": "ok",
        "events": events,
        "provider_note": f"TradingEconomics live calendar ({country.title()})",
        "source_kind": "live",
        "source_name": "tradingeconomics",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _severity_weight(sev: str) -> float:
    sev = str(sev or "low").lower()
    return 1.0 if sev == "high" else 0.6 if sev == "medium" else 0.3


def _format_time_to_event(hours: float | None) -> str:
    if hours is None:
        return "-"
    if hours < 1.0:
        return f"{int(round(hours * 60.0))}m"
    if hours < 24.0:
        return f"{hours:.1f}h"
    return f"{hours / 24.0:.1f}d"


def _finalize_snapshot(
    events: list[dict],
    *,
    lookahead_hours: int,
    status: str,
    source_kind: str,
    provider_note: str,
    fetched_at: str | None = None,
    refresh_error: str | None = None,
) -> dict:
    now = pd.Timestamp.now(tz="UTC")
    horizon = now + pd.Timedelta(hours=int(lookahead_hours))
    future = []
    for evt in events or []:
        ts = _normalize_ts(evt.get("scheduled_at"))
        if ts is None:
            continue
        if now <= ts <= horizon:
            evt = dict(evt)
            hours = max(0.0, (ts - now).total_seconds() / 3600.0)
            evt["hours_to_event"] = float(hours)
            evt["time_to_event_text"] = _format_time_to_event(hours)
            future.append(evt)
    future.sort(key=lambda x: x["scheduled_at"])
    risk_score = 0.0
    for evt in future[:5]:
        sev_mult = _severity_weight(evt.get("severity"))
        hours = float(evt.get("hours_to_event") or 0.0)
        urgency = 1.0 if hours <= 24.0 else 0.6 if hours <= 48.0 else 0.3
        risk_score = max(risk_score, sev_mult * urgency)
    next_event = future[0] if future else None
    out = {
        "status": status,
        "events": future,
        "next_event": next_event,
        "risk_score": float(max(0.0, min(1.0, risk_score))),
        "high_risk_24h": bool(any(str(evt.get("severity")).lower() == "high" and float(evt.get("hours_to_event") or 999.0) <= 24.0 for evt in future)),
        "provider_note": provider_note,
        "source_kind": source_kind,
        "fetched_at": fetched_at,
    }
    if next_event is not None:
        out["time_to_next_hours"] = float(next_event.get("hours_to_event") or 0.0)
        out["time_to_next_text"] = str(next_event.get("time_to_event_text") or "-")
    if refresh_error:
        out["refresh_error"] = refresh_error
    return out


def maybe_refresh_macro_events(lookahead_hours: int = 96, force: bool = False) -> dict:
    local_events = load_macro_events()
    if local_events:
        try:
            fetched_at = datetime.fromtimestamp(os.path.getmtime(_calendar_path()), tz=timezone.utc).isoformat()
        except Exception:
            fetched_at = None
        return _finalize_snapshot(
            local_events,
            lookahead_hours=lookahead_hours,
            status="ok",
            source_kind="local",
            provider_note="Local macro calendar file",
            fetched_at=fetched_at,
        )

    cached = _load_cached_snapshot()
    cached_events = []
    cached_age = None
    if isinstance(cached, dict):
        cached_events = [evt for evt in (cached.get("events") or []) if isinstance(evt, dict)]
        ts = _normalize_ts(cached.get("fetched_at"))
        if ts is not None:
            cached_age = max(0.0, time.time() - ts.timestamp())

    if (not force) and isinstance(cached_age, float) and cached_age <= float(MACRO_EVENTS_REFRESH_COOLDOWN_SEC):
        return _finalize_snapshot(
            cached_events,
            lookahead_hours=lookahead_hours,
            status="ok" if cached_events else "placeholder_only",
            source_kind="cache",
            provider_note=str(cached.get("provider_note") or "Cached macro calendar"),
            fetched_at=cached.get("fetched_at"),
        )

    remote_error = None
    try:
        remote = _fetch_remote_macro_events(lookahead_hours=lookahead_hours)
    except Exception as exc:
        remote = {"status": "error", "reason": str(exc)}
    if isinstance(remote, dict) and remote.get("status") == "ok":
        payload = {
            "events": remote.get("events") or [],
            "provider_note": remote.get("provider_note") or "TradingEconomics live calendar",
            "fetched_at": remote.get("fetched_at") or datetime.now(timezone.utc).isoformat(),
        }
        _save_cached_snapshot(payload)
        return _finalize_snapshot(
            payload["events"],
            lookahead_hours=lookahead_hours,
            status="ok",
            source_kind="live",
            provider_note=str(payload["provider_note"]),
            fetched_at=payload["fetched_at"],
        )
    if isinstance(remote, dict):
        remote_error = str(remote.get("reason") or remote.get("status") or "remote_unavailable")

    if cached_events:
        note = str(cached.get("provider_note") or "Cached macro calendar")
        if remote_error:
            note = f"{note} | live refresh failed"
        return _finalize_snapshot(
            cached_events,
            lookahead_hours=lookahead_hours,
            status="ok",
            source_kind="cache",
            provider_note=note,
            fetched_at=cached.get("fetched_at"),
            refresh_error=remote_error,
        )

    return _finalize_snapshot(
        [],
        lookahead_hours=lookahead_hours,
        status="placeholder_only",
        source_kind="placeholder",
        provider_note="Macro calendar adapter ready; no live or local feed loaded",
        refresh_error=remote_error,
    )


def get_macro_event_snapshot(lookahead_hours: int = 96) -> dict:
    return maybe_refresh_macro_events(lookahead_hours=lookahead_hours, force=False)
