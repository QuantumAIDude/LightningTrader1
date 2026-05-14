from __future__ import annotations

import contextlib
import io
import json
import os
import time
from urllib.request import Request, urlopen

import pandas as pd

from config import ALLOW_REMOTE_CATALYSTS, CATALYST_REFRESH_COOLDOWN_SEC, DATA_DIR, SEC_API_USER_AGENT
from news_pipeline import load_recent_headlines

try:
    import yfinance as yf
except Exception:
    yf = None

_SEC_SIGNIFICANT_FORMS = {"8-K", "10-K", "10-Q", "6-K", "S-3", "424B5", "13D", "SC 13G", "4"}
_SEC_FORM_PROFILE = {
    "8-K": {"severity_score": 0.80, "severity_label": "High", "risk_kind": "event risk", "ambiguity": "high"},
    "10-Q": {"severity_score": 0.48, "severity_label": "Medium", "risk_kind": "filing update", "ambiguity": "high"},
    "10-K": {"severity_score": 0.55, "severity_label": "Medium", "risk_kind": "filing update", "ambiguity": "high"},
    "6-K": {"severity_score": 0.50, "severity_label": "Medium", "risk_kind": "filing update", "ambiguity": "high"},
    "S-3": {"severity_score": 0.58, "severity_label": "Medium", "risk_kind": "financing risk", "ambiguity": "high"},
    "424B5": {"severity_score": 0.58, "severity_label": "Medium", "risk_kind": "offering risk", "ambiguity": "high"},
    "13D": {"severity_score": 0.52, "severity_label": "Medium", "risk_kind": "ownership change", "ambiguity": "medium"},
    "SC 13G": {"severity_score": 0.36, "severity_label": "Low", "risk_kind": "ownership update", "ambiguity": "medium"},
    "4": {"severity_score": 0.28, "severity_label": "Low", "risk_kind": "insider activity", "ambiguity": "high"},
}


def _safe_symbol(symbol: str) -> str:
    return str(symbol or "").upper()


def _provider_path(symbol: str, stem: str, ext: str) -> str:
    return os.path.join(DATA_DIR, f"{stem}_{_safe_symbol(symbol)}.{ext}")


def _cache_path(symbol: str, stem: str) -> str:
    return os.path.join(DATA_DIR, f"{stem}_{_safe_symbol(symbol)}_cache.json")


def _sec_company_cache_path() -> str:
    return os.path.join(DATA_DIR, "sec_company_tickers_cache.json")


def _load_json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _save_json(path: str, payload):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
    except Exception:
        pass


def _load_json_records(path: str) -> list[dict]:
    payload = _load_json(path)
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        items = payload.get("items") or payload.get("events") or payload.get("records")
        if isinstance(items, list):
            return [x for x in items if isinstance(x, dict)]
    return []


def _normalize_ts(value):
    try:
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts.isoformat()
    except Exception:
        return None


def _fetch_json_url(url: str, headers: dict | None = None, timeout: int = 8):
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _normalize_sec_record(record: dict, source: str = "local_file") -> dict | None:
    if not isinstance(record, dict):
        return None
    form = str(record.get("form") or record.get("title") or "").strip()
    ts = _normalize_ts(record.get("filed_at") or record.get("timestamp") or record.get("scheduled_at"))
    if not form or ts is None:
        return None
    profile = _SEC_FORM_PROFILE.get(form.upper(), {"severity_score": 0.35, "severity_label": "Low", "risk_kind": "filing update", "ambiguity": "high"})
    return {
        "type": "sec",
        "title": form,
        "scheduled_at": ts,
        "sentiment": record.get("sentiment"),
        "detail": str(record.get("detail") or record.get("headline") or ""),
        "source": str(record.get("source") or source),
        "severity_score": profile["severity_score"],
        "severity_label": profile["severity_label"],
        "risk_kind": profile["risk_kind"],
        "ambiguity": profile["ambiguity"],
    }


def _normalize_analyst_record(record: dict, source: str = "local_file") -> dict | None:
    if not isinstance(record, dict):
        return None
    action = str(record.get("action") or record.get("title") or "").strip()
    firm = str(record.get("firm") or "").strip()
    ts = _normalize_ts(record.get("published_at") or record.get("timestamp") or record.get("scheduled_at"))
    if not action or ts is None:
        return None
    sentiment = record.get("sentiment")
    if sentiment is None:
        text = action.lower()
        sentiment = "positive" if "up" in text or "buy" in text else "negative" if "down" in text or "sell" in text else "neutral"
    text = action.lower()
    if "down" in text or "downgrade" in text or "sell" in text or "underperform" in text:
        severity_score, severity_label, risk_kind, ambiguity = 0.42, "Medium", "analyst downgrade", "low"
    elif "up" in text or "upgrade" in text or "buy" in text or "outperform" in text:
        severity_score, severity_label, risk_kind, ambiguity = 0.34, "Low", "analyst upgrade", "low"
    elif "initiat" in text:
        severity_score, severity_label, risk_kind, ambiguity = 0.26, "Low", "coverage initiation", "medium"
    else:
        severity_score, severity_label, risk_kind, ambiguity = 0.22, "Low", "analyst action", "medium"
    return {
        "type": "analyst",
        "title": action,
        "scheduled_at": ts,
        "sentiment": sentiment,
        "detail": firm,
        "source": str(record.get("source") or source),
        "severity_score": severity_score,
        "severity_label": severity_label,
        "risk_kind": risk_kind,
        "ambiguity": ambiguity,
    }


def _normalize_social_record(record: dict, source: str = "local_file") -> dict | None:
    if not isinstance(record, dict):
        return None
    ts = _normalize_ts(record.get("as_of") or record.get("timestamp") or record.get("scheduled_at"))
    if ts is None:
        return None
    return {
        "type": "social",
        "title": "Social sentiment",
        "scheduled_at": ts,
        "sentiment": record.get("label"),
        "detail": f"score={record.get('score')}",
        "source": str(record.get("source") or source),
        "severity_score": 0.14,
        "severity_label": "Low",
        "risk_kind": "social chatter",
        "ambiguity": "high",
    }


def _normalize_news_meta_record(record: dict) -> dict | None:
    if not isinstance(record, dict):
        return None
    ts = _normalize_ts(record.get("published_at") or record.get("timestamp"))
    headline = str(record.get("headline") or "").strip()
    if ts is None or not headline:
        return None
    detail_bits = []
    if record.get("source"):
        detail_bits.append(str(record.get("source")))
    event_label = str(record.get("event_label") or record.get("event_type") or "headline").strip()
    if event_label:
        detail_bits.append(event_label)
    pol = record.get("political_relevance")
    try:
        if pol is not None:
            detail_bits.append(f"pol {float(pol):.2f}")
    except Exception:
        pass
    importance = 0.0
    impact = 0.0
    try:
        importance = float(record.get("importance_score") or 0.0)
    except Exception:
        importance = 0.0
    try:
        impact = float(record.get("impact_score") or 0.0)
    except Exception:
        impact = 0.0
    try:
        novelty = float(record.get("novelty_score") or 1.0)
    except Exception:
        novelty = 1.0
    sev_score = max(0.14, min(0.82, (0.18 + (0.72 * max(importance, 0.65 * impact))) * (0.85 + 0.15 * novelty)))
    if sev_score >= 0.55:
        severity_label = "High"
    elif sev_score >= 0.30:
        severity_label = "Medium"
    else:
        severity_label = "Low"
    ambiguity = "low" if sev_score >= 0.50 else "medium" if sev_score >= 0.28 else "high"
    risk_kind = f"{event_label.lower()} headline" if event_label else "headline risk"
    if record.get("signal_note"):
        detail_bits.append(str(record.get("signal_note"))[:46])
    return {
        "type": "news_meta",
        "title": headline[:72],
        "scheduled_at": ts,
        "sentiment": record.get("label"),
        "detail": " | ".join(detail_bits),
        "source": "news_store",
        "severity_score": sev_score,
        "severity_label": severity_label,
        "risk_kind": risk_kind,
        "ambiguity": ambiguity,
    }


def _load_cached_records(symbol: str, stem: str) -> tuple[list[dict], str | None]:
    payload = _load_json(_cache_path(symbol, stem))
    if not isinstance(payload, dict):
        return [], None
    fetched_at = payload.get("fetched_at")
    try:
        ts = pd.Timestamp(fetched_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age = max(0.0, time.time() - ts.timestamp())
        if age > float(CATALYST_REFRESH_COOLDOWN_SEC):
            return payload.get("records") or [], fetched_at
    except Exception:
        pass
    return payload.get("records") or [], fetched_at


def _cache_is_fresh(fetched_at: str | None) -> bool:
    try:
        ts = pd.Timestamp(fetched_at)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age = max(0.0, time.time() - ts.timestamp())
        return age <= float(CATALYST_REFRESH_COOLDOWN_SEC)
    except Exception:
        return False


def _save_cached_records(symbol: str, stem: str, records: list[dict]):
    _save_json(
        _cache_path(symbol, stem),
        {"fetched_at": pd.Timestamp.now(tz="UTC").isoformat(), "records": records},
    )


def _sec_company_map() -> dict[str, str]:
    payload = _load_json(_sec_company_cache_path())
    if not isinstance(payload, dict):
        if not ALLOW_REMOTE_CATALYSTS:
            return {}
        try:
            payload = _fetch_json_url(
                "https://www.sec.gov/files/company_tickers.json",
                headers={"User-Agent": SEC_API_USER_AGENT},
            )
            if isinstance(payload, dict):
                _save_json(_sec_company_cache_path(), payload)
        except Exception:
            return {}
    out = {}
    for item in (payload or {}).values():
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").upper()
        cik = item.get("cik_str")
        if ticker and cik is not None:
            out[ticker] = f"{int(cik):010d}"
    return out


def _fetch_sec_records_remote(symbol: str) -> list[dict]:
    if not ALLOW_REMOTE_CATALYSTS:
        return []
    cik = _sec_company_map().get(_safe_symbol(symbol))
    if not cik:
        return []
    payload = _fetch_json_url(
        f"https://data.sec.gov/submissions/CIK{cik}.json",
        headers={"User-Agent": SEC_API_USER_AGENT},
    )
    recent = (((payload or {}).get("filings") or {}).get("recent") or {})
    forms = list(recent.get("form") or [])
    dates = list(recent.get("filingDate") or [])
    accessions = list(recent.get("accessionNumber") or [])
    primary_docs = list(recent.get("primaryDocument") or [])
    out = []
    for idx, form in enumerate(forms):
        form_text = str(form or "").strip().upper()
        if not form_text:
            continue
        if form_text not in _SEC_SIGNIFICANT_FORMS and not form_text.startswith("8-K"):
            continue
        date_text = dates[idx] if idx < len(dates) else None
        ts = _normalize_ts(date_text)
        if ts is None:
            continue
        out.append(
            {
                "type": "sec",
                "title": form_text,
                "scheduled_at": ts,
                "sentiment": None,
                "detail": str(primary_docs[idx] if idx < len(primary_docs) else accessions[idx] if idx < len(accessions) else ""),
                "source": "sec_live",
            }
        )
        if len(out) >= 6:
            break
    return out


def _fetch_analyst_records_remote(symbol: str) -> list[dict]:
    if not ALLOW_REMOTE_CATALYSTS or yf is None:
        return []
    ticker = yf.Ticker(_safe_symbol(symbol))
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        frame = ticker.get_upgrades_downgrades(as_dict=False)
    if frame is None or frame.empty:
        return []
    rows = []
    for idx, row in frame.tail(8).iterrows():
        ts = _normalize_ts(getattr(idx, "isoformat", lambda: idx)())
        if ts is None:
            ts = _normalize_ts(row.get("GradeDate") or row.get("Date"))
        if ts is None:
            continue
        action_raw = str(row.get("Action") or row.get("action") or "").strip()
        to_grade = str(row.get("ToGrade") or row.get("toGrade") or "").strip()
        from_grade = str(row.get("FromGrade") or row.get("fromGrade") or "").strip()
        firm = str(row.get("Firm") or row.get("firm") or "").strip()
        action = action_raw or "Analyst change"
        if to_grade or from_grade:
            action = f"{action} {from_grade}->{to_grade}".strip()
        norm = _normalize_analyst_record(
            {
                "action": action,
                "firm": firm,
                "published_at": ts,
                "sentiment": "positive" if "up" in action.lower() or "buy" in to_grade.lower() else "negative" if "down" in action.lower() or "sell" in to_grade.lower() else "neutral",
                "source": "yfinance_live",
            },
            source="yfinance_live",
        )
        if norm is not None:
            rows.append(norm)
    rows.sort(key=lambda x: x["scheduled_at"], reverse=True)
    return rows[:6]


def _merge_records(*groups: list[dict]) -> list[dict]:
    merged: list[dict] = []
    seen = set()
    for group in groups:
        for item in group or []:
            if not isinstance(item, dict):
                continue
            key = (item.get("type"), item.get("title"), item.get("scheduled_at"))
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
    merged.sort(key=lambda x: x.get("scheduled_at") or "", reverse=True)
    return merged


def _provider_row(name: str, *, live: bool = False, cached: bool = False, local: bool = False, count: int = 0, note: str = "") -> tuple[str, str, str]:
    if live:
        status = "Live"
    elif cached:
        status = "Cached"
    elif local:
        status = "Local"
    else:
        status = "Stub"
    detail = note or (f"{count} records" if count else "Adapter ready")
    return (name, status, detail)


def _urgency_score(record: dict) -> float:
    try:
        ts = pd.Timestamp(record.get("scheduled_at"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_h = abs((pd.Timestamp.now(tz="UTC") - ts).total_seconds()) / 3600.0
    except Exception:
        return 0.35
    if age_h <= 6.0:
        return 1.0
    if age_h <= 24.0:
        return 0.75
    if age_h <= 48.0:
        return 0.45
    return 0.20


def _catalyst_age_text(record: dict) -> str:
    try:
        ts = pd.Timestamp(record.get("scheduled_at"))
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_h = abs((pd.Timestamp.now(tz="UTC") - ts).total_seconds()) / 3600.0
    except Exception:
        return "-"
    if age_h < 1.0:
        return f"{int(round(age_h * 60.0))}m ago"
    if age_h < 24.0:
        return f"{age_h:.1f}h ago"
    return f"{age_h / 24.0:.1f}d ago"


def _recent_risk_score(recent: list[dict]) -> float:
    now = pd.Timestamp.now(tz="UTC")
    score = 0.0
    fresh_count = 0
    for item in recent[:5]:
        try:
            ts = pd.Timestamp(item.get("scheduled_at"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
        except Exception:
            continue
        age_h = abs((now - ts).total_seconds()) / 3600.0
        if age_h > 48.0:
            continue
        fresh_count += 1
        sev = float(item.get("severity_score") or 0.0)
        urg = _urgency_score(item)
        score = max(score, sev * urg)
    if fresh_count >= 2:
        score += 0.05
    return float(max(0.0, min(1.0, score)))


def _active_catalyst(recent: list[dict]) -> dict | None:
    ranked = []
    for item in recent[:6]:
        sev = float(item.get("severity_score") or 0.0)
        urg = _urgency_score(item)
        ranked.append((sev * urg, item))
    ranked = [x for x in ranked if x[0] > 0]
    if not ranked:
        return None
    ranked.sort(key=lambda x: x[0], reverse=True)
    top = dict(ranked[0][1])
    top["risk_score"] = float(ranked[0][0])
    top["age_text"] = _catalyst_age_text(top)
    return top


def get_catalyst_provider_snapshot(symbol: str) -> dict:
    sym = _safe_symbol(symbol)

    local_sec = [_normalize_sec_record(x) for x in _load_json_records(_provider_path(sym, "sec_events", "json"))]
    local_analyst = [_normalize_analyst_record(x) for x in _load_json_records(_provider_path(sym, "analyst_actions", "json"))]
    local_social = [_normalize_social_record(x) for x in _load_json_records(_provider_path(sym, "social_sentiment", "json"))]
    local_sec = [x for x in local_sec if x is not None]
    local_analyst = [x for x in local_analyst if x is not None]
    local_social = [x for x in local_social if x is not None]

    cached_sec, sec_cached_at = _load_cached_records(sym, "sec_events_live")
    cached_analyst, analyst_cached_at = _load_cached_records(sym, "analyst_actions_live")
    cached_sec = [_normalize_sec_record(x, source="sec_cache") for x in cached_sec]
    cached_analyst = [_normalize_analyst_record(x, source="analyst_cache") for x in cached_analyst]
    cached_sec = [x for x in cached_sec if x is not None]
    cached_analyst = [x for x in cached_analyst if x is not None]

    remote_sec: list[dict] = []
    remote_analyst: list[dict] = []
    if ALLOW_REMOTE_CATALYSTS and not local_sec and not _cache_is_fresh(sec_cached_at):
        try:
            remote_sec = _fetch_sec_records_remote(sym)
            if remote_sec:
                _save_cached_records(sym, "sec_events_live", remote_sec)
        except Exception:
            remote_sec = []
    if ALLOW_REMOTE_CATALYSTS and not local_analyst and not _cache_is_fresh(analyst_cached_at):
        try:
            remote_analyst = _fetch_analyst_records_remote(sym)
            if remote_analyst:
                _save_cached_records(sym, "analyst_actions_live", remote_analyst)
        except Exception:
            remote_analyst = []

    sec_records = _merge_records(local_sec, remote_sec, cached_sec)
    analyst_records = _merge_records(local_analyst, remote_analyst, cached_analyst)
    news_meta = [_normalize_news_meta_record(x) for x in load_recent_headlines(sym, limit=6)]
    news_meta = [x for x in news_meta if x is not None]
    social_records = local_social

    provider_rows = [
        _provider_row(
            "SEC Feed",
            live=bool(remote_sec),
            cached=bool((not remote_sec) and cached_sec and not local_sec),
            local=bool(local_sec),
            count=len(sec_records),
        ),
        _provider_row(
            "Analyst Feed",
            live=bool(remote_analyst),
            cached=bool((not remote_analyst) and cached_analyst and not local_analyst),
            local=bool(local_analyst),
            count=len(analyst_records),
        ),
        _provider_row(
            "News Meta",
            local=bool(news_meta),
            count=len(news_meta),
            note=f"{len(news_meta)} recent headlines" if news_meta else "Backed by news store",
        ),
        _provider_row(
            "Social Feed",
            local=bool(social_records),
            count=len(social_records),
        ),
    ]
    recent = _merge_records(sec_records, analyst_records, news_meta, social_records)[:6]
    risk_score = _recent_risk_score(recent)
    active = _active_catalyst(recent)
    has_fresh = False
    now = pd.Timestamp.now(tz="UTC")
    for item in recent:
        try:
            ts = pd.Timestamp(item.get("scheduled_at"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            has_fresh = has_fresh or abs((now - ts).total_seconds()) <= 86400.0
        except Exception:
            continue
    status = "ok" if recent else "placeholder_only"
    if status == "placeholder_only" and (cached_sec or cached_analyst):
        status = "partial"
    risk_label = "Elevated" if risk_score >= 0.45 else "Watch" if risk_score >= 0.25 else "Low"
    rows = []
    if active is not None:
        rows.append(
            (
                "Active Catalyst",
                f"{str(active.get('title') or '-')[:18]} | {active.get('severity_label') or 'Low'}",
                f"{active.get('age_text') or '-'} | {active.get('risk_kind') or 'event risk'} | ambiguity {active.get('ambiguity') or 'high'}",
            )
        )
    rows.append(("Catalyst Risk", f"{int(round(100.0 * risk_score))}/100", f"{risk_label} | caution overlay, not direction"))
    rows.extend(provider_rows[:3])
    return {
        "status": status,
        "rows": rows,
        "provider_rows": provider_rows,
        "recent": recent,
        "catalyst_risk_score": risk_score,
        "has_fresh_catalyst": has_fresh,
        "active_catalyst": active,
        "risk_label": risk_label,
    }
