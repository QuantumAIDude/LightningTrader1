#!/usr/bin/env python3
"""
news_pipeline.py

QQQ-focused headline ingestion + sentiment features.
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf

from config import DATA_DIR, INSTRUMENTS, NEWS_PIPELINE_ENABLED, NEWS_PIPELINE_SYMBOL, NEWS_REFRESH_COOLDOWN_SEC

logger = logging.getLogger(__name__)

_POS_WORDS = {
    "beat", "beats", "surge", "surges", "record", "gain", "gains", "upgrade", "upgrades",
    "strong", "growth", "rally", "rallies", "bullish", "outperform", "innovate", "innovation",
    "expands", "expansion", "profit", "profits", "optimistic", "momentum", "upside",
}
_NEG_WORDS = {
    "miss", "misses", "drop", "drops", "weak", "downgrade", "downgrades", "lawsuit", "probe",
    "investigation", "risk", "risks", "bearish", "decline", "declines", "selloff", "cut", "cuts",
    "warning", "warns", "slowdown", "loss", "losses", "headwind", "volatile", "uncertain",
}
_NEGATORS = {"no", "not", "never", "without", "hardly"}
_INTENSIFIERS = {"strong", "sharply", "materially", "significantly", "record", "major"}

# Tech relevance signals for QQQ-context headlines.
_TECH_KEYWORDS = {
    "ai", "semiconductor", "chip", "chips", "cloud", "software", "hardware", "datacenter",
    "cybersecurity", "internet", "saas", "platform", "foundry", "gpu", "cpu", "bigtech", "nasdaq",
    "technology", "tech",
}
_TECH_ENTITIES = {
    "apple", "microsoft", "nvidia", "amazon", "alphabet", "google", "meta", "tesla",
    "amd", "intel", "broadcom", "qualcomm", "adobe", "oracle", "salesforce", "netflix", "qqq",
}

_POS_PHRASES = {
    "beats estimates": 1.25,
    "beat estimates": 1.15,
    "raises guidance": 1.20,
    "raise guidance": 1.10,
    "strong demand": 0.95,
    "record revenue": 1.15,
    "expands margin": 0.85,
    "buyback": 0.65,
}
_NEG_PHRASES = {
    "misses estimates": -1.25,
    "missed estimates": -1.20,
    "cuts guidance": -1.20,
    "cut guidance": -1.10,
    "antitrust probe": -1.10,
    "regulatory probe": -0.95,
    "weak demand": -0.95,
    "supply constraints": -0.70,
    "profit warning": -1.15,
}


def _news_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_{symbol.upper()}.csv")


def _state_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_state_{symbol.upper()}.json")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", str(text or "").lower())


def _phrase_score(text_norm: str) -> float:
    score = 0.0
    for phrase, w in _POS_PHRASES.items():
        if phrase in text_norm:
            score += float(w)
    for phrase, w in _NEG_PHRASES.items():
        if phrase in text_norm:
            score += float(w)
    return float(score)


def _tech_relevance(headline: str, toks: list[str]) -> float:
    text = str(headline or "").lower()
    hit_terms = sum(1 for t in toks if t in _TECH_KEYWORDS or t in _TECH_ENTITIES)
    entity_hits = sum(1 for e in _TECH_ENTITIES if e in text)
    if hit_terms == 0 and entity_hits == 0:
        return 0.0
    base = min(1.0, 0.28 * float(hit_terms) + 0.30 * float(entity_hits))
    # Headlines that mention the ETF explicitly are usually relevant.
    if "qqq" in text or "nasdaq 100" in text:
        base = min(1.0, base + 0.35)
    return float(max(0.0, min(1.0, base)))


def _headline_sentiment_detail(headline: str) -> dict:
    toks = _tokenize(headline)
    if not toks:
        return {"score": 0.0, "label": "neutral", "relevance": 0.0, "tech_relevant": False}

    text_norm = " ".join(toks)
    score_raw = _phrase_score(text_norm)
    for i, tok in enumerate(toks):
        weight = 0.0
        if tok in _POS_WORDS:
            weight = 1.0
        elif tok in _NEG_WORDS:
            weight = -1.0
        if weight == 0.0:
            continue
        prev = toks[i - 1] if i > 0 else ""
        prev2 = toks[i - 2] if i > 1 else ""
        if prev in _NEGATORS or prev2 in _NEGATORS:
            weight *= -0.8
        if prev in _INTENSIFIERS or prev2 in _INTENSIFIERS:
            weight *= 1.2
        score_raw += weight

    norm = max(2.0, float(len(toks)) ** 0.65)
    score = float(score_raw / norm)
    relevance = _tech_relevance(headline, toks)
    if relevance <= 0.0:
        score = 0.0
    else:
        # Downweight weakly relevant stories.
        score *= (0.45 + 0.55 * relevance)
    score = float(max(-1.0, min(1.0, score)))
    if score > 0.15:
        label = "positive"
    elif score < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return {
        "score": float(score),
        "label": label,
        "relevance": float(relevance),
        "tech_relevant": bool(relevance >= 0.25),
    }


def score_headline_sentiment(headline: str) -> tuple[float, str]:
    detail = _headline_sentiment_detail(headline)
    return float(detail["score"]), str(detail["label"])


def _parse_item(item: dict, symbol: str) -> dict | None:
    if not isinstance(item, dict):
        return None
    base = item
    if isinstance(item.get("content"), dict):
        base = item["content"]
    title = base.get("title") or item.get("title")
    if not title:
        return None
    provider = base.get("provider", {}) if isinstance(base.get("provider"), dict) else {}
    source = (
        base.get("publisher")
        or provider.get("displayName")
        or item.get("publisher")
        or "unknown"
    )
    canonical = base.get("canonicalUrl", {}) if isinstance(base.get("canonicalUrl"), dict) else {}
    url = canonical.get("url") or base.get("link") or item.get("link")
    raw_ts = (
        base.get("providerPublishTime")
        or base.get("pubDate")
        or item.get("providerPublishTime")
        or item.get("pubDate")
    )
    ts = None
    try:
        if raw_ts is not None and str(raw_ts).isdigit():
            ts = pd.to_datetime(int(raw_ts), unit="s", utc=True, errors="coerce")
        else:
            ts = pd.to_datetime(raw_ts, utc=True, errors="coerce")
    except Exception:
        ts = pd.NaT
    if pd.isna(ts):
        ts = pd.Timestamp.now(tz="UTC")
    sent = _headline_sentiment_detail(title)
    uid = str(base.get("id") or item.get("id") or item.get("uuid") or "") or None
    if uid is None:
        uid = f"{ts.isoformat()}|{title[:120]}"
    return {
        "symbol": symbol.upper(),
        "headline_id": uid,
        "published_at": ts.isoformat(),
        "headline": str(title),
        "source": str(source),
        "url": str(url) if url else None,
        "sentiment_score": float(sent["score"]),
        "sentiment_label": str(sent["label"]),
        "relevance_score": float(sent["relevance"]),
        "tech_relevant": int(1 if sent["tech_relevant"] else 0),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_existing_news(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
        return df
    except Exception:
        logger.exception("Failed reading news file %s", path)
        return pd.DataFrame()


def _save_news(path: str, df: pd.DataFrame):
    if df is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def fetch_latest_headlines(symbol: str = "QQQ", max_items: int = 200) -> list[dict]:
    if not NEWS_PIPELINE_ENABLED:
        return []
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception:
        logger.debug("News fetch failed for %s", symbol, exc_info=True)
        return []
    out = []
    for item in raw[:max_items]:
        parsed = _parse_item(item, symbol=symbol)
        if parsed:
            out.append(parsed)
    return out


def update_news_store(symbol: str = "QQQ", max_items: int = 200) -> dict:
    symbol = str(symbol or "QQQ").upper()
    path = _news_path(symbol)
    current = _load_existing_news(path)
    rows = fetch_latest_headlines(symbol=symbol, max_items=max_items)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return {"symbol": symbol, "added": 0, "rows": int(len(current))}
    incoming["published_at"] = pd.to_datetime(incoming["published_at"], utc=True, errors="coerce")
    if current is None or current.empty:
        merged = incoming
    else:
        merged = pd.concat([current, incoming], axis=0, ignore_index=True, sort=False)
    if "headline_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["headline_id"], keep="last")
    else:
        merged = merged.drop_duplicates(subset=["published_at", "headline"], keep="last")
    merged = merged.sort_values("published_at")
    _save_news(path, merged)
    added = max(0, int(len(merged) - len(current)))
    return {"symbol": symbol, "added": added, "rows": int(len(merged))}


def _load_state(symbol: str) -> dict:
    path = _state_path(symbol)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(symbol: str, payload: dict):
    path = _state_path(symbol)
    tmp = f"{path}.tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, path)
    except Exception:
        logger.debug("Failed writing news state", exc_info=True)


def maybe_refresh_news(symbol: str = None) -> dict:
    symbol = str(symbol or NEWS_PIPELINE_SYMBOL).upper()
    if not NEWS_PIPELINE_ENABLED:
        return {"symbol": symbol, "skipped": True, "reason": "disabled"}
    st = _load_state(symbol)
    now = pd.Timestamp.now(tz="UTC")
    try:
        last = pd.to_datetime(st.get("last_refresh"), utc=True, errors="coerce")
    except Exception:
        last = pd.NaT
    if pd.notna(last):
        age = (now - last).total_seconds()
        if age < max(60, int(NEWS_REFRESH_COOLDOWN_SEC)):
            return {"symbol": symbol, "skipped": True, "reason": "cooldown"}
    out = update_news_store(symbol=symbol, max_items=200)
    st["last_refresh"] = now.isoformat()
    st["last_rows"] = out.get("rows")
    _save_state(symbol, st)
    return out


def load_news_features_for_index(
    symbol: str,
    index_utc: pd.DatetimeIndex,
    tz: str = "America/New_York",
) -> pd.DataFrame:
    """
    Return lag-safe daily news features aligned to index_utc.
    Features are based on prior days only (shifted by 1 day).
    """
    cols = [
        "news_sent_mean_d1",
        "news_sent_sum_d3",
        "news_headline_count_d1",
        "news_pos_ratio_d3",
        "news_neg_ratio_d3",
    ]
    if index_utc is None or len(index_utc) == 0:
        return pd.DataFrame(columns=cols, index=index_utc)
    path = _news_path(symbol.upper())
    if not os.path.exists(path):
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.debug("Failed to read news store for features", exc_info=True)
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    if df.empty or "published_at" not in df.columns:
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    if df.empty:
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    df["sentiment_score"] = pd.to_numeric(df.get("sentiment_score"), errors="coerce").fillna(0.0)
    df["sentiment_label"] = df.get("sentiment_label", "neutral").astype(str).str.lower()
    if "relevance_score" in df.columns:
        df["relevance_score"] = pd.to_numeric(df.get("relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        df["relevance_score"] = 0.0
    # Backward compatibility: infer relevance when old rows lack relevance annotations.
    if float(df["relevance_score"].abs().sum()) == 0.0 and "headline" in df.columns:
        try:
            df["relevance_score"] = df["headline"].astype(str).apply(
                lambda h: float(_headline_sentiment_detail(h)["relevance"])
            ).clip(0.0, 1.0)
        except Exception:
            pass
    if "tech_relevant" in df.columns:
        df["tech_relevant"] = pd.to_numeric(df.get("tech_relevant"), errors="coerce").fillna(0).astype(int)
    else:
        df["tech_relevant"] = (df["relevance_score"] >= 0.25).astype(int)
    # Keep only relevant stories for QQQ signal extraction.
    df = df[(df["tech_relevant"] > 0) | (df["relevance_score"] >= 0.25)].copy()
    if df.empty:
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    # Relevance-weighted score reduces noise from loosely related articles.
    df["sentiment_score"] = df["sentiment_score"] * (0.5 + 0.5 * df["relevance_score"])
    local_dt = df["published_at"].dt.tz_convert(tz)
    df["news_date"] = pd.to_datetime(local_dt.dt.date)
    grouped = df.groupby("news_date", sort=True)
    daily = pd.DataFrame(index=grouped.size().index)
    daily["count"] = grouped.size().astype(float)
    daily["score_mean"] = grouped["sentiment_score"].mean()
    daily["score_sum"] = grouped["sentiment_score"].sum()
    daily["pos_count"] = grouped["sentiment_label"].apply(lambda s: float((s == "positive").sum()))
    daily["neg_count"] = grouped["sentiment_label"].apply(lambda s: float((s == "negative").sum()))
    daily = daily.sort_index()
    # Expand to a full daily calendar so lag features remain valid on no-news days.
    try:
        idx_local = pd.to_datetime(index_utc, utc=True, errors="coerce").tz_convert(tz)
        idx_dates = pd.to_datetime(idx_local.date)
        full_dates = pd.date_range(
            start=min(daily.index.min(), idx_dates.min()),
            end=max(daily.index.max(), idx_dates.max()),
            freq="D",
        )
        daily = daily.reindex(full_dates).fillna(0.0)
    except Exception:
        pass
    # Lag all features by one day to avoid peeking at same-day headlines in training labels.
    lag = pd.DataFrame(index=daily.index)
    lag["news_sent_mean_d1"] = daily["score_mean"].shift(1).fillna(0.0)
    lag["news_headline_count_d1"] = daily["count"].shift(1).fillna(0.0)
    lag["news_sent_sum_d3"] = daily["score_sum"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    pos_roll = daily["pos_count"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    neg_roll = daily["neg_count"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    den = (pos_roll + neg_roll).replace(0.0, np.nan)
    lag["news_pos_ratio_d3"] = (pos_roll / den).fillna(0.5)
    lag["news_neg_ratio_d3"] = (neg_roll / den).fillna(0.5)

    idx_local = pd.to_datetime(index_utc, utc=True, errors="coerce").tz_convert(tz)
    idx_dates = pd.to_datetime(idx_local.date)
    aligned = lag.reindex(idx_dates).fillna(0.0)
    aligned.index = index_utc
    for c in cols:
        if c not in aligned.columns:
            aligned[c] = 0.0
    return aligned[cols]
