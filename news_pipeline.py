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


def _news_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_{symbol.upper()}.csv")


def _state_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_state_{symbol.upper()}.json")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[A-Za-z]+", str(text or "").lower())


def score_headline_sentiment(headline: str) -> tuple[float, str]:
    toks = _tokenize(headline)
    if not toks:
        return 0.0, "neutral"
    pos = sum(1 for t in toks if t in _POS_WORDS)
    neg = sum(1 for t in toks if t in _NEG_WORDS)
    denom = max(1, pos + neg)
    score = float((pos - neg) / denom)
    if score > 0.15:
        return score, "positive"
    if score < -0.15:
        return score, "negative"
    return score, "neutral"


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
    score, label = score_headline_sentiment(title)
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
        "sentiment_score": float(score),
        "sentiment_label": label,
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
