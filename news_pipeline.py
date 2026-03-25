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
import ssl
from datetime import datetime, timezone
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf
try:
    import certifi
except Exception:
    certifi = None

from config import (
    DATA_DIR,
    INSTRUMENTS,
    NEW_INSTRUMENTS_DIR,
    NEWS_ARCHIVE_CHUNK_DAYS,
    NEWS_ARCHIVE_LOOKBACK_DAYS,
    NEWS_ARCHIVE_MAX_RECORDS_PER_CHUNK,
    NEWS_LONG_HORIZON_MIN_ACTIVE_DAYS,
    NEWS_LONG_HORIZON_MIN_COVERAGE,
    NEWS_LONG_HORIZON_MIN_ROWS,
    NEWS_PIPELINE_ENABLED,
    NEWS_PIPELINE_SYMBOL,
    NEWS_REFRESH_COOLDOWN_SEC,
)

logger = logging.getLogger(__name__)


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)

_POS_WORDS = {
    "beat", "beats", "surge", "surges", "record", "gain", "gains", "upgrade", "upgrades",
    "strong", "growth", "rally", "rallies", "bullish", "outperform", "innovate", "innovation",
    "expands", "expansion", "profit", "profits", "optimistic", "momentum", "upside",
    "rise", "rises", "rising", "higher", "winning", "outperforming",
}
_NEG_WORDS = {
    "miss", "misses", "drop", "drops", "weak", "downgrade", "downgrades", "lawsuit", "probe",
    "investigation", "risk", "risks", "bearish", "decline", "declines", "selloff", "cut", "cuts",
    "warning", "warns", "slowdown", "loss", "losses", "headwind", "volatile", "uncertain",
    "lower", "fall", "falls", "falling", "weakness", "lagging", "underperforming",
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
_MARKET_KEYWORDS = {
    "etf", "etfs", "fund", "funds", "equity", "equities", "future", "futures", "market", "markets",
    "sector", "sectors", "stock", "stocks", "index", "indexes", "midday", "bell",
}
_MACRO_KEYWORDS = {
    "cpi", "inflation", "fed", "rate", "rates", "yield", "yields", "treasury", "treasuries",
    "employment", "jobs", "oil",
}
_POLITICAL_KEYWORDS = {
    "administration", "antitrust", "ban", "bans", "bill", "budget", "ceasefire", "china", "congress",
    "conflict", "crackdown", "debt", "defense", "doj", "election", "embargo", "eu", "europe",
    "executive", "export", "exports", "ftc", "geopolitical", "government", "house", "lawmakers",
    "ministry", "president", "regulation", "regulator", "regulators", "sanction", "sanctions",
    "sec", "senate", "shutdown", "stimulus", "subsidy", "tariff", "tariffs", "tax", "trade",
    "treasury", "war", "white", "whitehouse",
}
_POLITICAL_ENTITIES = {
    "white house", "u.s. senate", "u.s. house", "department of justice", "federal trade commission",
    "securities and exchange commission", "commerce department", "treasury department",
    "european union", "prime minister", "president", "congress", "senate", "house republicans",
}
_DIRECT_QQQ_PHRASES = {
    "qqq", "invesco qqq", "nasdaq 100", "nasdaq-100",
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
    "move higher": 0.80,
    "still winning": 0.70,
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
    "move lower": -0.80,
    "lower pre bell": -0.90,
    "forget qqq": -1.00,
    "sector weakness": -0.65,
    "outperforming tech": -0.80,
    "export controls": -1.05,
    "trade war": -1.10,
    "government shutdown": -1.10,
    "debt ceiling": -0.90,
    "new tariffs": -1.00,
    "tariff threat": -0.90,
    "regulatory crackdown": -1.00,
    "geopolitical tension": -0.85,
    "chip restrictions": -0.95,
    "antitrust suit": -1.00,
    "sanctions risk": -0.90,
    "policy uncertainty": -0.75,
}
_POLITICAL_POS_PHRASES = {
    "tariff relief": 1.00,
    "trade deal": 0.95,
    "ceasefire talks": 0.75,
    "funding bill": 0.55,
    "budget deal": 0.75,
    "tax relief": 0.70,
    "regulatory approval": 0.90,
    "subsidy support": 0.75,
    "policy support": 0.65,
}
_POLITICAL_NEG_PHRASES = {
    "export ban": -1.10,
    "export curb": -1.00,
    "export controls": -1.05,
    "tariff hike": -1.05,
    "new tariffs": -1.00,
    "trade restriction": -0.95,
    "trade war": -1.10,
    "government shutdown": -1.10,
    "debt ceiling standoff": -1.00,
    "antitrust probe": -1.05,
    "regulatory crackdown": -1.00,
    "sanctions escalation": -1.00,
    "geopolitical escalation": -0.95,
    "war escalation": -1.10,
}


def _news_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_{symbol.upper()}.csv")


def _state_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_state_{symbol.upper()}.json")


def _archive_query(symbol: str) -> str:
    sym = str(symbol or "").upper()
    meta = INSTRUMENTS.get(sym, {}) if isinstance(INSTRUMENTS, dict) else {}
    aliases = []
    for raw in [sym, meta.get("display_name"), meta.get("company_name")] + list(meta.get("news_aliases", []) or []):
        text = str(raw or "").strip()
        if text:
            aliases.append(text)
    deduped = []
    seen = set()
    for item in aliases:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    if sym == "QQQ":
        return '("invesco qqq" OR "nasdaq 100" OR "nasdaq-100" OR qqq OR nvidia OR microsoft OR apple OR amazon OR alphabet OR meta OR semiconductor OR ai)'
    terms = [f'"{item}"' if " " in item else item for item in deduped[:8]]
    return "(" + " OR ".join(terms or [f'"{sym}"']) + ")"


def _archive_timestamp(raw_ts) -> pd.Timestamp:
    text = str(raw_ts or "").strip()
    if not text:
        return pd.Timestamp.now(tz="UTC")
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%dT%H%M%SZ", "%Y-%m-%dT%H:%M:%SZ"):
        try:
            return pd.Timestamp(datetime.strptime(text, fmt), tz="UTC")
        except Exception:
            continue
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.now(tz="UTC")
    return parsed


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


def _feed_relevance_prior(instrument_symbol: str | None, text_norm: str, toks: list[str]) -> float:
    sym = str(instrument_symbol or "").upper()
    if not sym:
        return 0.0
    base = 0.0
    if sym == "QQQ":
        direct = any(p in text_norm for p in _DIRECT_QQQ_PHRASES)
        market_hits = sum(1 for t in toks if t in _MARKET_KEYWORDS)
        if direct:
            base = 0.72
        elif market_hits >= 2:
            base = 0.30
        else:
            base = 0.18
        if any(t in _MACRO_KEYWORDS for t in toks) and base >= 0.25:
            base = min(1.0, base + 0.10)
    return float(max(0.0, min(1.0, base)))


def _tech_relevance(headline: str, toks: list[str], instrument_symbol: str | None = None) -> float:
    text = str(headline or "").lower()
    text_norm = " ".join(toks)
    hit_terms = sum(1 for t in toks if t in _TECH_KEYWORDS or t in _TECH_ENTITIES)
    entity_hits = sum(1 for e in _TECH_ENTITIES if e in text)
    market_hits = sum(1 for t in toks if t in _MARKET_KEYWORDS)
    macro_hits = sum(1 for t in toks if t in _MACRO_KEYWORDS)
    base = _feed_relevance_prior(instrument_symbol, text_norm, toks)
    if hit_terms > 0 or entity_hits > 0:
        base = min(1.0, base + 0.22 * float(hit_terms) + 0.26 * float(entity_hits))
    if market_hits >= 2 and (hit_terms > 0 or entity_hits > 0):
        base = min(1.0, base + 0.10)
    if macro_hits > 0 and (base >= 0.25 or market_hits >= 2):
        base = min(1.0, base + 0.08 * min(2.0, float(macro_hits)))
    if any(p in text for p in _DIRECT_QQQ_PHRASES):
        base = max(base, 0.75)
    if "exchange traded funds" in text_norm or "equity futures" in text_norm:
        base = min(1.0, max(base, 0.30) + 0.08)
    return float(max(0.0, min(1.0, base)))


def _political_relevance(headline: str, toks: list[str], instrument_symbol: str | None = None) -> float:
    text = str(headline or "").lower()
    text_norm = " ".join(toks)
    policy_hits = sum(1 for t in toks if t in _POLITICAL_KEYWORDS)
    entity_hits = sum(1 for e in _POLITICAL_ENTITIES if e in text)
    tech_hits = sum(1 for t in toks if t in _TECH_KEYWORDS or t in _TECH_ENTITIES)
    macro_hits = sum(1 for t in toks if t in _MACRO_KEYWORDS)
    market_hits = sum(1 for t in toks if t in _MARKET_KEYWORDS)
    phrase_hits = sum(1 for p in set(_POLITICAL_POS_PHRASES) | set(_POLITICAL_NEG_PHRASES) if p in text_norm)
    base = 0.0
    if policy_hits > 0 or entity_hits > 0 or phrase_hits > 0:
        base = 0.16 * float(policy_hits) + 0.22 * float(entity_hits) + 0.16 * float(phrase_hits)
    if tech_hits > 0:
        base += 0.14 * min(2.0, float(tech_hits))
    if market_hits > 0:
        base += 0.08 * min(2.0, float(market_hits))
    if macro_hits > 0:
        base += 0.06 * min(2.0, float(macro_hits))
    if str(instrument_symbol or "").upper() == "QQQ" and (tech_hits > 0 or any(p in text for p in _DIRECT_QQQ_PHRASES)):
        base += 0.10
    return float(max(0.0, min(1.0, base)))


def _price_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{str(symbol or '').upper()}.csv")


def _price_daily_long_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{str(symbol or '').upper()}_daily_long.csv")


def _load_daily_price_history(symbol: str, tz: str = "America/New_York") -> pd.DataFrame:
    symbol = str(symbol or "").upper()
    frames = []
    long_path = _price_daily_long_path(symbol)
    if os.path.exists(long_path):
        try:
            daily_long = pd.read_csv(long_path, index_col=0, parse_dates=True)
            daily_long.index = pd.to_datetime(daily_long.index, errors="coerce")
            daily_long.columns = [str(c).lower() for c in daily_long.columns]
            daily_long = daily_long.rename(columns={"adj close": "adj_close"})
            daily_long = daily_long[[c for c in ("open", "high", "low", "close", "volume") if c in daily_long.columns]].copy()
            daily_long = daily_long.dropna(subset=["close"]).sort_index()
            if not daily_long.empty:
                daily_long.index = pd.DatetimeIndex(pd.to_datetime(daily_long.index.date))
                daily_long["source"] = "daily_long"
                frames.append(daily_long)
        except Exception:
            logger.debug("Failed reading long daily history for %s", symbol, exc_info=True)
    path = _price_path(symbol)
    if os.path.exists(path):
        try:
            intraday = pd.read_csv(path)
            if "Datetime" in intraday.columns:
                idx = pd.to_datetime(intraday["Datetime"], utc=True, errors="coerce")
            else:
                idx = pd.to_datetime(intraday.iloc[:, 0], utc=True, errors="coerce")
            intraday.index = idx
            intraday.columns = [str(c).lower() for c in intraday.columns]
            intraday = intraday.dropna(subset=["close"]).sort_index()
            if not intraday.empty:
                local_idx = intraday.index.tz_convert(tz)
                intraday["session_date"] = pd.to_datetime(local_idx.date)
                grouped = intraday.groupby("session_date", sort=True)
                daily = pd.DataFrame(index=grouped.size().index)
                for col in ("open", "high", "low", "close", "volume"):
                    if col not in intraday.columns:
                        continue
                    series = pd.to_numeric(intraday[col], errors="coerce")
                    if col == "open":
                        daily[col] = grouped[col].first()
                    elif col == "high":
                        daily[col] = grouped[col].max()
                    elif col == "low":
                        daily[col] = grouped[col].min()
                    elif col == "close":
                        daily[col] = grouped[col].last()
                    elif col == "volume":
                        daily[col] = grouped[col].sum()
                daily = daily.dropna(subset=["close"]).sort_index()
                if not daily.empty:
                    daily["source"] = "intraday"
                    frames.append(daily)
        except Exception:
            logger.debug("Failed reading intraday price history for %s", symbol, exc_info=True)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=0, sort=False)
    merged = merged[~merged.index.duplicated(keep="last")].sort_index()
    return merged


def _rolling_corr_strength(x: pd.Series, y: pd.Series, lookback: int = 90, min_obs: int = 12) -> pd.Series:
    if not isinstance(x, pd.Series) or not isinstance(y, pd.Series):
        return pd.Series(dtype=float)
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    out = np.zeros(len(x), dtype=float)
    for i in range(len(x)):
        start = max(0, i - int(max(10, lookback)))
        xs = x.iloc[start:i]
        ys = y.iloc[start:i]
        valid = xs.notna() & ys.notna() & np.isfinite(xs) & np.isfinite(ys)
        if int(valid.sum()) < int(min_obs):
            out[i] = 0.0
            continue
        xv = xs[valid].to_numpy(dtype=float)
        yv = ys[valid].to_numpy(dtype=float)
        if np.nanstd(xv) < 1e-12 or np.nanstd(yv) < 1e-12:
            out[i] = 0.0
            continue
        corr = np.corrcoef(xv, yv)[0, 1]
        out[i] = float(abs(corr)) if np.isfinite(corr) else 0.0
    return pd.Series(out, index=x.index, dtype=float)


def _historical_news_weight_frame(daily_news: pd.DataFrame, symbol: str, tz: str = "America/New_York") -> pd.DataFrame:
    idx = pd.DatetimeIndex(daily_news.index)
    out = pd.DataFrame(index=idx)
    out["adaptive_weight"] = 1.0
    out["political_weight"] = 1.0
    prices = _load_daily_price_history(symbol, tz=tz)
    if prices.empty or "close" not in prices.columns:
        return out
    prices = prices.sort_index().copy()
    close = pd.to_numeric(prices.get("close"), errors="coerce")
    high = pd.to_numeric(prices.get("high"), errors="coerce")
    low = pd.to_numeric(prices.get("low"), errors="coerce")
    price_daily = pd.DataFrame(index=prices.index)
    price_daily["next_close_ret"] = (close.shift(-1) / close) - 1.0
    price_daily["next_abs_ret"] = price_daily["next_close_ret"].abs()
    if high is not None and low is not None:
        price_daily["next_range_ret"] = ((high.shift(-1) - low.shift(-1)) / close).replace([np.inf, -np.inf], np.nan)
    joined = daily_news.join(price_daily, how="left")
    if joined[["next_close_ret", "next_abs_ret"]].dropna(how="all").empty:
        return out
    dir_strength = _rolling_corr_strength(joined.get("score_sum"), joined.get("next_close_ret"))
    impact_strength = pd.concat(
        [
            _rolling_corr_strength(joined.get("hawkes_close"), joined.get("next_abs_ret")),
            _rolling_corr_strength(joined.get("count"), joined.get("next_abs_ret")),
            _rolling_corr_strength(joined.get("relevance_mean"), joined.get("next_range_ret")),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    political_strength = pd.concat(
        [
            _rolling_corr_strength(joined.get("political_score_sum"), joined.get("next_close_ret")),
            _rolling_corr_strength(joined.get("political_count"), joined.get("next_abs_ret")),
            _rolling_corr_strength(joined.get("political_relevance_mean"), joined.get("next_range_ret")),
        ],
        axis=1,
    ).max(axis=1).fillna(0.0)
    adaptive_strength = ((0.45 * dir_strength) + (0.35 * impact_strength) + (0.20 * political_strength)).clip(0.0, 1.0)
    out["adaptive_weight"] = (0.35 + (1.65 * adaptive_strength)).clip(0.25, 2.0).reindex(idx).fillna(1.0)
    out["political_weight"] = (0.25 + (1.75 * political_strength.clip(0.0, 1.0))).clip(0.15, 2.0).reindex(idx).fillna(1.0)
    return out


def _headline_sentiment_detail(headline: str, instrument_symbol: str | None = None) -> dict:
    toks = _tokenize(headline)
    if not toks:
        return {
            "score": 0.0,
            "label": "neutral",
            "relevance": 0.0,
            "tech_relevant": False,
            "political_relevance": 0.0,
            "political_relevant": False,
        }

    text_norm = " ".join(toks)
    score_raw = _phrase_score(text_norm)
    score_raw += sum(float(w) for phrase, w in _POLITICAL_POS_PHRASES.items() if phrase in text_norm)
    score_raw += sum(float(w) for phrase, w in _POLITICAL_NEG_PHRASES.items() if phrase in text_norm)
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
    tech_relevance = _tech_relevance(headline, toks, instrument_symbol=instrument_symbol)
    political_relevance = _political_relevance(headline, toks, instrument_symbol=instrument_symbol)
    relevance = float(max(tech_relevance, (0.85 * political_relevance) + (0.15 * tech_relevance)))
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
        "tech_relevant": bool(tech_relevance >= 0.25),
        "political_relevance": float(political_relevance),
        "political_relevant": bool(political_relevance >= 0.25),
    }


def score_headline_sentiment(headline: str) -> tuple[float, str]:
    detail = _headline_sentiment_detail(headline)
    return float(detail["score"]), str(detail["label"])


def _normalize_news_frame(df: pd.DataFrame, default_symbol: str | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    if "headline" not in out.columns:
        return out
    if "symbol" not in out.columns:
        out["symbol"] = str(default_symbol or "").upper()
    else:
        out["symbol"] = out["symbol"].fillna(str(default_symbol or "").upper()).astype(str).str.upper()
    if "published_at" in out.columns:
        out["published_at"] = pd.to_datetime(out["published_at"], utc=True, errors="coerce")
    try:
        sent_existing = pd.to_numeric(out.get("sentiment_score"), errors="coerce").fillna(0.0)
    except Exception:
        sent_existing = pd.Series(0.0, index=out.index)
    try:
        rel_existing = pd.to_numeric(out.get("relevance_score"), errors="coerce").fillna(0.0)
    except Exception:
        rel_existing = pd.Series(0.0, index=out.index)
    try:
        sent_nonzero_frac = float((sent_existing.abs() > 1e-12).mean())
    except Exception:
        sent_nonzero_frac = 0.0
    has_sent_score = "sentiment_score" in out.columns
    has_sent_label = "sentiment_label" in out.columns
    need_sentiment = (not has_sent_score) or (sent_nonzero_frac < 0.05)
    need_relevance = ("relevance_score" not in out.columns) or ("tech_relevant" not in out.columns)
    if not need_sentiment and need_relevance and sent_nonzero_frac < 0.20:
        need_sentiment = True
    if float(rel_existing.abs().sum()) == 0.0:
        need_relevance = True
    details = []
    for headline, sym in zip(out["headline"].astype(str).tolist(), out["symbol"].astype(str).tolist()):
        details.append(_headline_sentiment_detail(headline, instrument_symbol=sym or default_symbol))
    det_df = pd.DataFrame(details, index=out.index)
    if need_sentiment:
        out["sentiment_score"] = pd.to_numeric(det_df.get("score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
        out["sentiment_label"] = det_df.get("label", "neutral").astype(str).str.lower()
    else:
        out["sentiment_score"] = sent_existing.clip(-1.0, 1.0)
        if has_sent_label:
            out["sentiment_label"] = out.get("sentiment_label", "neutral").astype(str).str.lower()
        else:
            score_vals = pd.to_numeric(out["sentiment_score"], errors="coerce").fillna(0.0)
            out["sentiment_label"] = np.where(score_vals > 0.15, "positive", np.where(score_vals < -0.15, "negative", "neutral"))
    if need_relevance:
        out["relevance_score"] = pd.to_numeric(det_df.get("relevance"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
        out["tech_relevant"] = det_df.get("tech_relevant", False).astype(bool).astype(int)
    else:
        out["relevance_score"] = rel_existing.clip(0.0, 1.0)
        out["tech_relevant"] = pd.to_numeric(out.get("tech_relevant"), errors="coerce").fillna(0).astype(int)
    out["political_relevance_score"] = pd.to_numeric(
        det_df.get("political_relevance", out.get("political_relevance_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["political_relevant"] = pd.to_numeric(
        det_df.get("political_relevant", out.get("political_relevant")),
        errors="coerce",
    ).fillna(0).astype(int)
    return out


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
    sent = _headline_sentiment_detail(title, instrument_symbol=symbol)
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
        "political_relevance_score": float(sent.get("political_relevance", 0.0)),
        "political_relevant": int(1 if sent.get("political_relevant") else 0),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def _load_existing_news(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return _normalize_news_frame(df)
    except Exception:
        logger.exception("Failed reading news file %s", path)
        return pd.DataFrame()


def _save_news(path: str, df: pd.DataFrame):
    if df is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _local_archive_candidate_paths(symbol: str) -> list[str]:
    sym = str(symbol or "").upper()
    names = [
        f"news_history_{sym}.csv",
        f"news_archive_{sym}.csv",
        f"news_history_{sym}.jsonl",
        f"news_archive_{sym}.jsonl",
        f"{sym}_news.csv",
        f"{sym}_news.jsonl",
        f"news_{sym}.csv",
        f"news_{sym}.jsonl",
    ]
    out = [os.path.join(DATA_DIR, name) for name in names]
    out.extend([os.path.join(NEW_INSTRUMENTS_DIR, name) for name in names])
    return out


def _read_local_archive_frame(path: str, symbol: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        if str(path).lower().endswith(".jsonl"):
            rows = []
            with open(path, "r") as f:
                for raw in f:
                    raw = str(raw or "").strip()
                    if not raw:
                        continue
                    try:
                        item = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(item, dict):
                        rows.append(item)
            frame = pd.DataFrame(rows)
        else:
            frame = pd.read_csv(path)
    except Exception:
        logger.debug("Failed reading local historical news archive %s", path, exc_info=True)
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    rename_map = {
        "title": "headline",
        "headline_text": "headline",
        "headlineTitle": "headline",
        "published": "published_at",
        "publish_time": "published_at",
        "publishTime": "published_at",
        "datetime": "published_at",
        "timestamp": "published_at",
        "date": "published_at",
        "time": "published_at",
        "pubDate": "published_at",
        "link": "url",
        "article_url": "url",
        "site": "source",
        "publisher": "source",
        "provider": "source",
        "ticker": "symbol",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})
    if "headline" not in frame.columns or "published_at" not in frame.columns:
        return pd.DataFrame()
    frame["symbol"] = str(symbol or "").upper()
    frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["headline", "published_at"]).copy()
    if frame.empty:
        return pd.DataFrame()
    if "source" not in frame.columns:
        frame["source"] = "local_archive"
    if "url" not in frame.columns:
        frame["url"] = None
    if "headline_id" not in frame.columns:
        frame["headline_id"] = [
            f"local|{os.path.basename(path)}|{ts.isoformat()}|{str(head)[:120]}"
            for ts, head in zip(frame["published_at"], frame["headline"])
        ]
    if "ingested_at" not in frame.columns:
        now_iso = datetime.now(timezone.utc).isoformat()
        frame["ingested_at"] = now_iso
    return _normalize_news_frame(frame, default_symbol=str(symbol or "").upper())


def import_local_historical_news(symbol: str = "QQQ", extra_paths: list[str] | None = None) -> dict:
    symbol = str(symbol or "QQQ").upper()
    path = _news_path(symbol)
    current = _load_existing_news(path)
    imported_frames = []
    imported_files = []
    candidates = list(_local_archive_candidate_paths(symbol))
    for candidate in list(extra_paths or []):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        frame = _read_local_archive_frame(candidate, symbol)
        if frame.empty:
            continue
        imported_frames.append(frame)
        imported_files.append(candidate)
    if not imported_frames:
        return {
            "symbol": symbol,
            "added": 0,
            "rows": int(len(current)),
            "imported_rows": 0,
            "imported_files": [],
            "coverage": news_archive_coverage(symbol),
        }
    incoming = pd.concat(imported_frames, axis=0, ignore_index=True, sort=False)
    merged = incoming if current is None or current.empty else pd.concat([current, incoming], axis=0, ignore_index=True, sort=False)
    merged = _normalize_news_frame(merged, default_symbol=symbol)
    if "headline_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["headline_id"], keep="last")
    else:
        merged = merged.drop_duplicates(subset=["published_at", "headline"], keep="last")
    merged = merged.sort_values("published_at")
    _save_news(path, merged)
    return {
        "symbol": symbol,
        "added": max(0, int(len(merged) - len(current))),
        "rows": int(len(merged)),
        "imported_rows": int(len(incoming)),
        "imported_files": imported_files,
        "coverage": news_archive_coverage(symbol),
    }


def _parse_archive_item(item: dict, symbol: str) -> dict | None:
    if not isinstance(item, dict):
        return None
    title = item.get("title") or item.get("seendate_title") or item.get("urltitle")
    url = item.get("url") or item.get("url_mobile")
    if not title or not url:
        return None
    ts = _archive_timestamp(item.get("seendate") or item.get("date") or item.get("seenDate"))
    sent = _headline_sentiment_detail(title, instrument_symbol=symbol)
    uid = f"archive|{url}|{ts.isoformat()}"
    return {
        "symbol": symbol.upper(),
        "headline_id": uid,
        "published_at": ts.isoformat(),
        "headline": str(title),
        "source": str(item.get("domain") or item.get("sourcecountry") or "archive"),
        "url": str(url),
        "sentiment_score": float(sent["score"]),
        "sentiment_label": str(sent["label"]),
        "relevance_score": float(sent["relevance"]),
        "tech_relevant": int(1 if sent["tech_relevant"] else 0),
        "political_relevance_score": float(sent.get("political_relevance", 0.0)),
        "political_relevant": int(1 if sent.get("political_relevant") else 0),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_archive_headlines(symbol: str = "QQQ", start=None, end=None,
                            max_records: int = NEWS_ARCHIVE_MAX_RECORDS_PER_CHUNK) -> list[dict]:
    if not NEWS_PIPELINE_ENABLED:
        return []
    start_ts = pd.to_datetime(start, utc=True, errors="coerce")
    end_ts = pd.to_datetime(end, utc=True, errors="coerce")
    if pd.isna(end_ts):
        end_ts = pd.Timestamp.now(tz="UTC")
    if pd.isna(start_ts) or start_ts >= end_ts:
        return []
    query = _archive_query(symbol)
    params = (
        f"query={quote_plus(query)}"
        f"&mode=artlist&format=json&sort=datedesc"
        f"&maxrecords={int(max(10, min(250, max_records)))}"
        f"&startdatetime={start_ts.strftime('%Y%m%d%H%M%S')}"
        f"&enddatetime={end_ts.strftime('%Y%m%d%H%M%S')}"
    )
    url = f"https://api.gdeltproject.org/api/v2/doc/doc?{params}"
    try:
        req = Request(url, headers={"User-Agent": "LightningTrader4000/1.0"})
        ssl_ctx = None
        if certifi is not None:
            try:
                ssl_ctx = ssl.create_default_context(cafile=certifi.where())
            except Exception:
                ssl_ctx = None
        with urlopen(req, timeout=20, context=ssl_ctx) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception:
        logger.debug("Archive fetch failed for %s", symbol, exc_info=True)
        return []
    articles = payload.get("articles", []) if isinstance(payload, dict) else []
    out = []
    for item in articles:
        parsed = _parse_archive_item(item, symbol=symbol)
        if parsed:
            out.append(parsed)
    return out


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
    incoming = _normalize_news_frame(incoming, default_symbol=symbol)
    if current is None or current.empty:
        merged = incoming
    else:
        merged = pd.concat([current, incoming], axis=0, ignore_index=True, sort=False)
    merged = _normalize_news_frame(merged, default_symbol=symbol)
    if "headline_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["headline_id"], keep="last")
    else:
        merged = merged.drop_duplicates(subset=["published_at", "headline"], keep="last")
    merged = merged.sort_values("published_at")
    _save_news(path, merged)
    added = max(0, int(len(merged) - len(current)))
    return {"symbol": symbol, "added": added, "rows": int(len(merged))}


def backfill_news_archive(symbol: str = "QQQ", lookback_days: int = NEWS_ARCHIVE_LOOKBACK_DAYS,
                          chunk_days: int = NEWS_ARCHIVE_CHUNK_DAYS,
                          max_records_per_chunk: int = NEWS_ARCHIVE_MAX_RECORDS_PER_CHUNK) -> dict:
    symbol = str(symbol or "QQQ").upper()
    lookback_days = max(7, int(lookback_days))
    chunk_days = max(1, int(chunk_days))
    path = _news_path(symbol)
    local_import = import_local_historical_news(symbol=symbol)
    local_cov = local_import.get("coverage", {}) if isinstance(local_import, dict) else {}
    if bool(local_cov.get("sufficient_long_horizon", False)):
        current = _load_existing_news(path)
        return {
            "symbol": symbol,
            "added": int(local_import.get("added", 0) or 0),
            "rows": int(len(current)),
            "archive_backfill": True,
            "local_import": local_import,
            "coverage": local_cov,
        }
    current = _load_existing_news(path)
    end_ts = pd.Timestamp.now(tz="UTC")
    start_ts = end_ts - pd.Timedelta(days=lookback_days)
    rows = []
    cursor = start_ts
    while cursor < end_ts:
        nxt = min(end_ts, cursor + pd.Timedelta(days=chunk_days))
        rows.extend(fetch_archive_headlines(symbol=symbol, start=cursor, end=nxt, max_records=max_records_per_chunk))
        cursor = nxt
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        return {
            "symbol": symbol,
            "added": 0,
            "rows": int(len(current)),
            "archive_backfill": True,
            "local_import": local_import,
            "coverage": news_archive_coverage(symbol),
        }
    incoming = _normalize_news_frame(incoming, default_symbol=symbol)
    merged = incoming if current is None or current.empty else pd.concat([current, incoming], axis=0, ignore_index=True, sort=False)
    merged = _normalize_news_frame(merged, default_symbol=symbol)
    if "headline_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["headline_id"], keep="last")
    else:
        merged = merged.drop_duplicates(subset=["published_at", "headline"], keep="last")
    merged = merged.sort_values("published_at")
    _save_news(path, merged)
    return {
        "symbol": symbol,
        "added": max(0, int(len(merged) - len(current))),
        "rows": int(len(merged)),
        "archive_backfill": True,
        "local_import": local_import,
        "coverage": news_archive_coverage(symbol),
    }


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


def news_archive_coverage(symbol: str = "QQQ", trading_dates=None, tz: str = "America/New_York") -> dict:
    symbol = str(symbol or "QQQ").upper()
    path = _news_path(symbol)
    if not os.path.exists(path):
        return {
            "symbol": symbol,
            "rows": 0,
            "relevant_rows": 0,
            "active_days": 0,
            "trading_days": 0,
            "coverage_ratio": 0.0,
            "sufficient_long_horizon": False,
        }
    try:
        df = _normalize_news_frame(pd.read_csv(path), default_symbol=symbol)
    except Exception:
        logger.debug("Failed to read news archive coverage for %s", symbol, exc_info=True)
        return {
            "symbol": symbol,
            "rows": 0,
            "relevant_rows": 0,
            "active_days": 0,
            "trading_days": 0,
            "coverage_ratio": 0.0,
            "sufficient_long_horizon": False,
        }
    if df.empty:
        return {
            "symbol": symbol,
            "rows": 0,
            "relevant_rows": 0,
            "active_days": 0,
            "trading_days": 0,
            "coverage_ratio": 0.0,
            "sufficient_long_horizon": False,
        }
    df["published_at"] = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"])
    if df.empty:
        return {
            "symbol": symbol,
            "rows": 0,
            "relevant_rows": 0,
            "active_days": 0,
            "trading_days": 0,
            "coverage_ratio": 0.0,
            "sufficient_long_horizon": False,
        }
    df["relevance_score"] = pd.to_numeric(df.get("relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["tech_relevant"] = pd.to_numeric(df.get("tech_relevant"), errors="coerce").fillna(0).astype(int)
    df["political_relevant"] = pd.to_numeric(df.get("political_relevant"), errors="coerce").fillna(0).astype(int)
    relevant = df[
        (df["tech_relevant"] > 0)
        | (df["political_relevant"] > 0)
        | (df["relevance_score"] >= 0.25)
    ].copy()
    if relevant.empty:
        return {
            "symbol": symbol,
            "rows": int(len(df)),
            "relevant_rows": 0,
            "active_days": 0,
            "trading_days": 0,
            "coverage_ratio": 0.0,
            "sufficient_long_horizon": False,
        }
    relevant_local = relevant["published_at"].dt.tz_convert(tz)
    active_dates = pd.Index(pd.to_datetime(relevant_local.dt.date)).unique().sort_values()
    if trading_dates is None or len(trading_dates) == 0:
        start = active_dates.min()
        end = active_dates.max()
        trading_idx = pd.bdate_range(start=start, end=end) if start is not pd.NaT and end is not pd.NaT else pd.DatetimeIndex([])
    else:
        trading_idx = pd.DatetimeIndex(pd.to_datetime(trading_dates, utc=True, errors="coerce")).tz_convert(tz)
        trading_idx = pd.DatetimeIndex(pd.to_datetime(trading_idx.date)).unique().sort_values()
    trading_days = int(len(trading_idx))
    if trading_days <= 0:
        coverage = 0.0
    else:
        coverage = float(len(active_dates.intersection(trading_idx)) / trading_days)
    sufficient = bool(
        int(len(relevant)) >= int(NEWS_LONG_HORIZON_MIN_ROWS)
        and int(len(active_dates)) >= int(NEWS_LONG_HORIZON_MIN_ACTIVE_DAYS)
        and coverage >= float(NEWS_LONG_HORIZON_MIN_COVERAGE)
    )
    return {
        "symbol": symbol,
        "rows": int(len(df)),
        "relevant_rows": int(len(relevant)),
        "active_days": int(len(active_dates)),
        "trading_days": int(trading_days),
        "coverage_ratio": float(coverage),
        "start": active_dates.min().strftime("%Y-%m-%d") if len(active_dates) else None,
        "end": active_dates.max().strftime("%Y-%m-%d") if len(active_dates) else None,
        "sufficient_long_horizon": sufficient,
    }


def _hawkes_intensity(ts_ns: np.ndarray, weights: np.ndarray, half_life_hours: float = 6.0) -> np.ndarray:
    """
    Lightweight Hawkes-style excitation proxy.
    We use a fixed exponential decay instead of fitting a full point-process model.
    """
    n = len(ts_ns)
    if n == 0:
        return np.array([], dtype=float)
    try:
        decay = float(np.log(2.0) / max(1.0, float(half_life_hours)) / 3600.0)
    except Exception:
        decay = float(np.log(2.0) / 21600.0)
    out = np.zeros(n, dtype=float)
    level = 0.0
    last_ts = None
    for i in range(n):
        t_raw = ts_ns[i]
        try:
            t_sec = float(t_raw) / 1_000_000_000.0
        except Exception:
            t_sec = None
        if t_sec is None or not np.isfinite(t_sec):
            out[i] = level
            continue
        if last_ts is not None:
            dt = max(0.0, float(t_sec - last_ts))
            level *= float(np.exp(-decay * dt))
        try:
            w = max(0.0, float(weights[i]))
        except Exception:
            w = 0.0
        level += w
        out[i] = level
        last_ts = t_sec
    return out


def _normalized_entropy(counts: pd.DataFrame) -> pd.Series:
    if counts is None or counts.empty:
        return pd.Series(dtype=float)
    arr = np.asarray(counts, dtype=float)
    total = arr.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        probs = np.divide(arr, total, out=np.zeros_like(arr), where=total > 0)
    safe_probs = np.where(probs > 0, probs, 1.0)
    ent = -(probs * np.log(safe_probs)).sum(axis=1)
    norm = np.log(float(arr.shape[1])) if arr.shape[1] > 1 else 1.0
    if norm <= 0:
        norm = 1.0
    return pd.Series(ent / norm, index=counts.index).fillna(0.0)


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
    if index_utc is None or len(index_utc) == 0:
        return pd.DataFrame(columns=cols, index=index_utc)
    path = _news_path(symbol.upper())
    if not os.path.exists(path):
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    try:
        df = _normalize_news_frame(pd.read_csv(path), default_symbol=symbol.upper())
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
    df["relevance_score"] = pd.to_numeric(df.get("relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["tech_relevant"] = pd.to_numeric(df.get("tech_relevant"), errors="coerce").fillna(0).astype(int)
    df["political_relevance_score"] = pd.to_numeric(df.get("political_relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["political_relevant"] = pd.to_numeric(df.get("political_relevant"), errors="coerce").fillna(0).astype(int)
    # Keep only relevant stories for QQQ signal extraction.
    df = df[
        (df["tech_relevant"] > 0)
        | (df["political_relevant"] > 0)
        | (df["relevance_score"] >= 0.25)
    ].copy()
    if df.empty:
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    # Relevance-weighted score reduces noise from loosely related articles.
    df["sentiment_score"] = df["sentiment_score"] * (0.5 + 0.5 * df["relevance_score"])
    df["political_sentiment_score"] = df["sentiment_score"] * df["political_relevance_score"]
    df["event_weight"] = (
        (0.45 + 0.55 * df["relevance_score"])
        * (1.0 + df["sentiment_score"].abs())
        * (1.0 + (0.35 * df["political_relevance_score"]))
    )
    ts_ns = df["published_at"].astype("int64", copy=False).to_numpy()
    df["hawkes_intensity"] = _hawkes_intensity(ts_ns, df["event_weight"].to_numpy(dtype=float), half_life_hours=6.0)
    local_dt = df["published_at"].dt.tz_convert(tz)
    df["news_date"] = pd.to_datetime(local_dt.dt.date)
    grouped = df.groupby("news_date", sort=True)
    daily = pd.DataFrame(index=grouped.size().index)
    daily["count"] = grouped.size().astype(float)
    daily["score_mean"] = grouped["sentiment_score"].mean()
    daily["score_sum"] = grouped["sentiment_score"].sum()
    daily["pos_count"] = grouped["sentiment_label"].apply(lambda s: float((s == "positive").sum()))
    daily["neg_count"] = grouped["sentiment_label"].apply(lambda s: float((s == "negative").sum()))
    daily["neutral_count"] = grouped["sentiment_label"].apply(lambda s: float((s == "neutral").sum()))
    daily["relevance_mean"] = grouped["relevance_score"].mean()
    daily["political_count"] = grouped["political_relevant"].sum().astype(float)
    daily["political_score_sum"] = grouped["political_sentiment_score"].sum()
    daily["political_relevance_mean"] = grouped["political_relevance_score"].mean()
    daily["hawkes_close"] = grouped["hawkes_intensity"].last()
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
    weights = _historical_news_weight_frame(daily, symbol=symbol, tz=tz)
    daily["adaptive_weight"] = pd.to_numeric(weights.get("adaptive_weight"), errors="coerce").reindex(daily.index).fillna(1.0)
    daily["political_weight"] = pd.to_numeric(weights.get("political_weight"), errors="coerce").reindex(daily.index).fillna(1.0)
    daily["weighted_score_sum"] = daily["score_sum"] * daily["adaptive_weight"]
    daily["weighted_political_score_sum"] = daily["political_score_sum"] * daily["political_weight"]
    # Lag all features by one day to avoid peeking at same-day headlines in training labels.
    lag = pd.DataFrame(index=daily.index)
    lag["news_sent_mean_d1"] = daily["score_mean"].shift(1).fillna(0.0)
    lag["news_headline_count_d1"] = daily["count"].shift(1).fillna(0.0)
    lag["news_sent_sum_d3"] = daily["score_sum"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    pos_roll = daily["pos_count"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    neg_roll = daily["neg_count"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    neutral_roll = daily["neutral_count"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    den = (pos_roll + neg_roll).replace(0.0, np.nan)
    lag["news_pos_ratio_d3"] = (pos_roll / den).fillna(0.5)
    lag["news_neg_ratio_d3"] = (neg_roll / den).fillna(0.5)
    lag["news_sent_ewm_d7"] = daily["score_sum"].ewm(halflife=3.0, adjust=False, min_periods=1).mean().shift(1).fillna(0.0)
    lag["news_intensity_ewm_d7"] = daily["hawkes_close"].ewm(halflife=3.0, adjust=False, min_periods=1).mean().shift(1).fillna(0.0)
    entropy_roll = _normalized_entropy(pd.DataFrame({"pos": pos_roll, "neg": neg_roll, "neutral": neutral_roll}, index=daily.index))
    lag["news_entropy_d3"] = entropy_roll.fillna(0.0)
    lag["news_relevance_mean_d3"] = daily["relevance_mean"].rolling(3, min_periods=1).mean().shift(1).fillna(0.0)
    long_mean = daily["score_sum"].ewm(halflife=10.0, adjust=False, min_periods=3).mean()
    lag["news_sent_surprise_d1"] = (daily["score_sum"] - long_mean).shift(1).fillna(0.0)
    lag["news_political_count_d3"] = daily["political_count"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    lag["news_political_sent_d3"] = daily["political_score_sum"].rolling(3, min_periods=1).sum().shift(1).fillna(0.0)
    lag["news_political_relevance_d3"] = daily["political_relevance_mean"].rolling(3, min_periods=1).mean().shift(1).fillna(0.0)
    lag["news_adaptive_weight_d7"] = daily["adaptive_weight"].ewm(halflife=5.0, adjust=False, min_periods=1).mean().shift(1).fillna(1.0)
    lag["news_weighted_sent_d7"] = daily["weighted_score_sum"].ewm(halflife=3.0, adjust=False, min_periods=1).mean().shift(1).fillna(0.0)
    lag["news_weighted_political_d7"] = daily["weighted_political_score_sum"].ewm(halflife=3.0, adjust=False, min_periods=1).mean().shift(1).fillna(0.0)

    idx_local = pd.to_datetime(index_utc, utc=True, errors="coerce").tz_convert(tz)
    idx_dates = pd.to_datetime(idx_local.date)
    aligned = lag.reindex(idx_dates).fillna(0.0)
    aligned.index = index_utc
    for c in cols:
        if c not in aligned.columns:
            aligned[c] = 0.0
    return aligned[cols]


def load_recent_headlines(symbol: str = "QQQ", limit: int = 8) -> list[dict]:
    path = _news_path(symbol.upper())
    if not os.path.exists(path):
        return []
    try:
        df = _normalize_news_frame(pd.read_csv(path), default_symbol=symbol.upper())
    except Exception:
        logger.debug("Failed to read news store for recent headlines", exc_info=True)
        return []
    if df is None or df.empty:
        return []
    df = df.sort_values("published_at", ascending=False).head(max(1, int(limit))).copy()
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "published_at": row.get("published_at"),
                "headline": str(row.get("headline") or "").strip(),
                "score": _safe_float(row.get("sentiment_score")),
                "label": str(row.get("sentiment_label") or "").strip(),
                "relevance": _safe_float(row.get("relevance_score")),
                "political_relevance": _safe_float(row.get("political_relevance_score")),
                "source": str(row.get("source") or "").strip(),
                "url": str(row.get("url") or "").strip(),
            }
        )
    return rows
