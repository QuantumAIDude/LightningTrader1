#!/usr/bin/env python3
"""
news_pipeline.py

Multi-stock headline ingestion + sentiment features.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ssl
from datetime import datetime, timezone
from typing import Callable
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
from instrument_profile import build_instrument_meta, normalize_symbol

logger = logging.getLogger(__name__)
_OPTIONAL_NEWS_SCORERS: dict[str, Callable[[str, list[dict], dict], list[dict] | dict | None]] = {}


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _coerce_utc_ts(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")

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
_COMMUNICATIONS_KEYWORDS = {
    "telecom", "streaming", "advertising", "media", "social", "wireless", "broadband", "carrier",
}
_CONSUMER_KEYWORDS = {
    "retail", "consumer", "shopping", "apparel", "restaurant", "travel", "leisure", "ecommerce",
    "e-commerce", "warehouse", "membership",
}
_HEALTHCARE_KEYWORDS = {
    "biotech", "biotechnology", "pharma", "pharmaceutical", "drug", "medical", "hospital", "device",
    "fda", "medicare", "therapy",
}
_FINANCIALS_KEYWORDS = {
    "bank", "banking", "lender", "loan", "loans", "credit", "card", "payments", "insurance", "brokerage",
    "asset manager", "net interest income",
}
_INDUSTRIALS_KEYWORDS = {
    "airline", "airlines", "aerospace", "defense", "rail", "railroad", "logistics", "freight", "machinery",
    "manufacturing", "factory", "industrial",
}
_ENERGY_KEYWORDS = {
    "oil", "gas", "crude", "opec", "refinery", "refining", "drilling", "pipeline", "lng", "shale",
}
_MATERIALS_KEYWORDS = {
    "chemical", "chemicals", "mining", "metals", "steel", "copper", "aluminum", "lithium", "fertilizer",
}
_UTILITIES_KEYWORDS = {
    "utility", "utilities", "power", "grid", "electricity", "natural gas utility", "water utility",
}
_REAL_ESTATE_KEYWORDS = {
    "reit", "real estate", "property", "housing", "mortgage", "commercial real estate", "office market",
}
_SECTOR_THEME_KEYWORDS = {
    "technology": _TECH_KEYWORDS | _TECH_ENTITIES,
    "communications": _COMMUNICATIONS_KEYWORDS,
    "consumer": _CONSUMER_KEYWORDS,
    "healthcare": _HEALTHCARE_KEYWORDS,
    "financials": _FINANCIALS_KEYWORDS,
    "industrials": _INDUSTRIALS_KEYWORDS,
    "energy": _ENERGY_KEYWORDS,
    "materials": _MATERIALS_KEYWORDS,
    "utilities": _UTILITIES_KEYWORDS,
    "real_estate": _REAL_ESTATE_KEYWORDS,
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
_THEME_ALIAS_BY_SECTOR = {
    "technology": "technology",
    "communication services": "communications",
    "consumer discretionary": "consumer",
    "consumer staples": "consumer",
    "healthcare": "healthcare",
    "financials": "financials",
    "industrials": "industrials",
    "energy": "energy",
    "materials": "materials",
    "utilities": "utilities",
    "real estate": "real_estate",
    "broad market": "broad_market",
}
_SCOPE_LABELS = {
    "symbol_specific": "Symbol-specific",
    "sector_specific": "Sector-specific",
    "macro_political_regulatory": "Macro / political / regulatory",
    "broad_market": "Broad market",
    "low_value_noise": "Low-value / noise",
    "opinion_commentary": "Opinion / commentary",
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

_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from", "has", "have", "in", "into",
    "is", "it", "its", "of", "on", "or", "the", "their", "this", "to", "was", "will", "with",
}
_SOURCE_WEIGHT_OVERRIDES = {
    "reuters": 1.20,
    "associated press": 1.16,
    "dow jones": 1.18,
    "mt newswires": 1.12,
    "benzinga": 0.94,
    "yahoo finance": 0.94,
    "zacks": 0.76,
    "motley fool": 0.62,
    "24/7 wall st": 0.58,
    "247wallst": 0.58,
    "the daily upside": 0.78,
    "investorplace": 0.60,
    "seeking alpha": 0.74,
}
_LOW_SIGNAL_SOURCE_MARKERS = {
    "motley fool",
    "24/7 wall st",
    "247wallst",
    "investorplace",
}
_LOW_SIGNAL_PHRASES = {
    "should you buy",
    "clear answer",
    "id own",
    "i d own",
    "never look back",
    "top stock",
    "top stocks",
    "best etf",
    "what to know",
    "why this stock",
    "bullish asymmetric reward",
}
_MARKET_WRAP_PHRASES = {
    "pre bell",
    "pre-bell",
    "midday",
    "after hours",
    "before the bell",
    "equity futures",
    "markets open",
    "point higher",
    "point lower",
}
_EVENT_TYPE_PROFILES = [
    (
        "guidance",
        {
            "label": "Guidance",
            "horizon": "multi-day",
            "base_impact": 0.92,
            "patterns": (
                "raises guidance", "raise guidance", "cuts guidance", "cut guidance",
                "guidance", "outlook", "forecast", "sales outlook", "revenue outlook",
            ),
            "note": "Guidance changes often reset expectations for multiple sessions.",
        },
    ),
    (
        "earnings",
        {
            "label": "Earnings",
            "horizon": "multi-day",
            "base_impact": 0.88,
            "patterns": (
                "earnings", "quarter results", "quarterly results", "beats estimates",
                "misses estimates", "revenue beat", "eps beat", "profit warning",
            ),
            "note": "Earnings headlines can shift the next few sessions, not just the next bar.",
        },
    ),
    (
        "macro",
        {
            "label": "Macro",
            "horizon": "intraday",
            "base_impact": 0.82,
            "patterns": (
                "cpi", "inflation", "ppi", "pce", "fed", "fomc", "payrolls", "jobs report",
                "retail sales", "consumer confidence", "treasury yields", "treasury yield",
            ),
            "note": "Macro releases can dominate intraday tape even when single-name news is quiet.",
        },
    ),
    (
        "regulatory",
        {
            "label": "Regulatory",
            "horizon": "multi-day",
            "base_impact": 0.80,
            "patterns": (
                "antitrust", "doj", "ftc", "sec", "probe", "investigation", "lawsuit",
                "regulatory", "approval", "settlement", "ban", "crackdown",
            ),
            "note": "Regulatory headlines can reprice risk and compress multiples for days.",
        },
    ),
    (
        "geopolitical",
        {
            "label": "Geopolitical",
            "horizon": "intraday",
            "base_impact": 0.78,
            "patterns": (
                "sanctions", "tariff", "tariffs", "war", "ceasefire", "china",
                "export controls", "trade deal", "trade war", "iran", "ukraine",
            ),
            "note": "Geopolitical headlines usually matter through cross-asset and sector shock transmission.",
        },
    ),
    (
        "analyst",
        {
            "label": "Analyst",
            "horizon": "intraday",
            "base_impact": 0.66,
            "patterns": (
                "upgrade", "upgrades", "downgrade", "downgrades", "price target",
                "initiates", "initiation", "coverage", "outperform", "underperform",
            ),
            "note": "Analyst actions can matter tactically but usually need tape confirmation.",
        },
    ),
    (
        "merger",
        {
            "label": "M&A",
            "horizon": "multi-day",
            "base_impact": 0.84,
            "patterns": (
                "merger", "acquisition", "acquire", "acquires", "buyout", "takeover", "deal",
            ),
            "note": "Deal headlines often change valuation anchors and event-risk immediately.",
        },
    ),
    (
        "product",
        {
            "label": "Product",
            "horizon": "multi-day",
            "base_impact": 0.56,
            "patterns": (
                "launch", "launches", "unveil", "unveils", "introduces", "announces new",
                "platform", "product rollout", "roadmap",
            ),
            "note": "Product headlines matter when the tape confirms demand or margin implications.",
        },
    ),
    (
        "capital_flows",
        {
            "label": "Flows",
            "horizon": "multi-day",
            "base_impact": 0.34,
            "patterns": (
                "position in", "stake in", "inflows", "outflows", "holding", "initiates position",
                "etf", "fund",
            ),
            "note": "Flow headlines are usually slower-burn context unless size or source is exceptional.",
        },
    ),
    (
        "market_wrap",
        {
            "label": "Market Wrap",
            "horizon": "intraday",
            "base_impact": 0.26,
            "patterns": tuple(_MARKET_WRAP_PHRASES),
            "note": "Market-color headlines are useful context but rarely a standalone edge.",
        },
    ),
    (
        "opinion",
        {
            "label": "Opinion",
            "horizon": "intraday",
            "base_impact": 0.18,
            "patterns": tuple(_LOW_SIGNAL_PHRASES),
            "note": "Opinion headlines carry low tradable information density.",
        },
    ),
]


def _news_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_{symbol.upper()}.csv")


def _state_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"news_state_{symbol.upper()}.json")


def _instrument_meta(symbol: str | None) -> dict:
    sym = normalize_symbol(symbol)
    return build_instrument_meta(sym, overrides=INSTRUMENTS.get(sym, {}) if sym else {})


def _alias_terms(symbol: str | None) -> list[str]:
    meta = _instrument_meta(symbol)
    aliases = []
    for raw in [symbol, meta.get("display_name"), meta.get("company_name"), *list(meta.get("news_aliases", []) or [])]:
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
    return deduped


def _theme_terms_for_symbol(symbol: str | None, meta: dict | None = None) -> list[str]:
    payload = dict(meta or _instrument_meta(symbol))
    weights = dict(payload.get("news_theme_weights") or {})
    terms = []
    ranked = sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
    theme_seed = {
        "technology": ["technology", "semiconductor", "ai"],
        "communications": ["media", "telecom", "streaming"],
        "consumer": ["retail", "consumer", "e-commerce"],
        "healthcare": ["healthcare", "pharma", "biotech"],
        "financials": ["banking", "payments", "credit"],
        "industrials": ["industrial", "defense", "transport"],
        "energy": ["oil", "gas", "energy"],
        "materials": ["metals", "mining", "materials"],
        "utilities": ["utilities", "power", "grid"],
        "real_estate": ["real estate", "reit", "property"],
        "broad_market": ["stocks", "market", "futures"],
    }
    for theme, score in ranked[:2]:
        if float(score) < 0.55:
            continue
        terms.extend(theme_seed.get(str(theme), []))
    sector_etf = str((payload.get("context_daily_symbols") or {}).get("sector") or "").strip()
    if sector_etf and sector_etf.upper() != normalize_symbol(symbol):
        terms.append(sector_etf.upper())
    return terms[:8]


def _archive_query(symbol: str) -> str:
    sym = normalize_symbol(symbol)
    meta = _instrument_meta(sym)
    aliases = _alias_terms(sym)
    theme_terms = _theme_terms_for_symbol(sym, meta=meta)
    deduped = []
    seen = set()
    for item in aliases + theme_terms:
        key = str(item or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(str(item).strip())
    terms = [f'"{item}"' if " " in item else item for item in deduped[:10]]
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


def _normalized_headline_signature(text: str) -> str:
    toks = [tok for tok in _tokenize(text) if len(tok) > 2 and tok not in _STOPWORDS]
    return " ".join(toks[:12])


def _phrase_score(text_norm: str) -> float:
    score = 0.0
    for phrase, w in _POS_PHRASES.items():
        if phrase in text_norm:
            score += float(w)
    for phrase, w in _NEG_PHRASES.items():
        if phrase in text_norm:
            score += float(w)
    return float(score)


def _source_weight(source: str | None) -> float:
    src = str(source or "").strip().lower()
    if not src:
        return 0.85
    for marker, weight in _SOURCE_WEIGHT_OVERRIDES.items():
        if marker in src:
            return float(weight)
    return 0.88


def _noise_score(headline: str, source: str | None = None) -> float:
    text = str(headline or "").strip().lower()
    score = 0.0
    for phrase in _LOW_SIGNAL_PHRASES:
        if phrase in text:
            score += 0.42
    for phrase in _MARKET_WRAP_PHRASES:
        if phrase in text:
            score += 0.16
    src = str(source or "").strip().lower()
    if any(marker in src for marker in _LOW_SIGNAL_SOURCE_MARKERS):
        score += 0.20
    if "opinion" in text or "editorial" in text:
        score += 0.18
    return float(max(0.0, min(0.95, score)))


def _headline_event_detail(headline: str, source: str | None = None, instrument_symbol: str | None = None) -> dict:
    text = str(headline or "").strip()
    if not text:
        return {
            "event_type": "general",
            "event_label": "General",
            "impact_score": 0.18,
            "impact_horizon": "intraday",
            "source_weight": _source_weight(source),
            "noise_score": 0.0,
            "signal_note": "No structured event tag could be inferred from the headline.",
            "headline_signature": "",
        }
    raw_lower = text.lower()
    signature = _normalized_headline_signature(text)
    toks = _tokenize(text)
    symbol_specific = _symbol_specific_relevance(text, toks, instrument_symbol=instrument_symbol)
    theme_scores = _theme_relevance_scores(text, toks, instrument_symbol=instrument_symbol)
    best = None
    best_score = -1.0
    for event_type, profile in _EVENT_TYPE_PROFILES:
        hits = [pat for pat in profile["patterns"] if pat and pat in raw_lower]
        if not hits:
            continue
        score = float(profile["base_impact"] + (0.07 * min(3, len(hits))))
        if symbol_specific >= 0.55:
            score += 0.06
        if max(list(theme_scores.values()) or [0.0]) >= 0.45:
            score += 0.03
        if score > best_score:
            best_score = score
            best = (event_type, profile)
    if best is None:
        best = (
            "general",
            {
                "label": "General",
                "horizon": "intraday",
                "base_impact": 0.24,
                "note": "General sector-tone headline; use only as low-conviction context.",
            },
        )
    event_type, profile = best
    source_weight = _source_weight(source)
    noise_score = _noise_score(text, source=source)
    impact_score = float(profile["base_impact"])
    if any(phrase in raw_lower for phrase in ("raises guidance", "cuts guidance", "beats estimates", "misses estimates", "antitrust probe", "export controls", "profit warning")):
        impact_score += 0.08
    if noise_score >= 0.55:
        impact_score *= 0.75
    impact_score = float(max(0.10, min(1.0, impact_score)))
    note = str(profile.get("note") or "")
    if noise_score >= 0.60:
        note = "Opinion or market-color headline; treat it as low-conviction context unless tape confirms."
    return {
        "event_type": str(event_type),
        "event_label": str(profile.get("label") or event_type.title()),
        "impact_score": impact_score,
        "impact_horizon": str(profile.get("horizon") or "intraday"),
        "source_weight": float(max(0.45, min(1.30, source_weight))),
        "noise_score": noise_score,
        "signal_note": note,
        "headline_signature": signature,
    }


def _novelty_scores(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype=float)
    if "headline_signature" in frame.columns:
        signature = frame["headline_signature"].fillna("").astype(str)
    else:
        signature = frame.get("headline", pd.Series("", index=frame.index)).fillna("").astype(str).map(_normalized_headline_signature)
    ts = pd.to_datetime(frame.get("published_at"), utc=True, errors="coerce")
    if ts is None or len(ts) != len(frame):
        ts = pd.Series(pd.Timestamp.now(tz="UTC"), index=frame.index)
    order = np.argsort(ts.fillna(pd.Timestamp.now(tz="UTC")).astype("int64", copy=False).to_numpy())
    out = np.ones(len(frame), dtype=float)
    seen: dict[str, list[pd.Timestamp]] = {}
    for pos in order:
        idx = frame.index[pos]
        sig = str(signature.loc[idx]).strip()
        cur_ts = ts.loc[idx]
        if pd.isna(cur_ts) or not sig:
            out[pos] = 1.0
            continue
        cur_ts = pd.Timestamp(cur_ts)
        history = [item for item in seen.get(sig, []) if (cur_ts - item).total_seconds() <= 7 * 86400.0]
        dup_recent = [item for item in history if (cur_ts - item).total_seconds() <= 72 * 3600.0]
        dup_count = len(dup_recent)
        if dup_count <= 0:
            novelty = 1.0
        elif dup_count == 1:
            novelty = 0.72
        elif dup_count == 2:
            novelty = 0.52
        else:
            novelty = 0.36
        out[pos] = novelty
        history.append(cur_ts)
        seen[sig] = history[-8:]
    return pd.Series(out, index=frame.index, dtype=float)


def _alias_hit_score(text: str, aliases: list[str]) -> float:
    text_lower = str(text or "").lower()
    best = 0.0
    for alias in aliases:
        raw = str(alias or "").strip()
        if not raw:
            continue
        pattern = rf"(?<![A-Za-z0-9]){re.escape(raw.lower())}(?![A-Za-z0-9])"
        if re.search(pattern, text_lower):
            best = max(best, 0.96 if len(raw) <= 6 else 0.88)
    return float(max(0.0, min(1.0, best)))


def _theme_relevance_scores(headline: str, toks: list[str], instrument_symbol: str | None = None) -> dict[str, float]:
    text = str(headline or "").lower()
    text_norm = " ".join(toks)
    meta = _instrument_meta(instrument_symbol)
    weights = dict(meta.get("news_theme_weights") or {})
    sector = str(meta.get("sector") or "").lower()
    sector_theme = _THEME_ALIAS_BY_SECTOR.get(sector)
    scores: dict[str, float] = {}
    token_set = set(toks)
    for theme, keywords in _SECTOR_THEME_KEYWORDS.items():
        hits = 0.0
        for term in keywords:
            term_text = str(term or "").lower()
            if not term_text:
                continue
            if " " in term_text:
                if term_text in text:
                    hits += 1.0
            elif term_text in token_set:
                hits += 1.0
        base = 0.18 * min(4.0, hits)
        if sector_theme == theme and hits > 0:
            base += 0.28
        if str((meta.get("context_daily_symbols") or {}).get("sector") or "").lower() in text:
            base += 0.18 if sector_theme == theme else 0.08
        if theme == "technology" and any(p in text for p in _DIRECT_QQQ_PHRASES):
            base = max(base, 0.78)
        weight = _safe_float(weights.get(theme)) or 0.0
        scores[theme] = float(max(0.0, min(1.0, base * (0.50 + 0.50 * weight))))
    scores["broad_market"] = float(
        max(0.0, min(1.0, (0.16 * min(4.0, sum(1 for t in toks if t in _MARKET_KEYWORDS))) * (0.45 + 0.55 * (_safe_float(weights.get("broad_market")) or 0.55))))
    )
    return scores


def _symbol_specific_relevance(headline: str, toks: list[str], instrument_symbol: str | None = None) -> float:
    sym = normalize_symbol(instrument_symbol)
    if not sym:
        return 0.0
    meta = _instrument_meta(sym)
    aliases = _alias_terms(sym)
    score = _alias_hit_score(headline, aliases)
    if sym == "QQQ" and any(p in str(headline or "").lower() for p in _DIRECT_QQQ_PHRASES):
        score = max(score, 0.96)
    sector_theme = _THEME_ALIAS_BY_SECTOR.get(str(meta.get("sector") or "").lower())
    weights = dict(meta.get("news_theme_weights") or {})
    if sector_theme == "technology" and score < 0.55:
        tech_hits = sum(1 for t in toks if t in _TECH_KEYWORDS or t in _TECH_ENTITIES)
        if tech_hits >= 3 and sym in {"QQQ", "SMH", "SOXX"}:
            score = max(score, 0.58)
    if str(meta.get("instrument_kind") or "").lower() == "etf" and score < 0.55:
        market_hits = sum(1 for t in toks if t in _MARKET_KEYWORDS)
        if market_hits >= 2:
            score = max(score, 0.22 + 0.38 * (_safe_float(weights.get("broad_market")) or 0.6))
    return float(max(0.0, min(1.0, score)))


def _tech_relevance(headline: str, toks: list[str], instrument_symbol: str | None = None) -> float:
    theme_scores = _theme_relevance_scores(headline, toks, instrument_symbol=instrument_symbol)
    return float(max(theme_scores.get("technology", 0.0), theme_scores.get("communications", 0.0) * 0.60))


def _political_relevance(headline: str, toks: list[str], instrument_symbol: str | None = None) -> float:
    text = str(headline or "").lower()
    text_norm = " ".join(toks)
    meta = _instrument_meta(instrument_symbol)
    policy_hits = sum(1 for t in toks if t in _POLITICAL_KEYWORDS)
    entity_hits = sum(1 for e in _POLITICAL_ENTITIES if e in text)
    macro_hits = sum(1 for t in toks if t in _MACRO_KEYWORDS)
    phrase_hits = sum(1 for p in set(_POLITICAL_POS_PHRASES) | set(_POLITICAL_NEG_PHRASES) if p in text_norm)
    base = (0.15 * min(4.0, float(policy_hits))) + (0.20 * min(3.0, float(entity_hits))) + (0.14 * min(3.0, float(phrase_hits)))
    if macro_hits > 0:
        base += 0.08 * min(3.0, float(macro_hits))
    base *= (0.40 + 0.60 * max(_safe_float(meta.get("macro_sensitivity")) or 0.0, _safe_float(meta.get("regulatory_sensitivity")) or 0.0))
    if _symbol_specific_relevance(headline, toks, instrument_symbol=instrument_symbol) >= 0.55 and ("sec" in toks or "ftc" in toks or "doj" in toks):
        base = max(base, 0.62)
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
            "symbol_specific_relevance": 0.0,
            "sector_relevance": 0.0,
            "macro_relevance": 0.0,
            "broad_market_relevance": 0.0,
            "relevance_scope": "low_value_noise",
            "relevance_scope_label": _SCOPE_LABELS["low_value_noise"],
            "theme_bucket": "broad_market",
            "theme_relevance": 0.0,
            "symbol_specific": False,
            "sector_specific": False,
            "macro_relevant": False,
            "broad_market_relevant": False,
            "opinionated": False,
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
    theme_scores = _theme_relevance_scores(headline, toks, instrument_symbol=instrument_symbol)
    theme_items = sorted(theme_scores.items(), key=lambda item: float(item[1]), reverse=True)
    theme_bucket, theme_relevance = theme_items[0] if theme_items else ("broad_market", 0.0)
    symbol_specific_relevance = _symbol_specific_relevance(headline, toks, instrument_symbol=instrument_symbol)
    sector_values = [value for key, value in theme_scores.items() if key != "broad_market"]
    sector_relevance = float(max(sector_values or [0.0]))
    broad_market_relevance = float(theme_scores.get("broad_market", 0.0))
    political_relevance = _political_relevance(headline, toks, instrument_symbol=instrument_symbol)
    opinion_like = bool(_noise_score(headline) >= 0.60 or "opinion" in text_norm or "editorial" in text_norm)
    relevance = float(max(symbol_specific_relevance, sector_relevance, political_relevance, broad_market_relevance))
    if opinion_like and symbol_specific_relevance < 0.55:
        relevance *= 0.72
    if relevance <= 0.0:
        score = 0.0
    else:
        # Downweight weakly relevant stories.
        score *= (0.45 + 0.55 * relevance)
    score = float(max(-1.0, min(1.0, score)))
    scope = "low_value_noise"
    scope_score = 0.0
    if opinion_like and symbol_specific_relevance < 0.55 and sector_relevance < 0.45 and political_relevance < 0.35:
        scope = "opinion_commentary"
        scope_score = max(relevance, 0.20)
    if symbol_specific_relevance >= max(sector_relevance, political_relevance, broad_market_relevance, 0.52):
        scope = "symbol_specific"
        scope_score = symbol_specific_relevance
    elif sector_relevance >= max(political_relevance, broad_market_relevance, 0.42):
        scope = "sector_specific"
        scope_score = sector_relevance
    elif political_relevance >= max(broad_market_relevance, 0.35):
        scope = "macro_political_regulatory"
        scope_score = political_relevance
    elif broad_market_relevance >= 0.28:
        scope = "broad_market"
        scope_score = broad_market_relevance
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
        "tech_relevant": bool(_tech_relevance(headline, toks, instrument_symbol=instrument_symbol) >= 0.25),
        "political_relevance": float(political_relevance),
        "political_relevant": bool(political_relevance >= 0.25),
        "symbol_specific_relevance": float(symbol_specific_relevance),
        "sector_relevance": float(sector_relevance),
        "macro_relevance": float(political_relevance),
        "broad_market_relevance": float(broad_market_relevance),
        "relevance_scope": scope,
        "relevance_scope_label": _SCOPE_LABELS.get(scope, "Relevant"),
        "scope_score": float(scope_score),
        "theme_bucket": str(theme_bucket),
        "theme_relevance": float(theme_relevance),
        "symbol_specific": bool(symbol_specific_relevance >= 0.52),
        "sector_specific": bool(sector_relevance >= 0.42),
        "macro_relevant": bool(political_relevance >= 0.25),
        "broad_market_relevant": bool(broad_market_relevance >= 0.28),
        "opinionated": opinion_like,
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
    event_details = []
    sources = out.get("source", pd.Series("", index=out.index)).fillna("").astype(str).tolist()
    for headline, sym, source in zip(out["headline"].astype(str).tolist(), out["symbol"].astype(str).tolist(), sources):
        details.append(_headline_sentiment_detail(headline, instrument_symbol=sym or default_symbol))
        event_details.append(_headline_event_detail(headline, source=source, instrument_symbol=sym or default_symbol))
    det_df = pd.DataFrame(details, index=out.index)
    evt_df = pd.DataFrame(event_details, index=out.index)
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
    out["symbol_specific_relevance"] = pd.to_numeric(
        det_df.get("symbol_specific_relevance", out.get("symbol_specific_relevance")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["sector_relevance_score"] = pd.to_numeric(
        det_df.get("sector_relevance", out.get("sector_relevance_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["macro_relevance_score"] = pd.to_numeric(
        det_df.get("macro_relevance", out.get("macro_relevance_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["broad_market_relevance_score"] = pd.to_numeric(
        det_df.get("broad_market_relevance", out.get("broad_market_relevance_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["symbol_specific"] = pd.to_numeric(
        det_df.get("symbol_specific", out.get("symbol_specific")),
        errors="coerce",
    ).fillna(0).astype(int)
    out["sector_specific"] = pd.to_numeric(
        det_df.get("sector_specific", out.get("sector_specific")),
        errors="coerce",
    ).fillna(0).astype(int)
    out["macro_relevant"] = pd.to_numeric(
        det_df.get("macro_relevant", out.get("macro_relevant")),
        errors="coerce",
    ).fillna(0).astype(int)
    out["broad_market_relevant"] = pd.to_numeric(
        det_df.get("broad_market_relevant", out.get("broad_market_relevant")),
        errors="coerce",
    ).fillna(0).astype(int)
    out["opinionated"] = pd.to_numeric(
        det_df.get("opinionated", out.get("opinionated")),
        errors="coerce",
    ).fillna(0).astype(int)
    out["relevance_scope"] = det_df.get("relevance_scope", out.get("relevance_scope", "low_value_noise")).astype(str)
    out["relevance_scope_label"] = det_df.get("relevance_scope_label", out.get("relevance_scope_label", "Low-value / noise")).astype(str)
    out["scope_score"] = pd.to_numeric(
        det_df.get("scope_score", out.get("scope_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["theme_bucket"] = det_df.get("theme_bucket", out.get("theme_bucket", "broad_market")).astype(str)
    out["theme_relevance_score"] = pd.to_numeric(
        det_df.get("theme_relevance", out.get("theme_relevance_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["event_type"] = evt_df.get("event_type", out.get("event_type", "general")).astype(str).str.lower()
    out["event_label"] = evt_df.get("event_label", out.get("event_label", "General")).astype(str)
    out["impact_score"] = pd.to_numeric(
        evt_df.get("impact_score", out.get("impact_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["impact_horizon"] = evt_df.get("impact_horizon", out.get("impact_horizon", "intraday")).astype(str).str.lower()
    out["source_weight"] = pd.to_numeric(
        evt_df.get("source_weight", out.get("source_weight")),
        errors="coerce",
    ).fillna(0.85).clip(0.45, 1.30)
    out["noise_score"] = pd.to_numeric(
        evt_df.get("noise_score", out.get("noise_score")),
        errors="coerce",
    ).fillna(0.0).clip(0.0, 1.0)
    out["signal_note"] = evt_df.get("signal_note", out.get("signal_note", "")).astype(str)
    out["headline_signature"] = evt_df.get("headline_signature", out.get("headline_signature", "")).astype(str)
    novelty = _novelty_scores(out)
    if novelty is None or novelty.empty:
        out["novelty_score"] = 1.0
    else:
        out["novelty_score"] = pd.to_numeric(novelty, errors="coerce").reindex(out.index).fillna(1.0).clip(0.0, 1.0)
    relevance = pd.to_numeric(out.get("relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    sentiment = pd.to_numeric(out.get("sentiment_score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    scope_score = pd.to_numeric(out.get("scope_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    symbol_specific = pd.to_numeric(out.get("symbol_specific"), errors="coerce").fillna(0).astype(int)
    opinionated = pd.to_numeric(out.get("opinionated"), errors="coerce").fillna(0).astype(int)
    base_importance = (0.30 + 0.70 * out["impact_score"]) * (0.30 + 0.70 * relevance)
    importance = (
        base_importance
        * (0.45 + 0.55 * scope_score)
        * (1.10 + 0.20 * symbol_specific)
        * out["source_weight"]
        * out["novelty_score"]
        * (1.0 - (0.65 * out["noise_score"]))
        * (0.80 + 0.20 * sentiment.abs())
    )
    low_value_mask = out["event_type"].isin({"opinion", "market_wrap", "capital_flows"}) & (relevance < 0.45)
    importance = importance.where(~low_value_mask, importance * 0.65)
    importance = importance.where(opinionated <= 0, importance * (0.78 + 0.22 * symbol_specific))
    out["importance_score"] = pd.to_numeric(importance, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    direction_score = sentiment * (0.35 + 0.65 * out["impact_score"]) * out["source_weight"] * out["novelty_score"] * (1.0 - 0.45 * out["noise_score"])
    out["direction_score"] = pd.to_numeric(direction_score, errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    out["signal_direction"] = np.where(
        out["direction_score"] > 0.05,
        "bullish",
        np.where(out["direction_score"] < -0.05, "bearish", "neutral"),
    )
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
    event = _headline_event_detail(title, source=source, instrument_symbol=symbol)
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
        "symbol_specific_relevance": float(sent.get("symbol_specific_relevance", 0.0)),
        "sector_relevance_score": float(sent.get("sector_relevance", 0.0)),
        "macro_relevance_score": float(sent.get("macro_relevance", 0.0)),
        "broad_market_relevance_score": float(sent.get("broad_market_relevance", 0.0)),
        "symbol_specific": int(1 if sent.get("symbol_specific") else 0),
        "sector_specific": int(1 if sent.get("sector_specific") else 0),
        "macro_relevant": int(1 if sent.get("macro_relevant") else 0),
        "broad_market_relevant": int(1 if sent.get("broad_market_relevant") else 0),
        "opinionated": int(1 if sent.get("opinionated") else 0),
        "relevance_scope": str(sent.get("relevance_scope") or "low_value_noise"),
        "relevance_scope_label": str(sent.get("relevance_scope_label") or _SCOPE_LABELS["low_value_noise"]),
        "scope_score": float(sent.get("scope_score", 0.0)),
        "theme_bucket": str(sent.get("theme_bucket") or "broad_market"),
        "theme_relevance_score": float(sent.get("theme_relevance", 0.0)),
        "event_type": str(event.get("event_type") or "general"),
        "event_label": str(event.get("event_label") or "General"),
        "impact_score": float(event.get("impact_score") or 0.0),
        "impact_horizon": str(event.get("impact_horizon") or "intraday"),
        "source_weight": float(event.get("source_weight") or 0.85),
        "noise_score": float(event.get("noise_score") or 0.0),
        "signal_note": str(event.get("signal_note") or ""),
        "headline_signature": str(event.get("headline_signature") or ""),
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
    src = str(item.get("domain") or item.get("sourcecountry") or "archive")
    event = _headline_event_detail(title, source=src, instrument_symbol=symbol)
    uid = f"archive|{url}|{ts.isoformat()}"
    return {
        "symbol": symbol.upper(),
        "headline_id": uid,
        "published_at": ts.isoformat(),
        "headline": str(title),
        "source": src,
        "url": str(url),
        "sentiment_score": float(sent["score"]),
        "sentiment_label": str(sent["label"]),
        "relevance_score": float(sent["relevance"]),
        "tech_relevant": int(1 if sent["tech_relevant"] else 0),
        "political_relevance_score": float(sent.get("political_relevance", 0.0)),
        "political_relevant": int(1 if sent.get("political_relevant") else 0),
        "symbol_specific_relevance": float(sent.get("symbol_specific_relevance", 0.0)),
        "sector_relevance_score": float(sent.get("sector_relevance", 0.0)),
        "macro_relevance_score": float(sent.get("macro_relevance", 0.0)),
        "broad_market_relevance_score": float(sent.get("broad_market_relevance", 0.0)),
        "symbol_specific": int(1 if sent.get("symbol_specific") else 0),
        "sector_specific": int(1 if sent.get("sector_specific") else 0),
        "macro_relevant": int(1 if sent.get("macro_relevant") else 0),
        "broad_market_relevant": int(1 if sent.get("broad_market_relevant") else 0),
        "opinionated": int(1 if sent.get("opinionated") else 0),
        "relevance_scope": str(sent.get("relevance_scope") or "low_value_noise"),
        "relevance_scope_label": str(sent.get("relevance_scope_label") or _SCOPE_LABELS["low_value_noise"]),
        "scope_score": float(sent.get("scope_score", 0.0)),
        "theme_bucket": str(sent.get("theme_bucket") or "broad_market"),
        "theme_relevance_score": float(sent.get("theme_relevance", 0.0)),
        "event_type": str(event.get("event_type") or "general"),
        "event_label": str(event.get("event_label") or "General"),
        "impact_score": float(event.get("impact_score") or 0.0),
        "impact_horizon": str(event.get("impact_horizon") or "intraday"),
        "source_weight": float(event.get("source_weight") or 0.85),
        "noise_score": float(event.get("noise_score") or 0.0),
        "signal_note": str(event.get("signal_note") or ""),
        "headline_signature": str(event.get("headline_signature") or ""),
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


def _fetch_latest_headlines_result(symbol: str = "QQQ", max_items: int = 200) -> tuple[list[dict], str | None]:
    if not NEWS_PIPELINE_ENABLED:
        return [], "disabled"
    try:
        ticker = yf.Ticker(symbol)
        raw = ticker.news or []
    except Exception as exc:
        logger.debug("News fetch failed for %s", symbol, exc_info=True)
        return [], str(exc)
    out = []
    for item in raw[:max_items]:
        parsed = _parse_item(item, symbol=symbol)
        if parsed:
            out.append(parsed)
    return out, None


def fetch_latest_headlines(symbol: str = "QQQ", max_items: int = 200) -> list[dict]:
    rows, _ = _fetch_latest_headlines_result(symbol=symbol, max_items=max_items)
    return rows


def update_news_store(symbol: str = "QQQ", max_items: int = 200) -> dict:
    symbol = str(symbol or "QQQ").upper()
    path = _news_path(symbol)
    current = _load_existing_news(path)
    rows, fetch_error = _fetch_latest_headlines_result(symbol=symbol, max_items=max_items)
    incoming = pd.DataFrame(rows)
    if incoming.empty:
        status = "error" if fetch_error else "ok"
        reason = fetch_error or "no_new_headlines"
        return {"symbol": symbol, "status": status, "reason": reason, "added": 0, "rows": int(len(current))}
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
    return {"symbol": symbol, "status": "ok", "added": added, "rows": int(len(merged))}


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
            health = get_news_health(symbol)
            return {
                "symbol": symbol,
                "skipped": True,
                "reason": "cooldown",
                "status": str(st.get("last_status") or "cooldown"),
                "health": health,
            }
    out = update_news_store(symbol=symbol, max_items=200)
    local_import = None
    coverage = news_archive_coverage(symbol)
    try:
        last_local_import = pd.to_datetime(st.get("last_local_import"), utc=True, errors="coerce")
    except Exception:
        last_local_import = pd.NaT
    import_due = pd.isna(last_local_import) or (now - last_local_import).total_seconds() >= max(6 * 3600, int(NEWS_REFRESH_COOLDOWN_SEC) * 4)
    if not bool((coverage or {}).get("sufficient_long_horizon", False)) and import_due:
        local_import = import_local_historical_news(symbol=symbol)
        coverage = (local_import or {}).get("coverage") or news_archive_coverage(symbol)
        st["last_local_import"] = now.isoformat()
    st["last_refresh"] = now.isoformat()
    st["last_rows"] = out.get("rows")
    st["last_status"] = str(out.get("status") or "ok")
    st["last_reason"] = str(out.get("reason") or "")
    st["history_ready"] = bool((coverage or {}).get("sufficient_long_horizon", False))
    st["coverage_ratio"] = _safe_float((coverage or {}).get("coverage_ratio"))
    _save_state(symbol, st)
    health = get_news_health(symbol)
    out["coverage"] = coverage
    out["history_ready"] = bool((coverage or {}).get("sufficient_long_horizon", False))
    out["local_import"] = local_import
    out["backfill_recommended"] = not bool((coverage or {}).get("sufficient_long_horizon", False))
    out["health"] = health
    return out


def news_archive_coverage(symbol: str = "QQQ", trading_dates=None, tz: str = "America/New_York", as_of=None) -> dict:
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
    cutoff_ts = _coerce_utc_ts(as_of)
    if cutoff_ts is not None:
        df = df[df["published_at"] <= cutoff_ts].copy()
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
    if cutoff_ts is not None and len(trading_idx):
        cutoff_local = cutoff_ts.tz_convert(tz)
        trading_idx = trading_idx[trading_idx <= pd.Timestamp(cutoff_local.date())]
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


def get_news_health(symbol: str = "QQQ", trading_dates=None, tz: str = "America/New_York", as_of=None) -> dict:
    symbol = str(symbol or "QQQ").upper()
    out = {
        "symbol": symbol,
        "enabled": bool(NEWS_PIPELINE_ENABLED),
        "status": "disabled" if not NEWS_PIPELINE_ENABLED else "missing",
        "headline_state": "disabled" if not NEWS_PIPELINE_ENABLED else "missing",
        "history_state": "missing",
        "signal_ready": False,
        "history_ready": False,
    }
    if not NEWS_PIPELINE_ENABLED:
        return out
    path = _news_path(symbol)
    state = _load_state(symbol)
    coverage = news_archive_coverage(symbol, trading_dates=trading_dates, tz=tz, as_of=as_of)
    out["coverage"] = coverage
    out["history_ready"] = bool((coverage or {}).get("sufficient_long_horizon", False))
    out["history_state"] = "ready" if out["history_ready"] else "insufficient_history"
    out["last_refresh"] = state.get("last_refresh")
    out["last_status"] = state.get("last_status")
    out["last_reason"] = state.get("last_reason")
    if not os.path.exists(path):
        if str(state.get("last_status") or "").lower() == "error":
            out["status"] = "failed"
            out["headline_state"] = "failed"
        return out
    try:
        df = _normalize_news_frame(pd.read_csv(path), default_symbol=symbol)
    except Exception as exc:
        out["status"] = "error"
        out["headline_state"] = "failed"
        out["last_reason"] = str(exc)
        return out
    if df is None or df.empty:
        return out
    df["published_at"] = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
    df = df.dropna(subset=["published_at"]).copy()
    now_utc = _coerce_utc_ts(as_of) or pd.Timestamp.now(tz="UTC")
    df = df[df["published_at"] <= now_utc].copy()
    if df.empty:
        return out
    df["importance_score"] = pd.to_numeric(df.get("importance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["direction_score"] = pd.to_numeric(df.get("direction_score"), errors="coerce").fillna(0.0).clip(-1.0, 1.0)
    df["relevance_score"] = pd.to_numeric(df.get("relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    meaningful = df[(df["importance_score"] >= 0.18) | (df["relevance_score"] >= 0.25) | (df["direction_score"].abs() >= 0.08)].copy()
    last_ts = meaningful["published_at"].max() if not meaningful.empty else df["published_at"].max()
    age_hours = float((now_utc - last_ts).total_seconds() / 3600.0) if pd.notna(last_ts) else None
    freshness_limit = 96.0 if int(now_utc.dayofweek) >= 5 else 72.0
    unique_days = int(df["published_at"].dt.floor("D").nunique())
    meaningful_days = int(meaningful["published_at"].dt.floor("D").nunique()) if not meaningful.empty else 0
    importance_std = float(df["importance_score"].std()) if len(df) else 0.0
    direction_std = float(df["direction_score"].std()) if len(df) else 0.0
    high_impact_recent = 0
    if not meaningful.empty:
        fresh_mask = (now_utc - meaningful["published_at"]).dt.total_seconds() <= 24 * 3600.0
        high_impact_recent = int((meaningful.loc[fresh_mask, "importance_score"] >= 0.40).sum())
    signal_ready = bool(
        age_hours is not None
        and age_hours <= freshness_limit
        and meaningful_days >= 3
        and max(importance_std, direction_std) > 1e-6
        and len(meaningful) >= 3
    )
    headline_state = "ok"
    if len(meaningful) <= 0:
        headline_state = "no_meaningful_news"
    elif age_hours is not None and age_hours > freshness_limit:
        headline_state = "stale"
    elif str(state.get("last_status") or "").lower() == "error" and age_hours is not None and age_hours > 24.0:
        headline_state = "failed"
    elif not signal_ready:
        headline_state = "thin"
    status = "ok" if headline_state == "ok" else "degraded"
    if headline_state in {"failed", "stale"}:
        status = headline_state
    out.update(
        {
            "status": status,
            "headline_state": headline_state,
            "signal_ready": signal_ready,
            "rows": int(len(df)),
            "meaningful_rows": int(len(meaningful)),
            "unique_days": unique_days,
            "meaningful_days": meaningful_days,
            "last_published_at": last_ts.isoformat() if pd.notna(last_ts) else None,
            "age_hours": round(age_hours, 2) if age_hours is not None else None,
            "importance_std": round(importance_std, 6),
            "direction_std": round(direction_std, 6),
            "high_impact_recent": int(high_impact_recent),
        }
    )
    return out


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
    df["symbol_specific"] = pd.to_numeric(df.get("symbol_specific"), errors="coerce").fillna(0).astype(int)
    df["sector_specific"] = pd.to_numeric(df.get("sector_specific"), errors="coerce").fillna(0).astype(int)
    df["macro_relevant"] = pd.to_numeric(df.get("macro_relevant"), errors="coerce").fillna(0).astype(int)
    df["broad_market_relevant"] = pd.to_numeric(df.get("broad_market_relevant"), errors="coerce").fillna(0).astype(int)
    df["political_relevance_score"] = pd.to_numeric(df.get("political_relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["political_relevant"] = pd.to_numeric(df.get("political_relevant"), errors="coerce").fillna(0).astype(int)
    df["importance_score"] = pd.to_numeric(df.get("importance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["impact_score"] = pd.to_numeric(df.get("impact_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["source_weight"] = pd.to_numeric(df.get("source_weight"), errors="coerce").fillna(0.85).clip(0.45, 1.30)
    df["novelty_score"] = pd.to_numeric(df.get("novelty_score"), errors="coerce").fillna(1.0).clip(0.0, 1.0)
    df["noise_score"] = pd.to_numeric(df.get("noise_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    # Keep only stories that are relevant for the active workspace profile.
    df = df[
        (df["symbol_specific"] > 0)
        | (df["sector_specific"] > 0)
        | (df["macro_relevant"] > 0)
        | (df["broad_market_relevant"] > 0)
        | (df["tech_relevant"] > 0)
        | (df["political_relevant"] > 0)
        | (df["relevance_score"] >= 0.22)
    ].copy()
    if df.empty:
        return pd.DataFrame(0.0, index=index_utc, columns=cols)
    # Combine relevance, event impact, novelty and source quality into the live signal weight.
    signal_weight = (
        (0.30 + 0.70 * df["relevance_score"])
        * (0.35 + 0.65 * df["importance_score"])
        * (0.40 + 0.60 * df["impact_score"])
        * (0.45 + 0.55 * df["novelty_score"])
        * (0.60 + 0.40 * df["source_weight"])
        * (1.0 - 0.50 * df["noise_score"])
    ).clip(lower=0.0)
    df["sentiment_score"] = df["sentiment_score"] * signal_weight
    df["political_sentiment_score"] = df["sentiment_score"] * df["political_relevance_score"] * (0.70 + 0.30 * df["importance_score"])
    df["event_weight"] = (
        (0.35 + 0.65 * df["importance_score"])
        * (0.35 + 0.65 * df["impact_score"])
        * (1.0 + df["sentiment_score"].abs())
        * (1.0 + (0.35 * df["political_relevance_score"]))
        * (0.55 + 0.45 * df["source_weight"])
        * (1.0 - 0.35 * df["noise_score"])
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


def load_recent_headlines(symbol: str = "QQQ", limit: int = 8, as_of=None) -> list[dict]:
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
    df["published_at"] = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
    now = _coerce_utc_ts(as_of) or pd.Timestamp.now(tz="UTC")
    df = df[df["published_at"] <= now].copy()
    if df.empty:
        return []
    age_hours = ((now - df["published_at"]).dt.total_seconds() / 3600.0).clip(lower=0.0)
    df["importance_score"] = pd.to_numeric(df.get("importance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["impact_score"] = pd.to_numeric(df.get("impact_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["noise_score"] = pd.to_numeric(df.get("noise_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    df["scope_score"] = pd.to_numeric(df.get("scope_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    freshness = np.exp(-(age_hours / 36.0))
    df["rank_score"] = (df["importance_score"] * freshness) + (0.20 * df["impact_score"]) + (0.15 * df["scope_score"]) - (0.10 * df["noise_score"])
    df = df.sort_values(["rank_score", "published_at"], ascending=[False, False]).head(max(1, int(limit))).copy()
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
                "symbol_specific_relevance": _safe_float(row.get("symbol_specific_relevance")),
                "sector_relevance": _safe_float(row.get("sector_relevance_score")),
                "macro_relevance": _safe_float(row.get("macro_relevance_score")),
                "broad_market_relevance": _safe_float(row.get("broad_market_relevance_score")),
                "source": str(row.get("source") or "").strip(),
                "url": str(row.get("url") or "").strip(),
                "event_type": str(row.get("event_type") or "general").strip(),
                "event_label": str(row.get("event_label") or "General").strip(),
                "impact_score": _safe_float(row.get("impact_score")),
                "impact_horizon": str(row.get("impact_horizon") or "").strip(),
                "source_weight": _safe_float(row.get("source_weight")),
                "novelty_score": _safe_float(row.get("novelty_score")),
                "noise_score": _safe_float(row.get("noise_score")),
                "importance_score": _safe_float(row.get("importance_score")),
                "direction_score": _safe_float(row.get("direction_score")),
                "signal_direction": str(row.get("signal_direction") or "").strip(),
                "signal_note": str(row.get("signal_note") or "").strip(),
                "relevance_scope": str(row.get("relevance_scope") or "low_value_noise").strip(),
                "relevance_scope_label": str(row.get("relevance_scope_label") or _SCOPE_LABELS["low_value_noise"]).strip(),
                "theme_bucket": str(row.get("theme_bucket") or "broad_market").strip(),
                "theme_relevance_score": _safe_float(row.get("theme_relevance_score")),
            }
        )
    return rows


def register_optional_news_scorer(name: str, scorer: Callable[[str, list[dict], dict], list[dict] | dict | None]):
    scorer_name = str(name or "").strip().lower()
    if not scorer_name:
        raise ValueError("A scorer name is required.")
    _OPTIONAL_NEWS_SCORERS[scorer_name] = scorer


def unregister_optional_news_scorer(name: str):
    _OPTIONAL_NEWS_SCORERS.pop(str(name or "").strip().lower(), None)


def list_optional_news_scorers() -> list[str]:
    return sorted(_OPTIONAL_NEWS_SCORERS.keys())


def _profile_exposure_summary(meta: dict | None) -> dict:
    payload = dict(meta or {})
    weights = dict(payload.get("news_theme_weights") or {})
    ranked_themes = [
        {"theme": str(theme), "weight": float(score)}
        for theme, score in sorted(weights.items(), key=lambda item: float(item[1]), reverse=True)
        if _safe_float(score) is not None
    ]
    context_symbols = []
    seen_context = set()
    for raw in list((payload.get("context_daily_symbols") or {}).values()):
        symbol = normalize_symbol(raw)
        if symbol and symbol not in seen_context:
            seen_context.add(symbol)
            context_symbols.append(symbol)
    aliases = []
    seen_alias = set()
    for raw in list(payload.get("news_aliases") or []):
        text = str(raw or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_alias:
            continue
        seen_alias.add(key)
        aliases.append(text)
    return {
        "aliases": aliases[:6],
        "context_symbols": context_symbols,
        "dominant_themes": ranked_themes[:3],
        "macro_sensitivity": _safe_float(payload.get("macro_sensitivity")),
        "regulatory_sensitivity": _safe_float(payload.get("regulatory_sensitivity")),
        "broad_market_sensitivity": _safe_float(payload.get("broad_market_sensitivity")),
    }


def _layer_b_candidate_score(item: dict, meta: dict | None = None) -> float:
    payload = dict(meta or {})
    weights = dict(payload.get("news_theme_weights") or {})
    scope = str(item.get("relevance_scope") or "low_value_noise").strip() or "low_value_noise"
    theme = str(item.get("theme_bucket") or "broad_market").strip() or "broad_market"
    relevance = _safe_float(item.get("relevance")) or _safe_float(item.get("relevance_score")) or 0.0
    impact = _safe_float(item.get("impact_score")) or 0.0
    importance = _safe_float(item.get("importance_score")) or 0.0
    scope_score = _safe_float(item.get("scope_score")) or 0.0
    theme_score = _safe_float(item.get("theme_relevance")) or _safe_float(item.get("theme_relevance_score")) or 0.0
    theme_weight = _safe_float(weights.get(theme)) or _safe_float(weights.get("broad_market")) or 0.0
    macro_weight = max(_safe_float(payload.get("macro_sensitivity")) or 0.0, _safe_float(weights.get("macro")) or 0.0)
    regulatory_weight = max(_safe_float(payload.get("regulatory_sensitivity")) or 0.0, _safe_float(weights.get("regulatory")) or 0.0)
    broad_weight = max(_safe_float(payload.get("broad_market_sensitivity")) or 0.0, _safe_float(weights.get("broad_market")) or 0.0)
    instrument_kind = str(payload.get("instrument_kind") or payload.get("type") or "equity").strip().lower()
    scope_boost = 0.55
    if scope == "symbol_specific":
        scope_boost = 1.28
    elif scope == "sector_specific":
        scope_boost = 0.92 + 0.38 * max(theme_weight, theme_score)
    elif scope == "macro_political_regulatory":
        scope_boost = 0.95 + 0.40 * max(macro_weight, regulatory_weight)
    elif scope == "broad_market":
        scope_boost = 0.88 + 0.32 * broad_weight
    elif scope == "opinion_commentary":
        scope_boost = 0.42
    if instrument_kind == "etf" and scope in {"macro_political_regulatory", "broad_market"}:
        scope_boost += 0.08
    if bool(item.get("opinionated")) and scope != "symbol_specific":
        scope_boost *= 0.82
    base = (0.42 * max(importance, impact, relevance)) + (0.24 * scope_score) + (0.18 * theme_score) + (0.16 * relevance)
    return float(max(0.0, base * scope_boost))


def build_news_runtime_summary(
    symbol: str,
    *,
    headlines: list[dict] | None = None,
    news_health: dict | None = None,
    as_of=None,
    include_optional: bool = True,
    optional_limit: int = 3,
) -> dict:
    sym = str(symbol or NEWS_PIPELINE_SYMBOL or "QQQ").upper()
    meta = _instrument_meta(sym)
    profile_exposures = _profile_exposure_summary(meta)
    health = dict(news_health or get_news_health(sym, as_of=as_of))
    items = list(headlines if headlines is not None else load_recent_headlines(sym, limit=10, as_of=as_of))
    scope_counts = {
        "symbol_specific": 0,
        "sector_specific": 0,
        "macro_political_regulatory": 0,
        "broad_market": 0,
        "low_value_noise": 0,
        "opinion_commentary": 0,
    }
    theme_counts: dict[str, int] = {}
    top_scope = ""
    top_scope_score = -1.0
    top_item: dict = {}
    high_impact = 0
    ranked_items: list[tuple[float, dict]] = []
    for item in items:
        scope = str(item.get("relevance_scope") or "low_value_noise").strip() or "low_value_noise"
        scope_counts[scope] = int(scope_counts.get(scope) or 0) + 1
        theme = str(item.get("theme_bucket") or "broad_market").strip() or "broad_market"
        theme_counts[theme] = int(theme_counts.get(theme) or 0) + 1
        score = max(
            _safe_float(item.get("relevance")) or 0.0,
            _safe_float(item.get("impact_score")) or 0.0,
            _safe_float(item.get("importance_score")) or 0.0,
            _safe_float(item.get("scope_score")) or 0.0,
        )
        ranked_items.append((score, item))
        if score > top_scope_score:
            top_scope_score = float(score)
            top_scope = scope
            top_item = dict(item)
        if (_safe_float(item.get("impact_score")) or 0.0) >= 0.45:
            high_impact += 1
    ranked_items.sort(key=lambda pair: float(pair[0]), reverse=True)
    scope_leaderboard = [
        {"scope": key, "label": _SCOPE_LABELS.get(key, key.replace("_", " ").title()), "count": int(value)}
        for key, value in sorted(scope_counts.items(), key=lambda item: (int(item[1]), item[0]), reverse=True)
        if int(value) > 0
    ]
    summary_bits = []
    if top_scope:
        summary_bits.append(str(_SCOPE_LABELS.get(top_scope, top_scope.replace("_", " ").title())))
    if theme_counts:
        theme = max(theme_counts.items(), key=lambda item: item[1])[0]
        summary_bits.append(f"theme {theme.replace('_', ' ')}")
    if high_impact:
        summary_bits.append(f"{int(high_impact)} high-impact")
    layer_a_summary = " | ".join(summary_bits) if summary_bits else "No structured headlines"

    candidate_rows = []
    for _rank_score, item in ranked_items:
        candidate_rows.append((_layer_b_candidate_score(item, meta), item))
    candidate_rows = [row for row in candidate_rows if float(row[0]) > 0.0]
    candidate_rows.sort(key=lambda pair: float(pair[0]), reverse=True)
    candidate_budget = max(1, int(optional_limit))
    candidate_payloads = []
    for candidate_score, item in candidate_rows[:candidate_budget]:
        candidate_payloads.append(
            {
                "headline": str(item.get("headline") or ""),
                "scope": str(item.get("relevance_scope") or "low_value_noise"),
                "scope_label": str(item.get("relevance_scope_label") or _SCOPE_LABELS.get(str(item.get("relevance_scope") or "low_value_noise"), "Relevant")),
                "theme_bucket": str(item.get("theme_bucket") or "broad_market"),
                "candidate_score": float(round(candidate_score, 4)),
                "relevance": _safe_float(item.get("relevance")) or _safe_float(item.get("relevance_score")),
                "importance_score": _safe_float(item.get("importance_score")),
                "impact_score": _safe_float(item.get("impact_score")),
                "direction_score": _safe_float(item.get("direction_score")),
                "signal_direction": str(item.get("signal_direction") or ""),
            }
        )

    optional_rows = []
    scorer_names = list_optional_news_scorers()
    layer_b_status = "disabled"
    if include_optional and scorer_names and candidate_rows:
        layer_b_status = "ready"
        candidates = [dict(item) for _, item in candidate_rows[:candidate_budget]]
        for scorer_name in scorer_names:
            scorer = _OPTIONAL_NEWS_SCORERS.get(scorer_name)
            if scorer is None:
                continue
            try:
                payload = scorer(sym, candidates, dict(meta or {}))
            except Exception as exc:
                optional_rows.append({"scorer": scorer_name, "status": "error", "detail": str(exc)})
                continue
            optional_rows.append(
                {
                    "scorer": scorer_name,
                    "status": "ok",
                    "items": payload if isinstance(payload, (dict, list)) else [],
                }
            )

    return {
        "symbol": sym,
        "profile": {
            "instrument_kind": str(meta.get("instrument_kind") or meta.get("type") or "equity"),
            "sector": str(meta.get("sector") or "Broad Market"),
            "industry": str(meta.get("industry") or meta.get("sector") or "-"),
            "macro_sensitivity": _safe_float(meta.get("macro_sensitivity")),
            "regulatory_sensitivity": _safe_float(meta.get("regulatory_sensitivity")),
            "broad_market_sensitivity": _safe_float(meta.get("broad_market_sensitivity")),
            "news_theme_weights": dict(meta.get("news_theme_weights") or {}),
            "context_daily_symbols": dict(meta.get("context_daily_symbols") or {}),
            "profile_exposures": profile_exposures,
        },
        "health": health,
        "layer_a": {
            "status": "ok" if items else str(health.get("headline_state") or "missing"),
            "scope_counts": scope_counts,
            "theme_counts": theme_counts,
            "scope_leaderboard": scope_leaderboard,
            "top_scope": top_scope,
            "top_scope_label": _SCOPE_LABELS.get(top_scope, top_scope.replace("_", " ").title()),
            "top_item": top_item,
            "high_impact_count": int(high_impact),
            "summary": layer_a_summary,
            "items": items,
        },
        "layer_b": {
            "status": layer_b_status,
            "available_scorers": scorer_names,
            "candidate_budget": int(candidate_budget),
            "candidate_count": int(len(candidate_payloads)),
            "candidates": candidate_payloads,
            "scored_count": int(len(optional_rows)),
            "results": optional_rows,
        },
        "summary": layer_a_summary,
    }
