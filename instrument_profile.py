from __future__ import annotations

from copy import deepcopy


_SECTOR_ALIASES = {
    "tech": "Technology",
    "technology": "Technology",
    "information technology": "Technology",
    "communication": "Communication Services",
    "communications": "Communication Services",
    "communication services": "Communication Services",
    "consumer": "Consumer Discretionary",
    "consumer discretionary": "Consumer Discretionary",
    "consumer staples": "Consumer Staples",
    "retail": "Consumer Discretionary",
    "health": "Healthcare",
    "healthcare": "Healthcare",
    "medical": "Healthcare",
    "finance": "Financials",
    "financial": "Financials",
    "financials": "Financials",
    "banking": "Financials",
    "industrial": "Industrials",
    "industrials": "Industrials",
    "energy": "Energy",
    "materials": "Materials",
    "real estate": "Real Estate",
    "reit": "Real Estate",
    "utilities": "Utilities",
    "broad market": "Broad Market",
    "index": "Broad Market",
}

_SECTOR_CONTEXT = {
    "Technology": {"sector": "XLK", "vol": "^VXN", "lead_fut": "NQ=F"},
    "Communication Services": {"sector": "XLC", "vol": "^VXN", "lead_fut": "NQ=F"},
    "Consumer Discretionary": {"sector": "XLY", "vol": "^VIX", "lead_fut": "ES=F"},
    "Consumer Staples": {"sector": "XLP", "vol": "^VIX", "lead_fut": "ES=F"},
    "Healthcare": {"sector": "XLV", "vol": "^VIX", "lead_fut": "ES=F"},
    "Financials": {"sector": "XLF", "vol": "^VIX", "lead_fut": "ES=F"},
    "Industrials": {"sector": "XLI", "vol": "^VIX", "lead_fut": "ES=F"},
    "Energy": {"sector": "XLE", "vol": "^VIX", "lead_fut": "CL=F"},
    "Materials": {"sector": "XLB", "vol": "^VIX", "lead_fut": "ES=F"},
    "Real Estate": {"sector": "XLRE", "vol": "^VIX", "lead_fut": "ES=F"},
    "Utilities": {"sector": "XLU", "vol": "^VIX", "lead_fut": "ES=F"},
    "Broad Market": {"sector": None, "vol": "^VIX", "lead_fut": "ES=F"},
}

_BASE_THEME_WEIGHTS = {
    "technology": 0.35,
    "consumer": 0.35,
    "healthcare": 0.30,
    "financials": 0.30,
    "industrials": 0.30,
    "energy": 0.25,
    "materials": 0.20,
    "utilities": 0.20,
    "real_estate": 0.20,
    "communications": 0.25,
    "macro": 0.55,
    "regulatory": 0.55,
    "broad_market": 0.55,
}

_SECTOR_THEME_OVERRIDES = {
    "Technology": {"technology": 1.0, "communications": 0.65, "macro": 0.78, "regulatory": 0.80, "broad_market": 0.72},
    "Communication Services": {"communications": 1.0, "technology": 0.72, "consumer": 0.42, "macro": 0.70, "regulatory": 0.75},
    "Consumer Discretionary": {"consumer": 1.0, "technology": 0.35, "macro": 0.70, "regulatory": 0.58, "broad_market": 0.62},
    "Consumer Staples": {"consumer": 0.85, "macro": 0.48, "regulatory": 0.50, "broad_market": 0.40},
    "Healthcare": {"healthcare": 1.0, "macro": 0.50, "regulatory": 0.88, "broad_market": 0.42},
    "Financials": {"financials": 1.0, "macro": 0.92, "regulatory": 0.82, "broad_market": 0.60},
    "Industrials": {"industrials": 1.0, "energy": 0.40, "macro": 0.72, "regulatory": 0.55, "broad_market": 0.55},
    "Energy": {"energy": 1.0, "materials": 0.45, "macro": 0.80, "regulatory": 0.62, "broad_market": 0.52},
    "Materials": {"materials": 1.0, "industrials": 0.45, "macro": 0.68, "regulatory": 0.52, "broad_market": 0.45},
    "Real Estate": {"real_estate": 1.0, "macro": 0.86, "regulatory": 0.58, "broad_market": 0.50},
    "Utilities": {"utilities": 1.0, "macro": 0.74, "regulatory": 0.58, "broad_market": 0.38},
    "Broad Market": {"technology": 0.55, "consumer": 0.55, "financials": 0.48, "macro": 0.88, "regulatory": 0.72, "broad_market": 1.0},
}

_SECTOR_STYLE = {
    "Technology": "growth",
    "Communication Services": "growth",
    "Consumer Discretionary": "growth",
    "Consumer Staples": "defensive",
    "Healthcare": "defensive",
    "Financials": "cyclical",
    "Industrials": "cyclical",
    "Energy": "cyclical",
    "Materials": "cyclical",
    "Real Estate": "yield",
    "Utilities": "defensive",
    "Broad Market": "balanced",
}

_ETF_SYMBOLS = {
    "QQQ",
    "SPY",
    "DIA",
    "IWM",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
    "TLT",
    "ARKK",
    "SMH",
    "SOXX",
}

_SYMBOL_OVERRIDES = {
    "QQQ": {
        "type": "etf",
        "display_name": "QQQ",
        "company_name": "Invesco QQQ Trust",
        "sector": "Technology",
        "industry": "Nasdaq 100 ETF",
        "news_aliases": ["QQQ", "Invesco QQQ", "Nasdaq 100", "Nasdaq-100"],
        "style_profile": "growth",
    },
    "SPY": {
        "type": "etf",
        "display_name": "SPY",
        "company_name": "SPDR S&P 500 ETF",
        "sector": "Broad Market",
        "industry": "S&P 500 ETF",
        "news_aliases": ["SPY", "S&P 500", "SPDR S&P 500"],
        "style_profile": "balanced",
    },
    "DIA": {
        "type": "etf",
        "display_name": "DIA",
        "company_name": "SPDR Dow Jones Industrial Average ETF",
        "sector": "Broad Market",
        "industry": "Dow ETF",
        "news_aliases": ["DIA", "Dow", "Dow Jones"],
    },
    "IWM": {
        "type": "etf",
        "display_name": "IWM",
        "company_name": "iShares Russell 2000 ETF",
        "sector": "Broad Market",
        "industry": "Russell 2000 ETF",
        "news_aliases": ["IWM", "Russell 2000"],
    },
}


def normalize_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _safe_float(value, fallback: float) -> float:
    try:
        out = float(value)
    except Exception:
        return float(fallback)
    if out != out:
        return float(fallback)
    return float(out)


def normalize_sector(value: str | None) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    lowered = raw.lower()
    mapped = _SECTOR_ALIASES.get(lowered)
    if mapped:
        return mapped
    return raw.title()


def infer_instrument_kind(symbol: str, meta: dict | None = None) -> str:
    sym = normalize_symbol(symbol)
    payload = dict(meta or {})
    raw_type = str(payload.get("type") or payload.get("instrument_kind") or "").strip().lower()
    raw_industry = str(payload.get("industry") or "").strip().lower()
    raw_company = str(payload.get("company_name") or payload.get("display_name") or "").strip().lower()
    if raw_type in {"etf", "fund"}:
        return "etf"
    if sym in _ETF_SYMBOLS:
        return "etf"
    if "etf" in raw_industry or "fund" in raw_industry or "trust" in raw_company:
        return "etf"
    return "equity"


def default_news_theme_weights(sector: str | None, instrument_kind: str = "equity") -> dict:
    weights = deepcopy(_BASE_THEME_WEIGHTS)
    sector_key = normalize_sector(sector) or "Broad Market"
    weights.update(_SECTOR_THEME_OVERRIDES.get(sector_key, {}))
    if str(instrument_kind).lower() == "etf":
        weights["broad_market"] = max(weights.get("broad_market", 0.0), 0.72)
        weights["macro"] = max(weights.get("macro", 0.0), 0.68)
    return {str(k): max(0.0, min(1.0, _safe_float(v, 0.0))) for k, v in weights.items()}


def default_context_daily_symbols(symbol: str, meta: dict | None = None) -> dict:
    sym = normalize_symbol(symbol)
    payload = dict(meta or {})
    override = dict(_SYMBOL_OVERRIDES.get(sym, {}))
    sector = normalize_sector(payload.get("sector") or override.get("sector")) or "Broad Market"
    kind = infer_instrument_kind(sym, payload or override)
    base = {
        "market": "SPY",
        "vol": "^VIX",
        "lead_fut": "ES=F",
        "rates": "TLT",
    }
    base.update(_SECTOR_CONTEXT.get(sector, {}))
    if sym in {"SPY", "VOO", "IVV"}:
        base["market"] = "SPY"
        base["sector"] = None
    elif sym in {"QQQ"}:
        base["market"] = "SPY"
        base["sector"] = "XLK"
        base["vol"] = "^VXN"
        base["lead_fut"] = "NQ=F"
    elif kind == "etf" and base.get("sector") == sym:
        base["sector"] = None
    custom = payload.get("context_daily_symbols")
    if isinstance(custom, dict):
        base.update(custom)
    return {str(k): str(v) for k, v in base.items() if v}


def _default_aliases(symbol: str, payload: dict) -> list[str]:
    aliases = []
    for raw in [
        symbol,
        payload.get("display_name"),
        payload.get("company_name"),
        *list(payload.get("news_aliases", []) or []),
    ]:
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
    return deduped or [symbol]


def build_instrument_meta(symbol: str, overrides: dict | None = None) -> dict:
    sym = normalize_symbol(symbol)
    payload = deepcopy(_SYMBOL_OVERRIDES.get(sym, {}))
    if isinstance(overrides, dict):
        payload.update(deepcopy(overrides))
    instrument_kind = infer_instrument_kind(sym, payload)
    sector = normalize_sector(payload.get("sector")) or normalize_sector(_SYMBOL_OVERRIDES.get(sym, {}).get("sector")) or "Broad Market"
    style = str(payload.get("style_profile") or _SECTOR_STYLE.get(sector) or ("balanced" if instrument_kind == "etf" else "growth")).strip().lower()
    theme_weights = default_news_theme_weights(sector, instrument_kind=instrument_kind)
    custom_weights = payload.get("news_theme_weights")
    if isinstance(custom_weights, dict):
        for key, value in custom_weights.items():
            theme_weights[str(key)] = max(0.0, min(1.0, _safe_float(value, theme_weights.get(str(key), 0.0))))

    macro_sensitivity = payload.get("macro_sensitivity")
    regulatory_sensitivity = payload.get("regulatory_sensitivity")
    broad_market_sensitivity = payload.get("broad_market_sensitivity")
    if macro_sensitivity is None:
        macro_sensitivity = 0.82 if sector in {"Financials", "Real Estate"} else 0.72 if instrument_kind == "etf" else 0.60
    if regulatory_sensitivity is None:
        regulatory_sensitivity = 0.88 if sector == "Healthcare" else 0.80 if sector in {"Technology", "Financials"} else 0.58
    if broad_market_sensitivity is None:
        broad_market_sensitivity = 0.85 if instrument_kind == "etf" else 0.62

    meta = {
        "type": "etf" if instrument_kind == "etf" else "equity",
        "instrument_kind": instrument_kind,
        "alpaca": sym,
        "yahoo": sym,
        "display_name": str(payload.get("display_name") or sym),
        "company_name": str(payload.get("company_name") or payload.get("display_name") or sym),
        "sector": sector,
        "industry": str(payload.get("industry") or (f"{sector} ETF" if instrument_kind == "etf" else sector)).strip(),
        "style_profile": style,
        "profile_label": f"{sector} {'ETF' if instrument_kind == 'etf' else 'Equity'}",
        "news_enabled": bool(payload.get("news_enabled", True)),
        "tz": str(payload.get("tz") or "America/New_York"),
        "session_open_minute": int(payload.get("session_open_minute", 9 * 60 + 30)),
        "session_close_minute": int(payload.get("session_close_minute", 16 * 60)),
        "tick_size": float(payload.get("tick_size", 0.01)),
        "vol_floor": float(payload.get("vol_floor", 0.00025)),
        "max_abs_ret_mult": float(payload.get("max_abs_ret_mult", 6.0)),
        "stale_multiplier": float(payload.get("stale_multiplier", 3.5)),
        "closed_max_stale_hours": float(payload.get("closed_max_stale_hours", 80.0)),
        "target_vol": float(payload.get("target_vol", 0.004)),
        "macro_sensitivity": max(0.0, min(1.0, _safe_float(macro_sensitivity, 0.60))),
        "regulatory_sensitivity": max(0.0, min(1.0, _safe_float(regulatory_sensitivity, 0.58))),
        "broad_market_sensitivity": max(0.0, min(1.0, _safe_float(broad_market_sensitivity, 0.62))),
        "news_theme_weights": theme_weights,
    }
    meta.update(payload)
    meta["type"] = "etf" if infer_instrument_kind(sym, meta) == "etf" else str(meta.get("type") or "equity").lower()
    meta["instrument_kind"] = "etf" if meta["type"] == "etf" else "equity"
    meta["sector"] = normalize_sector(meta.get("sector")) or sector
    meta["style_profile"] = str(meta.get("style_profile") or style).strip().lower()
    meta["display_name"] = str(meta.get("display_name") or sym)
    meta["company_name"] = str(meta.get("company_name") or meta.get("display_name") or sym)
    meta["profile_label"] = str(meta.get("profile_label") or f"{meta['sector']} {'ETF' if meta['instrument_kind'] == 'etf' else 'Equity'}")
    meta["news_aliases"] = _default_aliases(sym, meta)
    meta["context_daily_symbols"] = default_context_daily_symbols(sym, meta=meta)
    return meta


def summarize_instrument_meta(meta: dict | None, symbol: str | None = None) -> str:
    payload = build_instrument_meta(symbol or str((meta or {}).get("display_name") or ""), overrides=meta or {})
    kind = "ETF" if str(payload.get("instrument_kind") or "").lower() == "etf" else "Equity"
    sector = str(payload.get("sector") or "Unknown")
    style = str(payload.get("style_profile") or "balanced").replace("_", " ")
    return f"{kind} | {sector} | {style}"
