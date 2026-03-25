from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from advisory_engine import build_advisory_snapshot
from breadth_analytics import get_breadth_snapshot
from catalyst_providers import get_catalyst_provider_snapshot
from config import DATA_DIR, INSTRUMENTS, OPPORTUNITY_SCAN_CACHE_TTL_SEC, OPPORTUNITY_SCAN_MAX_SYMBOLS
from event_analytics import get_event_snapshot
from instrument_onboarding import list_runtime_instruments
from liquidity_analytics import get_liquidity_snapshot
from macro_events import get_macro_event_snapshot
from model_utils import load_models, predict_4min, predict_horizon
from news_pipeline import load_recent_headlines
from trader_profiles import get_trader_profile, score_style_fit

_SCAN_CACHE: dict[tuple, dict] = {}


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _price_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{str(symbol or '').upper()}.csv")


def _load_cached_intraday(symbol: str) -> pd.DataFrame:
    path = _price_path(symbol)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.columns = [str(c).lower() for c in df.columns]
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def _candidate_symbols(max_symbols: int) -> list[str]:
    try:
        symbols = list_runtime_instruments() or list(INSTRUMENTS.keys())
    except Exception:
        symbols = list(INSTRUMENTS.keys())
    out = []
    for sym in symbols:
        path = _price_path(sym)
        if os.path.exists(path):
            out.append(str(sym).upper())
    return out[: int(max_symbols)]


def _score_candidate(symbol: str, advisory: dict, profile_name: str) -> dict:
    quality = dict(advisory.get("quality_meta") or {})
    liquidity = dict(advisory.get("liquidity_meta") or {})
    catalyst = dict(advisory.get("catalyst_meta") or {})
    plan = dict(advisory.get("plan_meta") or {})
    profile = get_trader_profile(profile_name)
    weights = dict(profile.get("scan_weights") or {})
    quality_score = _safe_float(quality.get("score")) or 0.0
    agreement = _safe_float(quality.get("agreement")) or 0.0
    tradability = _safe_float(liquidity.get("tradability_score")) or 0.45
    catalyst_risk = _safe_float(catalyst.get("catalyst_risk_score")) or 0.0
    no_trade = bool(quality.get("no_trade"))
    strategy = str(quality.get("strategy") or "")
    horizon_key = "intraday"
    style_fit = score_style_fit(profile_name, strategy, horizon_key, options_bias=bool(profile.get("prefer_options")))
    score = (
        float(weights.get("quality", 0.38)) * quality_score
        + float(weights.get("tradability", 0.18)) * tradability
        + float(weights.get("agreement", 0.16)) * agreement
        + float(weights.get("catalyst", 0.10)) * max(0.0, 1.0 - catalyst_risk)
        + float(weights.get("style", 0.18)) * style_fit
    )
    if no_trade:
        score -= 0.30
    action = str(plan.get("action") or "No-trade")
    detail = f"{action} | Q {int(round(100.0 * quality_score))}/100 | T {int(round(100.0 * tradability))}/100"
    return {
        "symbol": symbol,
        "score": max(0.0, min(1.0, float(score))),
        "action": action,
        "detail": detail,
        "no_trade": no_trade,
        "strategy": strategy,
    }


def get_opportunity_scan(profile_name: str | None = None, max_symbols: int | None = None) -> dict:
    profile_key = get_trader_profile(profile_name).get("key")
    symbols = _candidate_symbols(max_symbols or OPPORTUNITY_SCAN_MAX_SYMBOLS)
    cache_key = (tuple(symbols), profile_key)
    cached = _SCAN_CACHE.get(cache_key)
    if isinstance(cached, dict) and (time.time() - float(cached.get("ts") or 0.0)) <= float(OPPORTUNITY_SCAN_CACHE_TTL_SEC):
        return dict(cached.get("payload") or {})

    candidates = []
    macro_snapshot = get_macro_event_snapshot()
    for symbol in symbols:
        df = _load_cached_intraday(symbol)
        if df.empty or "close" not in df.columns:
            continue
        models = load_models(instrument_key=symbol)
        if not models:
            continue
        try:
            intraday = predict_4min(models, df, instrument_key=symbol, log_side_effects=False)
            daily = predict_horizon(models, df, instrument_key=symbol, mode="DailyHigh", log_side_effects=False)
            weekend = predict_horizon(models, df, instrument_key=symbol, mode="WeekEnd", log_side_effects=False)
        except Exception:
            continue
        preds = {}
        if isinstance(intraday, dict) and not intraday.get("error"):
            preds["intraday"] = intraday
        if isinstance(daily, dict) and not daily.get("error"):
            preds["daily"] = daily
        if isinstance(weekend, dict) and not weekend.get("error"):
            preds["weekend"] = weekend
        if not preds:
            continue
        breadth = get_breadth_snapshot(symbol)
        catalyst = get_catalyst_provider_snapshot(symbol)
        liquidity = get_liquidity_snapshot(df, symbol)
        advisory = build_advisory_snapshot(
            df,
            symbol,
            predictions=preds,
            event_snapshot=get_event_snapshot(symbol),
            news_items=load_recent_headlines(symbol, limit=6),
            macro_snapshot=macro_snapshot,
            breadth_snapshot=breadth,
            liquidity_snapshot=liquidity,
            catalyst_snapshot=catalyst,
        )
        candidates.append(_score_candidate(symbol, advisory, profile_key))

    ordered = sorted(candidates, key=lambda x: (x.get("score") or 0.0), reverse=True)
    rows = [
        (
            str(item.get("symbol") or "-"),
            f"{int(round(100.0 * (_safe_float(item.get('score')) or 0.0)))}/100",
            str(item.get("detail") or "-"),
        )
        for item in ordered[:6]
    ]
    payload = {
        "summary": "No ranked opportunities" if not rows else f"{len(rows)} ranked opportunities",
        "rows": rows,
        "candidates": ordered,
        "profile": profile_key,
        "source_note": "Local cached scan using current models and cached data",
    }
    _SCAN_CACHE[cache_key] = {"ts": time.time(), "payload": payload}
    return payload
