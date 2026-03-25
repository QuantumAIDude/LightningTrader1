from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd

from breadth_analytics import get_breadth_snapshot
from catalyst_providers import get_catalyst_provider_snapshot
from config import DATA_DIR, DEFAULT_INSTRUMENT, INSTRUMENTS
from event_analytics import get_event_snapshot
from features import BAR_MINUTES
from liquidity_analytics import get_liquidity_snapshot
from macro_events import get_macro_event_snapshot
from news_pipeline import load_recent_headlines
from readiness_tracker import load_readiness_state

MODEL_KEYS = ("base", "forest", "boost", "extra_trees", "meta")
Z70 = 1.0364333894937898
Z80 = 1.281551565545
Z95 = 1.959963984540054


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


def _long_daily_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{str(instrument_key or '').upper()}_daily_long.csv")


def _load_long_daily_history(instrument_key: str) -> pd.DataFrame:
    path = _long_daily_path(instrument_key)
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df.columns = [str(c).lower() for c in df.columns]
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


def _fmt_dollar(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"${val:,.2f}"


def _fmt_pct(value, digits: int = 1) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{100.0 * val:+.{int(digits)}f}%"


def _fmt_prob(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{100.0 * val:.0f}%"


def _fmt_ratio(value, digits: int = 2) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{val:.{int(digits)}f}x"


def _fmt_bps(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{10000.0 * val:.1f} bps"


def _logistic(value: float) -> float:
    x = max(-12.0, min(12.0, float(value)))
    return float(1.0 / (1.0 + math.exp(-x)))


def _touch_probability(distance: float | None, sigma_move: float | None) -> float | None:
    dist = _safe_float(distance)
    sigma = _safe_float(sigma_move)
    if dist is None or sigma is None or sigma <= 0:
        return None
    if dist <= 0:
        return 1.0
    z = float(dist / max(1e-9, sigma))
    return float(max(0.0, min(1.0, math.erfc(z / math.sqrt(2.0)))))


def _tz_index(index_like, tz: str):
    idx = pd.to_datetime(index_like, utc=True, errors="coerce")
    try:
        return idx.tz_convert(tz)
    except Exception:
        try:
            return pd.to_datetime(index_like).tz_localize(tz)
        except Exception:
            return pd.to_datetime(index_like, errors="coerce")


def _session_dates(idx_local, session_open_minute: int, session_close_minute: int):
    minutes = idx_local.hour * 60 + idx_local.minute
    local_dates = pd.to_datetime(idx_local.date)
    if session_open_minute > session_close_minute:
        return pd.to_datetime(np.where(minutes >= session_open_minute, local_dates, local_dates - pd.Timedelta(days=1)))
    return local_dates


def _iter_block_items(block: dict) -> list[dict]:
    if not isinstance(block, dict):
        return []
    out = []
    for key in MODEL_KEYS:
        item = block.get(key)
        if isinstance(item, dict):
            out.append(item)
    return out


def _prob_from_item(item: dict) -> float | None:
    if not isinstance(item, dict):
        return None
    p_up = _safe_float(item.get("p_up"))
    if p_up is not None:
        return max(0.001, min(0.999, float(p_up)))
    direction = str(item.get("direction") or "-").upper()
    conf = _safe_float(item.get("confidence"))
    if conf is None:
        return None
    conf /= 100.0
    if direction == "UP":
        return max(0.001, min(0.999, float(conf)))
    if direction == "DOWN":
        return max(0.001, min(0.999, float(1.0 - conf)))
    return 0.5


def _sigma_price_from_item(item: dict, reference_price: float | None) -> float | None:
    if not isinstance(item, dict):
        return None
    lo = _safe_float(item.get("pred_range_80_low"))
    hi = _safe_float(item.get("pred_range_80_high"))
    if lo is not None and hi is not None and hi >= lo:
        return max(0.01, float((hi - lo) / (2.0 * Z80)))
    err_bps = _safe_float(item.get("expected_abs_err_bps"))
    ref = _safe_float(reference_price)
    if err_bps is not None and ref not in (None, 0.0):
        return max(0.01, float(ref * (err_bps / 10000.0)))
    return None


def _aggregate_block(block: dict, current_price: float | None, price_key: str = "pred_price") -> dict:
    prices = []
    probs = []
    sigmas = []
    directions = []
    for item in _iter_block_items(block):
        price = _safe_float(item.get(price_key))
        if price is not None:
            prices.append(float(price))
        prob = _prob_from_item(item)
        if prob is not None:
            probs.append(float(prob))
        sigma = _sigma_price_from_item(item, reference_price=current_price)
        if sigma is not None:
            sigmas.append(float(sigma))
        direction = str(item.get("direction") or "-").upper()
        if direction in ("UP", "DOWN"):
            directions.append(direction)
    mean_price = float(np.mean(prices)) if prices else None
    mean_prob = float(np.mean(probs)) if probs else None
    sigma_model = float(np.mean(sigmas)) if sigmas else None
    sigma_disp = float(np.std(prices)) if len(prices) >= 2 else None
    sigma_price = None
    if sigma_model is not None and sigma_disp is not None:
        sigma_price = float(max(sigma_model, sigma_disp))
    elif sigma_model is not None:
        sigma_price = float(sigma_model)
    elif sigma_disp is not None:
        sigma_price = float(sigma_disp)
    up_votes = sum(1 for d in directions if d == "UP")
    down_votes = sum(1 for d in directions if d == "DOWN")
    vote_total = up_votes + down_votes
    agreement = None
    if vote_total > 0:
        agreement = float(max(up_votes, down_votes) / vote_total)
    if mean_prob is None and vote_total > 0:
        mean_prob = float(up_votes / vote_total)
    return {
        "mean_price": mean_price,
        "sigma_price": sigma_price,
        "p_up": mean_prob,
        "p_down": (1.0 - mean_prob) if mean_prob is not None else None,
        "agreement": agreement,
        "n": len(prices),
    }


def _bands_from_center(center: float | None, sigma_price: float | None) -> dict:
    center = _safe_float(center)
    sigma = _safe_float(sigma_price)
    if center is None or sigma is None:
        return {}
    return {
        "70": (float(center - Z70 * sigma), float(center + Z70 * sigma)),
        "95": (float(center - Z95 * sigma), float(center + Z95 * sigma)),
    }


def _current_session_state(df: pd.DataFrame, instrument_key: str) -> dict:
    meta = _instrument_meta(instrument_key)
    tz = meta.get("tz", "America/New_York")
    session_open = int(meta.get("session_open_minute", 9 * 60 + 30))
    session_close = int(meta.get("session_close_minute", 16 * 60))
    idx_local = _tz_index(df.index, tz)
    frame = df.copy()
    frame["idx_local"] = idx_local
    frame["session_date"] = _session_dates(idx_local, session_open, session_close)
    frame["week_key"] = pd.to_datetime(frame["session_date"]).dt.strftime("%G-W%V")
    current_session = frame[frame["session_date"] == frame["session_date"].iloc[-1]].copy()
    current_week = frame[frame["week_key"] == frame["week_key"].iloc[-1]].copy()
    minutes = idx_local[-1].hour * 60 + idx_local[-1].minute if len(idx_local) else session_open
    sess_span = max(1, session_close - session_open)
    session_progress = max(0.0, min(1.0, float(minutes - session_open) / float(sess_span)))
    weekday = int(pd.Timestamp(frame["session_date"].iloc[-1]).weekday()) if len(frame) else 0
    week_progress = max(0.0, min(1.0, ((weekday + session_progress) / 5.0)))
    return {
        "frame": frame,
        "current_session": current_session,
        "current_week": current_week,
        "session_progress": float(session_progress),
        "week_progress": float(week_progress),
        "session_open_minute": session_open,
        "session_close_minute": session_close,
        "tz": tz,
    }


def _atr14(daily_long: pd.DataFrame) -> float | None:
    if daily_long is None or daily_long.empty:
        return None
    high = pd.to_numeric(daily_long.get("high"), errors="coerce")
    low = pd.to_numeric(daily_long.get("low"), errors="coerce")
    close = pd.to_numeric(daily_long.get("close"), errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return _safe_float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) else None


def _current_spread_info(df: pd.DataFrame, current_price: float | None) -> tuple[float | None, float | None]:
    if df is None or df.empty or current_price in (None, 0.0):
        return None, None
    try:
        if "ask" in df.columns and "bid" in df.columns:
            ask = _safe_float(df["ask"].iloc[-1])
            bid = _safe_float(df["bid"].iloc[-1])
            if ask is not None and bid is not None and ask > bid:
                spread = float(ask - bid)
                return spread, float(spread / current_price)
    except Exception:
        pass
    return None, None


def _intraday_noise(df: pd.DataFrame, current_price: float | None) -> dict:
    if df is None or df.empty or current_price in (None, 0.0):
        return {}
    try:
        bars_ahead = max(1, int(round(4.0 / max(1.0, float(BAR_MINUTES)))))
    except Exception:
        bars_ahead = 2
    close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
    if len(close) <= bars_ahead + 5:
        return {}
    move = close.pct_change(bars_ahead).abs().dropna()
    return {
        "median_abs_move": _safe_float(move.tail(120).median()),
        "mean_abs_move": _safe_float(move.tail(120).mean()),
        "bars_ahead": int(bars_ahead),
        "noise_dollars": _safe_float((move.tail(120).median() or np.nan) * current_price),
    }


def _volume_confirmation(state: dict, current_price: float | None) -> tuple[str, float | None]:
    session = state.get("current_session")
    if session is None or session.empty:
        return "Unknown", None
    try:
        pv = pd.to_numeric(session["close"], errors="coerce") * pd.to_numeric(session["volume"], errors="coerce").fillna(0.0)
        vol = pd.to_numeric(session["volume"], errors="coerce").fillna(0.0)
        vwap = _safe_float(pv.sum() / (vol.sum() + 1e-12))
    except Exception:
        vwap = None
    try:
        today_vol = _safe_float(pd.to_numeric(session["volume"], errors="coerce").sum())
        sess_vol = state["frame"].groupby("session_date")["volume"].sum().tail(21)
        avg_vol = _safe_float(sess_vol.iloc[:-1].mean()) if len(sess_vol) >= 2 else None
        vol_pace = float(today_vol / avg_vol) if today_vol is not None and avg_vol not in (None, 0.0) else None
    except Exception:
        vol_pace = None
    if current_price is None or vwap is None:
        return "Unknown", vol_pace
    if current_price >= vwap and (vol_pace or 0.0) >= 1.0:
        return "Bull confirmed", vol_pace
    if current_price < vwap and (vol_pace or 0.0) >= 1.0:
        return "Bear confirmed", vol_pace
    if (vol_pace or 0.0) < 0.8:
        return "Thin participation", vol_pace
    return "Mixed", vol_pace


def _prediction_state(predictions: dict, current_price: float | None) -> dict:
    out = {}
    intraday = predictions.get("intraday", {}) if isinstance(predictions, dict) else {}
    daily = predictions.get("daily", {}) if isinstance(predictions, dict) else {}
    weekend = predictions.get("weekend", {}) if isinstance(predictions, dict) else {}

    cons = intraday.get("consensus", {}) if isinstance(intraday, dict) else {}
    out["intraday"] = {
        "point": _safe_float(cons.get("pred_price")),
        "direction": str(cons.get("direction") or "-").upper(),
        "p_up": _prob_from_item(cons),
        "confidence": _safe_float(cons.get("confidence")),
        "agreement": _aggregate_block(intraday, current_price).get("agreement"),
        "bands": {
            "70": (_safe_float(cons.get("pred_range_70_low")), _safe_float(cons.get("pred_range_70_high"))),
            "95": (_safe_float(cons.get("pred_range_95_low")), _safe_float(cons.get("pred_range_95_high"))),
        },
        "edge_bps": _safe_float(cons.get("edge_bps")),
        "uncertainty_bps": _safe_float(cons.get("uncertainty_bps")),
        "note": str(cons.get("note") or ""),
        "position_size": _safe_float(cons.get("position_size")),
        "sig_bucket": str(cons.get("sig_bucket") or "-"),
    }

    daily_high = _aggregate_block(daily.get("high", {}), current_price)
    daily_low = _aggregate_block(daily.get("low", {}), current_price)
    daily_mid = None
    if daily_high.get("mean_price") is not None and daily_low.get("mean_price") is not None:
        daily_mid = 0.5 * (daily_high["mean_price"] + daily_low["mean_price"])
    out["daily"] = {
        "high": daily_high,
        "low": daily_low,
        "mid": daily_mid,
    }

    week_close = _aggregate_block(weekend.get("week", {}), current_price)
    week_high = _aggregate_block(weekend.get("week", {}), current_price, price_key="pred_high_price")
    week_low = _aggregate_block(weekend.get("week", {}), current_price, price_key="pred_low_price")
    out["weekend"] = {
        "close": week_close,
        "high": week_high,
        "low": week_low,
    }
    return out


def _daily_high_low_probabilities(state: dict, pred_state: dict, atr14: float | None) -> dict:
    current_session = state.get("current_session")
    if current_session is None or current_session.empty:
        return {}
    current_hi = _safe_float(pd.to_numeric(current_session.get("high"), errors="coerce").max())
    current_lo = _safe_float(pd.to_numeric(current_session.get("low"), errors="coerce").min())
    pred_hi = pred_state.get("daily", {}).get("high", {}).get("mean_price")
    pred_lo = pred_state.get("daily", {}).get("low", {}).get("mean_price")
    sigma_hi = pred_state.get("daily", {}).get("high", {}).get("sigma_price")
    sigma_lo = pred_state.get("daily", {}).get("low", {}).get("sigma_price")
    rem = max(0.05, 1.0 - float(state.get("session_progress", 0.0)))
    move_scale_hi = max(0.20, _safe_float(sigma_hi) or 0.0, (_safe_float(atr14) or 0.0) * 0.35)
    move_scale_lo = max(0.20, _safe_float(sigma_lo) or 0.0, (_safe_float(atr14) or 0.0) * 0.35)
    sigma_rem_hi = float(move_scale_hi * math.sqrt(rem))
    sigma_rem_lo = float(move_scale_lo * math.sqrt(rem))

    high_gap = max(0.0, (pred_hi or current_hi or 0.0) - (current_hi or 0.0)) if pred_hi is not None and current_hi is not None else None
    low_gap = max(0.0, (current_lo or 0.0) - (pred_lo or current_lo or 0.0)) if pred_lo is not None and current_lo is not None else None

    p_touch_high_remaining = _touch_probability(high_gap, sigma_rem_hi)
    p_touch_low_remaining = _touch_probability(low_gap, sigma_rem_lo)
    prob_high_set = None if p_touch_high_remaining is None else float(max(0.0, min(1.0, 1.0 - p_touch_high_remaining)))
    prob_low_set = None if p_touch_low_remaining is None else float(max(0.0, min(1.0, 1.0 - p_touch_low_remaining)))
    if pred_hi is not None and current_hi is not None and current_hi >= float(pred_hi - move_scale_hi * 0.15):
        prob_high_set = 0.97
    if pred_lo is not None and current_lo is not None and current_lo <= float(pred_lo + move_scale_lo * 0.15):
        prob_low_set = 0.97

    return {
        "prob_high_set": prob_high_set,
        "prob_low_set": prob_low_set,
        "current_high": current_hi,
        "current_low": current_lo,
        "estimate_type": "heuristic_vol_touch",
    }


def _weekly_touch_probabilities(state: dict, pred_state: dict, atr14: float | None) -> dict:
    current_week = state.get("current_week")
    if current_week is None or current_week.empty:
        return {}
    week_hi = _safe_float(pd.to_numeric(current_week.get("high"), errors="coerce").max())
    week_lo = _safe_float(pd.to_numeric(current_week.get("low"), errors="coerce").min())
    pred_hi = pred_state.get("weekend", {}).get("high", {}).get("mean_price")
    pred_lo = pred_state.get("weekend", {}).get("low", {}).get("mean_price")
    prog = float(state.get("week_progress", 0.0))
    days_left = max(1.0, 5.0 * (1.0 - prog))
    budget = max(0.50, (_safe_float(atr14) or 1.0) * math.sqrt(days_left))
    up_gap = max(0.0, (pred_hi or week_hi or 0.0) - (week_hi or 0.0)) if pred_hi is not None and week_hi is not None else None
    down_gap = max(0.0, (week_lo or 0.0) - (pred_lo or week_lo or 0.0)) if pred_lo is not None and week_lo is not None else None

    p_touch_hi = _touch_probability(up_gap, budget) if pred_hi is not None and week_hi is not None else None
    p_touch_lo = _touch_probability(down_gap, budget) if pred_lo is not None and week_lo is not None else None
    if pred_hi is not None and week_hi is not None and week_hi >= float(pred_hi - budget * 0.10):
        p_touch_hi = 0.96
    if pred_lo is not None and week_lo is not None and week_lo <= float(pred_lo + budget * 0.10):
        p_touch_lo = 0.96

    return {
        "prob_touch_high": p_touch_hi,
        "prob_touch_low": p_touch_lo,
        "current_week_high": week_hi,
        "current_week_low": week_lo,
        "estimate_type": "heuristic_vol_touch",
    }


def _regime_snapshot(predictions: dict) -> dict:
    for key in ("intraday", "daily", "weekend"):
        block = predictions.get(key)
        if isinstance(block, dict) and isinstance(block.get("regime"), dict):
            return block.get("regime")
    return {}


def _dominant_regime_label(regime: dict) -> str:
    bull = _safe_float((regime or {}).get("Bullish")) or 0.0
    bear = _safe_float((regime or {}).get("Bearish")) or 0.0
    neutral = _safe_float((regime or {}).get("Neutral")) or 0.0
    shock = _safe_float((regime or {}).get("shock_score"))
    if shock is not None and shock >= 0.75:
        return "News / shock"
    if bull >= max(bear, neutral):
        return "Trend up"
    if bear >= max(bull, neutral):
        return "Trend down"
    return "Range / neutral"


def _signal_quality(
    current_price: float | None,
    pred_state: dict,
    noise: dict,
    spread_frac: float | None,
    volume_state: tuple[str, float | None],
    regime: dict,
    event_snapshot: dict,
    news_items: list[dict],
    macro_snapshot: dict | None = None,
    breadth_snapshot: dict | None = None,
    liquidity_snapshot: dict | None = None,
    catalyst_snapshot: dict | None = None,
) -> dict:
    intr = pred_state.get("intraday", {})
    p_up = _safe_float(intr.get("p_up"))
    conf = None if p_up is None else max(p_up, 1.0 - p_up)
    agreement = _safe_float(intr.get("agreement"))
    point = _safe_float(intr.get("point"))
    move = abs(float(point - current_price)) if point is not None and current_price is not None else None
    noise_dollars = _safe_float(noise.get("noise_dollars"))
    noise_ratio = float(move / noise_dollars) if move is not None and noise_dollars not in (None, 0.0) else None
    move_vs_spread = None
    if move is not None and spread_frac is not None and current_price not in (None, 0.0):
        spread_dollars = float(current_price * spread_frac)
        move_vs_spread = float(move / max(0.01, spread_dollars))
    regime_conf = _safe_float((regime or {}).get("regime_conf"))
    vol_conf = _safe_float(volume_state[1])
    vol_score = None if vol_conf is None else max(0.0, min(1.0, 0.45 + 0.25 * vol_conf))
    noise_score = None if noise_ratio is None else max(0.0, min(1.0, noise_ratio / 1.6))
    spread_score = 0.6 if move_vs_spread is None else max(0.0, min(1.0, move_vs_spread / 3.0))
    regime_score = 0.5 if regime_conf is None else max(0.0, min(1.0, regime_conf))
    breadth_score = _safe_float((breadth_snapshot or {}).get("risk_on_score"))
    breadth_source_kind = str((breadth_snapshot or {}).get("status") or (breadth_snapshot or {}).get("source_kind") or "unavailable")
    liquidity_score = _safe_float((liquidity_snapshot or {}).get("tradability_score"))
    base_parts = [v for v in (conf, agreement, noise_score, spread_score, regime_score, vol_score, liquidity_score) if v is not None]
    score = float(np.mean(base_parts)) if base_parts else 0.45

    event_penalty = 0.0
    dte = _safe_float((event_snapshot or {}).get("days_to_earnings"))
    if dte is not None and dte <= 1.5:
        event_penalty += 0.18
    elif dte is not None and dte <= 4.0:
        event_penalty += 0.08
    fresh_headlines = 0
    neg_hits = 0
    for item in news_items[-4:]:
        try:
            ts = pd.Timestamp(item.get("published_at"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age_h = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0)
            if age_h <= 6.0:
                fresh_headlines += 1
                label = str(item.get("label") or "").lower()
                if "neg" in label or "bear" in label:
                    neg_hits += 1
        except Exception:
            continue
    if fresh_headlines >= 2:
        event_penalty += 0.06
    if neg_hits >= 2 and (intr.get("direction") == "UP"):
        event_penalty += 0.05
    macro_risk = _safe_float((macro_snapshot or {}).get("risk_score"))
    if macro_risk is not None and macro_risk >= 0.75:
        event_penalty += 0.18
    elif macro_risk is not None and macro_risk >= 0.45:
        event_penalty += 0.08
    catalyst_risk = _safe_float((catalyst_snapshot or {}).get("catalyst_risk_score"))
    if catalyst_risk is not None and catalyst_risk >= 0.45:
        event_penalty += 0.10
    elif catalyst_risk is not None and catalyst_risk >= 0.25:
        event_penalty += 0.05
    recent_catalysts = (catalyst_snapshot or {}).get("recent", []) if isinstance(catalyst_snapshot, dict) else []
    neg_catalysts = 0
    for item in recent_catalysts[:3]:
        sent = str(item.get("sentiment") or "").lower()
        if "neg" in sent:
            neg_catalysts += 1
    if neg_catalysts >= 1 and intr.get("direction") == "UP":
        event_penalty += 0.05
    if neg_catalysts >= 1 and intr.get("direction") == "DOWN":
        event_penalty -= 0.02
    breadth_penalty_mult = 1.0 if bool((breadth_snapshot or {}).get("direct_feed")) else 0.7 if breadth_source_kind == "live_partial" else 0.4 if breadth_source_kind == "proxy_only" else 0.0
    if breadth_score is not None and breadth_penalty_mult > 0.0:
        if intr.get("direction") == "UP" and breadth_score < 0.40:
            event_penalty += 0.06 * breadth_penalty_mult
        elif intr.get("direction") == "DOWN" and breadth_score > 0.60:
            event_penalty += 0.06 * breadth_penalty_mult
    if liquidity_score is not None and liquidity_score < 0.45:
        event_penalty += 0.12
    score = max(0.0, min(1.0, score - event_penalty))

    if score >= 0.78:
        grade = "A"
    elif score >= 0.66:
        grade = "B"
    elif score >= 0.54:
        grade = "C"
    else:
        grade = "D"

    whipsaw = 1.0 - max(0.0, min(1.0, (agreement or 0.5) * (noise_score if noise_score is not None else 0.6)))
    vol_state = str(volume_state[0] or "Unknown")
    regime_label = _dominant_regime_label(regime)
    strategy = "No-trade"
    if score >= 0.72 and "Trend" in regime_label and "confirmed" in vol_state.lower():
        strategy = "Breakout / trend"
    elif score >= 0.62 and "Range" in regime_label:
        strategy = "Mean reversion"
    elif score >= 0.60:
        strategy = "Scalp"
    elif score >= 0.52:
        strategy = "Small / selective"

    reasons = []
    if conf is not None and conf < 0.56:
        reasons.append("direction probability is weak")
    if agreement is not None and agreement < 0.55:
        reasons.append("models disagree")
    if noise_ratio is not None and noise_ratio < 1.0:
        reasons.append("predicted move is near normal noise")
    if move_vs_spread is not None and move_vs_spread < 1.5:
        reasons.append("move is small vs spread")
    if dte is not None and dte <= 1.5:
        reasons.append("earnings risk is imminent")
    if macro_risk is not None and macro_risk >= 0.75:
        reasons.append("major macro event is close")
    if catalyst_risk is not None and catalyst_risk >= 0.45:
        reasons.append("fresh catalysts are active")
    if vol_state == "Thin participation":
        reasons.append("volume confirmation is weak")
    if liquidity_score is not None and liquidity_score < 0.45:
        reasons.append("execution quality is weak")
    breadth_desc = "direct breadth" if bool((breadth_snapshot or {}).get("direct_feed")) else "partial breadth" if breadth_source_kind == "live_partial" else "proxy breadth"
    if breadth_score is not None and breadth_penalty_mult > 0.0:
        if intr.get("direction") == "UP" and breadth_score < 0.40:
            reasons.append(f"{breadth_desc} does not confirm upside")
        elif intr.get("direction") == "DOWN" and breadth_score > 0.60:
            reasons.append(f"{breadth_desc} does not confirm downside")
    hard_macro = bool((macro_snapshot or {}).get("high_risk_24h"))
    no_trade = bool(score < 0.58 or len(reasons) >= 2 or hard_macro)

    return {
        "score": score,
        "grade": grade,
        "strategy": strategy,
        "whipsaw": whipsaw,
        "noise_ratio": noise_ratio,
        "move_vs_spread": move_vs_spread,
        "agreement": agreement,
        "confidence": conf,
        "no_trade": no_trade,
        "reasons": reasons,
        "volume_state": vol_state,
        "regime_label": regime_label,
        "macro_risk": macro_risk,
        "catalyst_risk": catalyst_risk,
        "breadth_score": breadth_score,
        "breadth_source_kind": breadth_source_kind,
        "liquidity_score": liquidity_score,
    }


def _pick_level(level_map: dict, names: list[str]) -> float | None:
    for name in names:
        row = level_map.get(name)
        if not row:
            continue
        val = row[0] if isinstance(row, tuple) else row
        text = str(val)
        if "/" in text:
            text = text.split("/")[0].strip()
        if text.startswith("$"):
            text = text.replace("$", "").replace(",", "").strip()
        number = _safe_float(text)
        if number is not None:
            return number
    return None


def _build_trade_plan(
    current_price: float | None,
    level_map: dict,
    pred_state: dict,
    quality: dict,
    atr14: float | None,
) -> dict:
    intr = pred_state.get("intraday", {})
    daily = pred_state.get("daily", {})
    week = pred_state.get("weekend", {})
    direction = str(intr.get("direction") or "-").upper()
    quality_score = _safe_float(quality.get("score")) or 0.0
    no_trade = bool(quality.get("no_trade"))
    vwap = _pick_level(level_map, ["VWAP"])
    opening_range = level_map.get("Opening Range")
    or_high = None
    or_low = None
    if isinstance(opening_range, tuple):
        parts = [p.strip() for p in str(opening_range[0]).split("/")]
        if parts:
            or_high = _safe_float(parts[0].replace("$", "").replace(",", ""))
        if len(parts) > 1:
            or_low = _safe_float(parts[1].replace("$", "").replace(",", ""))
    prior_session = level_map.get("Prior Session")
    prior_high = None
    prior_low = None
    if isinstance(prior_session, tuple):
        parts = [p.strip() for p in str(prior_session[0]).split("/")]
        if parts:
            prior_high = _safe_float(parts[0].replace("$", "").replace(",", ""))
        if len(parts) > 1:
            prior_low = _safe_float(parts[1].replace("$", "").replace(",", ""))
    daily_high = _safe_float(daily.get("high", {}).get("mean_price"))
    daily_low = _safe_float(daily.get("low", {}).get("mean_price"))
    week_high = _safe_float(week.get("high", {}).get("mean_price"))
    week_low = _safe_float(week.get("low", {}).get("mean_price"))
    intr_point = _safe_float(intr.get("point"))
    sigma = None
    bands = intr.get("bands", {})
    if isinstance(bands, dict):
        lo70 = _safe_float((bands.get("70") or (None, None))[0])
        hi70 = _safe_float((bands.get("70") or (None, None))[1])
        if lo70 is not None and hi70 is not None:
            sigma = max(0.01, float((hi70 - lo70) / (2.0 * Z70)))
    plan = {
        "action": "No-trade",
        "detail": "Edge is not strong enough to force a setup.",
        "entry": None,
        "stop": None,
        "target1": None,
        "target2": None,
        "rr": None,
        "bias": "-",
    }
    if current_price is None or no_trade:
        return plan

    if direction == "UP" and quality_score >= 0.58:
        entry = max(v for v in (current_price, vwap, or_high) if v is not None)
        stop_candidates = [v for v in (vwap, or_low, prior_low, current_price - (0.35 * (atr14 or 0.0))) if v is not None and v < entry]
        stop = max(stop_candidates) if stop_candidates else max(0.01, entry - max(0.25, 0.35 * (atr14 or 1.0)))
        target1_candidates = [v for v in (intr_point, daily_high, week_high, entry + (sigma or 0.0)) if v is not None and v > entry]
        target1 = min(target1_candidates) if target1_candidates else entry + max(0.25, 0.30 * (atr14 or 1.0))
        target2_candidates = [v for v in (daily_high, week_high, entry + 2.0 * (sigma or 0.0)) if v is not None and v > target1]
        target2 = min(target2_candidates) if target2_candidates else target1 + max(0.25, 0.35 * (atr14 or 1.0))
        risk = max(0.01, entry - stop)
        reward = max(0.0, target1 - entry)
        rr = reward / risk if risk > 0 else None
        plan.update(
            {
                "action": "Long setup",
                "detail": "Use only if price holds above key support and participation stays firm.",
                "entry": entry,
                "stop": stop,
                "target1": target1,
                "target2": target2,
                "rr": rr,
                "bias": "Bullish",
            }
        )
    elif direction == "DOWN" and quality_score >= 0.58:
        entry_pool = [v for v in (current_price, vwap, or_low) if v is not None]
        entry = min(entry_pool) if entry_pool else current_price
        stop_candidates = [v for v in (vwap, or_high, prior_high, current_price + (0.35 * (atr14 or 0.0))) if v is not None and v > entry]
        stop = min(stop_candidates) if stop_candidates else entry + max(0.25, 0.35 * (atr14 or 1.0))
        target1_candidates = [v for v in (intr_point, daily_low, week_low, entry - (sigma or 0.0)) if v is not None and v < entry]
        target1 = max(target1_candidates) if target1_candidates else entry - max(0.25, 0.30 * (atr14 or 1.0))
        target2_candidates = [v for v in (daily_low, week_low, entry - 2.0 * (sigma or 0.0)) if v is not None and v < target1]
        target2 = max(target2_candidates) if target2_candidates else target1 - max(0.25, 0.35 * (atr14 or 1.0))
        risk = max(0.01, stop - entry)
        reward = max(0.0, entry - target1)
        rr = reward / risk if risk > 0 else None
        plan.update(
            {
                "action": "Short setup",
                "detail": "Use only if price stays below resistance and downside volume remains present.",
                "entry": entry,
                "stop": stop,
                "target1": target1,
                "target2": target2,
                "rr": rr,
                "bias": "Bearish",
            }
        )
    return plan


def _scenario_rows(current_price: float | None, pred_state: dict) -> list[tuple[str, str, str]]:
    rows = []
    intr = pred_state.get("intraday", {})
    bands = intr.get("bands", {})
    lo70 = _safe_float((bands.get("70") or (None, None))[0]) if isinstance(bands, dict) else None
    hi70 = _safe_float((bands.get("70") or (None, None))[1]) if isinstance(bands, dict) else None
    if lo70 is not None or hi70 is not None or intr.get("point") is not None:
        rows.append(("4m Bull / Base / Bear", f"{_fmt_dollar(hi70)} / {_fmt_dollar(intr.get('point'))} / {_fmt_dollar(lo70)}", "70% path around the 4m forecast"))
    daily = pred_state.get("daily", {})
    rows.append(
        (
            "1D Bull / Base / Bear",
            f"{_fmt_dollar(daily.get('high', {}).get('mean_price'))} / {_fmt_dollar(daily.get('mid'))} / {_fmt_dollar(daily.get('low', {}).get('mean_price'))}",
            "Bull = target high, base = midpoint, bear = target low",
        )
    )
    week = pred_state.get("weekend", {})
    rows.append(
        (
            "1W Bull / Base / Bear",
            f"{_fmt_dollar(week.get('high', {}).get('mean_price'))} / {_fmt_dollar(week.get('close', {}).get('mean_price'))} / {_fmt_dollar(week.get('low', {}).get('mean_price'))}",
            "Bull = projected week high, base = week close, bear = projected week low",
        )
    )
    if current_price is not None:
        rows.append(("Anchor", _fmt_dollar(current_price), "Current spot used for setup and invalidation"))
    return rows[:5]


def _historical_analog_rows(df: pd.DataFrame, instrument_key: str, current_price: float | None, state: dict) -> list[tuple[str, str, str]]:
    rows = []
    try:
        bars_ahead = max(1, int(round(4.0 / max(1.0, float(BAR_MINUTES)))))
    except Exception:
        bars_ahead = 2
    if df is not None and not df.empty and "close" in df.columns:
        close = pd.to_numeric(df["close"], errors="coerce")
        vol = pd.to_numeric(df.get("volume"), errors="coerce").fillna(0.0)
        px = close
        roll = px.rolling(20, min_periods=5).mean()
        rel = (px > roll).astype(int)
        vol_bucket = (vol > vol.rolling(20, min_periods=5).median()).astype(int)
        future = close.shift(-bars_ahead) / close - 1.0
        mask = rel == rel.iloc[-1]
        mask &= vol_bucket == vol_bucket.iloc[-1]
        mask.iloc[-bars_ahead:] = False
        sample = future[mask].dropna()
        if len(sample) >= 12:
            up_rate = float((sample > 0).mean())
            avg_abs = float(np.mean(np.abs(sample)) * (current_price or float(close.iloc[-1])))
            rows.append(("4m Analog", _fmt_prob(up_rate), f"{len(sample)} similar bars | abs move {_fmt_dollar(avg_abs)}"))

    daily_long = _load_long_daily_history(instrument_key)
    if daily_long is not None and not daily_long.empty and "close" in daily_long.columns:
        close = pd.to_numeric(daily_long["close"], errors="coerce")
        high = pd.to_numeric(daily_long.get("high"), errors="coerce")
        low = pd.to_numeric(daily_long.get("low"), errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr_pct = (tr.rolling(14, min_periods=5).mean() / close).replace([np.inf, -np.inf], np.nan)
        vol_bucket = (atr_pct >= atr_pct.median()).astype(int)
        trend20 = close.pct_change(20)
        trend_bucket = np.where(trend20 > 0.01, 1, np.where(trend20 < -0.01, -1, 0))
        mask = pd.Series(trend_bucket, index=close.index) == trend_bucket[-1]
        mask &= vol_bucket == vol_bucket.iloc[-1]
        fut1 = close.shift(-1) / close - 1.0
        fut5 = close.shift(-5) / close - 1.0
        daily_sample = fut1[mask].dropna()
        week_sample = fut5[mask].dropna()
        if len(daily_sample) >= 12:
            rows.append(("1D Analog", _fmt_prob((daily_sample > 0).mean()), f"{len(daily_sample)} similar days | avg {_fmt_pct(daily_sample.mean())}"))
        if len(week_sample) >= 10:
            rows.append(("1W Analog", _fmt_prob((week_sample > 0).mean()), f"{len(week_sample)} similar weeks | avg {_fmt_pct(week_sample.mean())}"))
    return rows[:4]


def _readiness_rows(instrument_key: str) -> list[tuple[str, str, str]]:
    rows = []
    state = load_readiness_state(str(instrument_key or "").upper()) or {}
    evaluation = state.get("evaluation", {}) if isinstance(state, dict) else {}
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    if metrics:
        rows.append(
            (
                "4m History",
                f"{_fmt_dollar(metrics.get('intraday_avg_discrepancy_dollars'))} / {_fmt_dollar(metrics.get('intraday_target_window_dollars'))}",
                f"Hit {_fmt_prob(metrics.get('intraday_window_hit_rate'))} | Dir {_fmt_prob(metrics.get('intraday_updown_acc'))}",
            )
        )
        rows.append(
            (
                "1D History",
                f"{_fmt_dollar(metrics.get('daily_avg_discrepancy_dollars'))} / {_fmt_dollar(metrics.get('daily_target_window_dollars'))}",
                f"Hit {_fmt_prob(metrics.get('daily_window_hit_rate'))} | Dir {_fmt_prob(metrics.get('daily_updown_acc'))}",
            )
        )
        rows.append(
            (
                "1W History",
                f"{_fmt_dollar(metrics.get('weekend_avg_discrepancy_dollars'))} / {_fmt_dollar(metrics.get('weekend_target_window_dollars'))}",
                f"Hit {_fmt_prob(metrics.get('weekend_window_hit_rate'))} | Dir {_fmt_prob(metrics.get('weekend_updown_acc'))}",
            )
        )
    if isinstance(state, dict):
        rows.append(("Readiness", "Ready" if bool(state.get("live_ready")) else "Not ready", f"Fresh {bool(state.get('fresh'))} | Streak {state.get('pass_streak_days', 0)}"))
    return rows[:4]


def build_advisory_snapshot(
    df: pd.DataFrame,
    instrument_key: str,
    predictions: dict | None = None,
    options_snapshot: dict | None = None,
    event_snapshot: dict | None = None,
    news_items: list[dict] | None = None,
    macro_snapshot: dict | None = None,
    breadth_snapshot: dict | None = None,
    liquidity_snapshot: dict | None = None,
    catalyst_snapshot: dict | None = None,
) -> dict:
    if df is None or df.empty:
        return {
            "action_card": "Action: unavailable",
            "action_detail": "No data",
            "quality_card": "Quality: unavailable",
            "quality_detail": "No data",
            "probability_rows": [],
            "quality_rows": [],
            "setup_rows": [],
            "scenario_rows": [],
            "invalidation_rows": [],
            "reliability_rows": [],
            "breadth_rows": [],
            "macro_rows": [],
            "liquidity_rows": [],
            "catalyst_rows": [],
            "quality_meta": {},
            "plan_meta": {},
            "pred_state": {},
            "macro_meta": {},
            "breadth_meta": {},
            "liquidity_meta": {},
            "catalyst_meta": {},
        }

    instrument_key = str(instrument_key or DEFAULT_INSTRUMENT).upper()
    current_price = _safe_float(pd.to_numeric(df.get("close"), errors="coerce").iloc[-1]) if "close" in df.columns else None
    state = _current_session_state(df, instrument_key)
    daily_long = _load_long_daily_history(instrument_key)
    atr14 = _atr14(daily_long)
    predictions = predictions or {}
    pred_state = _prediction_state(predictions, current_price)
    regime = _regime_snapshot(predictions)
    spread_dollars, spread_frac = _current_spread_info(df, current_price)
    noise = _intraday_noise(df, current_price)
    volume_state = _volume_confirmation(state, current_price)
    if not isinstance(event_snapshot, dict):
        event_snapshot = get_event_snapshot(instrument_key)
    if news_items is None:
        news_items = load_recent_headlines(instrument_key, limit=8) or []
    if not isinstance(macro_snapshot, dict):
        macro_snapshot = get_macro_event_snapshot()
    if not isinstance(breadth_snapshot, dict):
        breadth_snapshot = get_breadth_snapshot(instrument_key)
    if not isinstance(liquidity_snapshot, dict):
        liquidity_snapshot = get_liquidity_snapshot(df, instrument_key)
    if not isinstance(catalyst_snapshot, dict):
        catalyst_snapshot = get_catalyst_provider_snapshot(instrument_key)
    quality = _signal_quality(
        current_price,
        pred_state,
        noise,
        spread_frac,
        volume_state,
        regime,
        event_snapshot,
        news_items,
        macro_snapshot=macro_snapshot,
        breadth_snapshot=breadth_snapshot,
        liquidity_snapshot=liquidity_snapshot,
        catalyst_snapshot=catalyst_snapshot,
    )

    daily_probs = _daily_high_low_probabilities(state, pred_state, atr14)
    week_probs = _weekly_touch_probabilities(state, pred_state, atr14)

    intr = pred_state.get("intraday", {})
    daily = pred_state.get("daily", {})
    week = pred_state.get("weekend", {})
    intr_band70 = intr.get("bands", {}).get("70", (None, None))
    intr_band95 = intr.get("bands", {}).get("95", (None, None))
    day_hi_bands = _bands_from_center(daily.get("high", {}).get("mean_price"), daily.get("high", {}).get("sigma_price"))
    day_lo_bands = _bands_from_center(daily.get("low", {}).get("mean_price"), daily.get("low", {}).get("sigma_price"))
    week_close_bands = _bands_from_center(week.get("close", {}).get("mean_price"), week.get("close", {}).get("sigma_price"))
    week_hi_bands = _bands_from_center(week.get("high", {}).get("mean_price"), week.get("high", {}).get("sigma_price"))
    week_lo_bands = _bands_from_center(week.get("low", {}).get("mean_price"), week.get("low", {}).get("sigma_price"))

    probability_rows = [
        ("4m Up / Down (model)", f"{_fmt_prob(intr.get('p_up'))} / {_fmt_prob(intr.get('p_down'))}", f"70% {_fmt_dollar(intr_band70[0])}..{_fmt_dollar(intr_band70[1])} | 95% {_fmt_dollar(intr_band95[0])}..{_fmt_dollar(intr_band95[1])}"),
        ("1D High Set? (heur.)", _fmt_prob(daily_probs.get("prob_high_set")), f"Vol-touch est | now {_fmt_dollar(daily_probs.get('current_high'))} vs tgt {_fmt_dollar(daily.get('high', {}).get('mean_price'))}"),
        ("1D Low Set? (heur.)", _fmt_prob(daily_probs.get("prob_low_set")), f"Vol-touch est | now {_fmt_dollar(daily_probs.get('current_low'))} vs tgt {_fmt_dollar(daily.get('low', {}).get('mean_price'))}"),
        ("1W Close Up / Down (model)", f"{_fmt_prob(week.get('close', {}).get('p_up'))} / {_fmt_prob(week.get('close', {}).get('p_down'))}", f"70% {_fmt_dollar((week_close_bands.get('70') or (None, None))[0])}..{_fmt_dollar((week_close_bands.get('70') or (None, None))[1])}"),
        ("1W Touch High / Low (heur.)", f"{_fmt_prob(week_probs.get('prob_touch_high'))} / {_fmt_prob(week_probs.get('prob_touch_low'))}", f"Vol-touch est | H {_fmt_dollar(week.get('high', {}).get('mean_price'))} | L {_fmt_dollar(week.get('low', {}).get('mean_price'))}"),
        ("1W 95% Extremes (model)", f"{_fmt_dollar((week_hi_bands.get('95') or (None, None))[1])} / {_fmt_dollar((week_lo_bands.get('95') or (None, None))[0])}", "Upper high band / lower low band"),
    ]

    expected_move = _safe_float((options_snapshot or {}).get("expected_move_dollars"))
    predicted_move = abs(float((intr.get("point") or current_price or 0.0) - (current_price or 0.0))) if intr.get("point") is not None and current_price is not None else None
    liquidity_flag = "Tradable"
    liq_score = _safe_float((liquidity_snapshot or {}).get("tradability_score"))
    if liq_score is not None and liq_score < 0.45:
        liquidity_flag = "Fragile"
    elif spread_frac is not None and spread_frac >= 0.0012:
        liquidity_flag = "Wide spread"
    elif spread_frac is None:
        liquidity_flag = "Spread unknown"
    quality_rows = [
        ("Signal Quality", f"{quality.get('grade', '-')}", f"{int(round(100.0 * (_safe_float(quality.get('score')) or 0.0)))} / 100 | {quality.get('strategy', '-')}"),
        ("Agreement", _fmt_prob(quality.get("agreement")), f"Regime {quality.get('regime_label', '-')} | Breadth {_fmt_prob(quality.get('breadth_score'))} ({str(quality.get('breadth_source_kind') or '-').replace('_', ' ')})"),
        ("Move vs Noise", _fmt_ratio(quality.get("noise_ratio")), f"Pred {_fmt_dollar(predicted_move)} | Noise {_fmt_dollar(noise.get('noise_dollars'))}"),
        ("Move vs Spread", _fmt_ratio(quality.get("move_vs_spread")), f"Spread {_fmt_dollar(spread_dollars)} | {liquidity_flag}"),
        ("ATR / Opt Move", f"{_fmt_dollar(atr14)} / {_fmt_dollar(expected_move)}", f"4m edge {_fmt_bps((_safe_float(intr.get('edge_bps')) or 0.0) / 10000.0)} | unc {_fmt_bps((_safe_float(intr.get('uncertainty_bps')) or 0.0) / 10000.0)}"),
        ("Whipsaw Risk", _fmt_prob(quality.get("whipsaw")), "Higher means more likely to chop / reverse"),
        ("Macro / Catalyst Risk", _fmt_prob(max((_safe_float(quality.get("macro_risk")) or 0.0), (_safe_float(quality.get("catalyst_risk")) or 0.0))), "Risk overlay only; not a price forecast"),
    ]

    level_map = {
        "VWAP": (_safe_float(pd.to_numeric(state["current_session"]["close"], errors="coerce").mul(pd.to_numeric(state["current_session"]["volume"], errors="coerce").fillna(0.0)).sum() / (pd.to_numeric(state["current_session"]["volume"], errors="coerce").fillna(0.0).sum() + 1e-12)) if not state["current_session"].empty else None, ""),
    }
    if not state["current_session"].empty:
        try:
            or_minutes = state["current_session"]["idx_local"].dt.hour * 60 + state["current_session"]["idx_local"].dt.minute
            or_block = state["current_session"][(or_minutes >= state["session_open_minute"]) & (or_minutes < state["session_open_minute"] + 30)]
            if not or_block.empty:
                level_map["Opening Range"] = (
                    f"{_fmt_dollar(pd.to_numeric(or_block['high'], errors='coerce').max())} / {_fmt_dollar(pd.to_numeric(or_block['low'], errors='coerce').min())}",
                    "",
                )
        except Exception:
            pass
    frame = state["frame"]
    prior_sessions = frame["session_date"].drop_duplicates()
    if len(prior_sessions) >= 2:
        prev_session = prior_sessions.iloc[-2]
        prev = frame[frame["session_date"] == prev_session]
        if not prev.empty:
            level_map["Prior Session"] = (
                f"{_fmt_dollar(pd.to_numeric(prev['high'], errors='coerce').max())} / {_fmt_dollar(pd.to_numeric(prev['low'], errors='coerce').min())}",
                "",
            )

    plan = _build_trade_plan(current_price, level_map, pred_state, quality, atr14)
    reasons = quality.get("reasons") or []
    if bool(quality.get("no_trade")) and not reasons:
        reasons = ["signal quality is below the action threshold"]
    earnings_near = bool(_safe_float((event_snapshot or {}).get("days_to_earnings")) not in (None,) and (_safe_float((event_snapshot or {}).get("days_to_earnings")) or 99.0) <= 4.0)
    active_catalyst = (catalyst_snapshot or {}).get("active_catalyst") if isinstance(catalyst_snapshot, dict) else None
    catalyst_risk_label = "Elevated" if (
        earnings_near
        or bool((macro_snapshot or {}).get("high_risk_24h"))
        or ((_safe_float((catalyst_snapshot or {}).get("catalyst_risk_score")) or 0.0) >= 0.35)
    ) else "Normal"
    catalyst_detail = "Fresh news, earnings, or macro releases can override technical setups"
    if isinstance(active_catalyst, dict):
        catalyst_detail = f"{str(active_catalyst.get('title') or '-')[:18]} | {active_catalyst.get('risk_kind') or 'event risk'} | ambiguity {active_catalyst.get('ambiguity') or 'high'}"
    invalidation_rows = [
        ("Primary", "Hold VWAP" if plan.get("bias") == "Bullish" else "Reject VWAP" if plan.get("bias") == "Bearish" else "Wait", "Intraday bias fails if price loses the control level"),
        ("Opening Range", "Keep OR support" if plan.get("bias") == "Bullish" else "Keep OR resistance" if plan.get("bias") == "Bearish" else "No break", "Range failure downgrades the setup"),
        ("Participation", volume_state[0], "If volume confirmation fades, reduce conviction"),
        ("Catalyst Risk", catalyst_risk_label, catalyst_detail),
        ("No-trade", "Yes" if quality.get("no_trade") else "No", "; ".join(reasons[:2]) if reasons else "Edge is strong enough to monitor"),
    ]

    setup_rows = [
        ("Bias", plan.get("action", "No-trade"), plan.get("detail", "-")),
        ("Entry", _fmt_dollar(plan.get("entry")), "Wait for the level, do not chase if price is extended"),
        ("Stop", _fmt_dollar(plan.get("stop")), "Invalidate quickly if the setup loses structure"),
        ("Target 1 / 2", f"{_fmt_dollar(plan.get('target1'))} / {_fmt_dollar(plan.get('target2'))}", f"Reward/Risk {_fmt_ratio(plan.get('rr'))}"),
        ("Size Hint", f"{(_safe_float(intr.get('confidence')) or 0.0):.0f}% conf", f"Model size {_safe_float(intr.get('position_size')) or 0.0:.2f} | {intr.get('sig_bucket', '-')}"),
    ]

    reliability_rows = _readiness_rows(instrument_key)
    reliability_rows.extend(_historical_analog_rows(df, instrument_key, current_price, state))
    reliability_rows = reliability_rows[:7]

    breadth_rows = list((breadth_snapshot or {}).get("rows") or [])[:5]
    liquidity_rows = list((liquidity_snapshot or {}).get("rows") or [])[:6]
    catalyst_rows = list((catalyst_snapshot or {}).get("rows") or [])[:5]
    macro_rows = []
    macro_source = str((macro_snapshot or {}).get("source_kind") or "placeholder").title()
    macro_rows.append(("Macro Feed", macro_source, str((macro_snapshot or {}).get("provider_note") or "-")))
    next_macro = (macro_snapshot or {}).get("next_event") if isinstance(macro_snapshot, dict) else None
    if isinstance(next_macro, dict):
        next_ts = str(pd.Timestamp(next_macro.get("scheduled_at")).strftime("%Y-%m-%d %H:%M UTC")) if next_macro.get("scheduled_at") else "-"
        time_text = str(next_macro.get("time_to_event_text") or (macro_snapshot or {}).get("time_to_next_text") or "-")
        macro_rows.append(("Next Macro", str(next_macro.get("title") or "-"), f"{time_text} | {next_ts}"))
    macro_rows.append(("Macro Risk", f"{int(round(100.0 * (_safe_float((macro_snapshot or {}).get('risk_score')) or 0.0)))}/100", "Confidence downgrade only"))
    for evt in ((macro_snapshot or {}).get("events") or [])[:2]:
        try:
            label = str(evt.get("title") or "-")
            when = str(evt.get("time_to_event_text") or pd.Timestamp(evt.get("scheduled_at")).strftime("%m-%d %H:%M UTC"))
            macro_rows.append((str(evt.get("severity") or "-").title(), label[:28], when))
        except Exception:
            continue

    action_card = f"Action: {plan.get('action', 'No-trade')}"
    action_detail = plan.get("detail", "-")
    quality_card = f"Quality: {quality.get('grade', '-')}"
    quality_detail = f"{quality.get('strategy', '-')} | Agreement {_fmt_prob(quality.get('agreement'))} | No-trade {'Yes' if quality.get('no_trade') else 'No'}"

    return {
        "action_card": action_card,
        "action_detail": action_detail,
        "quality_card": quality_card,
        "quality_detail": quality_detail,
        "probability_rows": probability_rows,
        "quality_rows": quality_rows,
        "setup_rows": setup_rows,
        "scenario_rows": _scenario_rows(current_price, pred_state),
        "invalidation_rows": invalidation_rows,
        "reliability_rows": reliability_rows,
        "breadth_rows": breadth_rows,
        "macro_rows": macro_rows[:5],
        "liquidity_rows": liquidity_rows,
        "catalyst_rows": catalyst_rows,
        "quality_meta": quality,
        "plan_meta": plan,
        "pred_state": pred_state,
        "macro_meta": macro_snapshot or {},
        "breadth_meta": breadth_snapshot or {},
        "liquidity_meta": liquidity_snapshot or {},
        "catalyst_meta": catalyst_snapshot or {},
        "predictions_summary": {
            "intraday_position_size": _safe_float(intr.get("position_size")),
            "intraday_direction": intr.get("direction"),
        },
    }
