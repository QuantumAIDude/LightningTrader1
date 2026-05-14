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
from instrument_profile import build_instrument_meta
from liquidity_analytics import get_liquidity_snapshot
from macro_events import get_macro_event_snapshot
from news_pipeline import build_news_runtime_summary, get_news_health, load_recent_headlines
from onboarding_jobs import summarize_onboarding_state
from readiness_tracker import load_readiness_state, summarize_deployment_readiness
from risk_engine import classify_market_regime, risk_gate_guidance, risk_summary_for_symbol

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
    return build_instrument_meta(key, overrides=INSTRUMENTS.get(key, {}))


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


def _fmt_bps_num(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{val:.1f} bps"


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
        "direction_stage": str(cons.get("direction_stage") or "blocked").lower(),
        "raw_direction": str(cons.get("raw_direction") or "-").upper(),
        "horizon_support_stage": str(cons.get("horizon_support_stage") or "unavailable").lower(),
        "horizon_support_direction": str(cons.get("horizon_support_direction") or "-").upper(),
        "horizon_support_score": _safe_float(cons.get("horizon_support_score")),
        "horizon_blended_norm": _safe_float(cons.get("horizon_blended_norm")),
        "p_up": _prob_from_item(cons),
        "raw_p_up": _safe_float(cons.get("raw_p_up")),
        "confidence": _safe_float(cons.get("confidence")),
        "raw_confidence": _safe_float(cons.get("raw_confidence")),
        "agreement": _aggregate_block(intraday, current_price).get("agreement"),
        "bands": {
            "70": (_safe_float(cons.get("pred_range_70_low")), _safe_float(cons.get("pred_range_70_high"))),
            "95": (_safe_float(cons.get("pred_range_95_low")), _safe_float(cons.get("pred_range_95_high"))),
        },
        "edge_bps": _safe_float(cons.get("edge_bps")),
        "raw_edge_bps": _safe_float(cons.get("raw_edge_bps")),
        "raw_pred_ret_bps": None if _safe_float(cons.get("raw_pred_ret")) is None else float(_safe_float(cons.get("raw_pred_ret")) * 10000.0),
        "cost_bps_est": _safe_float(cons.get("cost_bps_est")),
        "uncertainty_penalty_bps": _safe_float(cons.get("uncertainty_penalty_bps")),
        "uncertainty_bps": _safe_float(cons.get("uncertainty_bps")),
        "note": str(cons.get("note") or ""),
        "position_size": _safe_float(cons.get("position_size")),
        "raw_position_size": _safe_float(cons.get("raw_position_size")),
        "sig_bucket": str(cons.get("sig_bucket") or "-"),
        "shrink_factor": _safe_float(cons.get("shrink_factor")),
        "align_score": _safe_float(cons.get("align_score")),
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


def _signed_label(score: float | None, positive: str = "Bullish", negative: str = "Bearish", neutral: str = "Neutral") -> str:
    val = _safe_float(score)
    if val is None:
        return neutral
    if val > 0.05:
        return positive
    if val < -0.05:
        return negative
    return neutral


def _recent_tape_context(df: pd.DataFrame, current_price: float | None) -> dict:
    if df is None or df.empty or current_price in (None, 0.0) or "close" not in df.columns:
        return {}
    close = pd.to_numeric(df.get("close"), errors="coerce").dropna()
    if close.empty:
        return {}
    try:
        bars_30m = max(1, int(round(30.0 / max(1.0, float(BAR_MINUTES)))))
    except Exception:
        bars_30m = 15
    try:
        bars_60m = max(1, int(round(60.0 / max(1.0, float(BAR_MINUTES)))))
    except Exception:
        bars_60m = 30
    ret_30m = None
    ret_60m = None
    if len(close) > bars_30m:
        ret_30m = _safe_float((close.iloc[-1] / close.iloc[-bars_30m - 1]) - 1.0)
    if len(close) > bars_60m:
        ret_60m = _safe_float((close.iloc[-1] / close.iloc[-bars_60m - 1]) - 1.0)
    try:
        session_open = _safe_float(close.iloc[0])
        session_ret = None if session_open in (None, 0.0) else _safe_float((close.iloc[-1] / session_open) - 1.0)
    except Exception:
        session_ret = None
    return {"ret_30m": ret_30m, "ret_60m": ret_60m, "session_ret": session_ret}


def _news_context_snapshot(
    df: pd.DataFrame,
    instrument_key: str,
    news_items: list[dict],
    current_price: float | None,
    pred_state: dict,
    news_health: dict | None = None,
    news_runtime: dict | None = None,
    now_utc: pd.Timestamp | None = None,
) -> dict:
    health = dict(news_health or get_news_health(instrument_key))
    runtime = dict(news_runtime or build_news_runtime_summary(instrument_key, headlines=news_items, news_health=health, as_of=now_utc))
    layer_a = dict(runtime.get("layer_a") or {})
    layer_b = dict(runtime.get("layer_b") or {})
    profile = dict(runtime.get("profile") or {})
    profile_exposures = dict(profile.get("profile_exposures") or {})
    dominant_themes = [str(item.get("theme") or "").replace("_", " ") for item in list(profile_exposures.get("dominant_themes") or []) if str(item.get("theme") or "").strip()]
    profile_focus = ", ".join(dominant_themes[:2]) or str(profile.get("sector") or "Broad Market")
    if not news_items:
        return {
            "health": health,
            "headline_state": str((health or {}).get("headline_state") or "missing"),
            "history_state": str((health or {}).get("history_state") or "missing"),
            "dominant_event": "None",
            "impact_horizon": "-",
            "alignment": "neutral",
            "alignment_label": "Neutral",
            "tape_confirmation": "no_signal",
            "tape_label": "No strong news bias",
            "bias_score": 0.0,
            "fresh_high_impact_count": 0,
            "summary": "No fresh structured headlines in the local news store.",
            "layer_a_summary": str(layer_a.get("summary") or "No structured headlines"),
            "layer_b_status": str(layer_b.get("status") or "disabled"),
            "layer_b_candidate_count": int(layer_b.get("candidate_count") or 0),
            "profile_label": f"{str(profile.get('instrument_kind') or 'equity').title()} | {str(profile.get('sector') or 'Broad Market')}",
            "profile_focus": profile_focus,
        }
    now = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    weighted_items = []
    for item in list(news_items or []):
        try:
            ts = pd.Timestamp(item.get("published_at"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age_h = max(0.0, (now - ts).total_seconds() / 3600.0)
        except Exception:
            age_h = 999.0
        importance = _safe_float(item.get("importance_score")) or 0.0
        impact = _safe_float(item.get("impact_score")) or 0.0
        score = _safe_float(item.get("direction_score"))
        if score is None:
            score = (_safe_float(item.get("score")) or 0.0) * max(0.15, importance or impact or 0.25)
        freshness = float(np.exp(-(age_h / 18.0)))
        weight = freshness * max(0.10, importance, 0.65 * impact)
        weighted_items.append(
            {
                "item": item,
                "age_h": age_h,
                "weight": weight,
                "importance": importance,
                "impact": impact,
                "direction_score": score,
            }
        )
    weighted_items.sort(key=lambda x: (x["weight"], -x["age_h"]), reverse=True)
    bias_score = float(sum(x["direction_score"] * x["weight"] for x in weighted_items))
    dominant = weighted_items[0]["item"] if weighted_items else {}
    fresh_high_impact = sum(1 for x in weighted_items if x["age_h"] <= 24.0 and max(x["importance"], x["impact"]) >= 0.45)
    intr_block = ((pred_state or {}).get("intraday") or {})
    intr_direction = str(intr_block.get("direction") or "-").upper()
    if intr_direction not in {"UP", "DOWN"}:
        intr_direction = str(intr_block.get("raw_direction") or "-").upper()
    if intr_direction not in {"UP", "DOWN"}:
        raw_pred_bps = _safe_float(intr_block.get("raw_pred_ret_bps"))
        if raw_pred_bps is not None and abs(raw_pred_bps) >= 0.75:
            intr_direction = "UP" if raw_pred_bps > 0.0 else "DOWN"
    alignment = "neutral"
    if bias_score >= 0.05 and intr_direction == "UP":
        alignment = "aligned"
    elif bias_score <= -0.05 and intr_direction == "DOWN":
        alignment = "aligned"
    elif bias_score >= 0.05 and intr_direction == "DOWN":
        alignment = "conflict"
    elif bias_score <= -0.05 and intr_direction == "UP":
        alignment = "conflict"
    alignment_label = "Aligned" if alignment == "aligned" else "Conflict" if alignment == "conflict" else "Neutral"
    tape = _recent_tape_context(df, current_price)
    tape_move = _safe_float(tape.get("ret_30m"))
    if tape_move is None:
        tape_move = _safe_float(tape.get("session_ret"))
    tape_confirmation = "no_signal"
    tape_label = "No strong news bias"
    if abs(bias_score) >= 0.05 and tape_move is not None:
        if abs(tape_move) < 0.0015 and fresh_high_impact >= 1:
            tape_confirmation = "underreacting"
            tape_label = "Price has not fully reacted yet"
        elif np.sign(tape_move) == np.sign(bias_score):
            tape_confirmation = "confirming"
            tape_label = "Tape is confirming the news bias"
        else:
            tape_confirmation = "conflicting"
            tape_label = "Tape is moving against the news bias"
    dominant_event = str(dominant.get("event_label") or dominant.get("event_type") or "General")
    dominant_scope = str(dominant.get("relevance_scope_label") or dominant.get("relevance_scope") or "Relevant").replace("_", " ").title()
    impact_horizon = str(dominant.get("impact_horizon") or "-")
    bias_label = _signed_label(bias_score)
    headline_state = str((health or {}).get("headline_state") or "missing")
    history_state = str((health or {}).get("history_state") or "missing")
    summary = f"{dominant_scope} {dominant_event} | {bias_label} | {alignment_label} vs model | {tape_label}"
    if headline_state not in {"ok", "thin"}:
        summary = f"{headline_state.replace('_', ' ').title()} | {summary}"
    if history_state != "ready":
        summary = f"{summary} | hist {history_state.replace('_', ' ')}"
    layer_a_summary = str(layer_a.get("summary") or "")
    if layer_a_summary and layer_a_summary.lower() not in summary.lower():
        summary = f"{summary} | {layer_a_summary}"
    return {
        "health": health,
        "headline_state": headline_state,
        "history_state": history_state,
        "dominant_event": dominant_event,
        "dominant_scope": dominant_scope,
        "impact_horizon": impact_horizon,
        "alignment": alignment,
        "alignment_label": alignment_label,
        "tape_confirmation": tape_confirmation,
        "tape_label": tape_label,
        "bias_score": bias_score,
        "bias_label": bias_label,
        "fresh_high_impact_count": int(fresh_high_impact),
        "top_item": dominant,
        "tape": tape,
        "summary": summary,
        "layer_a_summary": layer_a_summary,
        "layer_b_status": str(layer_b.get("status") or "disabled"),
        "layer_b_candidate_count": int(layer_b.get("candidate_count") or 0),
        "profile_label": f"{str(profile.get('instrument_kind') or 'equity').title()} | {str(profile.get('sector') or 'Broad Market')}",
        "profile_focus": profile_focus,
    }


def _signal_quality(
    current_price: float | None,
    pred_state: dict,
    noise: dict,
    spread_frac: float | None,
    volume_state: tuple[str, float | None],
    regime: dict,
    event_snapshot: dict,
    news_items: list[dict],
    news_snapshot: dict | None = None,
    macro_snapshot: dict | None = None,
    breadth_snapshot: dict | None = None,
    liquidity_snapshot: dict | None = None,
    catalyst_snapshot: dict | None = None,
    now_utc: pd.Timestamp | None = None,
) -> dict:
    intr = pred_state.get("intraday", {})
    signal_direction = str(intr.get("direction") or "-").upper()
    if signal_direction not in {"UP", "DOWN"}:
        signal_direction = str(intr.get("raw_direction") or "-").upper()
    if signal_direction not in {"UP", "DOWN"}:
        raw_pred_hint = _safe_float(intr.get("raw_pred_ret_bps"))
        if raw_pred_hint is not None and abs(raw_pred_hint) >= 0.75:
            signal_direction = "UP" if raw_pred_hint > 0.0 else "DOWN"
    p_up = _safe_float(intr.get("p_up"))
    raw_p_up = _safe_float(intr.get("raw_p_up"))
    conf = None if p_up is None else max(p_up, 1.0 - p_up)
    raw_conf = None if raw_p_up is None else max(raw_p_up, 1.0 - raw_p_up)
    signal_conf = conf
    if signal_conf in (None, 0.5) and raw_conf is not None:
        signal_conf = raw_conf
    elif raw_conf is not None and (signal_conf is None or signal_conf < 0.53):
        signal_conf = max(float(signal_conf or 0.0), float(raw_conf))
    agreement = _safe_float(intr.get("agreement"))
    point = _safe_float(intr.get("point"))
    raw_direction = str(intr.get("raw_direction") or "-").upper()
    raw_edge_bps = _safe_float(intr.get("raw_edge_bps"))
    raw_pred_ret_bps = _safe_float(intr.get("raw_pred_ret_bps"))
    edge_bps = _safe_float(intr.get("edge_bps"))
    uncertainty_penalty_bps = _safe_float(intr.get("uncertainty_penalty_bps"))
    cost_bps_est = _safe_float(intr.get("cost_bps_est"))
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
    horizon_support_stage = str(intr.get("horizon_support_stage") or "unavailable").lower()
    horizon_support_score = _safe_float(intr.get("horizon_support_score")) or 0.0
    base_parts = [v for v in (signal_conf, agreement, noise_score, spread_score, regime_score, vol_score, liquidity_score) if v is not None]
    score = float(np.mean(base_parts)) if base_parts else 0.45

    now_ref = pd.Timestamp(now_utc) if now_utc is not None else pd.Timestamp.now(tz="UTC")
    if now_ref.tzinfo is None:
        now_ref = now_ref.tz_localize("UTC")
    else:
        now_ref = now_ref.tz_convert("UTC")
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
            age_h = max(0.0, (now_ref - ts).total_seconds() / 3600.0)
            if age_h <= 6.0:
                fresh_headlines += 1
                label = str(item.get("label") or "").lower()
                if "neg" in label or "bear" in label:
                    neg_hits += 1
        except Exception:
            continue
    if fresh_headlines >= 2:
        event_penalty += 0.06
    if neg_hits >= 2 and signal_direction == "UP":
        event_penalty += 0.05
    news_snapshot = news_snapshot or {}
    news_alignment = str(news_snapshot.get("alignment") or "neutral")
    news_tape = str(news_snapshot.get("tape_confirmation") or "no_signal")
    news_state = str(news_snapshot.get("headline_state") or "missing")
    fresh_high_impact = int(news_snapshot.get("fresh_high_impact_count") or 0)
    if news_state in {"failed", "stale"}:
        event_penalty += 0.05
    elif news_state == "thin":
        event_penalty += 0.02
    if news_alignment == "conflict":
        event_penalty += 0.08
    elif news_alignment == "aligned" and fresh_high_impact >= 1:
        event_penalty -= 0.03
    if news_tape == "conflicting":
        event_penalty += 0.05
    elif news_tape == "confirming" and news_alignment == "aligned":
        event_penalty -= 0.02
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
    if neg_catalysts >= 1 and signal_direction == "UP":
        event_penalty += 0.05
    if neg_catalysts >= 1 and signal_direction == "DOWN":
        event_penalty -= 0.02
    breadth_penalty_mult = 1.0 if bool((breadth_snapshot or {}).get("direct_feed")) else 0.7 if breadth_source_kind == "live_partial" else 0.4 if breadth_source_kind == "proxy_only" else 0.0
    if breadth_score is not None and breadth_penalty_mult > 0.0:
        if signal_direction == "UP" and breadth_score < 0.40:
            event_penalty += 0.06 * breadth_penalty_mult
        elif signal_direction == "DOWN" and breadth_score > 0.60:
            event_penalty += 0.06 * breadth_penalty_mult
    if liquidity_score is not None and liquidity_score < 0.45:
        event_penalty += 0.12
    score = max(0.0, min(1.0, score - event_penalty))
    if (
        horizon_support_stage in {"supportive", "confirmed"}
        and signal_direction in {"UP", "DOWN"}
        and news_alignment != "conflict"
        and (((raw_edge_bps or 0.0) > 0.0) or ((edge_bps or 0.0) > 0.0))
    ):
        support_bonus = (0.02 if horizon_support_stage == "supportive" else 0.035) + (0.03 * float(horizon_support_score))
        score = max(0.0, min(1.0, score + support_bonus))

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

    hard_reasons = []
    soft_reasons = []

    def _add_reason(reason: str, hard: bool = False) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        if hard:
            hard_reasons.append(text)
        else:
            soft_reasons.append(text)

    if signal_conf is not None and signal_conf < 0.56:
        _add_reason("direction probability is weak")
    if agreement is not None and agreement < 0.55:
        _add_reason("models disagree")
    if raw_pred_ret_bps is not None and abs(raw_pred_ret_bps) < 1.0:
        _add_reason("raw model move is very small")
    if raw_direction not in {"UP", "DOWN"} and raw_pred_ret_bps is not None and abs(raw_pred_ret_bps) >= 1.0:
        _add_reason("directional lean exists, but it is not surviving signal translation")
    if noise_ratio is not None and noise_ratio < 1.0:
        _add_reason("predicted move is near normal noise")
    if move_vs_spread is not None and move_vs_spread < 1.5:
        _add_reason("move is small vs spread")
    if raw_edge_bps is not None and edge_bps is not None and raw_edge_bps > 0.0 and edge_bps <= 0.0:
        _add_reason("raw signal exists, but cost and uncertainty remove the tradable edge")
    elif edge_bps is not None and edge_bps <= 0.0 and uncertainty_penalty_bps is not None and cost_bps_est is not None:
        if uncertainty_penalty_bps >= max(0.5, 0.75 * abs(cost_bps_est)):
            _add_reason("uncertainty is overwhelming the forecast edge")
    if dte is not None and dte <= 1.5:
        _add_reason("earnings risk is imminent", hard=True)
    if macro_risk is not None and macro_risk >= 0.75:
        _add_reason("major macro event is close", hard=True)
    if catalyst_risk is not None and catalyst_risk >= 0.45:
        _add_reason("fresh catalysts are active", hard=True)
    if vol_state == "Thin participation":
        _add_reason("volume confirmation is weak")
    if liquidity_score is not None and liquidity_score < 0.45:
        _add_reason("execution quality is weak", hard=True)
    if news_alignment == "conflict":
        _add_reason("fresh news conflicts with the model bias", hard=True)
    elif news_state in {"failed", "stale"}:
        _add_reason("news pipeline is degraded")
    elif news_tape == "underreacting":
        _add_reason("fresh news has not been confirmed by tape yet")
    breadth_desc = "direct breadth" if bool((breadth_snapshot or {}).get("direct_feed")) else "partial breadth" if breadth_source_kind == "live_partial" else "proxy breadth"
    if breadth_score is not None and breadth_penalty_mult > 0.0:
        if signal_direction == "UP" and breadth_score < 0.40:
            _add_reason(f"{breadth_desc} does not confirm upside")
        elif signal_direction == "DOWN" and breadth_score > 0.60:
            _add_reason(f"{breadth_desc} does not confirm downside")
    hard_macro = bool((macro_snapshot or {}).get("high_risk_24h"))
    reasons = hard_reasons + soft_reasons
    no_trade = bool(
        hard_macro
        or len(hard_reasons) >= 1
        or score < 0.54
        or (score < 0.60 and len(soft_reasons) >= 2)
        or (
            (edge_bps is None or edge_bps <= 0.0)
            and (raw_edge_bps is None or raw_edge_bps <= 0.0)
            and len(soft_reasons) >= 1
        )
    )

    return {
        "score": score,
        "grade": grade,
        "strategy": strategy,
        "whipsaw": whipsaw,
        "noise_ratio": noise_ratio,
        "move_vs_spread": move_vs_spread,
        "agreement": agreement,
        "confidence": signal_conf,
        "raw_signal_confidence": raw_conf,
        "no_trade": no_trade,
        "reasons": reasons,
        "hard_reasons": hard_reasons,
        "soft_reasons": soft_reasons,
        "volume_state": vol_state,
        "regime_label": regime_label,
        "macro_risk": macro_risk,
        "catalyst_risk": catalyst_risk,
        "breadth_score": breadth_score,
        "breadth_source_kind": breadth_source_kind,
        "liquidity_score": liquidity_score,
        "news_alignment": news_alignment,
        "news_tape": news_tape,
        "news_state": news_state,
        "news_event": str(news_snapshot.get("dominant_event") or "-"),
        "news_bias_score": _safe_float(news_snapshot.get("bias_score")),
    }


def _signal_translation(pred_state: dict, quality: dict, news_snapshot: dict | None = None) -> dict:
    intr = (pred_state or {}).get("intraday", {}) if isinstance(pred_state, dict) else {}
    news_meta = news_snapshot or {}
    direction = str(intr.get("direction") or "-").upper()
    direction_stage = str(intr.get("direction_stage") or "blocked").lower()
    raw_direction = str(intr.get("raw_direction") or "-").upper()
    raw_pred_ret_bps = _safe_float(intr.get("raw_pred_ret_bps"))
    raw_edge_bps = _safe_float(intr.get("raw_edge_bps"))
    edge_bps = _safe_float(intr.get("edge_bps"))
    cost_bps_est = _safe_float(intr.get("cost_bps_est"))
    uncertainty_penalty_bps = _safe_float(intr.get("uncertainty_penalty_bps"))
    confidence = _safe_float(intr.get("confidence"))
    raw_confidence = _safe_float(intr.get("raw_confidence"))
    position_size = _safe_float(intr.get("position_size"))
    raw_position_size = _safe_float(intr.get("raw_position_size"))
    shrink_factor = _safe_float(intr.get("shrink_factor"))
    horizon_support_stage = str(intr.get("horizon_support_stage") or "unavailable").lower()
    horizon_support_score = _safe_float(intr.get("horizon_support_score")) or 0.0
    quality_score = _safe_float((quality or {}).get("score")) or 0.0
    no_trade = bool((quality or {}).get("no_trade"))
    reasons = [str(r).strip() for r in list((quality or {}).get("reasons") or []) if str(r).strip()]
    news_alignment = str(news_meta.get("alignment") or "neutral").lower()
    news_tape = str(news_meta.get("tape_confirmation") or "no_signal").lower()

    bias = direction
    if bias not in {"UP", "DOWN"} and raw_pred_ret_bps is not None and abs(raw_pred_ret_bps) >= 0.75:
        bias = "UP" if raw_pred_ret_bps > 0.0 else "DOWN"
    if bias not in {"UP", "DOWN"} and raw_direction in {"UP", "DOWN"}:
        bias = raw_direction
    bias_label = "Bullish" if bias == "UP" else "Bearish" if bias == "DOWN" else "Neutral"
    operator_side = "buy" if bias == "UP" else "sell" if bias == "DOWN" else ""

    kind = "weak_signal"
    kind_label = "Weak signal"
    headline = "Stand Aside"
    action_label = "Stand Aside"
    summary = "The stock path is not offering a finished directional edge yet."
    primary_blocker = reasons[0] if reasons else "wait for a clearer setup"
    watchlist = False
    economics_state_key = "unfinished"
    economics_state_label = "Unfinished edge"
    economics_detail = "The model lean has not become a finished tradeable edge yet."

    if news_alignment == "conflict" and bias in {"UP", "DOWN"}:
        kind = "news_conflicted_signal"
        kind_label = "News-conflicted signal"
        headline = f"{bias_label} but News-Conflicted"
        action_label = "Stand Aside"
        summary = f"{bias_label} model bias exists, but fresh news conflicts with it."
        primary_blocker = "fresh news conflicts with the model bias"
        watchlist = bool((raw_edge_bps or 0.0) > 0.0 or quality_score >= 0.58)
        economics_state_key = "news_conflicted"
        economics_state_label = "News-conflicted"
        economics_detail = "Fresh headlines conflict with the model bias, so the setup is watchlist-only until tape and news realign."
    elif bias in {"UP", "DOWN"} and not no_trade and (edge_bps or 0.0) > 0.0 and (position_size or 0.0) > 0.0:
        kind = "actionable_signal"
        kind_label = "Finished tradable signal"
        headline = f"Tradable {bias_label} Bias"
        action_label = f"Lean {'Long' if bias == 'UP' else 'Short'}"
        summary = f"Finished edge survives costs and uncertainty for a {bias_label.lower()} stock setup."
        primary_blocker = "edge remains positive after costs"
        economics_state_key = "finished_edge"
        economics_state_label = "Finished edge survives"
        economics_detail = "Raw edge still remains positive after estimated costs and uncertainty penalties."
    elif bias in {"UP", "DOWN"} and direction_stage == "tentative":
        kind = "tentative_signal"
        kind_label = "Tentative finished direction"
        headline = f"Tentative {bias_label} Bias"
        if horizon_support_stage in {"supportive", "confirmed"}:
            headline += " (10m/30m Confirmed)"
        action_label = f"Watchlist {'Long' if bias == 'UP' else 'Short'}"
        summary = f"The {bias_label.lower()} stock direction survives translation, but conviction is still too low to treat it as a finished trade."
        if horizon_support_stage in {"supportive", "confirmed"}:
            summary += " The 10m/30m stock horizon is confirming that lean."
        primary_blocker = reasons[0] if reasons else "confidence is still too low"
        watchlist = True
        economics_state_key = "tentative_edge"
        economics_state_label = "Finished edge, low conviction"
        economics_detail = "The finished edge exists, but conviction and sizing are still too light for a cleaner trade."
    elif bias in {"UP", "DOWN"} and ((raw_edge_bps or 0.0) > 0.0 or (quality_score >= 0.60 and (confidence or 0.0) >= 52.0)):
        watchlist = True
        headline = f"{bias_label} Watchlist"
        if horizon_support_stage in {"supportive", "confirmed"}:
            headline += " (10m/30m Confirmed)"
        action_label = f"Watchlist {'Long' if bias == 'UP' else 'Short'}"
        if raw_edge_bps is not None and raw_edge_bps > 0.0 and (edge_bps or 0.0) <= 0.0:
            if uncertainty_penalty_bps is not None and cost_bps_est is not None and uncertainty_penalty_bps >= max(0.5, cost_bps_est):
                kind = "uncertain_signal"
                kind_label = "Uncertain signal"
                summary = f"A {bias_label.lower()} raw edge exists, but uncertainty is still too large."
                if horizon_support_stage in {"supportive", "confirmed"}:
                    summary += " The 10m/30m stock horizon agrees, but the near-term edge still does not survive risk translation."
                primary_blocker = "uncertainty is still overwhelming the forecast edge"
                economics_state_key = "uncertainty_crushed"
                economics_state_label = "Uncertainty-crushed"
                economics_detail = "The forecast direction exists, but uncertainty suppresses sizing and removes the finished trade."
            else:
                kind = "costly_signal"
                kind_label = "Cost-crushed signal"
                summary = f"A {bias_label.lower()} raw edge exists, but costs and uncertainty still remove the finished edge."
                if horizon_support_stage in {"supportive", "confirmed"}:
                    summary += " The 10m/30m stock horizon agrees, but the immediate trade still does not clear costs."
                primary_blocker = "cost and uncertainty still remove the tradable edge"
                economics_state_key = "cost_crushed"
                economics_state_label = "Cost-crushed"
                economics_detail = "Direction exists, but estimated costs and uncertainty consume the trade economics."
        elif no_trade:
            kind = "blocked_signal"
            kind_label = "Blocked but interesting"
            summary = f"The {bias_label.lower()} setup is worth watching, but it is not clean enough to act on yet."
            primary_blocker = reasons[0] if reasons else "setup is not clean enough yet"
            economics_state_key = "watchlist_only"
            economics_state_label = "Watchlist only"
            economics_detail = "The setup is directionally interesting, but quality blockers still outweigh execution value."
        else:
            kind = "tentative_signal"
            kind_label = "Tentative signal"
            summary = f"The {bias_label.lower()} lean is present, but it still needs confirmation."
            primary_blocker = reasons[0] if reasons else "wait for confirmation"
            economics_state_key = "directional_watchlist"
            economics_state_label = "Directional watchlist"
            economics_detail = "The model sees a lean, but the setup is still below the current execution threshold."
    elif bias in {"UP", "DOWN"}:
        kind = "weak_signal"
        kind_label = "Weak signal"
        headline = f"{bias_label} Lean, Not Finished"
        action_label = "Stand Aside"
        summary = f"The {bias_label.lower()} lean is too small or too noisy to treat as a finished setup."
        primary_blocker = reasons[0] if reasons else "model edge is too small"
        economics_state_key = "weak_edge"
        economics_state_label = "Weak edge"
        economics_detail = "A directional lean exists, but the move is too small or noisy to justify action."
    elif raw_pred_ret_bps is not None and abs(raw_pred_ret_bps) >= 1.0:
        kind = "directional_signal_unfinished"
        kind_label = "Directional signal not finished"
        headline = "Directional Lean, Not Finished"
        action_label = "Watchlist"
        summary = "A directional lean exists before translation, but it is not surviving into a finished stock signal."
        primary_blocker = reasons[0] if reasons else "direction did not survive signal translation"
        watchlist = True
        economics_state_key = "translation_failed"
        economics_state_label = "Direction failed translation"
        economics_detail = "A move may be coming, but it is not surviving signal translation into a finished trade."

    detail_bits = [summary]
    if horizon_support_stage in {"supportive", "confirmed"}:
        detail_bits.append(
            f"10m/30m support {horizon_support_stage} ({round(float(horizon_support_score) * 100.0, 1)} score)"
        )
    if raw_edge_bps is not None or edge_bps is not None:
        detail_bits.append(f"raw {_fmt_bps_num(raw_edge_bps)} -> finished {_fmt_bps_num(edge_bps)}")
    if cost_bps_est is not None or uncertainty_penalty_bps is not None:
        detail_bits.append(f"cost {_fmt_bps_num(cost_bps_est)} | unc {_fmt_bps_num(uncertainty_penalty_bps)}")
    if confidence is not None:
        detail_bits.append(f"conf {confidence:.0f}%")
    if raw_confidence is not None and (confidence is None or abs(raw_confidence - confidence) >= 3.0):
        detail_bits.append(f"raw conf {raw_confidence:.0f}%")
    if position_size is not None:
        detail_bits.append(f"size {position_size:.2f}")
    if raw_position_size is not None and (position_size is None or abs(raw_position_size - position_size) >= 0.05):
        detail_bits.append(f"raw size {raw_position_size:.2f}")
    if shrink_factor is not None and shrink_factor < 0.50:
        detail_bits.append(f"shrink {shrink_factor:.2f}")
    if news_tape == "underreacting" and watchlist:
        detail_bits.append("tape has not confirmed the headline yet")

    return {
        "headline": headline,
        "detail": " | ".join(bit for bit in detail_bits if bit),
        "kind": kind,
        "kind_label": kind_label,
        "direction_stage": direction_stage,
        "horizon_support_stage": horizon_support_stage,
        "horizon_support_score": horizon_support_score,
        "primary_blocker": primary_blocker,
        "watchlist": bool(watchlist),
        "action_label": action_label,
        "operator_side": operator_side,
        "bias_label": bias_label,
        "quality_score": quality_score,
        "raw_edge_bps": raw_edge_bps,
        "finished_edge_bps": edge_bps,
        "raw_pred_ret_bps": raw_pred_ret_bps,
        "economics_state_key": economics_state_key,
        "economics_state_label": economics_state_label,
        "economics_detail": economics_detail,
    }


def _deployment_execution_posture(signal_translation: dict, deployment_meta: dict, quality: dict) -> dict:
    signal_translation = signal_translation if isinstance(signal_translation, dict) else {}
    deployment_meta = deployment_meta if isinstance(deployment_meta, dict) else {}
    operator_key = str(deployment_meta.get("operator_state_key") or "")
    operator_label = str(deployment_meta.get("operator_state_label") or "Operator state unknown")
    operator_detail = str(deployment_meta.get("operator_state_detail") or deployment_meta.get("detail") or "-")
    evidence_label = str(deployment_meta.get("evidence_state_label") or "Evidence unknown")
    stage_label = str(deployment_meta.get("stage_label") or "-")
    next_requirement = str(deployment_meta.get("primary_next_requirement") or "")
    signal_headline = str(signal_translation.get("headline") or "-")
    signal_detail = str(signal_translation.get("detail") or "-")
    watchlist = bool(signal_translation.get("watchlist"))
    hard_block = operator_key in {
        "unsupported",
        "blocked_incomplete_onboarding",
        "blocked_missing_maturity",
        "blocked_missing_models",
        "blocked_missing_evidence",
        "blocked_stale_data",
        "blocked_stale_news",
        "blocked_calibration_drift",
        "blocked_stale_evidence",
    }
    if hard_block:
        action_label = "Stand Aside"
        label = operator_label
        detail_bits = []
        if watchlist or str(signal_translation.get("kind") or "") in {"actionable_signal", "tentative_signal", "blocked_signal", "costly_signal", "uncertain_signal"}:
            detail_bits.append(f"{signal_headline} exists, but execution is blocked by platform evidence.")
        detail_bits.append(operator_detail)
        if next_requirement:
            detail_bits.append(f"Next: {next_requirement}")
        return {
            "key": operator_key or "blocked",
            "label": label,
            "detail": " | ".join(bit for bit in detail_bits if bit),
            "hard_block": True,
            "action_label": action_label,
        }

    detail_bits = [signal_detail]
    if operator_key in {"advisory_only", "advisory_weak_evidence"}:
        detail_bits.append(operator_detail)
        if next_requirement:
            detail_bits.append(f"Next: {next_requirement}")
    elif operator_key in {"shadow_candidate", "paper_candidate", "paper_candidate_rollout_gated", "live_candidate_rollout_gated", "rollout_allowed"}:
        detail_bits.append(f"{operator_label} | {stage_label} stage")
    elif evidence_label:
        detail_bits.append(f"Evidence: {evidence_label}")
    return {
        "key": operator_key or "advisory",
        "label": operator_label,
        "detail": " | ".join(bit for bit in detail_bits if bit),
        "hard_block": False,
        "action_label": str(signal_translation.get("action_label") or "Stand Aside"),
    }


def _advisory_decision_payload(
    signal_translation: dict,
    regime_meta: dict,
    risk_meta: dict,
    deployment_meta: dict,
    execution_posture: dict,
    quality: dict,
    plan: dict,
    pred_state: dict,
) -> dict:
    signal_translation = dict(signal_translation or {})
    regime_meta = dict(regime_meta or {})
    risk_meta = dict(risk_meta or {})
    deployment_meta = dict(deployment_meta or {})
    validation_evidence = dict(deployment_meta.get("validation_evidence") or {})
    execution_posture = dict(execution_posture or {})
    quality = dict(quality or {})
    plan = dict(plan or {})
    intr = dict((pred_state or {}).get("intraday") or {})

    operator_side = str(signal_translation.get("operator_side") or "").strip().lower()
    watchlist = bool(signal_translation.get("watchlist"))
    regime_label = str(regime_meta.get("label") or "unknown_or_insufficient_evidence")
    risk_blocked = bool(risk_meta.get("blocked"))
    risk_guidance = risk_gate_guidance(risk_meta)
    posture_blocked = bool(execution_posture.get("hard_block"))
    if posture_blocked or risk_blocked or regime_label == "unknown_or_insufficient_evidence":
        action = "watchlist" if watchlist else "stand_aside"
    elif operator_side in {"buy", "sell"} and str(signal_translation.get("kind") or "") == "actionable_signal":
        action = operator_side
    elif watchlist:
        action = "watchlist"
    else:
        action = "stand_aside"

    blockers = []
    for item in [
        signal_translation.get("primary_blocker"),
        risk_meta.get("primary_blocker"),
        deployment_meta.get("primary_blocker") if posture_blocked else "",
    ]:
        text = str(item or "").strip()
        if text and text not in blockers:
            blockers.append(text)
    contradictory = []
    for item in list(quality.get("hard_reasons") or []) + list(quality.get("soft_reasons") or []):
        text = str(item or "").strip()
        if text and text not in blockers and text not in contradictory:
            contradictory.append(text)

    next_best_action = ""
    if regime_label == "unknown_or_insufficient_evidence":
        next_best_action = "Wait for enough recent bars and stable context to classify the market regime safely."
    elif str(signal_translation.get("economics_state_key") or "") in {"cost_crushed", "uncertainty_crushed"}:
        next_best_action = "Wait for a finished edge that stays positive after estimated costs and uncertainty."
    elif risk_blocked:
        next_best_action = str(risk_guidance.get("next_requirement") or "Resolve the active risk blocker before promoting this setup beyond advisory review.")
    else:
        next_best_action = str(deployment_meta.get("primary_next_requirement") or "").strip()
        if not next_best_action:
            if action in {"buy", "sell"}:
                next_best_action = "Use the suggested stop and size discipline; do not override the risk budget."
            else:
                next_best_action = "Monitor for clearer confirmation before taking risk."

    return {
        "action": action,
        "thesis": str(signal_translation.get("economics_detail") or signal_translation.get("detail") or "-"),
        "market_regime": regime_label,
        "regime_confidence": _safe_float(regime_meta.get("confidence")),
        "model_edge": {
            "direction": str(intr.get("direction") or intr.get("raw_direction") or "").upper(),
            "raw_edge_bps": _safe_float(signal_translation.get("raw_edge_bps")),
            "finished_edge_bps": _safe_float(signal_translation.get("finished_edge_bps")),
            "confidence": _safe_float(intr.get("confidence")),
        },
        "economic_edge": {
            "state_key": str(signal_translation.get("economics_state_key") or ""),
            "state_label": str(signal_translation.get("economics_state_label") or ""),
            "detail": str(signal_translation.get("economics_detail") or ""),
            "cost_bps_est": _safe_float(intr.get("cost_bps_est")),
            "uncertainty_penalty_bps": _safe_float(intr.get("uncertainty_penalty_bps")),
        },
        "risk_context": {
            "blocked": risk_blocked,
            "primary_blocker": str(risk_meta.get("primary_blocker") or ""),
            "guidance_key": str(risk_guidance.get("key") or ""),
            "guidance_detail": str(risk_guidance.get("detail") or ""),
            "next_requirement": str(risk_guidance.get("next_requirement") or ""),
            "suggested_qty": int(risk_meta.get("suggested_qty") or 0),
            "stop_price": _safe_float(risk_meta.get("stop_price") or plan.get("stop")),
            "risk_budget": _safe_float(((risk_meta.get("risk_budget") or {}).get("risk_budget"))),
        },
        "evidence_strength": {
            "quality_score": _safe_float(quality.get("score")),
            "signal_grade": str(quality.get("grade") or ""),
            "symbol_maturity": str(deployment_meta.get("symbol_maturity_label") or ""),
            "evidence_state": str(deployment_meta.get("evidence_state_label") or ""),
        },
        "validation_status": str(validation_evidence.get("validation_status") or deployment_meta.get("validation_status") or ""),
        "validation_method": str(validation_evidence.get("validation_method") or deployment_meta.get("validation_method") or ""),
        "validation_summary": str(validation_evidence.get("summary") or deployment_meta.get("validation_summary") or ""),
        "economic_validation": dict(validation_evidence.get("economic_metrics") or {}),
        "leakage_controls_summary": str(validation_evidence.get("leakage_controls_summary") or ""),
        "validation_blockers": list(
            validation_evidence.get("blockers")
            or validation_evidence.get("autonomy_blockers")
            or deployment_meta.get("validation_blockers")
            or []
        ),
        "validation_next_best_action": str(validation_evidence.get("next_best_action") or deployment_meta.get("validation_next_best_action") or ""),
        "blockers": blockers,
        "contradictory_evidence": contradictory[:4],
        "next_best_action": next_best_action,
        "autonomy_posture": {
            "operator_state": str(deployment_meta.get("operator_state_label") or ""),
            "rollout": str(deployment_meta.get("rollout_label") or ""),
            "stage": str(deployment_meta.get("stage_label") or ""),
        },
    }


def advisory_decision_rows(decision_payload: dict | None) -> list[tuple[str, str, str]]:
    payload = dict(decision_payload or {})
    if not payload:
        return []

    def _clean(value, fallback: str = "-") -> str:
        if value is None:
            return fallback
        text = str(value).strip()
        if not text:
            return fallback
        return text if len(text) <= 180 else f"{text[:177]}..."

    def _joined(values, fallback: str = "-") -> str:
        items = []
        for item in values or []:
            text = str(item or "").strip()
            if text and text not in items:
                items.append(text)
        return _clean("; ".join(items), fallback=fallback)

    model_edge = dict(payload.get("model_edge") or {})
    economic_edge = dict(payload.get("economic_edge") or {})
    risk_context = dict(payload.get("risk_context") or {})
    evidence = dict(payload.get("evidence_strength") or {})
    economic_validation = dict(payload.get("economic_validation") or {})
    posture = dict(payload.get("autonomy_posture") or {})
    rows = [
        (
            "Decision Action",
            _clean(payload.get("action")).replace("_", " ").title(),
            _clean(payload.get("thesis")),
        ),
        (
            "Market Regime",
            _clean(payload.get("market_regime")).replace("_", " ").title(),
            f"Confidence {_fmt_prob(payload.get('regime_confidence'))}",
        ),
        (
            "Model Edge",
            _clean(model_edge.get("direction")),
            f"raw {_fmt_bps_num(model_edge.get('raw_edge_bps'))} | finished {_fmt_bps_num(model_edge.get('finished_edge_bps'))} | conf {_fmt_prob(model_edge.get('confidence'))}",
        ),
        (
            "Economic Edge",
            _clean(economic_edge.get("state_label") or economic_edge.get("state_key")),
            f"{_clean(economic_edge.get('detail'))} | cost {_fmt_bps_num(economic_edge.get('cost_bps_est'))} | unc {_fmt_bps_num(economic_edge.get('uncertainty_penalty_bps'))}",
        ),
        (
            "Risk Context",
            "Blocked" if bool(risk_context.get("blocked")) else "Open",
            _clean(
                risk_context.get("guidance_detail")
                or risk_context.get("primary_blocker")
                or (
                    f"Budget {_clean(risk_context.get('risk_budget'))} | qty {int(risk_context.get('suggested_qty') or 0)}"
                    if risk_context
                    else "-"
                )
            ),
        ),
        (
            "Evidence Strength",
            _clean(evidence.get("signal_grade") or evidence.get("evidence_state")),
            f"quality {_fmt_prob(evidence.get('quality_score'))} | maturity {_clean(evidence.get('symbol_maturity'))} | evidence {_clean(evidence.get('evidence_state'))}",
        ),
    ]
    validation_status = _clean(payload.get("validation_status"), fallback="")
    validation_summary = _clean(payload.get("validation_summary"), fallback="")
    if validation_status or validation_summary:
        rows.append(
            (
                "Validation Evidence",
                _clean(validation_status or "missing").replace("_", " ").title(),
                _clean(
                    validation_summary
                    or (
                        f"{_clean(payload.get('validation_method'))} | net {_clean(economic_validation.get('net_return_after_costs'))}"
                        if economic_validation
                        else ""
                    )
                ),
            )
        )
    leakage_summary = _clean(payload.get("leakage_controls_summary"), fallback="")
    validation_blockers = _joined(payload.get("validation_blockers") or [], fallback="")
    if leakage_summary or validation_blockers:
        rows.append(
            (
                "Validation Controls",
                _clean(payload.get("validation_method") or "none").replace("_", " ").title(),
                _clean("; ".join(item for item in [leakage_summary, validation_blockers] if item)),
            )
        )
    blockers = _joined(payload.get("blockers") or [])
    if blockers != "-":
        rows.append(("Blockers", str(len(payload.get("blockers") or [])), blockers))
    contradictions = _joined(payload.get("contradictory_evidence") or [])
    if contradictions != "-":
        rows.append(("Contradictions", str(len(payload.get("contradictory_evidence") or [])), contradictions))
    rows.append(("Next Best Action", "-", _clean(payload.get("next_best_action"))))
    rows.append(
        (
            "Autonomy Posture",
            _clean(posture.get("operator_state") or posture.get("stage")),
            f"{_clean(posture.get('stage'))} | {_clean(posture.get('rollout'))}",
        )
    )
    return rows[:12]


def _history_compact_row(metrics: dict) -> tuple[str, str, str] | None:
    if not isinstance(metrics, dict) or not metrics:
        return None
    i_hit = _fmt_prob(metrics.get("intraday_window_hit_rate"))
    d_hit = _fmt_prob(metrics.get("daily_window_hit_rate"))
    w_hit = _fmt_prob(metrics.get("weekend_window_hit_rate"))
    detail = (
        f"4m {_fmt_dollar(metrics.get('intraday_avg_discrepancy_dollars'))}/{_fmt_dollar(metrics.get('intraday_target_window_dollars'))}"
        f" | 1D {_fmt_dollar(metrics.get('daily_avg_discrepancy_dollars'))}/{_fmt_dollar(metrics.get('daily_target_window_dollars'))}"
        f" | 1W {_fmt_dollar(metrics.get('weekend_avg_discrepancy_dollars'))}/{_fmt_dollar(metrics.get('weekend_target_window_dollars'))}"
    )
    return ("History Windows", f"4m {i_hit} | 1D {d_hit} | 1W {w_hit}", detail)


def _expression_paths(
    instrument_key: str,
    pred_state: dict,
    quality: dict,
    options_snapshot: dict | None = None,
    liquidity_snapshot: dict | None = None,
) -> dict:
    meta = _instrument_meta(instrument_key)
    instrument_label = str(meta.get("display_name") or instrument_key or "This stock").strip() or str(instrument_key or "This stock")
    intr = (pred_state or {}).get("intraday", {}) if isinstance(pred_state, dict) else {}
    direction = str(intr.get("direction") or "-").upper()
    quality_score = _safe_float((quality or {}).get("score")) or 0.0
    no_trade = bool((quality or {}).get("no_trade"))
    liq_score = _safe_float((liquidity_snapshot or {}).get("tradability_score"))
    stock_status = "Primary"
    stock_detail = f"Direct {instrument_label} stock is the primary trading expression for this workspace."
    if direction not in {"UP", "DOWN"}:
        stock_status = "Blocked"
        stock_detail = f"No finished directional bias is available for {instrument_label}."
    elif no_trade:
        stock_status = "Blocked"
        stock_detail = f"The advisory stack marked the {instrument_label} setup as no-trade."
    elif quality_score < 0.58:
        stock_status = "Selective"
        stock_detail = f"The {instrument_label} stock path is available only if the tape strengthens from here."
    elif liq_score is not None and liq_score < 0.45:
        stock_status = "Fragile"
        stock_detail = f"The {instrument_label} stock path exists, but execution quality is too weak for high conviction."

    opt_status = str((options_snapshot or {}).get("status") or "unavailable")
    dte = _safe_float((options_snapshot or {}).get("days_to_expiry"))
    expected_move = _safe_float((options_snapshot or {}).get("expected_move_dollars"))
    iv_vs_rv20 = _safe_float((options_snapshot or {}).get("iv_vs_rv20"))
    option_label = "Context only"
    option_detail = "Options remain advisory-only here; contract selection, spread gating, and Greeks filters are not automation-ready."
    if opt_status != "ok":
        option_label = "Unavailable"
        option_detail = "No current options surface is available, so options should not drive execution decisions."
    else:
        notes = []
        if dte is None or dte > 7:
            notes.append("nearest expiry is not tightly intraday-focused")
        if expected_move is not None:
            notes.append(f"ATM move {_fmt_dollar(expected_move)}")
        if iv_vs_rv20 is not None:
            notes.append(f"IV/RV20 {_fmt_ratio(iv_vs_rv20)}")
        notes.append("still missing contract-level spread and Greeks gating")
        option_detail = " | ".join(notes[:4])
    return {
        "stock": {"label": stock_status, "detail": stock_detail},
        "options": {"label": option_label, "detail": option_detail},
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
    symbol = str(instrument_key or "").upper()
    state = load_readiness_state(symbol) or {}
    deployment = summarize_deployment_readiness(symbol, state=state)
    onboarding = summarize_onboarding_state(symbol)
    evaluation = state.get("evaluation", {}) if isinstance(state, dict) else {}
    metrics = evaluation.get("metrics", {}) if isinstance(evaluation, dict) else {}
    onboarding_status = str(onboarding.get("status_label") or "Not started")
    onboarding_detail = str(onboarding.get("summary") or onboarding.get("current_stage_label") or "-")
    if str(onboarding.get("status") or "") != "not_started":
        rows.append(("Onboarding", onboarding_status, onboarding_detail))
    rows.append(
        (
            "Symbol Maturity",
            str(deployment.get("symbol_maturity_label") or deployment.get("stage_label") or "Unknown"),
            str(deployment.get("symbol_maturity_detail") or deployment.get("summary") or "-"),
        )
    )
    rows.append(
        (
            "Evidence State",
            str(deployment.get("evidence_state_label") or "-"),
            str(deployment.get("evidence_state_detail") or deployment.get("detail") or "-"),
        )
    )
    rows.append(
        (
            "Deployment",
            str(deployment.get("operator_state_label") or deployment.get("stage_label") or "-"),
            f"{str(deployment.get('stage_label') or '-')} | {str(deployment.get('summary') or deployment.get('detail') or '-')}",
        )
    )
    rows.append(
        (
            "Rollout",
            str(deployment.get("rollout_label") or "-"),
            str(deployment.get("rollout_detail") or "-"),
        )
    )
    primary_next = str(deployment.get("primary_next_requirement") or "")
    if primary_next:
        rows.append(
            (
                "Next Requirement",
                str(deployment.get("next_requirement_category") or "other").replace("_", " ").title(),
                primary_next,
            )
        )
    compact_history = _history_compact_row(metrics)
    if compact_history:
        rows.append(compact_history)
    if isinstance(state, dict) and not primary_next:
        rows.append(
            (
                "Readiness State",
                "Fresh" if bool(state.get("fresh")) else "Stale",
                f"Live-ready {'Yes' if bool(state.get('live_ready')) else 'No'} | Streak {state.get('pass_streak_days', 0)}",
            )
        )
    return rows[:7]


def build_advisory_snapshot(
    df: pd.DataFrame,
    instrument_key: str,
    predictions: dict | None = None,
    options_snapshot: dict | None = None,
    event_snapshot: dict | None = None,
    news_items: list[dict] | None = None,
    news_health: dict | None = None,
    macro_snapshot: dict | None = None,
    breadth_snapshot: dict | None = None,
    liquidity_snapshot: dict | None = None,
    catalyst_snapshot: dict | None = None,
    now_utc: pd.Timestamp | None = None,
    daily_long_history: pd.DataFrame | None = None,
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
            "regime_meta": {},
            "risk_meta": {},
            "decision_payload": {},
            "decision_rows": [],
            "news_meta": {},
            "macro_meta": {},
            "breadth_meta": {},
            "liquidity_meta": {},
            "catalyst_meta": {},
            "expression_meta": {},
        }

    instrument_key = str(instrument_key or DEFAULT_INSTRUMENT).upper()
    current_price = _safe_float(pd.to_numeric(df.get("close"), errors="coerce").iloc[-1]) if "close" in df.columns else None
    state = _current_session_state(df, instrument_key)
    daily_long = daily_long_history if isinstance(daily_long_history, pd.DataFrame) else _load_long_daily_history(instrument_key)
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
        try:
            news_items = load_recent_headlines(instrument_key, limit=8, as_of=now_utc) or []
        except TypeError:
            news_items = load_recent_headlines(instrument_key, limit=8) or []
    if not isinstance(news_health, dict):
        try:
            news_health = get_news_health(instrument_key, as_of=now_utc)
        except TypeError:
            news_health = get_news_health(instrument_key)
    news_runtime = build_news_runtime_summary(
        instrument_key,
        headlines=news_items,
        news_health=news_health,
        as_of=now_utc,
    )
    if not isinstance(macro_snapshot, dict):
        macro_snapshot = get_macro_event_snapshot()
    if not isinstance(breadth_snapshot, dict):
        breadth_snapshot = get_breadth_snapshot(instrument_key)
    if not isinstance(liquidity_snapshot, dict):
        liquidity_snapshot = get_liquidity_snapshot(df, instrument_key)
    if not isinstance(catalyst_snapshot, dict):
        catalyst_snapshot = get_catalyst_provider_snapshot(instrument_key)
    news_snapshot = _news_context_snapshot(
        df,
        instrument_key,
        news_items,
        current_price,
        pred_state,
        news_health=news_health,
        news_runtime=news_runtime,
        now_utc=now_utc,
    )
    quality = _signal_quality(
        current_price,
        pred_state,
        noise,
        spread_frac,
        volume_state,
        regime,
        event_snapshot,
        news_items,
        news_snapshot=news_snapshot,
        macro_snapshot=macro_snapshot,
        breadth_snapshot=breadth_snapshot,
        liquidity_snapshot=liquidity_snapshot,
        catalyst_snapshot=catalyst_snapshot,
        now_utc=now_utc,
    )
    signal_translation = _signal_translation(pred_state, quality, news_snapshot=news_snapshot)
    readiness_state = load_readiness_state(instrument_key) or {}
    deployment_meta = summarize_deployment_readiness(instrument_key, state=readiness_state, news_health=news_health)
    execution_posture = _deployment_execution_posture(signal_translation, deployment_meta, quality)
    expression_meta = _expression_paths(
        instrument_key,
        pred_state,
        quality,
        options_snapshot=options_snapshot,
        liquidity_snapshot=liquidity_snapshot,
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
        ("Trader Lens", str(signal_translation.get("headline") or "-"), str(signal_translation.get("detail") or "-")),
        ("Execution Posture", str(execution_posture.get("label") or "-"), str(execution_posture.get("detail") or "-")),
        ("Symbol Maturity", str(deployment_meta.get("symbol_maturity_label") or "-"), str(deployment_meta.get("symbol_maturity_detail") or "-")),
        ("Evidence State", str(deployment_meta.get("evidence_state_label") or "-"), str(deployment_meta.get("evidence_state_detail") or "-")),
        ("Deployment Stage", str(deployment_meta.get("stage_label") or "-"), f"{str(deployment_meta.get('operator_state_label') or '-')} | {str(deployment_meta.get('rollout_label') or '-')}"),
        ("Signal Quality", f"{quality.get('grade', '-')}", f"{int(round(100.0 * (_safe_float(quality.get('score')) or 0.0)))} / 100 | {quality.get('strategy', '-')}"),
        ("Agreement", _fmt_prob(quality.get("agreement")), f"Regime {quality.get('regime_label', '-')} | Breadth {_fmt_prob(quality.get('breadth_score'))} ({str(quality.get('breadth_source_kind') or '-').replace('_', ' ')})"),
        ("News Context", str(news_snapshot.get("alignment_label") or "-"), f"{str(news_snapshot.get('dominant_event') or '-')} | {str(news_snapshot.get('tape_label') or '-')}"),
        ("Stock Path", str((expression_meta.get("stock") or {}).get("label") or "-"), str((expression_meta.get("stock") or {}).get("detail") or "-")),
        ("Options Path", str((expression_meta.get("options") or {}).get("label") or "-"), str((expression_meta.get("options") or {}).get("detail") or "-")),
        (
            "Edge Translation",
            str(signal_translation.get("economics_state_label") or _fmt_bps_num(intr.get("edge_bps"))),
            f"Raw {_fmt_bps_num(intr.get('raw_edge_bps'))} -> finished {_fmt_bps_num(intr.get('edge_bps'))} | cost {_fmt_bps_num(intr.get('cost_bps_est'))} | unc {_fmt_bps_num(intr.get('uncertainty_penalty_bps'))}",
        ),
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
        ("News Check", str(news_snapshot.get("alignment_label") or "-"), str(news_snapshot.get("summary") or "-")),
        ("Catalyst Risk", catalyst_risk_label, catalyst_detail),
        ("No-trade", "Yes" if quality.get("no_trade") or bool(execution_posture.get("hard_block")) else "No", f"{str(signal_translation.get('kind_label') or '-')} | {str(signal_translation.get('primary_blocker') or '; '.join(reasons[:2]) or 'Edge is strong enough to monitor')}"),
    ]

    setup_rows = [
        ("Stock Signal", str(signal_translation.get("headline") or "-"), str(signal_translation.get("primary_blocker") or "-")),
        ("Operator State", str(deployment_meta.get("operator_state_label") or "-"), str(deployment_meta.get("operator_state_detail") or "-")),
        ("Bias", plan.get("action", "No-trade"), plan.get("detail", "-")),
        ("Entry", _fmt_dollar(plan.get("entry")), "Wait for the level, do not chase if price is extended"),
        ("Stop", _fmt_dollar(plan.get("stop")), "Invalidate quickly if the setup loses structure"),
        ("Target 1 / 2", f"{_fmt_dollar(plan.get('target1'))} / {_fmt_dollar(plan.get('target2'))}", f"Reward/Risk {_fmt_ratio(plan.get('rr'))}"),
        ("Size Hint", f"{(_safe_float(intr.get('confidence')) or 0.0):.0f}% conf", f"Model size {_safe_float(intr.get('position_size')) or 0.0:.2f} | {intr.get('sig_bucket', '-')}"),
    ]

    regime_meta = classify_market_regime(df, predictions=predictions, news_meta=news_snapshot)
    risk_meta = risk_summary_for_symbol(
        instrument_key,
        df=df,
        advisory={
            "quality_meta": quality,
            "plan_meta": plan,
            "pred_state": pred_state,
            "news_meta": news_snapshot,
        },
        predictions=predictions,
        current_price=current_price,
        atr14=atr14,
        require_portfolio=False,
    )
    quality_rows.append(
        (
            "Regime State",
            str(regime_meta.get("label") or "-").replace("_", " ").title(),
            f"{str(regime_meta.get('evidence_strength') or '-').title()} confidence | {str(regime_meta.get('reason') or '-')}",
        )
    )
    quality_rows.append(
        (
            "Risk Context",
            "Blocked" if bool(risk_meta.get("blocked")) else f"Qty {int(risk_meta.get('suggested_qty') or 0)}",
            str(risk_meta.get("detail") or risk_meta.get("primary_blocker") or "-"),
        )
    )

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

    action_label = plan.get("action", "No-trade")
    action_detail = plan.get("detail", "-")
    if bool(quality.get("no_trade")) or bool(execution_posture.get("hard_block")):
        action_label = str(execution_posture.get("action_label") or signal_translation.get("action_label") or "Stand Aside")
        action_detail = str(execution_posture.get("detail") or signal_translation.get("detail") or action_detail)
    elif str(signal_translation.get("kind") or "") == "actionable_signal":
        action_label = str(signal_translation.get("action_label") or action_label)
        action_detail = str(execution_posture.get("detail") or signal_translation.get("detail") or action_detail)
    else:
        action_label = str(signal_translation.get("action_label") or action_label)
        action_detail = str(execution_posture.get("detail") or signal_translation.get("detail") or action_detail)
    action_card = f"Action: {action_label}"
    if news_snapshot.get("alignment") == "conflict":
        action_detail = f"{action_detail} | Fresh news conflicts with the model bias."
    elif news_snapshot.get("tape_confirmation") == "underreacting":
        action_detail = f"{action_detail} | Watch for delayed tape response to fresh news."
    quality_card = f"Quality: {quality.get('grade', '-')}"
    quality_detail = (
        f"{str(signal_translation.get('headline') or quality.get('strategy', '-'))}"
        f" | {str(signal_translation.get('kind_label') or '-')}"
        f" | {str(deployment_meta.get('operator_state_label') or '-')}"
        f" | {str(deployment_meta.get('evidence_state_label') or '-')}"
        f" | No-trade {'Yes' if quality.get('no_trade') or bool(execution_posture.get('hard_block')) else 'No'}"
    )
    decision_payload = _advisory_decision_payload(
        signal_translation,
        regime_meta,
        risk_meta,
        deployment_meta,
        execution_posture,
        quality,
        plan,
        pred_state,
    )
    decision_rows = advisory_decision_rows(decision_payload)

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
        "regime_meta": regime_meta,
        "risk_meta": risk_meta,
        "news_meta": news_snapshot,
        "news_runtime": news_runtime,
        "macro_meta": macro_snapshot or {},
        "breadth_meta": breadth_snapshot or {},
        "liquidity_meta": liquidity_snapshot or {},
        "catalyst_meta": catalyst_snapshot or {},
        "deployment_meta": deployment_meta,
        "expression_meta": expression_meta,
        "execution_posture": execution_posture,
        "signal_translation": signal_translation,
        "decision_payload": decision_payload,
        "decision_rows": decision_rows,
        "predictions_summary": {
            "intraday_position_size": _safe_float(intr.get("position_size")),
            "intraday_direction": intr.get("direction"),
        },
    }
