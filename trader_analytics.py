from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from config import DATA_DIR, DEFAULT_INSTRUMENT, INSTRUMENTS
from market_context import load_context_daily_history
from news_pipeline import load_recent_headlines
from options_analytics import get_options_snapshot


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


def _fmt_pct(value, digits: int = 2) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{100.0 * val:+.{int(digits)}f}%"


def _fmt_bps(value) -> str:
    val = _safe_float(value)
    return "-" if val is None else f"{10000.0 * val:+.1f} bps"


def _age_text(ts_value) -> str:
    try:
        ts = pd.Timestamp(ts_value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        age_hours = max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0)
        if age_hours < 1.0:
            return f"{int(round(age_hours * 60.0))}m"
        return f"{age_hours:.1f}h"
    except Exception:
        return "-"


def _avg_prediction_price(block: dict, key: str = "pred_price") -> float | None:
    if not isinstance(block, dict):
        return None
    vals = []
    for model_key in ("base", "forest", "boost", "extra_trees"):
        item = block.get(model_key)
        if not isinstance(item, dict):
            continue
        val = _safe_float(item.get(key))
        if val is not None:
            vals.append(val)
    if not vals:
        return None
    return float(np.mean(vals))


def _prediction_alignment(price: float | None, preds: dict | None) -> tuple[str, str]:
    if price is None or not isinstance(preds, dict):
        return "Alignment: unavailable", "No prediction context"
    signals = []
    details = []
    intraday = preds.get("intraday")
    if isinstance(intraday, dict):
        cons = intraday.get("consensus", {})
        direction = str(cons.get("direction") or "-")
        conf = _safe_float(cons.get("confidence"))
        if direction == "UP":
            signals.append(1.0)
        elif direction == "DOWN":
            signals.append(-1.0)
        if direction != "-":
            details.append(f"4m {direction} {conf:.0f}%" if conf is not None else f"4m {direction}")
    daily = preds.get("daily")
    if isinstance(daily, dict):
        hi = _avg_prediction_price(daily.get("high"))
        lo = _avg_prediction_price(daily.get("low"))
        if hi is not None and lo is not None:
            mid = 0.5 * (hi + lo)
            if mid > price:
                signals.append(0.75)
            elif mid < price:
                signals.append(-0.75)
            details.append(f"1D mid {_fmt_dollar(mid)}")
    week = preds.get("weekend")
    if isinstance(week, dict):
        close_avg = _avg_prediction_price(week.get("week"))
        if close_avg is not None:
            if close_avg > price:
                signals.append(1.0)
            elif close_avg < price:
                signals.append(-1.0)
            details.append(f"1W {_fmt_dollar(close_avg)}")
    if not signals:
        return "Alignment: unavailable", "No aligned horizon signals"
    score = float(np.sum(signals))
    if score >= 1.5:
        head = "Alignment: bullish"
    elif score <= -1.5:
        head = "Alignment: bearish"
    else:
        head = "Alignment: mixed"
    return head, " | ".join(details[:3])


def _market_context_rows(instrument_key: str) -> list[tuple[str, str, str]]:
    meta = _instrument_meta(instrument_key)
    mapping = meta.get("context_daily_symbols", {}) if isinstance(meta, dict) else {}
    role_labels = {
        "market": "Market",
        "sector": "Sector",
        "vol": "Vol",
        "lead_fut": "Lead Fut",
        "rates": "Rates",
    }
    rows = []
    for role in ("market", "sector", "vol", "lead_fut", "rates"):
        if role not in mapping:
            continue
        hist = load_context_daily_history(str(instrument_key).upper(), role, allow_remote=False)
        if hist is None or hist.empty or "close" not in hist.columns:
            continue
        close = pd.to_numeric(hist["close"], errors="coerce")
        ret_1 = _safe_float(close.pct_change().iloc[-1]) if len(close) >= 2 else None
        ret_5 = _safe_float(close.pct_change(5).iloc[-1]) if len(close) >= 6 else None
        last_px = _safe_float(close.iloc[-1]) if len(close) else None
        label = f"{role_labels.get(role, role.title())} {mapping.get(role)}"
        value = _fmt_pct(ret_1)
        detail = f"5D {_fmt_pct(ret_5)} | {_fmt_dollar(last_px)}"
        rows.append((label, value, detail))
    return rows


def _levels_rows(df: pd.DataFrame, instrument_key: str) -> list[tuple[str, str, str]]:
    if df is None or df.empty:
        return []
    meta = _instrument_meta(instrument_key)
    tz = meta.get("tz", "America/New_York")
    session_open = int(meta.get("session_open_minute", 9 * 60 + 30))
    session_close = int(meta.get("session_close_minute", 16 * 60))
    idx_local = _tz_index(df.index, tz)
    session_dates = _session_dates(idx_local, session_open, session_close)
    frame = df.copy()
    frame["session_date"] = session_dates
    current_price = _safe_float(frame["close"].iloc[-1])
    rows: list[tuple[str, str, str]] = []

    current_session = frame[frame["session_date"] == frame["session_date"].iloc[-1]].copy()
    prior_sessions = frame["session_date"].drop_duplicates()
    prev_session = None
    if len(prior_sessions) >= 2:
        prev_session = prior_sessions.iloc[-2]
    if current_session is not None and not current_session.empty and current_price is not None:
        try:
            pv = pd.to_numeric(current_session["close"], errors="coerce") * pd.to_numeric(current_session["volume"], errors="coerce").fillna(0.0)
            vol = pd.to_numeric(current_session["volume"], errors="coerce").fillna(0.0)
            vwap = _safe_float(pv.sum() / (vol.sum() + 1e-12))
        except Exception:
            vwap = None
        if vwap is not None:
            rows.append(("VWAP", _fmt_dollar(vwap), f"Dist {_fmt_pct((current_price / vwap) - 1.0)}"))
        try:
            sess_hi = _safe_float(pd.to_numeric(current_session["high"], errors="coerce").max())
            sess_lo = _safe_float(pd.to_numeric(current_session["low"], errors="coerce").min())
            rows.append(("Today H/L", f"{_fmt_dollar(sess_hi)} / {_fmt_dollar(sess_lo)}", f"Range {_fmt_pct((sess_hi / sess_lo) - 1.0) if sess_hi and sess_lo else None}"))
        except Exception:
            pass
        try:
            minutes = idx_local[-len(current_session):].hour * 60 + idx_local[-len(current_session):].minute
            or_end = session_open + 30
            or_block = current_session[(minutes >= session_open) & (minutes < or_end)].copy()
            if not or_block.empty:
                or_hi = _safe_float(pd.to_numeric(or_block["high"], errors="coerce").max())
                or_lo = _safe_float(pd.to_numeric(or_block["low"], errors="coerce").min())
                state = "Inside"
                if current_price is not None and or_hi is not None and current_price > or_hi:
                    state = "Above OR high"
                elif current_price is not None and or_lo is not None and current_price < or_lo:
                    state = "Below OR low"
                rows.append(("Opening Range", f"{_fmt_dollar(or_hi)} / {_fmt_dollar(or_lo)}", state))
        except Exception:
            pass
        try:
            vol_today = _safe_float(pd.to_numeric(current_session["volume"], errors="coerce").sum())
            sess_vol = frame.groupby("session_date")["volume"].sum().tail(21)
            avg_vol = _safe_float(sess_vol.iloc[:-1].mean()) if len(sess_vol) >= 2 else None
            pace = float(vol_today / avg_vol) if vol_today is not None and avg_vol not in (None, 0.0) else None
            rows.append(("Volume Pace", f"{pace:.2f}x" if pace is not None else "-", f"Today {vol_today:,.0f}" if vol_today is not None else "-"))
        except Exception:
            pass

    if prev_session is not None:
        prev = frame[frame["session_date"] == prev_session].copy()
        if not prev.empty:
            prev_hi = _safe_float(pd.to_numeric(prev["high"], errors="coerce").max())
            prev_lo = _safe_float(pd.to_numeric(prev["low"], errors="coerce").min())
            state = "Inside"
            if current_price is not None and prev_hi is not None and current_price > prev_hi:
                state = "Above prior high"
            elif current_price is not None and prev_lo is not None and current_price < prev_lo:
                state = "Below prior low"
            rows.append(("Prior Session", f"{_fmt_dollar(prev_hi)} / {_fmt_dollar(prev_lo)}", state))

    daily_long = _load_long_daily_history(instrument_key)
    if daily_long is not None and not daily_long.empty and "close" in daily_long.columns:
        close = pd.to_numeric(daily_long["close"], errors="coerce")
        high = pd.to_numeric(daily_long.get("high"), errors="coerce")
        low = pd.to_numeric(daily_long.get("low"), errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = _safe_float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) else None
        hi20 = _safe_float(high.rolling(20, min_periods=5).max().iloc[-1]) if len(high) else None
        lo20 = _safe_float(low.rolling(20, min_periods=5).min().iloc[-1]) if len(low) else None
        rows.append(
            (
                "ATR(14)",
                _fmt_dollar(atr14),
                f"1D ±{_fmt_dollar(atr14)} | 1W ±{_fmt_dollar((atr14 * np.sqrt(5.0)) if atr14 is not None else None)}",
            )
        )
        if hi20 is not None or lo20 is not None:
            dist = None
            if current_price is not None and hi20 is not None:
                dist = (current_price / hi20) - 1.0
            rows.append(("20D Breakout", f"{_fmt_dollar(hi20)} / {_fmt_dollar(lo20)}", f"Vs 20D high {_fmt_pct(dist)}" if dist is not None else "-"))
    return rows[:8]


def _options_rows(symbol: str, spot_price: float | None) -> tuple[dict, list[tuple[str, str, str]], str]:
    snap = get_options_snapshot(symbol, spot_price=spot_price)
    rows: list[tuple[str, str, str]] = []
    card = "Options: unavailable"
    if not isinstance(snap, dict):
        return {"status": "unavailable"}, rows, card
    if snap.get("status") == "ok":
        exp = str(snap.get("expiry") or "-")
        dte = snap.get("days_to_expiry")
        exp_move = snap.get("expected_move_dollars")
        exp_pct = snap.get("expected_move_pct")
        pcr_oi = snap.get("put_call_oi_ratio")
        card = f"Options {exp} | ±{_fmt_dollar(exp_move)} ({_fmt_pct(exp_pct)})" if exp_move is not None else f"Options {exp}"
        rows.append(("Nearest Expiry", f"{exp} ({dte}d)" if dte is not None else exp, str(snap.get("source", "-"))))
        rows.append(("ATM Expected Move", f"±{_fmt_dollar(exp_move)}", _fmt_pct(exp_pct)))
        rows.append(("Put/Call OI", f"{pcr_oi:.2f}" if _safe_float(pcr_oi) is not None else "-", f"Vol {snap.get('put_call_volume_ratio'):.2f}" if _safe_float(snap.get("put_call_volume_ratio")) is not None else "-"))
        rows.append(("IV Skew", _fmt_pct(snap.get("iv_skew_put_minus_call")), "Put IV - Call IV"))
        rows.append(("Call / Put Wall", f"{_fmt_dollar(snap.get('call_wall_oi'))} / {_fmt_dollar(snap.get('put_wall_oi'))}", f"Max pain {_fmt_dollar(snap.get('max_pain'))}"))
    else:
        reason = str(snap.get("reason") or snap.get("status") or "unavailable")
        if "Could not resolve host" in reason or "curl:" in reason:
            reason = "Remote feed unavailable"
        elif len(reason) > 48:
            reason = reason[:45] + "..."
        rows.append(("Options", "-", reason))
    return snap, rows, card


def _news_rows(symbol: str) -> tuple[list[tuple[str, str, str]], str]:
    headlines = load_recent_headlines(symbol, limit=6)
    rows = []
    if not headlines:
        return rows, "News: unavailable"
    scores = []
    fresh_hours = []
    for item in headlines:
        ts = item.get("published_at")
        age = _age_text(ts)
        label = str(item.get("label") or "neutral").lower()
        head = str(item.get("headline") or "").strip()
        if len(head) > 88:
            head = f"{head[:85]}..."
        rows.append((age, label[:8] or "-", head))
        sc = _safe_float(item.get("score"))
        if sc is not None:
            scores.append(sc)
        try:
            tsv = pd.Timestamp(ts)
            if tsv.tzinfo is None:
                tsv = tsv.tz_localize("UTC")
            fresh_hours.append(max(0.0, (pd.Timestamp.now(tz="UTC") - tsv).total_seconds() / 3600.0))
        except Exception:
            pass
    sent_mean = float(np.mean(scores)) if scores else None
    fresh = min(fresh_hours) if fresh_hours else None
    meta = f"News {len(rows)} items | Sent {_fmt_pct(sent_mean)} | Fresh {fresh:.1f}h" if fresh is not None else f"News {len(rows)} items"
    return rows, meta


def build_trader_dashboard(df: pd.DataFrame, instrument_key: str, predictions: dict | None = None) -> dict:
    if df is None or df.empty:
        return {
            "setup_card": "Alignment: unavailable",
            "setup_detail": "No data",
            "vol_card": "Volatility: unavailable",
            "vol_detail": "No data",
            "options_card": "Options: unavailable",
            "news_card": "News: unavailable",
            "market_rows": [],
            "levels_rows": [],
            "options_rows": [],
            "news_rows": [],
        }
    price = _safe_float(df["close"].iloc[-1]) if "close" in df.columns else None
    align_head, align_detail = _prediction_alignment(price, predictions or {})
    levels = _levels_rows(df, instrument_key)
    market = _market_context_rows(instrument_key)
    opt_snap, opt_rows, opt_card = _options_rows(str(instrument_key).upper(), spot_price=price)
    news_rows, news_card = _news_rows(str(instrument_key).upper())

    atr_row = next((r for r in levels if r[0] == "ATR(14)"), None)
    vol_card = f"Volatility: {atr_row[1]}" if atr_row else "Volatility: unavailable"
    vol_detail = atr_row[2] if atr_row else "ATR unavailable"
    if market:
        first_market = market[0]
        vol_detail = f"{vol_detail} | {first_market[0]} {first_market[1]}"

    return {
        "setup_card": align_head,
        "setup_detail": align_detail,
        "vol_card": vol_card,
        "vol_detail": vol_detail,
        "options_card": opt_card,
        "news_card": news_card,
        "market_rows": market,
        "levels_rows": levels,
        "options_rows": opt_rows,
        "news_rows": news_rows,
        "options_snapshot": opt_snap,
    }
