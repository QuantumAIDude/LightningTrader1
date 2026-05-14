from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd

from advisory_engine import build_advisory_snapshot
from alert_engine import get_alert_snapshot
from breadth_analytics import get_breadth_snapshot
from catalyst_providers import get_catalyst_provider_snapshot
from config import DATA_DIR, DEFAULT_INSTRUMENT, INSTRUMENTS
from execution_engine import get_execution_dashboard_snapshot
from event_analytics import get_event_snapshot
from instrument_profile import build_instrument_meta
from liquidity_analytics import get_liquidity_snapshot
from market_context import load_context_daily_history
from macro_events import get_macro_event_snapshot
from news_pipeline import build_news_runtime_summary, get_news_health, load_recent_headlines, news_archive_coverage
from onboarding_jobs import summarize_onboarding_state
from options_analytics import get_options_snapshot
from opportunity_scanner import get_opportunity_scan
from performance_analytics import get_performance_snapshot
from readiness_tracker import load_readiness_state, summarize_deployment_readiness
from replay_engine import get_replay_dashboard_snapshot
from symbol_lifecycle import build_symbol_lifecycle_snapshot, lifecycle_rows
from trader_profiles import get_trader_profile


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


def _clean_reason_text(value, fallback: str = "Unavailable") -> str:
    text = str(value or fallback).strip() or fallback
    if "Could not resolve host" in text or "curl:" in text or "DNSError" in text:
        return "Remote feed unavailable"
    if text.lower() in {"none", "nan"}:
        return fallback
    if len(text) > 110:
        return text[:107] + "..."
    return text


def _info_rows(rows: list[tuple[str, str, str]] | None, label: str, reason: str) -> list[tuple[str, str, str]]:
    if rows:
        return rows
    return [(label, "-", _clean_reason_text(reason))]


def _news_info_rows(rows: list[tuple[str, str, str]] | None, reason: str) -> list[tuple[str, str, str]]:
    if rows:
        return rows
    return [("-", "info", _clean_reason_text(reason))]


def _news_event_tag(item: dict) -> str:
    tone = str(item.get("label") or item.get("signal_direction") or "neutral").lower()
    event = str(item.get("event_label") or item.get("event_type") or "General").strip()
    scope = str(item.get("relevance_scope_label") or item.get("relevance_scope") or "Relevant").strip()
    impact = _safe_float(item.get("impact_score")) or 0.0
    impact_label = "H" if impact >= 0.70 else "M" if impact >= 0.40 else "L"
    event_short = event[:10]
    scope_short = scope.split("/", 1)[0][:3]
    return f"{tone[:3]}/{scope_short}/{event_short}/{impact_label}"


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


def _signal_rows(df: pd.DataFrame, instrument_key: str, options_snapshot: dict | None = None) -> list[tuple[str, str, str]]:
    rows = []
    daily_long = _load_long_daily_history(instrument_key)
    if daily_long is None or daily_long.empty or "close" not in daily_long.columns:
        return rows
    close = pd.to_numeric(daily_long["close"], errors="coerce").dropna()
    if len(close) < 5:
        return rows
    ret = close.pct_change().dropna()
    mapping = _instrument_meta(instrument_key).get("context_daily_symbols", {})
    market_hist = load_context_daily_history(str(instrument_key).upper(), "market", allow_remote=False)
    if market_hist is not None and not market_hist.empty and "close" in market_hist.columns:
        m_close = pd.to_numeric(market_hist["close"], errors="coerce").dropna()
        joined = pd.concat([ret.rename("asset"), m_close.pct_change().rename("market")], axis=1).dropna()
        if len(joined) >= 5:
            beta20 = None
            tail = joined.tail(20)
            if len(tail) >= 8 and float(tail["market"].var()) > 1e-12:
                beta20 = _safe_float(tail["asset"].cov(tail["market"]) / tail["market"].var())
            rs20 = _safe_float((1.0 + tail["asset"]).prod() - (1.0 + tail["market"]).prod())
            rows.append((f"RS vs {mapping.get('market', 'Mkt')}", _fmt_pct(rs20), f"Beta20 {beta20:.2f}" if beta20 is not None else "Beta20 -"))
    sector_hist = load_context_daily_history(str(instrument_key).upper(), "sector", allow_remote=False)
    if sector_hist is not None and not sector_hist.empty and "close" in sector_hist.columns:
        s_close = pd.to_numeric(sector_hist["close"], errors="coerce").dropna()
        joined = pd.concat([ret.rename("asset"), s_close.pct_change().rename("sector")], axis=1).dropna()
        if len(joined) >= 5:
            tail = joined.tail(20)
            rs20 = _safe_float((1.0 + tail["asset"]).prod() - (1.0 + tail["sector"]).prod())
            rows.append((f"RS vs {mapping.get('sector', 'Sector')}", _fmt_pct(rs20), f"20D {_fmt_pct(((1.0 + tail['asset']).prod() - 1.0))}"))
    high = pd.to_numeric(daily_long["high"], errors="coerce") if "high" in daily_long.columns else None
    low = pd.to_numeric(daily_long["low"], errors="coerce") if "low" in daily_long.columns else None
    atr14 = None
    if isinstance(high, pd.Series) and isinstance(low, pd.Series):
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr14 = _safe_float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) else None
    exp_move = _safe_float((options_snapshot or {}).get("expected_move_dollars"))
    if atr14 is not None or exp_move is not None:
        ratio = float(exp_move / atr14) if exp_move is not None and atr14 not in (None, 0.0) else None
        rows.append(("Move Budget", f"ATR {_fmt_dollar(atr14)}", f"Opt/ATR {ratio:.2f}x" if ratio is not None else f"Opt {_fmt_dollar(exp_move)}"))
    ret20 = _safe_float((1.0 + ret.tail(20)).prod() - 1.0) if len(ret) >= 2 else None
    rv20 = _safe_float(ret.tail(20).std() * np.sqrt(252.0)) if len(ret) >= 5 else None
    rows.append(("Trend / RV", _fmt_pct(ret20), f"RV20 {_fmt_pct(rv20)}"))
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta.clip(upper=0.0))
    rs = gain.rolling(14, min_periods=5).mean() / loss.rolling(14, min_periods=5).mean().replace(0.0, np.nan)
    rsi14 = _safe_float(100.0 - (100.0 / (1.0 + rs.iloc[-1]))) if len(rs) else None
    state = "Neutral"
    if rsi14 is not None and rsi14 >= 70.0:
        state = "Overbought"
    elif rsi14 is not None and rsi14 <= 30.0:
        state = "Oversold"
    rows.append(("RSI(14)", f"{rsi14:.1f}" if rsi14 is not None else "-", state))
    return rows[:6]


def _stack_rows(df: pd.DataFrame, instrument_key: str, options_snapshot: dict | None = None, event_snapshot: dict | None = None) -> list[tuple[str, str, str]]:
    symbol = str(instrument_key or "").upper()
    try:
        news_health = get_news_health(symbol)
    except Exception:
        news_health = {}
    onboarding = summarize_onboarding_state(symbol)
    readiness_state = load_readiness_state(symbol) or {}
    deployment = summarize_deployment_readiness(symbol, state=readiness_state, news_health=news_health)
    lifecycle = build_symbol_lifecycle_snapshot(
        symbol,
        runtime_bundle={"df": df, "news_health": news_health},
        onboarding_summary=onboarding,
        deployment_summary=deployment,
        news_health=news_health,
    )
    rows = lifecycle_rows(lifecycle)
    try:
        coverage = news_archive_coverage(symbol)
    except Exception:
        coverage = {}
    if isinstance(coverage, dict):
        cov_ratio = _safe_float(coverage.get("coverage_ratio"))
        active = coverage.get("active_days")
        relevant = coverage.get("relevant_rows")
        rows.insert(
            1,
            ("News Depth", f"{active or 0} days", f"Cov {_fmt_pct(cov_ratio)} | Rows {relevant or 0}"),
        )
    return rows[:6]


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
    current_week = frame[frame["session_date"].dt.strftime("%G-W%V") == frame["session_date"].iloc[-1].strftime("%G-W%V")].copy()
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
            premarket = current_session[(idx_local[-len(current_session):].hour * 60 + idx_local[-len(current_session):].minute) < session_open].copy()
            if not premarket.empty:
                pm_hi = _safe_float(pd.to_numeric(premarket["high"], errors="coerce").max())
                pm_lo = _safe_float(pd.to_numeric(premarket["low"], errors="coerce").min())
                rows.append(("Premarket H/L", f"{_fmt_dollar(pm_hi)} / {_fmt_dollar(pm_lo)}", "Pre-open reference"))
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
        if current_week is not None and not current_week.empty:
            try:
                week_hi = _safe_float(pd.to_numeric(current_week["high"], errors="coerce").max())
                week_lo = _safe_float(pd.to_numeric(current_week["low"], errors="coerce").min())
                rows.append(("Week So Far", f"{_fmt_dollar(week_hi)} / {_fmt_dollar(week_lo)}", "Current trading week range"))
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

    try:
        if frame is not None and not frame.empty and "close" in frame.columns and "volume" in frame.columns:
            recent = frame.tail(min(len(frame), 3000)).copy()
            tp = (
                pd.to_numeric(recent.get("high"), errors="coerce").fillna(pd.to_numeric(recent.get("close"), errors="coerce"))
                + pd.to_numeric(recent.get("low"), errors="coerce").fillna(pd.to_numeric(recent.get("close"), errors="coerce"))
                + pd.to_numeric(recent.get("close"), errors="coerce")
            ) / 3.0
            vol = pd.to_numeric(recent.get("volume"), errors="coerce").fillna(0.0)
            if tp.notna().sum() >= 20 and float(vol.sum()) > 0:
                bins = np.linspace(float(tp.min()), float(tp.max()), 16)
                if len(np.unique(bins)) >= 4:
                    bucket = pd.cut(tp, bins=bins, include_lowest=True)
                    prof = vol.groupby(bucket, observed=False).sum()
                    if not prof.empty:
                        hvn_bin = prof.idxmax()
                        hvn = _safe_float((hvn_bin.left + hvn_bin.right) * 0.5)
                        rows.append(("Comp HVN", _fmt_dollar(hvn), "Approx volume node from loaded intraday bars"))
                        total_vol = float(prof.sum())
                        order = prof.sort_values(ascending=False)
                        keep = order.cumsum() <= (0.70 * total_vol)
                        kept = order[keep]
                        if kept.empty:
                            kept = order.head(1)
                        va_low = _safe_float(min(iv.left for iv in kept.index))
                        va_high = _safe_float(max(iv.right for iv in kept.index))
                        rows.append(("Value Area", f"{_fmt_dollar(va_low)} / {_fmt_dollar(va_high)}", "Approx 70% volume area"))
    except Exception:
        pass

    daily_long = _load_long_daily_history(instrument_key)
    if daily_long is not None and not daily_long.empty and "close" in daily_long.columns:
        close = pd.to_numeric(daily_long["close"], errors="coerce")
        high = pd.to_numeric(daily_long["high"], errors="coerce") if "high" in daily_long.columns else None
        low = pd.to_numeric(daily_long["low"], errors="coerce") if "low" in daily_long.columns else None
        atr14 = None
        hi20 = None
        lo20 = None
        if isinstance(high, pd.Series) and isinstance(low, pd.Series):
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
        sma20 = _safe_float(close.rolling(20, min_periods=5).mean().iloc[-1]) if len(close) else None
        sma50 = _safe_float(close.rolling(50, min_periods=10).mean().iloc[-1]) if len(close) else None
        if atr14 is not None:
            rows.append(
                (
                    "ATR(14)",
                    _fmt_dollar(atr14),
                    f"1D ±{_fmt_dollar(atr14)} | 1W ±{_fmt_dollar((atr14 * np.sqrt(5.0)) if atr14 is not None else None)}",
                )
            )
        if sma20 is not None or sma50 is not None:
            rows.append(("20D / 50D SMA", f"{_fmt_dollar(sma20)} / {_fmt_dollar(sma50)}", "Trend anchors"))
        if hi20 is not None or lo20 is not None:
            dist = None
            if current_price is not None and hi20 is not None:
                dist = (current_price / hi20) - 1.0
            rows.append(("20D Breakout", f"{_fmt_dollar(hi20)} / {_fmt_dollar(lo20)}", f"Vs 20D high {_fmt_pct(dist)}" if dist is not None else "-"))
        try:
            recent = daily_long.tail(20).copy()
            if not recent.empty:
                px = pd.to_numeric(recent["close"], errors="coerce").ffill()
                vol20 = pd.to_numeric(recent["volume"], errors="coerce").fillna(0.0)
                if px.notna().sum() >= 5 and vol20.sum() > 0:
                    bins = np.linspace(float(px.min()), float(px.max()), 9)
                    if len(np.unique(bins)) >= 3:
                        bucket = pd.cut(px, bins=bins, include_lowest=True)
                        grouped = vol20.groupby(bucket, observed=False).sum()
                        if not grouped.empty:
                            top_bin = grouped.idxmax()
                            hvn = _safe_float((top_bin.left + top_bin.right) * 0.5)
                            rows.append(("HVN 20D", _fmt_dollar(hvn), "Highest recent volume node"))
        except Exception:
            pass
    return rows[:14]


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
        rows.append(("Nearest Expiry", f"{exp} ({dte}d)" if dte is not None else exp, f"{str(snap.get('source_kind') or snap.get('source') or '-')}"))
        rows.append(("ATM Expected Move", f"±{_fmt_dollar(exp_move)}", _fmt_pct(exp_pct)))
        rows.append(("Put/Call OI", f"{pcr_oi:.2f}" if _safe_float(pcr_oi) is not None else "-", f"Vol {snap.get('put_call_volume_ratio'):.2f}" if _safe_float(snap.get("put_call_volume_ratio")) is not None else "-"))
        rows.append(("IV Skew", _fmt_pct(snap.get("iv_skew_put_minus_call")), "Put IV - Call IV"))
        rows.append(("IV Rank / Pctl", f"{_fmt_pct(snap.get('iv_rank'))} / {_fmt_pct(snap.get('iv_percentile'))}", f"IV/RV20 {(_safe_float(snap.get('iv_vs_rv20')) or 0.0):.2f}x | hist {int(_safe_float(snap.get('history_points')) or 0)}"))
        rows.append(("Call / Put Wall", f"{_fmt_dollar(snap.get('call_wall_oi'))} / {_fmt_dollar(snap.get('put_wall_oi'))}", f"Max pain {_fmt_dollar(snap.get('max_pain'))}"))
        rows.append(("Autonomy Use", "Context only", "Use the options panel for context; stock is the only autonomy-ready intraday path right now"))
    else:
        reason = str(snap.get("reason") or snap.get("status") or "unavailable")
        if "Could not resolve host" in reason or "curl:" in reason:
            reason = "Remote feed unavailable"
        elif len(reason) > 48:
            reason = reason[:45] + "..."
        rows.append(("Options", "-", f"{reason} | {str(snap.get('source_kind') or snap.get('source') or 'unavailable')}"))
        rows.append(("Autonomy Use", "Blocked", "Without a valid options surface, options should not influence automated expression choice"))
    return snap, rows, card


def _news_rows(
    symbol: str,
    headlines: list[dict] | None = None,
    news_health: dict | None = None,
    news_meta: dict | None = None,
    news_runtime: dict | None = None,
) -> tuple[list[tuple[str, str, str]], str, str]:
    headlines = headlines if headlines is not None else load_recent_headlines(symbol, limit=6)
    rows = []
    health = news_health if isinstance(news_health, dict) else get_news_health(symbol)
    meta = news_meta if isinstance(news_meta, dict) else {}
    runtime = news_runtime if isinstance(news_runtime, dict) else build_news_runtime_summary(symbol, headlines=headlines, news_health=health)
    layer_a = dict(runtime.get("layer_a") or {})
    layer_b = dict(runtime.get("layer_b") or {})
    profile = dict(runtime.get("profile") or {})
    headline_state = str((health or {}).get("headline_state") or "missing")
    history_state = str((health or {}).get("history_state") or "missing")
    if not headlines:
        card = "News: unavailable" if headline_state in {"missing", "failed"} else f"News: {headline_state.replace('_', ' ')}"
        detail = str(meta.get("summary") or layer_a.get("summary") or f"{headline_state.replace('_', ' ').title()} | hist {history_state.replace('_', ' ')}")
        return rows, card, detail
    scores = []
    fresh_hours = []
    political = []
    impact_scores = []
    high_impact = 0
    for item in headlines:
        ts = item.get("published_at")
        age = _age_text(ts)
        label = _news_event_tag(item)
        head = str(item.get("headline") or "").strip()
        if len(head) > 88:
            head = f"{head[:85]}..."
        event_label = str(item.get("event_label") or item.get("event_type") or "").strip()
        horizon = str(item.get("impact_horizon") or "").strip()
        scope = str(item.get("relevance_scope_label") or item.get("relevance_scope") or "").strip()
        note = str(item.get("signal_note") or "").strip()
        tail = " | ".join(part for part in (scope, event_label, horizon, note[:40] if note else "") if part)
        headline_text = head if not tail else f"{head} [{tail}]"
        rows.append((age, label[:16] or "-", headline_text))
        sc = _safe_float(item.get("score"))
        if sc is not None:
            scores.append(sc)
        impact = _safe_float(item.get("impact_score"))
        if impact is not None:
            impact_scores.append(impact)
            if impact >= 0.45:
                high_impact += 1
        political.append(1.0 if (_safe_float(item.get("political_relevance")) or 0.0) >= 0.25 else 0.0)
        try:
            tsv = pd.Timestamp(ts)
            if tsv.tzinfo is None:
                tsv = tsv.tz_localize("UTC")
            fresh_hours.append(max(0.0, (pd.Timestamp.now(tz="UTC") - tsv).total_seconds() / 3600.0))
        except Exception:
            pass
    sent_mean = float(np.mean(scores)) if scores else None
    fresh = min(fresh_hours) if fresh_hours else None
    pol_share = float(np.mean(political)) if political else None
    impact_mean = float(np.mean(impact_scores)) if impact_scores else None
    headline_label = "News"
    if headline_state not in {"ok", "thin"}:
        headline_label = f"News {headline_state.replace('_', ' ').title()}"
    elif high_impact > 0:
        headline_label = f"News {high_impact} hi-impact"
    if fresh is not None:
        card = f"{headline_label} | Fresh {fresh:.1f}h"
    else:
        card = f"{headline_label} | {len(rows)} items"
    detail_bits = []
    if sent_mean is not None:
        detail_bits.append(f"sent {_fmt_pct(sent_mean)}")
    if impact_mean is not None:
        detail_bits.append(f"impact {_fmt_pct(impact_mean)}")
    detail_bits.append(f"pol {_fmt_pct(pol_share)}")
    if layer_a.get("summary"):
        detail_bits.append(str(layer_a.get("summary")))
    if str(layer_b.get("status") or "disabled") != "disabled":
        detail_bits.append(f"layer B {str(layer_b.get('status') or '-')}")
    detail_bits.append(f"{str(profile.get('instrument_kind') or 'equity').title()} / {str(profile.get('sector') or 'Broad Market')}")
    if history_state != "ready":
        detail_bits.append(f"hist {history_state.replace('_', ' ')}")
    summary = str(meta.get("summary") or "")
    if summary:
        detail_bits.append(summary)
    return rows, card, " | ".join(bit for bit in detail_bits if bit)


def build_trader_dashboard(df: pd.DataFrame, instrument_key: str, predictions: dict | None = None, profile_name: str | None = None) -> dict:
    if df is None or df.empty:
        lifecycle = build_symbol_lifecycle_snapshot(str(instrument_key or "").upper())
        stack_rows = lifecycle_rows(lifecycle)
        quality_rows = [
            ("Operator State", lifecycle.get("operator_state_label") or "-", lifecycle.get("operator_state_detail") or "-"),
            ("Evidence State", lifecycle.get("evidence_state_label") or "-", lifecycle.get("evidence_state_detail") or "-"),
            ("Next Requirement", lifecycle.get("next_requirement_category") or "-", lifecycle.get("primary_next_requirement") or lifecycle.get("onboarding_summary") or "-"),
        ]
        return {
            "setup_card": f"Lifecycle: {lifecycle.get('lifecycle_label') or 'Unavailable'}",
            "setup_detail": lifecycle.get("lifecycle_detail") or "No local data",
            "vol_card": "Volatility: unavailable",
            "vol_detail": lifecycle.get("data_state_detail") or "No data",
            "options_card": "Options: unavailable",
            "options_detail": lifecycle.get("model_state_detail") or "No data",
            "news_card": f"News: {lifecycle.get('news_state_label') or 'unavailable'}",
            "news_detail": lifecycle.get("news_state_detail") or "No data",
            "market_rows": [("Status", "-", lifecycle.get("data_state_detail") or "No local data loaded")],
            "breadth_rows": [("Status", "-", lifecycle.get("evidence_state_detail") or "No breadth context available")],
            "macro_rows": [("Status", "-", lifecycle.get("operator_state_detail") or "No macro context available")],
            "liquidity_rows": [("Status", "-", lifecycle.get("data_state_detail") or "No liquidity context available")],
            "catalyst_rows": [("Status", "-", lifecycle.get("news_state_detail") or "No catalyst context available")],
            "signal_rows": [("Ticker Lifecycle", lifecycle.get("lifecycle_label") or "-", lifecycle.get("workspace_summary") or "-")],
            "stack_rows": stack_rows,
            "levels_rows": [],
            "options_rows": [],
            "news_rows": [("-", "info", lifecycle.get("news_state_detail") or "No news context staged")],
            "action_card": f"Action: {lifecycle.get('operator_state_label') or 'Unavailable'}",
            "action_detail": lifecycle.get("operator_state_detail") or "No data",
            "quality_card": f"Quality: {lifecycle.get('evidence_state_label') or 'Unavailable'}",
            "quality_detail": lifecycle.get("evidence_state_detail") or "No data",
            "probability_rows": [],
            "quality_rows": quality_rows,
            "setup_rows": [("Next Requirement", lifecycle.get("next_requirement_category") or "-", lifecycle.get("primary_next_requirement") or lifecycle.get("onboarding_summary") or "-")],
            "decision_rows": [],
            "scenario_rows": [],
            "invalidation_rows": [],
            "reliability_rows": [("Deployment", lifecycle.get("stage_label") or "-", f"{lifecycle.get('rollout_label') or '-'} | {lifecycle.get('symbol_maturity_label') or '-'}")],
            "execution_rows": [],
            "portfolio_rows": [],
            "performance_rows": [],
            "replay_rows": [],
            "alerts_rows": [],
            "opportunity_rows": [],
            "profile_card": "Profile: unavailable",
            "profile_detail": lifecycle.get("workspace_summary") or "No data",
            "broker_card": "Broker: unavailable",
            "broker_detail": lifecycle.get("operator_state_detail") or "No data",
            "replay_card": "Replay: unavailable",
            "replay_detail": "No data",
            "advisory_meta": {"deployment_meta": lifecycle},
            "execution_meta": {},
            "performance_meta": {},
            "replay_meta": {},
            "alerts_meta": {},
            "scan_meta": {},
        }
    price = _safe_float(df["close"].iloc[-1]) if "close" in df.columns else None
    align_head, align_detail = _prediction_alignment(price, predictions or {})
    symbol = str(instrument_key).upper()
    profile = get_trader_profile(profile_name)
    section_errors: dict[str, str] = {}

    try:
        levels = _levels_rows(df, instrument_key)
    except Exception as exc:
        section_errors["levels"] = _clean_reason_text(exc, "Key levels unavailable")
        levels = []
    levels = _info_rows(levels, "Levels", section_errors.get("levels", "Need more OHLC history for key levels"))

    try:
        market = _market_context_rows(instrument_key)
    except Exception as exc:
        section_errors["market"] = _clean_reason_text(exc, "Market context unavailable")
        market = []
    market = _info_rows(market, "Market Context", section_errors.get("market", "No cached context history loaded"))

    try:
        breadth_snapshot = get_breadth_snapshot(symbol)
    except Exception as exc:
        section_errors["breadth"] = _clean_reason_text(exc, "Breadth snapshot unavailable")
        breadth_snapshot = {"status": "error", "reason": section_errors["breadth"], "rows": []}

    try:
        macro_snapshot = get_macro_event_snapshot()
    except Exception as exc:
        section_errors["macro"] = _clean_reason_text(exc, "Macro snapshot unavailable")
        macro_snapshot = {"status": "error", "reason": section_errors["macro"], "rows": []}

    try:
        catalyst_snapshot = get_catalyst_provider_snapshot(symbol)
    except Exception as exc:
        section_errors["catalyst"] = _clean_reason_text(exc, "Catalyst snapshot unavailable")
        catalyst_snapshot = {"status": "error", "reason": section_errors["catalyst"], "rows": []}

    try:
        opt_snap, opt_rows, opt_card = _options_rows(symbol, spot_price=price)
    except Exception as exc:
        section_errors["options"] = _clean_reason_text(exc, "Options feed unavailable")
        opt_snap = {"status": "error", "reason": section_errors["options"], "source_kind": "error"}
        opt_rows = []
        opt_card = "Options: unavailable"
    opt_rows = _info_rows(
        opt_rows,
        "Options",
        section_errors.get("options")
        or opt_snap.get("refresh_error")
        or opt_snap.get("reason")
        or opt_snap.get("source")
        or opt_snap.get("source_kind")
        or "No options snapshot cached yet",
    )
    options_detail = str(opt_rows[0][2]) if opt_rows else "No options snapshot cached yet"

    try:
        liquidity_snapshot = get_liquidity_snapshot(df, symbol)
    except Exception as exc:
        section_errors["liquidity"] = _clean_reason_text(exc, "Liquidity snapshot unavailable")
        liquidity_snapshot = {"status": "error", "reason": section_errors["liquidity"], "rows": []}

    try:
        event_snapshot = get_event_snapshot(symbol)
    except Exception as exc:
        section_errors["event"] = _clean_reason_text(exc, "Event snapshot unavailable")
        event_snapshot = {"status": "error", "reason": section_errors["event"]}

    try:
        news_items = load_recent_headlines(symbol, limit=6)
    except Exception as exc:
        section_errors["news"] = _clean_reason_text(exc, "News store unavailable")
        news_items = []
    try:
        news_health = get_news_health(symbol)
    except Exception as exc:
        section_errors["news_health"] = _clean_reason_text(exc, "News health unavailable")
        news_health = {"status": "error", "headline_state": "failed", "history_state": "missing", "last_reason": section_errors["news_health"]}
    try:
        news_runtime = build_news_runtime_summary(symbol, headlines=news_items, news_health=news_health)
    except Exception as exc:
        section_errors["news_runtime"] = _clean_reason_text(exc, "News routing summary unavailable")
        news_runtime = {}

    try:
        signal_rows = _signal_rows(df, instrument_key, options_snapshot=opt_snap)
    except Exception as exc:
        section_errors["signal"] = _clean_reason_text(exc, "Signal context unavailable")
        signal_rows = []
    signal_rows = _info_rows(signal_rows, "Signal Context", section_errors.get("signal", "Need daily history and context data"))

    try:
        stack_rows = _stack_rows(df, instrument_key, options_snapshot=opt_snap, event_snapshot=event_snapshot)
    except Exception as exc:
        section_errors["stack"] = _clean_reason_text(exc, "Advisory stack unavailable")
        stack_rows = []
    stack_rows = _info_rows(stack_rows, "Data Stack", section_errors.get("stack", "Support feeds are still warming up"))

    try:
        advisory = build_advisory_snapshot(
            df,
            instrument_key,
            predictions=predictions or {},
            options_snapshot=opt_snap,
            event_snapshot=event_snapshot,
            news_items=news_items,
            news_health=news_health,
            macro_snapshot=macro_snapshot,
            breadth_snapshot=breadth_snapshot,
            liquidity_snapshot=liquidity_snapshot,
            catalyst_snapshot=catalyst_snapshot,
        )
    except Exception as exc:
        section_errors["advisory"] = _clean_reason_text(exc, "Advisory context unavailable")
        advisory = {
            "action_card": "Action: unavailable",
            "action_detail": section_errors["advisory"],
            "quality_card": "Quality: unavailable",
            "quality_detail": section_errors["advisory"],
            "probability_rows": [],
            "quality_rows": [],
            "setup_rows": [],
            "decision_rows": [],
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
            "news_meta": {},
            "macro_meta": macro_snapshot if isinstance(macro_snapshot, dict) else {},
            "breadth_meta": breadth_snapshot if isinstance(breadth_snapshot, dict) else {},
            "liquidity_meta": liquidity_snapshot if isinstance(liquidity_snapshot, dict) else {},
            "catalyst_meta": catalyst_snapshot if isinstance(catalyst_snapshot, dict) else {},
            "signal_translation": {},
        }
    signal_translation = dict(advisory.get("signal_translation") or {})
    if signal_translation:
        signal_rows = [
            (
                "Trader Lens",
                str(signal_translation.get("headline") or "-"),
                str(signal_translation.get("detail") or "-"),
            )
        ] + list(signal_rows or [])

    try:
        news_rows, news_card, news_detail = _news_rows(
            symbol,
            headlines=news_items,
            news_health=news_health,
            news_meta=advisory.get("news_meta"),
            news_runtime=news_runtime or advisory.get("news_runtime"),
        )
    except Exception as exc:
        section_errors["news"] = _clean_reason_text(exc, "News summary unavailable")
        news_rows, news_card, news_detail = [], "News: unavailable", section_errors["news"]
    news_rows = _news_info_rows(
        news_rows,
        section_errors.get("news")
        or str((news_health or {}).get("last_reason") or "No local headlines yet; feed will populate after refresh"),
    )
    if not news_detail:
        news_detail = str(news_rows[0][2]) if news_rows else "No local headlines yet"

    try:
        execution_snapshot = get_execution_dashboard_snapshot(symbol, current_price=price, advisory=advisory, profile_name=profile.get("key"))
    except Exception as exc:
        section_errors["execution"] = _clean_reason_text(exc, "Execution panel unavailable")
        execution_snapshot = {
            "broker_mode": "unavailable",
            "rows": [],
            "portfolio_rows": [],
            "runtime": {"route_label": "Unavailable", "detail": section_errors["execution"]},
            "diagnostics": {},
            "verification": {},
        }

    try:
        performance_snapshot = get_performance_snapshot()
    except Exception as exc:
        section_errors["performance"] = _clean_reason_text(exc, "Performance panel unavailable")
        performance_snapshot = {"summary": section_errors["performance"], "rows": []}

    try:
        replay_snapshot = get_replay_dashboard_snapshot(symbol)
    except Exception as exc:
        section_errors["replay"] = _clean_reason_text(exc, "Replay panel unavailable")
        replay_snapshot = {"card": "Replay: unavailable", "detail": section_errors["replay"], "rows": []}

    try:
        alerts_snapshot = get_alert_snapshot(symbol, current_price=price, advisory=advisory, profile_name=profile.get("key"), persist=True)
    except Exception as exc:
        section_errors["alerts"] = _clean_reason_text(exc, "Alerts unavailable")
        alerts_snapshot = {"summary": section_errors["alerts"], "rows": []}

    try:
        scan_snapshot = get_opportunity_scan(profile_name=profile.get("key"))
    except Exception as exc:
        section_errors["scan"] = _clean_reason_text(exc, "Opportunity scan unavailable")
        scan_snapshot = {"summary": section_errors["scan"], "rows": []}

    atr_row = next((r for r in levels if r[0] == "ATR(14)"), None)
    vol_card = f"Volatility: {atr_row[1]}" if atr_row else "Volatility: unavailable"
    vol_detail = atr_row[2] if atr_row else section_errors.get("levels", "ATR unavailable")
    if market:
        first_market = market[0]
        vol_detail = f"{vol_detail} | {first_market[0]} {first_market[1]}"

    diagnostics = dict(execution_snapshot.get("diagnostics") or {})
    verification = dict(execution_snapshot.get("verification") or {})
    verify_status = str(verification.get("status") or "").replace("_", " ").strip()
    broker_detail = str((execution_snapshot.get("runtime") or {}).get("detail") or execution_snapshot.get("live_detail") or performance_snapshot.get("summary", "No closed paper trades yet"))
    operator_hint = str(diagnostics.get("operator_hint") or "").strip()
    checked_at = str(diagnostics.get("checked_at") or "")
    credentials_source = str(diagnostics.get("credentials_source") or "").strip()
    if checked_at:
        broker_detail = f"{broker_detail} | checked {checked_at}"
    if verify_status:
        broker_detail = f"{broker_detail} | verify {verify_status}"
    if credentials_source == "local_store":
        broker_detail = f"{broker_detail} | saved on this computer"
    elif credentials_source == "env":
        broker_detail = f"{broker_detail} | using runtime environment"
    if operator_hint:
        broker_detail = f"{broker_detail} | {operator_hint}"

    breadth_rows = _info_rows(
        list(advisory.get("breadth_rows") or []),
        "Breadth",
        section_errors.get("breadth")
        or breadth_snapshot.get("provider_note")
        or breadth_snapshot.get("reason")
        or "No breadth cache loaded yet",
    )
    macro_rows = _info_rows(
        list(advisory.get("macro_rows") or []),
        "Macro",
        section_errors.get("macro")
        or macro_snapshot.get("provider_note")
        or macro_snapshot.get("reason")
        or "No macro calendar loaded yet",
    )
    liquidity_rows = _info_rows(
        list(advisory.get("liquidity_rows") or []),
        "Liquidity",
        section_errors.get("liquidity")
        or liquidity_snapshot.get("reason")
        or "Need more price history for tradability checks",
    )
    catalyst_rows = _info_rows(
        list(advisory.get("catalyst_rows") or []),
        "Catalysts",
        section_errors.get("catalyst")
        or catalyst_snapshot.get("status")
        or "No fresh catalyst records cached yet",
    )
    probability_rows = _info_rows(
        list(advisory.get("probability_rows") or []),
        "Probabilities",
        section_errors.get("advisory", "Predictions are still warming up"),
    )
    quality_rows = _info_rows(
        list(advisory.get("quality_rows") or []),
        "Signal Quality",
        section_errors.get("advisory", "Quality analysis unavailable right now"),
    )
    decision_rows = list(advisory.get("decision_rows") or [])
    setup_rows = _info_rows(
        decision_rows + list(advisory.get("setup_rows") or []),
        "Trade Plan",
        section_errors.get("advisory", "No actionable setup summary yet"),
    )
    scenario_rows = _info_rows(
        list(advisory.get("scenario_rows") or []),
        "Scenario Map",
        section_errors.get("advisory", "Scenario map unavailable right now"),
    )
    invalidation_rows = _info_rows(
        list(advisory.get("invalidation_rows") or []),
        "Invalidation",
        section_errors.get("advisory", "Invalidation rules unavailable right now"),
    )
    reliability_rows = _info_rows(
        list(advisory.get("reliability_rows") or []),
        "Reliability",
        section_errors.get("advisory", "No readiness history loaded yet"),
    )
    execution_rows = _info_rows(
        list(execution_snapshot.get("rows") or []),
        "Execution",
        section_errors.get("execution", "Execution panel unavailable"),
    )
    portfolio_rows = _info_rows(
        list(execution_snapshot.get("portfolio_rows") or []),
        "Portfolio",
        section_errors.get("execution", "Portfolio summary unavailable"),
    )
    performance_rows = _info_rows(
        list(performance_snapshot.get("rows") or []),
        "Performance",
        section_errors.get("performance", "No closed paper trades yet"),
    )
    replay_rows = _info_rows(
        list(replay_snapshot.get("rows") or []),
        "Replay",
        section_errors.get("replay", "No replay session"),
    )
    alerts_rows = _info_rows(
        list(alerts_snapshot.get("rows") or []),
        "Alerts",
        section_errors.get("alerts", "No active trader alerts"),
    )
    opportunity_rows = _info_rows(
        list(scan_snapshot.get("rows") or []),
        "Scan",
        section_errors.get("scan", "Opportunity scan unavailable"),
    )

    return {
        "setup_card": align_head,
        "setup_detail": align_detail,
        "vol_card": vol_card,
        "vol_detail": vol_detail,
        "options_card": opt_card,
        "options_detail": options_detail,
        "news_card": news_card,
        "news_detail": news_detail,
        "market_rows": market,
        "breadth_rows": breadth_rows,
        "macro_rows": macro_rows,
        "liquidity_rows": liquidity_rows,
        "catalyst_rows": catalyst_rows,
        "signal_rows": signal_rows,
        "stack_rows": stack_rows,
        "levels_rows": levels,
        "options_rows": opt_rows,
        "news_rows": news_rows,
        "options_snapshot": opt_snap,
        "action_card": advisory.get("action_card", "Action: unavailable"),
        "action_detail": advisory.get("action_detail", "-"),
        "quality_card": advisory.get("quality_card", "Quality: unavailable"),
        "quality_detail": advisory.get("quality_detail", "-"),
        "probability_rows": probability_rows,
        "quality_rows": quality_rows,
        "setup_rows": setup_rows,
        "decision_rows": decision_rows,
        "scenario_rows": scenario_rows,
        "invalidation_rows": invalidation_rows,
        "reliability_rows": reliability_rows,
        "execution_rows": execution_rows,
        "portfolio_rows": portfolio_rows,
        "performance_rows": performance_rows,
        "replay_rows": replay_rows,
        "alerts_rows": alerts_rows,
        "opportunity_rows": opportunity_rows,
        "profile_card": f"Profile: {profile.get('label', 'Balanced')}",
        "profile_detail": f"Risk/trade {100.0 * float(profile.get('risk_per_trade') or 0.0):.2f}% | gross cap {100.0 * float(profile.get('max_gross_exposure') or 0.0):.0f}%",
        "broker_card": f"Broker: {str((execution_snapshot.get('runtime') or {}).get('route_label') or execution_snapshot.get('broker_mode') or 'paper')}",
        "broker_detail": broker_detail,
        "replay_card": replay_snapshot.get("card", "Replay: unavailable"),
        "replay_detail": replay_snapshot.get("detail", "No replay session"),
        "advisory_meta": advisory,
        "deployment_meta": advisory.get("deployment_meta") or summarize_deployment_readiness(symbol, state=load_readiness_state(symbol) or {}),
        "execution_meta": execution_snapshot,
        "performance_meta": performance_snapshot,
        "replay_meta": replay_snapshot,
        "alerts_meta": alerts_snapshot,
        "scan_meta": scan_snapshot,
        "section_errors": section_errors,
    }
