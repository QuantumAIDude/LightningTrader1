from __future__ import annotations

import math
from collections import Counter, defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

import autonomy_engine as autonomy
import advisory_engine as advisory_module
import features as feature_module
import news_pipeline as news_module
from config import HORIZON_MINUTES, PAPER_STARTING_CASH, SLIPPAGE_PCT, TRANSACTION_COST_PER_TRADE
from execution_engine import new_paper_state, recommend_order_for_state
from features import BAR_MINUTES, DEFAULT_FEATURES, build_feature_frame, build_features_for_training
from forecast_calibration import fit_intraday_activity_gate, fit_return_calibration
from model_experience_ledger import (
    append_records as append_experience_records,
    build_experience_record,
    resolve_one_step_outcome,
)
from model_trainers import (
    _build_calibration,
    _boost_ctor,
    _compute_recency_weights,
    _extra_trees_ctor,
    _forest_ctor,
    _resolve_active_features,
    _ridge_pipeline,
)
from model_utils import predict_4min
from news_pipeline import load_news_features_for_index
from regime_model import train_regime


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
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


def _normalize_frame(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    if out.empty:
        return out
    try:
        out.columns = [str(c).lower() for c in out.columns]
    except Exception:
        pass
    ts_col = next((col for col in ("timestamp", "datetime", "date") if col in out.columns), None)
    if ts_col is not None:
        out.index = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
        out = out.drop(columns=[ts_col], errors="ignore")
    else:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")
    out = out[~out.index.isna()].sort_index()
    for col in ("open", "high", "low", "close", "volume"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["close"])
    return out


def _load_default_intraday_frame(symbol: str = "QQQ") -> pd.DataFrame:
    path = f"data/prices_{str(symbol or 'QQQ').upper()}.csv"
    try:
        return _normalize_frame(pd.read_csv(path, index_col=0, parse_dates=True))
    except Exception:
        return pd.DataFrame()


def _load_truncated_daily_history(symbol: str, as_of: pd.Timestamp | None) -> pd.DataFrame:
    path = f"data/prices_{str(symbol or 'QQQ').upper()}_daily_long.csv"
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    # Daily cache files may carry explicit UTC timestamps while historical replay
    # cutoff dates are local-date based. Normalize daily bars to timezone-naive
    # calendar dates so the comparison stays about available daily sessions, not
    # pandas timezone metadata.
    df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert(None).normalize()
    df = df[~df.index.isna()].sort_index()
    try:
        df.columns = [str(c).lower() for c in df.columns]
    except Exception:
        pass
    if as_of is None:
        return df
    cutoff_local = _coerce_utc_ts(as_of)
    if cutoff_local is None:
        return df
    cutoff_date = pd.Timestamp(cutoff_local.tz_convert("America/New_York").date())
    return df[df.index <= cutoff_date].copy()


def _session_dates(index_utc: pd.DatetimeIndex, tz: str = "America/New_York") -> pd.Series:
    idx_local = pd.to_datetime(index_utc, utc=True, errors="coerce").tz_convert(tz)
    return pd.Series(pd.to_datetime(idx_local.date), index=index_utc)


def _fold_specs(df: pd.DataFrame, n_splits: int) -> list[dict]:
    if df is None or df.empty:
        return []
    session_series = _session_dates(df.index)
    sessions = list(pd.Index(session_series.dropna().unique()).sort_values())
    if len(sessions) < 12:
        return []
    target_splits = max(2, int(n_splits))
    max_splits = max(2, min(target_splits, len(sessions) - 1))
    while max_splits > 2:
        block = len(sessions) // (max_splits + 1)
        if block >= 6:
            break
        max_splits -= 1
    block = max(1, len(sessions) // (max_splits + 1))
    folds = []
    for i in range(1, max_splits + 1):
        train_sessions = sessions[: block * i]
        test_start = block * i
        test_end = block * (i + 1) if i < max_splits else len(sessions)
        test_sessions = sessions[test_start:test_end]
        if len(train_sessions) < 6 or len(test_sessions) < 2:
            continue
        folds.append(
            {
                "fold": len(folds) + 1,
                "train_sessions": train_sessions,
                "test_sessions": test_sessions,
                "train_start": str(train_sessions[0].date()),
                "train_end": str(train_sessions[-1].date()),
                "test_start": str(test_sessions[0].date()),
                "test_end": str(test_sessions[-1].date()),
            }
        )
    return folds


def _time_bucket(ts_utc: pd.Timestamp, tz: str = "America/New_York") -> str:
    local = _coerce_utc_ts(ts_utc).tz_convert(tz)
    hour = int(local.hour)
    if hour < 11:
        return "AM"
    if hour < 14:
        return "Midday"
    return "PM"


def _regime_label(predictions: dict) -> str:
    regime = dict((predictions or {}).get("intraday", {}).get("regime") or {})
    scores = {
        "Bullish": _safe_float(regime.get("Bullish")) or 0.0,
        "Neutral": _safe_float(regime.get("Neutral")) or 0.0,
        "Bearish": _safe_float(regime.get("Bearish")) or 0.0,
    }
    return max(scores.items(), key=lambda kv: kv[1])[0]


def _row_frame(df: pd.DataFrame, ts_value: pd.Timestamp) -> pd.DataFrame:
    row = df.loc[[ts_value]].copy()
    row.attrs["data_source"] = "backtest"
    row.attrs["data_error"] = False
    return row


def _neutral_breadth_snapshot() -> dict:
    return {
        "status": "historical_unavailable",
        "source_kind": "historical_unavailable",
        "risk_on_score": None,
        "direct_feed": False,
        "rows": [],
    }


def _neutral_macro_snapshot() -> dict:
    return {
        "status": "historical_unavailable",
        "source_kind": "historical_unavailable",
        "provider_note": "Historical macro calendar was not cached for this replay.",
        "risk_score": 0.0,
        "high_risk_24h": False,
        "events": [],
        "rows": [],
    }


def _news_event_snapshot(news_items: list[dict], as_of: pd.Timestamp) -> dict:
    if not news_items:
        return {"status": "historical_unavailable", "days_to_earnings": 99.0}
    earnings_like = any(str(item.get("event_type") or "").lower() in {"earnings", "guidance"} for item in news_items)
    return {
        "status": "historical_news",
        "days_to_earnings": 0.0 if earnings_like else 99.0,
    }


def _news_catalyst_snapshot(news_items: list[dict], as_of: pd.Timestamp) -> dict:
    now = _coerce_utc_ts(as_of)
    recent = []
    risk_score = 0.0
    top_item = None
    for item in list(news_items or []):
        ts = _coerce_utc_ts(item.get("published_at"))
        if ts is None or now is None:
            age_h = 999.0
        else:
            age_h = max(0.0, float((now - ts).total_seconds()) / 3600.0)
        importance = _safe_float(item.get("importance_score")) or 0.0
        impact = _safe_float(item.get("impact_score")) or 0.0
        sent = str(item.get("label") or item.get("signal_direction") or "").lower()
        if age_h <= 24.0:
            recent.append(
                {
                    "title": str(item.get("headline") or "")[:80],
                    "sentiment": sent,
                    "event_type": str(item.get("event_type") or "general"),
                    "importance": importance,
                    "impact": impact,
                }
            )
            risk_score = max(risk_score, 0.55 * max(importance, impact))
        if top_item is None or max(importance, impact) > max(
            _safe_float(top_item.get("importance_score")) or 0.0,
            _safe_float(top_item.get("impact_score")) or 0.0,
        ):
            top_item = item
    active = None
    if top_item is not None:
        active = {
            "title": str(top_item.get("headline") or "")[:80],
            "risk_kind": str(top_item.get("event_type") or "general"),
            "ambiguity": "medium" if (_safe_float(top_item.get("noise_score")) or 0.0) > 0.35 else "low",
        }
    rows = []
    if active is not None:
        rows.append(("News Catalyst", str(active.get("risk_kind") or "-").replace("_", " ").title(), str(active.get("title") or "-")))
    return {
        "status": "historical_news",
        "catalyst_risk_score": float(min(1.0, risk_score)),
        "active_catalyst": active,
        "recent": recent[:4],
        "rows": rows,
    }


def _readiness_override() -> dict:
    return {"live_ready": True, "fresh": True, "reason": "historical fold replay"}


def _historical_replay_context(
    df: pd.DataFrame | None,
    as_of: pd.Timestamp | None,
    instrument_key: str = "QQQ",
    min_bars: int = 160,
) -> dict:
    frame = _normalize_frame(df)
    as_of_ts = _coerce_utc_ts(as_of) or pd.Timestamp.now(tz="UTC")
    last_ts = None
    age_sec = None
    if not frame.empty:
        last_ts = _coerce_utc_ts(frame.index[-1])
        if last_ts is not None:
            try:
                age_sec = float((as_of_ts - last_ts).total_seconds())
            except Exception:
                age_sec = None
    try:
        threshold_sec = max(180.0, float(BAR_MINUTES) * 60.0 * 1.5)
    except Exception:
        threshold_sec = 180.0
    bars_available = int(len(frame))
    enough_bars = bool(bars_available >= int(min_bars))
    timestamp_valid = bool(age_sec is not None and age_sec >= 0.0)
    fresh_as_of = bool(enough_bars and timestamp_valid and age_sec <= threshold_sec)
    if not enough_bars:
        reason = "insufficient prior bars as of replay timestamp"
        severity = "insufficient_history"
    elif not timestamp_valid:
        reason = "latest replay bar timestamp is unavailable or after replay timestamp"
        severity = "invalid_timestamp"
    elif not fresh_as_of:
        reason = "historical data is stale relative to replay timestamp"
        severity = "critical"
    else:
        reason = "fresh as of replay timestamp"
        severity = "fresh"
    return {
        "historical_mode": True,
        "freshness_is_as_of": True,
        "instrument": str(instrument_key or "QQQ").upper(),
        "replay_as_of": as_of_ts.isoformat(),
        "bars_available_as_of": bars_available,
        "min_bars_required": int(min_bars),
        "enough_prior_bars": enough_bars,
        "data_last_timestamp_as_of": last_ts.isoformat() if last_ts is not None else None,
        "age_sec_as_of": round(float(age_sec), 3) if age_sec is not None else None,
        "freshness_threshold_sec": round(float(threshold_sec), 3),
        "fresh_as_of": fresh_as_of,
        "stale_as_of": not fresh_as_of,
        "stale_severity_as_of": severity,
        "reason": reason,
    }


def _historical_system_report(
    instrument_key: str,
    base_state: dict | None,
    replay_context: dict | None,
    predictions: dict | None = None,
) -> dict:
    symbol = str(instrument_key or "QQQ").upper()
    base_state = dict(base_state or {})
    current_system = dict(base_state.get("system_checks") or base_state.get("system_report") or {})
    context = dict(replay_context or {})
    fresh_as_of = bool(context.get("fresh_as_of"))
    age_sec = _safe_float(context.get("age_sec_as_of"))
    threshold_sec = _safe_float(context.get("freshness_threshold_sec"))
    pred_meta = dict(((predictions or {}).get("intraday") or {}).get("_meta") or {})
    system = {
        "status": "ok" if fresh_as_of else "fail",
        "instrument": symbol,
        "ts": str(context.get("replay_as_of") or ""),
        "historical_replay": True,
        "data": {
            "rows": int(context.get("bars_available_as_of") or 0),
            "last_ts": context.get("data_last_timestamp_as_of"),
            "stale_severity": str(context.get("stale_severity_as_of") or ""),
            "error": None if fresh_as_of else str(context.get("reason") or "historical replay data is stale"),
            "staleness": {
                "is_stale": not fresh_as_of,
                "age_sec": age_sec,
                "threshold_sec": threshold_sec,
                "market_open": True,
                "market_state": "historical_replay",
                "freshness_is_as_of": True,
            },
        },
        "predictions": {
            "4min": {
                "diagnostics": {
                    "critical_stale": bool((not fresh_as_of) or pred_meta.get("critical_stale")),
                    "readiness_locked": bool((not fresh_as_of) or pred_meta.get("readiness_locked")),
                    "diversity": dict(pred_meta.get("diversity") or {}),
                    "historical_replay": True,
                    "freshness_is_as_of": True,
                }
            }
        },
    }
    for key in ("calibration", "horizon_accuracy", "models"):
        if isinstance(current_system.get(key), dict):
            system[key] = dict(current_system.get(key) or {})
    return system


def _historical_readiness_state(base_state: dict | None, replay_context: dict | None) -> dict:
    out = dict(base_state or {})
    context = dict(replay_context or {})
    out["fresh"] = bool(context.get("fresh_as_of"))
    out["historical_replay"] = True
    out["historical_freshness"] = context
    return out


def _load_historical_news_store(symbol: str) -> dict:
    symbol = str(symbol or "QQQ").upper()
    state = news_module._load_state(symbol)
    path = news_module._news_path(symbol)
    if not news_module.NEWS_PIPELINE_ENABLED or not path or not news_module.os.path.exists(path):
        return {"symbol": symbol, "state": state, "frame": pd.DataFrame()}
    try:
        df = news_module._normalize_news_frame(pd.read_csv(path), default_symbol=symbol)
    except Exception:
        df = pd.DataFrame()
    if not df.empty:
        df["published_at"] = pd.to_datetime(df.get("published_at"), utc=True, errors="coerce")
        df = df.dropna(subset=["published_at"]).copy()
    return {"symbol": symbol, "state": state, "frame": df}


def _cached_recent_headlines(store: dict, as_of: pd.Timestamp | None, limit: int = 8) -> list[dict]:
    df = dict(store or {}).get("frame")
    symbol = str(dict(store or {}).get("symbol") or "QQQ").upper()
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    now = _coerce_utc_ts(as_of) or pd.Timestamp.now(tz="UTC")
    rows = df[df["published_at"] <= now].copy()
    if rows.empty:
        return []
    age_hours = ((now - rows["published_at"]).dt.total_seconds() / 3600.0).clip(lower=0.0)
    rows["importance_score"] = pd.to_numeric(rows.get("importance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    rows["impact_score"] = pd.to_numeric(rows.get("impact_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    rows["noise_score"] = pd.to_numeric(rows.get("noise_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
    freshness = np.exp(-(age_hours / 36.0))
    rows["rank_score"] = (rows["importance_score"] * freshness) + (0.20 * rows["impact_score"]) - (0.10 * rows["noise_score"])
    rows = rows.sort_values(["rank_score", "published_at"], ascending=[False, False]).head(max(1, int(limit))).copy()
    out = []
    for _, row in rows.iterrows():
        out.append(
            {
                "published_at": row.get("published_at"),
                "headline": str(row.get("headline") or "").strip(),
                "score": _safe_float(row.get("sentiment_score")),
                "label": str(row.get("sentiment_label") or "").strip(),
                "relevance": _safe_float(row.get("relevance_score")),
                "political_relevance": _safe_float(row.get("political_relevance_score")),
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
                "symbol": symbol,
            }
        )
    return out


def _cached_news_health(store: dict, trading_dates=None, tz: str = "America/New_York", as_of=None) -> dict:
    symbol = str(dict(store or {}).get("symbol") or "QQQ").upper()
    state = dict(dict(store or {}).get("state") or {})
    df = dict(store or {}).get("frame")
    out = {
        "symbol": symbol,
        "enabled": bool(news_module.NEWS_PIPELINE_ENABLED),
        "status": "disabled" if not news_module.NEWS_PIPELINE_ENABLED else "missing",
        "headline_state": "disabled" if not news_module.NEWS_PIPELINE_ENABLED else "missing",
        "history_state": "missing",
        "signal_ready": False,
        "history_ready": False,
    }
    if not news_module.NEWS_PIPELINE_ENABLED:
        return out
    out["last_refresh"] = state.get("last_refresh")
    out["last_status"] = state.get("last_status")
    out["last_reason"] = state.get("last_reason")
    coverage = {
        "symbol": symbol,
        "rows": 0,
        "relevant_rows": 0,
        "active_days": 0,
        "trading_days": 0,
        "coverage_ratio": 0.0,
        "sufficient_long_horizon": False,
    }
    now_utc = _coerce_utc_ts(as_of) or pd.Timestamp.now(tz="UTC")
    if isinstance(df, pd.DataFrame) and not df.empty:
        rows = df[df["published_at"] <= now_utc].copy()
        if not rows.empty:
            rows["relevance_score"] = pd.to_numeric(rows.get("relevance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            rows["tech_relevant"] = pd.to_numeric(rows.get("tech_relevant"), errors="coerce").fillna(0).astype(int)
            rows["political_relevant"] = pd.to_numeric(rows.get("political_relevant"), errors="coerce").fillna(0).astype(int)
            relevant = rows[
                (rows["tech_relevant"] > 0)
                | (rows["political_relevant"] > 0)
                | (rows["relevance_score"] >= 0.25)
            ].copy()
            relevant_local = relevant["published_at"].dt.tz_convert(tz) if not relevant.empty else pd.Series(dtype="datetime64[ns, UTC]")
            active_dates = pd.Index(pd.to_datetime(relevant_local.dt.date)).unique().sort_values() if not relevant.empty else pd.DatetimeIndex([])
            if trading_dates is None or len(trading_dates) == 0:
                start = active_dates.min() if len(active_dates) else pd.NaT
                end = active_dates.max() if len(active_dates) else pd.NaT
                trading_idx = pd.bdate_range(start=start, end=end) if start is not pd.NaT and end is not pd.NaT else pd.DatetimeIndex([])
            else:
                trading_idx = pd.DatetimeIndex(pd.to_datetime(trading_dates, utc=True, errors="coerce")).tz_convert(tz)
                trading_idx = pd.DatetimeIndex(pd.to_datetime(trading_idx.date)).unique().sort_values()
            if len(trading_idx):
                cutoff_local = now_utc.tz_convert(tz)
                trading_idx = trading_idx[trading_idx <= pd.Timestamp(cutoff_local.date())]
            trading_days = int(len(trading_idx))
            coverage_ratio = float(len(active_dates.intersection(trading_idx)) / trading_days) if trading_days > 0 else 0.0
            sufficient = bool(
                int(len(relevant)) >= int(news_module.NEWS_LONG_HORIZON_MIN_ROWS)
                and int(len(active_dates)) >= int(news_module.NEWS_LONG_HORIZON_MIN_ACTIVE_DAYS)
                and coverage_ratio >= float(news_module.NEWS_LONG_HORIZON_MIN_COVERAGE)
            )
            coverage = {
                "symbol": symbol,
                "rows": int(len(rows)),
                "relevant_rows": int(len(relevant)),
                "active_days": int(len(active_dates)),
                "trading_days": trading_days,
                "coverage_ratio": float(coverage_ratio),
                "start": active_dates.min().strftime("%Y-%m-%d") if len(active_dates) else None,
                "end": active_dates.max().strftime("%Y-%m-%d") if len(active_dates) else None,
                "sufficient_long_horizon": sufficient,
            }
            latest_ts = rows["published_at"].max()
            age_minutes = float((now_utc - latest_ts).total_seconds() / 60.0) if latest_ts is not pd.NaT else None
            importance = pd.to_numeric(rows.get("importance_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)
            recent = rows[rows["published_at"] >= (now_utc - pd.Timedelta(hours=24))].copy()
            high_impact_recent = int((pd.to_numeric(recent.get("importance_score"), errors="coerce").fillna(0.0) >= 0.55).sum()) if not recent.empty else 0
            out.update(
                {
                    "status": "ok",
                    "headline_state": "ok" if high_impact_recent > 0 or not rows.empty else "no_meaningful_news",
                    "headline_count": int(len(rows)),
                    "recent_headline_count": int(len(recent)),
                    "high_impact_recent_count": high_impact_recent,
                    "latest_headline_at": latest_ts.isoformat() if latest_ts is not pd.NaT else None,
                    "latest_headline_age_min": age_minutes,
                    "signal_ready": bool(high_impact_recent > 0 or len(recent) > 0),
                }
            )
            if age_minutes is not None and age_minutes > float(news_module.NEWS_REFRESH_COOLDOWN_SEC) / 60.0:
                out["headline_state"] = "stale"
                out["status"] = "stale"
        elif str(state.get("last_status") or "").lower() == "error":
            out["status"] = "failed"
            out["headline_state"] = "failed"
    elif str(state.get("last_status") or "").lower() == "error":
        out["status"] = "failed"
        out["headline_state"] = "failed"
    out["coverage"] = coverage
    out["history_ready"] = bool((coverage or {}).get("sufficient_long_horizon", False))
    out["history_state"] = "ready" if out["history_ready"] else "insufficient_history"
    return out


def _historical_liquidity_snapshot(df: pd.DataFrame, daily_long_history: pd.DataFrame) -> dict:
    current_price = _safe_float(pd.to_numeric(df.get("close"), errors="coerce").iloc[-1]) if df is not None and not df.empty and "close" in df.columns else None
    spread_bps = None
    adv = None
    dollar_vol = None
    atr14 = None
    if daily_long_history is not None and not daily_long_history.empty and "close" in daily_long_history.columns:
        close = pd.to_numeric(daily_long_history["close"], errors="coerce")
        volume = pd.to_numeric(daily_long_history.get("volume"), errors="coerce").fillna(0.0)
        adv = _safe_float(volume.tail(20).mean())
        dollar_vol = _safe_float((close * volume).tail(20).mean())
        high = pd.to_numeric(daily_long_history.get("high"), errors="coerce")
        low = pd.to_numeric(daily_long_history.get("low"), errors="coerce")
        prev_close = close.shift(1)
        tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
        atr14 = _safe_float(tr.rolling(14, min_periods=5).mean().iloc[-1]) if len(tr) else None
    rel_vol = None
    if df is not None and not df.empty and "volume" in df.columns:
        try:
            session_dates = _session_dates(pd.DatetimeIndex(df.index))
            today = session_dates.iloc[-1]
            today_vol = _safe_float(pd.to_numeric(df.loc[session_dates == today, "volume"], errors="coerce").sum())
            session_vol = (
                pd.Series(pd.to_numeric(df["volume"], errors="coerce").fillna(0.0).values, index=session_dates.values)
                .groupby(level=0)
                .sum()
                .tail(21)
            )
            avg_session_vol = _safe_float(session_vol.iloc[:-1].mean()) if len(session_vol) >= 2 else None
            if today_vol is not None and avg_session_vol not in (None, 0.0):
                rel_vol = float(today_vol / avg_session_vol)
        except Exception:
            pass
    slippage_bps = None
    score_parts = []
    if dollar_vol is not None:
        score_parts.append(max(0.0, min(1.0, np.log10(max(1.0, dollar_vol)) / 9.0)))
    if rel_vol is not None:
        score_parts.append(max(0.0, min(1.0, rel_vol / 1.4)))
    tradability_score = float(np.mean(score_parts)) if score_parts else 0.45
    label = "Good" if tradability_score >= 0.72 else "Fair" if tradability_score >= 0.55 else "Fragile"
    warnings = ["live spread unavailable"]
    if dollar_vol is not None and dollar_vol < 15_000_000:
        warnings.append("light dollar volume")
    if rel_vol is not None and rel_vol < 0.75:
        warnings.append("subpar participation")
    return {
        "rows": [
            ("Spread", "-", "-"),
            ("Slippage Est", "-" if slippage_bps is None else f"{slippage_bps:.1f} bps", "Historical replay; no book depth"),
            ("ADV / $Vol", "-" if adv is None else f"{adv:,.0f}", "-" if dollar_vol is None else f"${dollar_vol/1_000_000:.1f}M"),
            ("Rel Volume", "-" if rel_vol is None else f"{rel_vol:.2f}x", f"ATR {_safe_float(atr14) or 0.0:.2f}"),
            ("Tradability", f"{int(round(100.0 * tradability_score))}/100", label),
            ("Depth Feed", "Unavailable", "Historical replay"),
        ],
        "tradability_score": tradability_score,
        "slippage_bps": slippage_bps,
        "spread_bps": spread_bps,
        "warnings": warnings,
        "depth_supported": False,
        "orderflow_supported": False,
    }


def _fast_session_state(df: pd.DataFrame, instrument_key: str) -> dict:
    meta = advisory_module._instrument_meta(instrument_key)
    tz = meta.get("tz", "America/New_York")
    session_open = int(meta.get("session_open_minute", 9 * 60 + 30))
    session_close = int(meta.get("session_close_minute", 16 * 60))
    idx_local = advisory_module._tz_index(df.index, tz)
    idx_local_series = pd.Series(idx_local, index=df.index)
    frame = df.copy()
    frame["session_date"] = pd.to_datetime(idx_local.date)
    current_session_date = frame["session_date"].iloc[-1] if not frame.empty else None
    current_session = frame[frame["session_date"] == current_session_date].copy()
    current_session["idx_local"] = idx_local_series.loc[current_session.index].values
    minutes = idx_local[-1].hour * 60 + idx_local[-1].minute if len(idx_local) else session_open
    sess_span = max(1, session_close - session_open)
    session_progress = max(0.0, min(1.0, float(minutes - session_open) / float(sess_span)))
    return {
        "frame": frame,
        "current_session": current_session,
        "session_progress": float(session_progress),
        "session_open_minute": session_open,
        "session_close_minute": session_close,
        "tz": tz,
    }


def _controller_advisory_snapshot(
    df: pd.DataFrame,
    instrument_key: str,
    predictions: dict,
    news_items: list[dict],
    news_health: dict,
    event_snapshot: dict,
    macro_snapshot: dict,
    breadth_snapshot: dict,
    liquidity_snapshot: dict,
    catalyst_snapshot: dict,
    now_utc: pd.Timestamp,
    daily_long_history: pd.DataFrame,
    replay_context: dict | None = None,
) -> dict:
    replay_context = dict(replay_context or _historical_replay_context(df, now_utc, instrument_key=instrument_key))
    current_price = _safe_float(pd.to_numeric(df.get("close"), errors="coerce").iloc[-1]) if "close" in df.columns else None
    state = _fast_session_state(df, instrument_key)
    atr14 = advisory_module._atr14(daily_long_history)
    pred_state = advisory_module._prediction_state(predictions, current_price)
    regime = advisory_module._regime_snapshot(predictions)
    _, spread_frac = advisory_module._current_spread_info(df, current_price)
    noise = advisory_module._intraday_noise(df, current_price)
    volume_state = advisory_module._volume_confirmation(state, current_price)
    news_snapshot = advisory_module._news_context_snapshot(
        df,
        instrument_key,
        news_items,
        current_price,
        pred_state,
        news_health=news_health,
        now_utc=now_utc,
    )
    quality = advisory_module._signal_quality(
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
    signal_translation = advisory_module._signal_translation(pred_state, quality, news_snapshot=news_snapshot)
    readiness_state = _historical_readiness_state(
        advisory_module.load_readiness_state(instrument_key) or {},
        replay_context,
    )
    historical_system = _historical_system_report(instrument_key, readiness_state, replay_context, predictions=predictions)
    deployment_meta = advisory_module.summarize_deployment_readiness(
        instrument_key,
        state=readiness_state,
        system_report=historical_system,
        news_health=news_health,
    )
    deployment_meta["historical_freshness"] = replay_context
    execution_posture = advisory_module._deployment_execution_posture(signal_translation, deployment_meta, quality)
    expression_meta = advisory_module._expression_paths(
        instrument_key,
        pred_state,
        quality,
        options_snapshot={"status": "historical_unavailable"},
        liquidity_snapshot=liquidity_snapshot,
    )
    current_session = state.get("current_session")
    level_map = {
        "VWAP": (
            (
                _safe_float(
                    pd.to_numeric(current_session.get("close"), errors="coerce").mul(
                        pd.to_numeric(current_session.get("volume"), errors="coerce").fillna(0.0)
                    ).sum()
                    / (pd.to_numeric(current_session.get("volume"), errors="coerce").fillna(0.0).sum() + 1e-12)
                )
                if current_session is not None and not current_session.empty
                else None
            ),
            "",
        ),
    }
    if current_session is not None and not current_session.empty:
        try:
            or_minutes = current_session["idx_local"].dt.hour * 60 + current_session["idx_local"].dt.minute
            or_block = current_session[
                (or_minutes >= state["session_open_minute"]) & (or_minutes < state["session_open_minute"] + 30)
            ]
            if not or_block.empty:
                level_map["Opening Range"] = (
                    f"{advisory_module._fmt_dollar(pd.to_numeric(or_block['high'], errors='coerce').max())} / "
                    f"{advisory_module._fmt_dollar(pd.to_numeric(or_block['low'], errors='coerce').min())}",
                    "",
                )
        except Exception:
            pass
    frame = state.get("frame")
    try:
        prior_sessions = frame["session_date"].drop_duplicates()
        if len(prior_sessions) >= 2:
            prev_session = prior_sessions.iloc[-2]
            prev = frame[frame["session_date"] == prev_session]
            if not prev.empty:
                level_map["Prior Session"] = (
                    f"{advisory_module._fmt_dollar(pd.to_numeric(prev['high'], errors='coerce').max())} / "
                    f"{advisory_module._fmt_dollar(pd.to_numeric(prev['low'], errors='coerce').min())}",
                    "",
                )
    except Exception:
            pass
    plan = advisory_module._build_trade_plan(current_price, level_map, pred_state, quality, atr14)
    regime_meta = advisory_module.classify_market_regime(df, predictions=predictions, news_meta=news_snapshot)
    risk_meta = advisory_module.risk_summary_for_symbol(
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
    decision_payload = advisory_module._advisory_decision_payload(
        signal_translation,
        regime_meta,
        risk_meta,
        deployment_meta,
        execution_posture,
        quality,
        plan,
        pred_state,
    )
    return {
        "quality_meta": quality,
        "plan_meta": plan,
        "pred_state": pred_state,
        "regime_meta": regime_meta,
        "risk_meta": risk_meta,
        "news_meta": news_snapshot,
        "macro_meta": macro_snapshot or {},
        "breadth_meta": breadth_snapshot or {},
        "liquidity_meta": liquidity_snapshot or {},
        "catalyst_meta": catalyst_snapshot or {},
        "deployment_meta": deployment_meta,
        "execution_posture": execution_posture,
        "signal_translation": signal_translation,
        "decision_payload": decision_payload,
        "decision_rows": advisory_module.advisory_decision_rows(decision_payload),
        "historical_freshness": replay_context,
        "expression_meta": expression_meta,
        "predictions_summary": {
            "intraday_position_size": _safe_float(((pred_state or {}).get("intraday") or {}).get("position_size")),
            "intraday_direction": str(((pred_state or {}).get("intraday") or {}).get("direction") or "-"),
        },
    }


def _train_fold_models_fast(train_df: pd.DataFrame, instrument_key: str, train_horizons: bool = True) -> dict:
    df_full = build_features_for_training(train_df, predict_minutes=4, instrument_key=instrument_key)
    if df_full is None or df_full.empty:
        return {}
    try:
        df_full = df_full.replace([np.inf, -np.inf], np.nan).dropna()
    except Exception:
        pass
    if len(df_full) < 240:
        return {}
    target_col = "target_ret_norm"
    try:
        target_mean = float(np.nanmean(df_full[target_col]))
    except Exception:
        target_mean = 0.0
    df_full["target_ret_norm_demean"] = pd.to_numeric(df_full[target_col], errors="coerce").fillna(0.0) - target_mean
    y = pd.to_numeric(df_full["target_ret_norm_demean"], errors="coerce").fillna(0.0)
    base_feats, base_diag = _resolve_active_features(df_full, DEFAULT_FEATURES)
    forest_feats, forest_diag = _resolve_active_features(df_full, DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3"])
    boost_feats, boost_diag = _resolve_active_features(df_full, DEFAULT_FEATURES + ["ret_2", "ret_4"])
    extra_feats, extra_diag = _resolve_active_features(
        df_full,
        DEFAULT_FEATURES + ["ret_2", "ret_4", "mom_3", "ema_5", "ema_20", "trend_strength", "vol_rolling_skew", "vol_rolling_kurt"],
    )
    weights = _compute_recency_weights(df_full.index, half_life_days=7.0)
    gate = {"scale": None, "quantile": None, "selected": "disabled"}
    try:
        close_full = pd.to_numeric(df_full.get("close"), errors="coerce").values.astype(float)
        target_price = pd.to_numeric(df_full.get("target_price"), errors="coerce").values.astype(float)
        gate_proxy = pd.to_numeric(df_full.get("ewm_vol"), errors="coerce").fillna(0.001).values.astype(float)
        gate = fit_intraday_activity_gate(close_full, target_price, gate_proxy, instrument_key=instrument_key, min_active_floor=0.05)
    except Exception:
        pass
    try:
        regime_payload = train_regime(df_full)
    except Exception:
        regime_payload = None

    def _fit_payload(feats, ctor, *, kind: str, store_as_model: bool) -> dict:
        feats = [f for f in list(feats or []) if f in df_full.columns]
        if not feats:
            return {}
        X = df_full[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        calibration = {"scale": 0.0, "bias": 0.0, "selected": "anchor_only"}
        cal_sig = {"Weak": 0.5, "Medium": 0.5, "Strong": 0.5}
        try:
            cal_model = ctor()
            _, cal_sig = _build_calibration(cal_model, X, y, df_full, None)
        except Exception:
            cal_sig = {"Weak": 0.5, "Medium": 0.5, "Strong": 0.5}
        try:
            cal_len = max(24, len(X) // 5)
            cal_start = len(X) - cal_len
            if cal_start < max(40, int(len(X) * 0.60)):
                cal_start = max(40, int(len(X) * 0.75))
            if cal_start < len(X) - 8:
                X_core = X.iloc[:cal_start]
                X_cal = X.iloc[cal_start:]
                y_core = y.iloc[:cal_start]
                if len(X_core) >= 40 and len(X_cal) >= 12:
                    core_model = ctor()
                    try:
                        sw_core = _compute_recency_weights(X_core.index, half_life_days=7.0)
                        core_model.fit(X_core, y_core, sample_weight=sw_core)
                    except Exception:
                        core_model.fit(X_core, y_core)
                    pred_cal_norm = np.asarray(core_model.predict(X_cal), dtype=float)
                    pred_cal_norm = pred_cal_norm + float(target_mean)
                    vol_cal = np.maximum(
                        pd.to_numeric(df_full.get("ewm_vol"), errors="coerce").iloc[cal_start:].fillna(0.001).values.astype(float),
                        1e-6,
                    )
                    pred_cal_raw = pred_cal_norm * vol_cal
                    actual_cal_raw = pd.to_numeric(df_full.get("target_ret"), errors="coerce").iloc[cal_start:].fillna(0.0).values.astype(float)
                    close_cal = pd.to_numeric(df_full.get("close"), errors="coerce").iloc[cal_start:].ffill().bfill().values.astype(float)
                    calibration = fit_return_calibration(
                        pred_cal_raw,
                        actual_cal_raw,
                        close_cal,
                        horizon="intraday",
                        instrument_key=instrument_key,
                        allow_flip=True,
                        min_samples=max(12, len(X_cal) // 2),
                    )
        except Exception:
            calibration = {"scale": 0.0, "bias": 0.0, "selected": "anchor_only"}
        model = ctor()
        try:
            model.fit(X, y, sample_weight=weights)
        except Exception:
            model.fit(X, y)
        try:
            preds = np.asarray(model.predict(X), dtype=float)
            resid_std = max(1e-6, float(np.std(y.values - preds)))
            pred_bias = float(np.mean(preds - y.values))
        except Exception:
            resid_std = 1e-6
            pred_bias = 0.0
        payload = {
            "features": feats,
            "calibration": calibration,
            "calibration_sig": cal_sig,
            "regime_models": {},
            "ensemble_models": {},
            "window_calibration": {},
            "performance": {},
            "performance_mae": {},
            "meta": {
                "resid_std": float(resid_std),
                "pred_bias": float(pred_bias),
                "version": "backtest-fast",
                "dropped_features": {},
                "target_norm": True,
                "demeaned": True,
                "target_mean": float(target_mean),
                "predictability_gate": gate,
                "mode": "4min",
            },
        }
        if store_as_model:
            payload["model"] = model
        else:
            payload["model"] = None
            payload["fallback_model"] = model
        return payload

    def _fit_horizon_payload(df_h: pd.DataFrame, feats, ctor, *, store_as_model: bool) -> dict:
        feats = [f for f in list(feats or []) if f in df_h.columns]
        if not feats:
            return {}
        target_col_h = "target_ret_norm"
        try:
            target_mean_h = float(np.nanmean(df_h[target_col_h]))
        except Exception:
            target_mean_h = 0.0
        y_h = pd.to_numeric(df_h[target_col_h], errors="coerce").fillna(0.0) - float(target_mean_h)
        X_h = df_h[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if X_h.empty:
            return {}
        try:
            weights_h = _compute_recency_weights(df_h.index, half_life_days=9.0)
        except Exception:
            weights_h = None
        model_h = ctor()
        try:
            if weights_h is not None:
                model_h.fit(X_h, y_h, sample_weight=weights_h)
            else:
                model_h.fit(X_h, y_h)
        except Exception:
            try:
                model_h.fit(X_h, y_h)
            except Exception:
                return {}
        try:
            preds_h = np.asarray(model_h.predict(X_h), dtype=float)
            resid_std_h = max(1e-6, float(np.std(y_h.values - preds_h)))
            pred_bias_h = float(np.mean(preds_h - y_h.values))
        except Exception:
            resid_std_h = 1e-6
            pred_bias_h = 0.0
        payload_h = {
            "features": feats,
            "ensemble_models": {},
            "regime_models": {},
            "window_calibration": {},
            "performance": {},
            "performance_mae": {},
            "meta": {
                "resid_std": float(resid_std_h),
                "pred_bias": float(pred_bias_h),
                "version": "backtest-fast-horizon",
                "target_norm": True,
                "demeaned": True,
                "target_mean": float(target_mean_h),
            },
        }
        if store_as_model:
            payload_h["model"] = model_h
        else:
            payload_h["model"] = None
            payload_h["fallback_model"] = model_h
        return payload_h

    models = {
        "base": _fit_payload(base_feats, _ridge_pipeline, kind="base", store_as_model=True),
        "forest": _fit_payload(forest_feats, _forest_ctor, kind="forest", store_as_model=False),
        "boost": _fit_payload(boost_feats, _boost_ctor, kind="boost", store_as_model=False),
        "extra_trees": _fit_payload(extra_feats, _extra_trees_ctor, kind="extra_trees", store_as_model=False),
    }
    models["base"].setdefault("meta", {})
    models["base"]["meta"]["dropped_features"] = base_diag
    models["forest"].setdefault("meta", {})
    models["forest"]["meta"]["dropped_features"] = forest_diag
    models["boost"].setdefault("meta", {})
    models["boost"]["meta"]["dropped_features"] = boost_diag
    models["extra_trees"].setdefault("meta", {})
    models["extra_trees"]["meta"]["dropped_features"] = extra_diag
    for key in ("base", "forest", "boost", "extra_trees"):
        models[key].setdefault("horizon_models", {})
    if train_horizons:
        try:
            horizons = [int(h) for h in list(HORIZON_MINUTES or []) if int(h) > 4]
        except Exception:
            horizons = [10, 30]
        for horizon in horizons:
            try:
                df_h = build_features_for_training(train_df, predict_minutes=int(horizon), instrument_key=instrument_key)
            except Exception:
                df_h = None
            if df_h is None or df_h.empty:
                continue
            try:
                df_h = df_h.replace([np.inf, -np.inf], np.nan).dropna()
            except Exception:
                pass
            if len(df_h) < 200:
                continue
            base_h_feats, _ = _resolve_active_features(df_h, DEFAULT_FEATURES)
            # Keep replay practical: use a lightweight ridge horizon model as the
            # supplementary stock-horizon signal, while live/prod artifacts still
            # retain the fuller per-model horizon stack.
            models["base"]["horizon_models"][str(horizon)] = _fit_horizon_payload(
                df_h,
                base_h_feats,
                _ridge_pipeline,
                store_as_model=True,
            )
    try:
        split_n = 4 if len(df_full) >= 600 else 3 if len(df_full) >= 320 else 2
        tss = TimeSeriesSplit(n_splits=max(2, split_n))
        oof = {k: np.full(len(df_full), np.nan) for k in ("base", "forest", "boost", "extra_trees")}
        feature_map = {
            "base": base_feats,
            "forest": forest_feats,
            "boost": boost_feats,
            "extra_trees": extra_feats,
        }
        ctor_map = {
            "base": _ridge_pipeline,
            "forest": _forest_ctor,
            "boost": _boost_ctor,
            "extra_trees": _extra_trees_ctor,
        }
        for train_idx, test_idx in tss.split(df_full):
            train_fold = df_full.iloc[train_idx]
            test_fold = df_full.iloc[test_idx]
            y_train = y.iloc[train_idx]
            for key, ctor in ctor_map.items():
                feats = [f for f in list(feature_map.get(key) or []) if f in train_fold.columns]
                if not feats:
                    continue
                X_train = train_fold[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                X_test = test_fold[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
                if X_train.empty or X_test.empty:
                    continue
                try:
                    sw_train = _compute_recency_weights(X_train.index, half_life_days=7.0)
                except Exception:
                    sw_train = None
                model = ctor()
                try:
                    if sw_train is not None:
                        model.fit(X_train, y_train, sample_weight=sw_train)
                    else:
                        model.fit(X_train, y_train)
                except Exception:
                    try:
                        model.fit(X_train, y_train)
                    except Exception:
                        continue
                try:
                    oof[key][test_idx] = np.asarray(model.predict(X_test), dtype=float)
                except Exception:
                    continue
        meta_df = pd.DataFrame(oof, index=df_full.index)
        meta_df["ewm_vol"] = pd.to_numeric(df_full.get("ewm_vol"), errors="coerce").fillna(0.0).values
        meta_df["trend_strength"] = pd.to_numeric(df_full.get("trend_strength"), errors="coerce").fillna(0.0).values
        meta_df = meta_df.dropna()
        if len(meta_df) >= 120:
            y_meta = y.loc[meta_df.index]
            meta_model = _ridge_pipeline()
            try:
                sw_meta = _compute_recency_weights(meta_df.index, half_life_days=7.0)
                meta_model.fit(meta_df, y_meta, sample_weight=sw_meta)
            except Exception:
                meta_model.fit(meta_df, y_meta)
            preds_meta = np.asarray(meta_model.predict(meta_df), dtype=float)
            meta_resid = max(1e-6, float(np.std(y_meta.values - preds_meta)))
            models["meta"] = {
                "model": meta_model,
                "features": list(meta_df.columns),
                "meta": {
                    "resid_std": float(meta_resid),
                    "version": "backtest-fast-meta",
                    "target_norm": True,
                    "demeaned": True,
                    "target_mean": float(target_mean),
                    "predictability_gate": gate,
                    "mode": "4min",
                },
            }
    except Exception:
        pass
    if regime_payload is not None:
        models["regime"] = regime_payload
    return models


def _build_records_for_fold(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    instrument_key: str,
    stock_horizon_blend: bool = True,
    max_records: int | None = None,
) -> dict:
    models = _train_fold_models_fast(train_df, instrument_key, train_horizons=stock_horizon_blend)
    if not isinstance(models, dict) or not models.get("regime"):
        return {"status": "error", "reason": "4min fold training failed", "records": [], "models": None}
    records = []
    symbol = str(instrument_key or "QQQ").upper()
    daily_full = _load_truncated_daily_history(symbol, None)
    news_store = _load_historical_news_store(symbol)
    feature_frame_full = build_feature_frame(full_df.copy(), instrument_key=symbol)
    eval_step = max(1, int(round(4.0 / max(1.0, float(BAR_MINUTES)))))
    max_count = None
    if max_records is not None:
        try:
            max_count = max(0, int(max_records))
        except Exception:
            max_count = None
    for ts_value in list(test_df.index)[::eval_step]:
        if max_count is not None and len(records) >= max_count:
            break
        prefix = full_df.loc[:ts_value].tail(2000).copy()
        if prefix.empty or len(prefix) < 160:
            continue
        as_of = _coerce_utc_ts(ts_value) + pd.Timedelta(minutes=max(1, int(BAR_MINUTES)))
        replay_context = _historical_replay_context(prefix, as_of, instrument_key=symbol)
        feature_frame = feature_frame_full.loc[feature_frame_full.index <= ts_value].tail(2000).copy() if feature_frame_full is not None and not feature_frame_full.empty else None
        predictions_intraday = predict_4min(
            models,
            prefix,
            instrument_key=symbol,
            log_side_effects=False,
            now_utc=as_of,
            use_live_artifacts=False,
            readiness_override=_readiness_override(),
            live_calibration_override={},
            feature_frame_override=feature_frame,
            stock_horizon_blend=stock_horizon_blend,
        )
        if not isinstance(predictions_intraday, dict) or predictions_intraday.get("error"):
            continue
        predictions = {"intraday": predictions_intraday}
        news_items = _cached_recent_headlines(news_store, as_of=as_of, limit=8)
        news_health = _cached_news_health(news_store, trading_dates=prefix.index, as_of=as_of)
        event_snapshot = _news_event_snapshot(news_items, as_of)
        catalyst_snapshot = _news_catalyst_snapshot(news_items, as_of)
        cutoff_date = pd.Timestamp(as_of.tz_convert("America/New_York").date())
        daily_slice = daily_full[daily_full.index <= cutoff_date].copy() if not daily_full.empty else pd.DataFrame()
        liquidity_snapshot = _historical_liquidity_snapshot(prefix, daily_slice)
        advisory = _controller_advisory_snapshot(
            prefix,
            symbol,
            predictions,
            news_items,
            news_health,
            event_snapshot,
            _neutral_macro_snapshot(),
            _neutral_breadth_snapshot(),
            liquidity_snapshot,
            catalyst_snapshot,
            as_of,
            daily_slice,
            replay_context=replay_context,
        )
        records.append(
            {
                "ts": _coerce_utc_ts(ts_value),
                "as_of": as_of,
                "market_df": _row_frame(test_df, ts_value),
                "current_price": _safe_float(test_df.loc[ts_value, "close"]),
                "advisory": advisory,
                "predictions": predictions,
                "event_snapshot": event_snapshot,
                "catalyst_snapshot": catalyst_snapshot,
                "news_health": news_health,
                "news_items": news_items,
                "historical_freshness": replay_context,
                "regime_label": _regime_label(predictions),
                "time_bucket": _time_bucket(as_of),
            }
        )
    return {"status": "ok", "reason": "", "records": records, "models": models}


def _portfolio_equity(state: dict, symbol: str, current_price: float | None) -> float:
    cash = float(_safe_float((state or {}).get("cash")) or 0.0)
    positions = dict((state or {}).get("positions") or {})
    pos = dict(positions.get(str(symbol or "").upper()) or {})
    qty = float(_safe_float(pos.get("qty")) or 0.0)
    px = float(_safe_float(current_price) or _safe_float(pos.get("avg_price")) or 0.0)
    return float(cash + (qty * px))


def _build_execution_snapshot(
    state: dict,
    symbol: str,
    current_price: float | None,
    advisory: dict,
    as_of: pd.Timestamp,
    profile_name: str | None = None,
) -> dict:
    rec = recommend_order_for_state(
        state,
        symbol,
        current_price=current_price,
        advisory=advisory,
        profile_name=profile_name,
    )
    equity = _portfolio_equity(state, symbol, current_price)
    cash = float(_safe_float((state or {}).get("cash")) or 0.0)
    return {
        "broker_mode": "local_paper",
        "route_label": "Backtest Paper",
        "runtime": {
            "routing_mode": "local_paper",
            "route_label": "Backtest Paper",
            "live_snapshot": {
                "account": {"equity": equity, "cash": cash},
                "fetched_at": _coerce_utc_ts(as_of).isoformat(),
            },
        },
        "diagnostics": {"checked_at": _coerce_utc_ts(as_of).isoformat()},
        "portfolio": {"equity": equity, "cash": cash},
        "state": state,
        "recommendation": rec,
    }


def _fill_price(raw_price: float, side: str, intent: str) -> float:
    px = float(max(0.01, raw_price))
    slip = abs(float(SLIPPAGE_PCT))
    if str(intent).lower() == "entry":
        return float(px * (1.0 + slip)) if str(side).lower() == "buy" else float(px * (1.0 - slip))
    return float(px * (1.0 - slip)) if str(side).lower() == "buy" else float(px * (1.0 + slip))


def _apply_entry(
    state: dict,
    symbol: str,
    side: str,
    qty: int,
    raw_price: float,
    as_of: pd.Timestamp,
    meta: dict,
) -> dict:
    out = dict(state or {})
    positions = dict(out.get("positions") or {})
    fill_px = _fill_price(raw_price, side, "entry")
    signed_qty = int(qty) if str(side).lower() == "buy" else -int(qty)
    fee = abs(float(fill_px * qty * float(TRANSACTION_COST_PER_TRADE)))
    cash = float(_safe_float(out.get("cash")) or 0.0)
    cash = cash - (fill_px * qty) - fee if signed_qty > 0 else cash + (fill_px * qty) - fee
    out["cash"] = float(cash)
    positions[str(symbol).upper()] = {
        "qty": float(signed_qty),
        "avg_price": float(fill_px),
        "opened_at": _coerce_utc_ts(as_of).isoformat(),
        "stop_price": _safe_float(meta.get("stop_price")),
        "target_price": _safe_float(meta.get("target_price")),
    }
    out["positions"] = positions
    out["updated_at"] = _coerce_utc_ts(as_of).isoformat()
    return {
        "state": out,
        "position": {
            "symbol": str(symbol).upper(),
            "side": str(side).lower(),
            "qty": int(qty),
            "signed_qty": int(signed_qty),
            "entry_time": _coerce_utc_ts(as_of),
            "entry_price_raw": float(raw_price),
            "entry_price_fill": float(fill_px),
            "entry_fee": float(fee),
            **dict(meta or {}),
        },
    }


def _apply_exit(
    state: dict,
    position: dict,
    raw_price: float,
    as_of: pd.Timestamp,
    reason: str,
) -> dict:
    out = dict(state or {})
    positions = dict(out.get("positions") or {})
    side = str(position.get("side") or "buy").lower()
    qty = int(abs(_safe_float(position.get("qty")) or 0))
    fill_px = _fill_price(raw_price, side, "exit")
    fee = abs(float(fill_px * qty * float(TRANSACTION_COST_PER_TRADE)))
    cash = float(_safe_float(out.get("cash")) or 0.0)
    if side == "buy":
        cash += (fill_px * qty) - fee
    else:
        cash -= (fill_px * qty) + fee
    out["cash"] = float(cash)
    positions.pop(str(position.get("symbol") or "").upper(), None)
    out["positions"] = positions
    out["updated_at"] = _coerce_utc_ts(as_of).isoformat()
    entry_px = float(_safe_float(position.get("entry_price_fill")) or 0.0)
    gross_pnl = ((float(fill_px) - entry_px) * qty) if side == "buy" else ((entry_px - float(fill_px)) * qty)
    total_fees = float(_safe_float(position.get("entry_fee")) or 0.0) + fee
    net_pnl = float(gross_pnl - total_fees)
    entry_notional = max(0.01, abs(entry_px * qty))
    return {
        "state": out,
        "trade": {
            **dict(position or {}),
            "exit_time": _coerce_utc_ts(as_of),
            "exit_price_raw": float(raw_price),
            "exit_price_fill": float(fill_px),
            "exit_fee": float(fee),
            "gross_pnl": float(gross_pnl),
            "net_pnl": float(net_pnl),
            "total_fees": float(total_fees),
            "return_pct": float(net_pnl / entry_notional),
            "hold_minutes": float(
                max(0.0, (_coerce_utc_ts(as_of) - _coerce_utc_ts(position.get("entry_time"))).total_seconds() / 60.0)
            ) if _coerce_utc_ts(position.get("entry_time")) is not None else None,
            "exit_reason": str(reason or "-"),
        },
    }


def _mark_equity(state: dict, symbol: str, current_price: float | None, as_of: pd.Timestamp) -> dict:
    return {
        "ts": _coerce_utc_ts(as_of),
        "equity": _portfolio_equity(state, symbol, current_price),
    }


@contextmanager
def _autonomy_overrides(overrides: dict | None = None):
    mapping = dict(overrides or {})
    if not mapping:
        yield
        return
    previous = {}
    try:
        for key, value in mapping.items():
            previous[key] = getattr(autonomy, key)
            setattr(autonomy, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(autonomy, key, value)


@contextmanager
def _cached_news_feature_loader(symbol: str, index_utc: pd.DatetimeIndex):
    old_loader = getattr(feature_module, "load_news_features_for_index", None)
    cached = None
    try:
        cached = load_news_features_for_index(symbol=symbol, index_utc=index_utc, tz="America/New_York")
    except Exception:
        cached = None

    def _loader(symbol: str, index_utc: pd.DatetimeIndex, tz: str = "America/New_York"):
        idx = pd.DatetimeIndex(pd.to_datetime(index_utc, utc=True, errors="coerce"))
        if cached is None or cached.empty:
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
            return pd.DataFrame(0.0, index=idx, columns=cols)
        return cached.reindex(idx).fillna(0.0)

    try:
        feature_module.load_news_features_for_index = _loader
        yield
    finally:
        feature_module.load_news_features_for_index = old_loader


def simulate_controller_records(
    records: list[dict] | None,
    instrument_key: str = "QQQ",
    profile_name: str | None = None,
    starting_cash: float = PAPER_STARTING_CASH,
    overrides: dict | None = None,
) -> dict:
    symbol = str(instrument_key or "QQQ").upper()
    controller_state = autonomy._default_state(symbol)
    controller_state["mode"] = "shadow"
    sim_state = new_paper_state(starting_cash=starting_cash, broker_mode="backtest_shadow")
    open_position = None
    trades = []
    equity_curve = []
    block_counts = Counter()
    block_reason_counts = Counter()
    block_combo_counts = Counter()
    action_counts = Counter()
    entry_tags = defaultdict(int)
    daily_realized = defaultdict(float)
    with _autonomy_overrides(overrides):
        for record in list(records or []):
            advisory = dict(record.get("advisory") or {})
            as_of = _coerce_utc_ts(record.get("as_of"))
            current_price = _safe_float(record.get("current_price"))
            if as_of is None or current_price is None:
                continue
            exec_snapshot = _build_execution_snapshot(sim_state, symbol, current_price, advisory, as_of, profile_name=profile_name)
            snapshot = autonomy.evaluate_autonomy_cycle(
                symbol,
                record.get("market_df"),
                advisory=advisory,
                execution_snapshot=exec_snapshot,
                profile_name=profile_name,
                replay_snapshot={"active": False},
                options_snapshot={"status": "historical_unavailable"},
                now_utc=as_of,
                state_override=controller_state,
                persist=False,
                readiness_override={"live_ready": True, "fresh": True},
            )
            controller_state = dict(snapshot.get("state") or controller_state)
            decision = dict(snapshot.get("decision") or {})
            stock_path = dict(snapshot.get("stock_path") or {})
            action_counts[str(decision.get("status") or "idle")] += 1
            if not stock_path.get("allowed"):
                label = str(stock_path.get("label") or "Blocked")
                block_counts[label] += 1
                reasons = [str(r) for r in list(stock_path.get("reasons") or []) if str(r).strip()]
                for reason in reasons:
                    block_reason_counts[reason] += 1
                if reasons:
                    block_combo_counts[" | ".join(reasons[:3])] += 1
            if str(decision.get("status") or "") == "shadow_entry" and open_position is None:
                meta = {
                    "entry_bucket": str(record.get("time_bucket") or "-"),
                    "entry_regime": str(record.get("regime_label") or "-"),
                    "quality_score": _safe_float((advisory.get("quality_meta") or {}).get("score")),
                    "edge_bps": _safe_float(((advisory.get("pred_state") or {}).get("intraday") or {}).get("edge_bps")),
                    "news_alignment": str((advisory.get("news_meta") or {}).get("alignment") or "neutral"),
                    "entry_reason": str(stock_path.get("detail") or ""),
                    "target_price": _safe_float((advisory.get("plan_meta") or {}).get("target1")),
                    "stop_price": _safe_float((advisory.get("plan_meta") or {}).get("stop")),
                }
                entry = _apply_entry(
                    sim_state,
                    symbol,
                    stock_path.get("side"),
                    max(1, int(stock_path.get("qty") or 0)),
                    current_price,
                    as_of,
                    meta,
                )
                sim_state = entry["state"]
                open_position = entry["position"]
                entry_tags[f"{meta['entry_bucket']}|{meta['entry_regime']}|{meta['news_alignment']}"] += 1
            elif str(decision.get("status") or "") == "shadow_exit" and open_position is not None:
                exit_event = next((evt for evt in list(snapshot.get("events") or []) if str(evt.get("kind") or "") == "shadow_exit"), {})
                reason = str(exit_event.get("reason") or decision.get("detail") or "controller exit")
                closed = _apply_exit(sim_state, open_position, current_price, as_of, reason)
                sim_state = closed["state"]
                trade = closed["trade"]
                trades.append(trade)
                session_key = str(controller_state.get("session_date") or "")
                daily_realized[session_key] += float(trade.get("net_pnl") or 0.0)
                controller_state["shadow_realized_pnl_today"] = float(daily_realized[session_key])
                if float(trade.get("net_pnl") or 0.0) < 0.0:
                    controller_state["cooldown_until"] = (
                        as_of + pd.Timedelta(minutes=int(autonomy.AUTONOMY_LOSS_COOLDOWN_MIN))
                    ).isoformat()
                open_position = None
            equity_curve.append(_mark_equity(sim_state, symbol, current_price, as_of))

    if open_position is not None and records:
        last_record = dict(records[-1])
        as_of = _coerce_utc_ts(last_record.get("as_of"))
        current_price = _safe_float(last_record.get("current_price"))
        if as_of is not None and current_price is not None:
            closed = _apply_exit(sim_state, open_position, current_price, as_of, "fold end")
            sim_state = closed["state"]
            trade = closed["trade"]
            trades.append(trade)
            equity_curve.append(_mark_equity(sim_state, symbol, current_price, as_of))
            open_position = None

    return {
        "trades": trades,
        "equity_curve": equity_curve,
        "block_counts": dict(block_counts),
        "block_reason_counts": dict(block_reason_counts),
        "block_combo_counts": dict(block_combo_counts),
        "action_counts": dict(action_counts),
        "entry_tags": dict(entry_tags),
        "ending_equity": _portfolio_equity(sim_state, symbol, _safe_float((records[-1] or {}).get("current_price")) if records else None),
        "starting_cash": float(starting_cash),
    }


def _max_drawdown(equity_curve: list[dict]) -> float | None:
    values = [float(_safe_float(item.get("equity")) or 0.0) for item in list(equity_curve or [])]
    if not values:
        return None
    peak = values[0]
    max_dd = 0.0
    for value in values:
        peak = max(peak, value)
        if peak > 0:
            max_dd = min(max_dd, (value / peak) - 1.0)
    return float(max_dd)


def _aggregate_trade_bucket(trades: list[dict], key: str) -> dict:
    out = {}
    grouped = defaultdict(list)
    for trade in list(trades or []):
        grouped[str(trade.get(key) or "-")].append(trade)
    for bucket, items in grouped.items():
        pnl = sum(float(_safe_float(item.get("net_pnl")) or 0.0) for item in items)
        out[bucket] = {
            "trades": len(items),
            "net_pnl": float(pnl),
            "win_rate": float(sum(1 for item in items if float(_safe_float(item.get("net_pnl")) or 0.0) > 0.0) / len(items)),
        }
    return out


def summarize_simulation(sim: dict | None) -> dict:
    sim = dict(sim or {})
    trades = list(sim.get("trades") or [])
    equity_curve = list(sim.get("equity_curve") or [])
    gross_pnl = float(sum(float(_safe_float(t.get("gross_pnl")) or 0.0) for t in trades))
    net_pnl = float(sum(float(_safe_float(t.get("net_pnl")) or 0.0) for t in trades))
    total_fees = float(sum(float(_safe_float(t.get("total_fees")) or 0.0) for t in trades))
    long_trades = [t for t in trades if str(t.get("side") or "").lower() == "buy"]
    short_trades = [t for t in trades if str(t.get("side") or "").lower() == "sell"]
    wins = [t for t in trades if float(_safe_float(t.get("net_pnl")) or 0.0) > 0.0]
    losses = [t for t in trades if float(_safe_float(t.get("net_pnl")) or 0.0) <= 0.0]
    avg_trade = float(net_pnl / len(trades)) if trades else None
    avg_trade_ret = None
    if trades:
        avg_trade_ret = float(np.mean([float(_safe_float(t.get("return_pct")) or 0.0) for t in trades]))
    profit_factor = None
    loss_total = abs(sum(float(_safe_float(t.get("net_pnl")) or 0.0) for t in losses))
    if loss_total > 0:
        profit_factor = float(sum(float(_safe_float(t.get("net_pnl")) or 0.0) for t in wins) / loss_total)
    expectancy = avg_trade
    start_cash = float(_safe_float(sim.get("starting_cash")) or PAPER_STARTING_CASH)
    end_equity = float(_safe_float(sim.get("ending_equity")) or start_cash)
    return {
        "trade_count": len(trades),
        "gross_pnl": gross_pnl,
        "net_pnl": net_pnl,
        "fees": total_fees,
        "ending_equity": end_equity,
        "return_pct": float((end_equity / max(1.0, start_cash)) - 1.0),
        "win_rate": float(len(wins) / len(trades)) if trades else None,
        "avg_trade_pnl": avg_trade,
        "avg_trade_return_pct": avg_trade_ret,
        "expectancy": expectancy,
        "max_drawdown_pct": _max_drawdown(equity_curve),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "long_net_pnl": float(sum(float(_safe_float(t.get("net_pnl")) or 0.0) for t in long_trades)),
        "short_net_pnl": float(sum(float(_safe_float(t.get("net_pnl")) or 0.0) for t in short_trades)),
        "long_win_rate": float(sum(1 for t in long_trades if float(_safe_float(t.get("net_pnl")) or 0.0) > 0.0) / len(long_trades)) if long_trades else None,
        "short_win_rate": float(sum(1 for t in short_trades if float(_safe_float(t.get("net_pnl")) or 0.0) > 0.0) / len(short_trades)) if short_trades else None,
        "avg_hold_minutes": float(np.mean([float(_safe_float(t.get("hold_minutes")) or 0.0) for t in trades])) if trades else None,
        "profit_factor": profit_factor,
        "time_of_day": _aggregate_trade_bucket(trades, "entry_bucket"),
        "regimes": _aggregate_trade_bucket(trades, "entry_regime"),
        "news_alignment": _aggregate_trade_bucket(trades, "news_alignment"),
        "exit_reasons": dict(Counter(str(t.get("exit_reason") or "-") for t in trades)),
        "block_counts": dict(sim.get("block_counts") or {}),
        "block_reason_counts": dict(sim.get("block_reason_counts") or {}),
        "block_combo_counts": dict(sim.get("block_combo_counts") or {}),
        "action_counts": dict(sim.get("action_counts") or {}),
    }


def summarize_record_diagnostics(records: list[dict] | None) -> dict:
    rows = []
    for record in list(records or []):
        advisory = dict(record.get("advisory") or {})
        intr = dict((advisory.get("pred_state") or {}).get("intraday") or {})
        quality = dict(advisory.get("quality_meta") or {})
        news_meta = dict(advisory.get("news_meta") or {})
        pred_meta = dict((((record.get("predictions") or {}).get("intraday") or {}).get("_meta") or {}).get("predictability_gate") or {})
        rows.append(
            {
                "direction": str(intr.get("direction") or "-").upper(),
                "direction_stage": str(intr.get("direction_stage") or "blocked").lower(),
                "raw_direction": str(intr.get("raw_direction") or "-").upper(),
                "raw_pred_bps": float(_safe_float(intr.get("raw_pred_ret_bps")) or 0.0),
                "raw_edge_bps": float(_safe_float(intr.get("raw_edge_bps")) or 0.0),
                "edge_bps": float(_safe_float(intr.get("edge_bps")) or 0.0),
                "conf": float(_safe_float(intr.get("confidence")) or 0.0),
                "raw_conf": float(_safe_float(intr.get("raw_confidence")) or 0.0),
                "size": float(_safe_float(intr.get("position_size")) or 0.0),
                "raw_size": float(_safe_float(intr.get("raw_position_size")) or 0.0),
                "unc_bps": float(_safe_float(intr.get("uncertainty_bps")) or 0.0),
                "horizon_stage": str(intr.get("horizon_support_stage") or "unavailable").lower(),
                "horizon_score": float(_safe_float(intr.get("horizon_support_score")) or 0.0),
                "quality": float(_safe_float(quality.get("score")) or 0.0),
                "no_trade": bool(quality.get("no_trade")),
                "news_conflict": str(news_meta.get("alignment") or "").lower() == "conflict",
                "news_degraded": str(news_meta.get("headline_state") or "").lower() in {"failed", "stale"},
                "predictable": bool(pred_meta.get("predictable_now")),
            }
        )
    if not rows:
        return {
            "evaluations": 0,
            "direction_non_dash": 0,
            "raw_direction_non_dash": 0,
            "tentative_direction_non_dash": 0,
            "horizon_supportive": 0,
            "horizon_confirmed": 0,
            "raw_edge_positive": 0,
            "finished_edge_positive": 0,
            "positive_edge_blanked": 0,
            "size_positive": 0,
            "quality_above_action": 0,
            "confidence_above_autonomy": 0,
            "predictable_now": 0,
            "no_trade": 0,
            "news_conflict": 0,
            "news_degraded": 0,
            "means": {},
        }
    frame = pd.DataFrame(rows)
    return {
        "evaluations": int(len(frame)),
        "direction_non_dash": int(frame["direction"].isin(["UP", "DOWN"]).sum()),
        "raw_direction_non_dash": int(frame["raw_direction"].isin(["UP", "DOWN"]).sum()),
        "tentative_direction_non_dash": int(((frame["direction_stage"] == "tentative") & frame["direction"].isin(["UP", "DOWN"])).sum()),
        "horizon_supportive": int(frame["horizon_stage"].isin(["supportive", "confirmed"]).sum()),
        "horizon_confirmed": int((frame["horizon_stage"] == "confirmed").sum()),
        "raw_edge_positive": int((frame["raw_edge_bps"] > 0.0).sum()),
        "finished_edge_positive": int((frame["edge_bps"] > 0.0).sum()),
        "positive_edge_blanked": int(((frame["edge_bps"] > 0.0) & (~frame["direction"].isin(["UP", "DOWN"]))).sum()),
        "size_positive": int((frame["size"] > 0.0).sum()),
        "quality_above_action": int((frame["quality"] >= 0.58).sum()),
        "confidence_above_autonomy": int((frame["conf"] >= float(autonomy.AUTONOMY_MIN_CONFIDENCE)).sum()),
        "predictable_now": int(frame["predictable"].sum()),
        "no_trade": int(frame["no_trade"].sum()),
        "news_conflict": int(frame["news_conflict"].sum()),
        "news_degraded": int(frame["news_degraded"].sum()),
        "means": {
            "raw_pred_bps": float(frame["raw_pred_bps"].mean()),
            "raw_edge_bps": float(frame["raw_edge_bps"].mean()),
            "edge_bps": float(frame["edge_bps"].mean()),
            "confidence": float(frame["conf"].mean()),
            "raw_confidence": float(frame["raw_conf"].mean()),
            "position_size": float(frame["size"].mean()),
            "raw_position_size": float(frame["raw_size"].mean()),
            "uncertainty_bps": float(frame["unc_bps"].mean()),
            "horizon_support_score": float(frame["horizon_score"].mean()),
            "quality": float(frame["quality"].mean()),
        },
    }


def build_replay_experience_records(
    records: list[dict] | None,
    instrument_key: str = "QQQ",
    source: str = "backtest",
    horizon: str = "4min",
    interval: str | None = None,
    max_records: int | None = None,
) -> list[dict]:
    symbol = str(instrument_key or "QQQ").upper()
    items = list(records or [])
    if max_records is not None:
        try:
            items = items[: max(0, int(max_records))]
        except Exception:
            pass
    out = []
    round_trip_cost = 2.0 * (float(SLIPPAGE_PCT) + float(TRANSACTION_COST_PER_TRADE))
    for idx, record in enumerate(items):
        advisory = dict(record.get("advisory") or {})
        predictions = dict(record.get("predictions") or {})
        prediction_meta = dict(((predictions.get("intraday") or {}).get("_meta") or {}))
        deployment = dict(advisory.get("deployment_meta") or {})
        validation = dict(deployment.get("validation_evidence") or {})
        experience = build_experience_record(
            symbol=symbol,
            as_of=record.get("as_of") or record.get("ts"),
            source=source,
            horizon=horizon,
            interval=interval or f"{int(BAR_MINUTES)}m",
            predictions=predictions,
            advisory=advisory,
            risk_summary=advisory.get("risk_meta"),
            validation_evidence=validation,
            calibration_state={
                "staleness": prediction_meta.get("staleness"),
                "critical_stale": prediction_meta.get("critical_stale"),
                "readiness_locked": prediction_meta.get("readiness_locked"),
                "signal_diagnostics": prediction_meta.get("signal_diagnostics"),
            },
            market_context={
                "current_price": _safe_float(record.get("current_price")),
                "time_bucket": record.get("time_bucket"),
                "regime_label": record.get("regime_label"),
                "historical_freshness": record.get("historical_freshness"),
            },
            news_context=advisory.get("news_meta"),
            event_context=record.get("event_snapshot"),
            macro_context=advisory.get("macro_meta"),
        )
        next_record = items[idx + 1] if idx + 1 < len(items) else None
        if isinstance(next_record, dict):
            as_of = _coerce_utc_ts(record.get("as_of") or record.get("ts"))
            next_as_of = _coerce_utc_ts(next_record.get("as_of") or next_record.get("ts"))
            if as_of is not None and next_as_of is not None and next_as_of > as_of:
                experience = resolve_one_step_outcome(
                    experience,
                    current_price=_safe_float(record.get("current_price")),
                    future_price=_safe_float(next_record.get("current_price")),
                    future_as_of=next_as_of,
                    cost_pct=round_trip_cost,
                    horizon="1_step",
                )
        out.append(experience)
    return out


def _experience_horizon_label() -> str:
    value = HORIZON_MINUTES
    if isinstance(value, (list, tuple)) and value:
        value = value[0]
    try:
        return f"{int(value)}min"
    except Exception:
        return str(value or "4min")


def _mean(values):
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(sum(vals) / len(vals)) if vals else None


def summarize_folds(folds: list[dict] | None) -> dict:
    items = list(folds or [])
    if not items:
        return {
            "folds": 0,
            "mean_return_pct": None,
            "mean_net_pnl": None,
            "positive_fold_share": None,
            "mean_trade_count": None,
            "mean_win_rate": None,
            "mean_max_drawdown_pct": None,
            "long_pnl_total": None,
            "short_pnl_total": None,
            "stability_std_return": None,
        }
    rets = [item.get("summary", {}).get("return_pct") for item in items]
    ret_clean = [float(v) for v in rets if v is not None and np.isfinite(v)]
    mean_ret = _mean(ret_clean)
    std_ret = float(np.std(ret_clean)) if len(ret_clean) >= 2 else None
    return {
        "folds": len(items),
        "mean_return_pct": mean_ret,
        "mean_net_pnl": _mean([item.get("summary", {}).get("net_pnl") for item in items]),
        "positive_fold_share": float(sum(1 for v in ret_clean if v > 0.0) / len(ret_clean)) if ret_clean else None,
        "mean_trade_count": _mean([item.get("summary", {}).get("trade_count") for item in items]),
        "mean_win_rate": _mean([item.get("summary", {}).get("win_rate") for item in items]),
        "mean_max_drawdown_pct": _mean([item.get("summary", {}).get("max_drawdown_pct") for item in items]),
        "long_pnl_total": float(sum(float(_safe_float(item.get("summary", {}).get("long_net_pnl")) or 0.0) for item in items)),
        "short_pnl_total": float(sum(float(_safe_float(item.get("summary", {}).get("short_net_pnl")) or 0.0) for item in items)),
        "stability_std_return": std_ret,
    }


def _sensitivity_variants() -> list[tuple[str, dict]]:
    return [
        ("baseline", {}),
        ("quality_looser", {"AUTONOMY_MIN_QUALITY_SCORE": max(0.50, float(autonomy.AUTONOMY_MIN_QUALITY_SCORE) - 0.04)}),
        ("quality_tighter", {"AUTONOMY_MIN_QUALITY_SCORE": min(0.90, float(autonomy.AUTONOMY_MIN_QUALITY_SCORE) + 0.04)}),
        ("edge_looser", {"AUTONOMY_MIN_EDGE_BPS": max(0.5, float(autonomy.AUTONOMY_MIN_EDGE_BPS) - 1.0)}),
        ("edge_tighter", {"AUTONOMY_MIN_EDGE_BPS": float(autonomy.AUTONOMY_MIN_EDGE_BPS) + 1.0}),
        ("flat_later", {"AUTONOMY_FLAT_BEFORE_CLOSE_MIN": max(5, int(autonomy.AUTONOMY_FLAT_BEFORE_CLOSE_MIN) - 5)}),
    ]


def run_stock_autonomy_validation(
    df,
    instrument_key: str = "QQQ",
    n_splits: int = 5,
    holdout_folds: int = 1,
    profile_name: str | None = None,
    include_sensitivity: bool = True,
    stock_horizon_blend: bool = True,
    experience_ledger_path: str | None = None,
    return_experience_records: bool = False,
    max_replay_records: int | None = None,
) -> dict:
    symbol = str(instrument_key or "QQQ").upper()
    frame = _normalize_frame(df if df is not None else _load_default_intraday_frame(symbol))
    if frame.empty:
        return {
            "instrument": symbol,
            "path": "stock",
            "status": "error",
            "reason": "No intraday price history available",
            "folds": [],
            "walk_forward": summarize_folds([]),
            "holdout": summarize_folds([]),
            "sensitivity": {},
            "options_path": {
                "status": "unsupported",
                "reason": "Historical contract-level options data is not available here for a credible autonomy replay.",
            },
        }
    fold_specs = _fold_specs(frame, n_splits=max(2, int(n_splits)))
    fold_results = []
    experience_records_all = []
    experience_records_written = 0
    replay_records_seen = 0
    max_records = None
    if max_replay_records is not None:
        try:
            max_records = max(0, int(max_replay_records))
        except Exception:
            max_records = None
    with _cached_news_feature_loader(symbol, frame.index):
        for spec in fold_specs:
            remaining_records = None
            if max_records is not None:
                remaining_records = max_records - replay_records_seen
                if remaining_records <= 0:
                    break
            session_series = _session_dates(frame.index)
            train_mask = session_series.isin(spec["train_sessions"]).values
            test_mask = session_series.isin(spec["test_sessions"]).values
            train_df = frame.loc[train_mask].copy()
            test_df = frame.loc[test_mask].copy()
            built = _build_records_for_fold(
                frame.loc[: test_df.index.max()].copy(),
                train_df,
                test_df,
                symbol,
                stock_horizon_blend=stock_horizon_blend,
                max_records=remaining_records,
            )
            if built.get("status") != "ok":
                fold_results.append({**spec, "status": "error", "reason": built.get("reason"), "summary": {}})
                continue
            sim = simulate_controller_records(
                built.get("records"),
                instrument_key=symbol,
                profile_name=profile_name,
                starting_cash=PAPER_STARTING_CASH,
            )
            summary = summarize_simulation(sim)
            experience_records = build_replay_experience_records(
                built.get("records"),
                instrument_key=symbol,
                source="backtest",
                horizon=_experience_horizon_label(),
                interval=f"{int(BAR_MINUTES)}m",
                max_records=remaining_records,
            )
            replay_records_seen += int(len(experience_records))
            if experience_ledger_path:
                experience_records_written += append_experience_records(experience_records, path=experience_ledger_path)
            if return_experience_records:
                experience_records_all.extend(experience_records)
            fold_results.append(
                {
                    **spec,
                    "status": "ok",
                    "record_count": int(len(built.get("records") or [])),
                    "records": list(built.get("records") or []),
                    "summary": summary,
                    "simulation": sim,
                    "diagnostics": summarize_record_diagnostics(built.get("records")),
                    "experience_record_count": int(len(experience_records)),
                    **({"experience_records": experience_records} if return_experience_records else {}),
                }
            )

    holdout_n = max(0, min(int(holdout_folds), len(fold_results)))
    dev_folds = fold_results[:-holdout_n] if holdout_n else fold_results
    holdout = fold_results[-holdout_n:] if holdout_n else []
    sensitivity = {}
    if include_sensitivity:
        for name, overrides in _sensitivity_variants():
            variant_folds = []
            for item in dev_folds:
                if item.get("status") != "ok":
                    continue
                sim = simulate_controller_records(
                    list(item.get("records") or []),
                    instrument_key=symbol,
                    profile_name=profile_name,
                    starting_cash=PAPER_STARTING_CASH,
                    overrides=overrides,
                )
                variant_folds.append({"summary": summarize_simulation(sim)})
            sensitivity[name] = summarize_folds(variant_folds)

    # Preserve raw records for downstream analysis while keeping the summary compact.
    compact_folds = []
    for item in fold_results:
        compact_folds.append(
            {
                "fold": item.get("fold"),
                "train_start": item.get("train_start"),
                "train_end": item.get("train_end"),
                "test_start": item.get("test_start"),
                "test_end": item.get("test_end"),
                "status": item.get("status"),
                "record_count": item.get("record_count"),
                "summary": item.get("summary"),
            }
        )
    return {
        "instrument": symbol,
        "path": "stock",
        "status": "ok" if any(item.get("status") == "ok" for item in fold_results) else "error",
        "folds": compact_folds,
        "walk_forward": summarize_folds(dev_folds),
        "holdout": summarize_folds(holdout),
        "raw_folds": fold_results,
        "sensitivity": sensitivity,
        "assumptions": {
            "decision_mode": "shadow controller replay",
            "fill_model": "bar-close decision price with adverse slippage and per-side transaction cost",
            "bar_minutes": int(BAR_MINUTES),
            "slippage_pct_per_side": float(SLIPPAGE_PCT),
            "transaction_cost_pct_per_side": float(TRANSACTION_COST_PER_TRADE),
            "stock_horizon_blend": bool(stock_horizon_blend),
            "historical_macro_context": "neutral placeholder because no historical macro cache was available",
            "historical_breadth_context": "neutral placeholder because no historical breadth cache was available",
            "readiness_mode": "fold-ready override",
        },
        "experience_ledger": {
            "enabled": bool(experience_ledger_path),
            "path": str(experience_ledger_path) if experience_ledger_path else None,
            "max_replay_records": max_records,
            "records_built": int(sum(int(item.get("experience_record_count") or 0) for item in fold_results)),
            "records_written": int(experience_records_written),
        },
        **({"experience_records": experience_records_all} if return_experience_records else {}),
        "options_path": {
            "status": "unsupported",
            "reason": "Historical contract-level options data is not available here for a credible autonomy replay.",
        },
    }
