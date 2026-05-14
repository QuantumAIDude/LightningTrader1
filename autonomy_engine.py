from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from config import (
    AUTONOMY_ENTRY_COOLDOWN_MIN,
    AUTONOMY_FLAT_BEFORE_CLOSE_MIN,
    AUTONOMY_LOSS_COOLDOWN_MIN,
    AUTONOMY_MAX_DAILY_LOSS_PCT,
    AUTONOMY_MAX_HOLD_MIN,
    AUTONOMY_MAX_SPREAD_BPS,
    AUTONOMY_MAX_TRADES_PER_DAY,
    AUTONOMY_MIN_CONFIDENCE,
    AUTONOMY_MIN_EDGE_BPS,
    AUTONOMY_MIN_POSITION_SIZE,
    AUTONOMY_MIN_QUALITY_SCORE,
    AUTONOMY_CONTROLLER_STOP_LOSS_DOLLARS,
    AUTONOMY_CONTROLLER_STOP_LOSS_PCT,
    AUTONOMY_DECISION_STYLE_DEFAULT,
    AUTONOMY_MODE_DEFAULT,
    AUTONOMY_NEWS_CONFLICT_IMPACT,
    AUTONOMY_OPEN_DELAY_MIN,
    AUTONOMY_STALE_BROKER_MAX_SEC,
    AUTONOMY_STALE_MARKET_MAX_SEC,
    AUTONOMY_STRATEGY_DEFAULT,
    AUTONOMY_SUPPORTED_SYMBOLS,
    DATA_DIR,
    DEFAULT_INSTRUMENT,
    INSTRUMENTS,
)
from autonomy_strategies import (
    build_strategy_advisory,
    evaluate_strategy_signal,
    list_strategy_catalog,
    normalize_strategy_key,
    strategy_label,
)
from execution_engine import (
    flatten_routed_position,
    get_execution_dashboard_snapshot,
    place_routed_order,
)
from instrument_profile import build_instrument_meta
from readiness_tracker import load_readiness_state, summarize_deployment_readiness
from risk_engine import risk_summary_for_symbol

AUTONOMY_MODES = ("off", "shadow", "paper_auto", "live_auto")
AUTONOMY_MODE_LABELS = {
    "off": "OFF",
    "shadow": "SHADOW",
    "paper_auto": "PAPER AUTO",
    "live_auto": "LIVE AUTO",
}
AUTONOMY_DECISION_STYLES = ("full_autonomy", "strategy_autonomy")
AUTONOMY_DECISION_STYLE_LABELS = {
    "full_autonomy": "Full Autonomy",
    "strategy_autonomy": "Strategy Autonomy",
}
_ACTUAL_MODES = {"paper_auto", "live_auto"}


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if pd.isna(out):
        return None
    return float(out)


def _utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return repr(value)


def _state_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"autonomy_state_{str(symbol or '').upper()}.json")


def _audit_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"autonomy_audit_{str(symbol or '').upper()}.jsonl")


def _instrument_meta(symbol: str | None) -> dict:
    key = str(symbol or DEFAULT_INSTRUMENT or "").upper()
    return build_instrument_meta(key, overrides=INSTRUMENTS.get(key, {}) or {})


def _normalize_mode(mode_value: str | None) -> str:
    mode = str(mode_value or AUTONOMY_MODE_DEFAULT or "OFF").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "paper": "paper_auto",
        "paperauto": "paper_auto",
        "live": "live_auto",
        "liveauto": "live_auto",
        "shadow_mode": "shadow",
        "none": "off",
    }
    mode = aliases.get(mode, mode)
    return mode if mode in AUTONOMY_MODES else "off"


def _normalize_decision_style(value: str | None) -> str:
    style = str(value or AUTONOMY_DECISION_STYLE_DEFAULT or "full_autonomy").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "full": "full_autonomy",
        "predictive": "full_autonomy",
        "model": "full_autonomy",
        "model_autonomy": "full_autonomy",
        "strategy": "strategy_autonomy",
        "strategic": "strategy_autonomy",
        "rules": "strategy_autonomy",
        "rule": "strategy_autonomy",
    }
    style = aliases.get(style, style)
    return style if style in AUTONOMY_DECISION_STYLES else "full_autonomy"


def _decision_style_label(value: str | None) -> str:
    return AUTONOMY_DECISION_STYLE_LABELS.get(_normalize_decision_style(value), "Full Autonomy")


def _mode_label(mode_value: str | None) -> str:
    return AUTONOMY_MODE_LABELS.get(_normalize_mode(mode_value), "OFF")


def list_autonomy_strategies() -> list[dict]:
    return list_strategy_catalog()


def _default_state(symbol: str) -> dict:
    return {
        "symbol": str(symbol or DEFAULT_INSTRUMENT).upper(),
        "mode": _normalize_mode(AUTONOMY_MODE_DEFAULT),
        "decision_style": _normalize_decision_style(AUTONOMY_DECISION_STYLE_DEFAULT),
        "strategy_key": normalize_strategy_key(AUTONOMY_STRATEGY_DEFAULT),
        "loss_limit_dollars": float(AUTONOMY_CONTROLLER_STOP_LOSS_DOLLARS) if float(AUTONOMY_CONTROLLER_STOP_LOSS_DOLLARS or 0.0) > 0.0 else None,
        "loss_limit_pct": float(AUTONOMY_CONTROLLER_STOP_LOSS_PCT),
        "kill_switch": False,
        "session_date": None,
        "trades_today": 0,
        "shadow_trades_today": 0,
        "realized_pnl_today": 0.0,
        "shadow_realized_pnl_today": 0.0,
        "cooldown_until": None,
        "last_signal_key": None,
        "last_action": None,
        "last_action_detail": None,
        "last_order_id": None,
        "last_client_order_id": None,
        "last_decision_at": None,
        "last_block_reason": None,
        "open_trade": None,
        "pending_order": None,
        "shadow_position": None,
        "updated_at": _utc_now().isoformat(),
    }


def _load_state(symbol: str) -> dict:
    path = _state_path(symbol)
    state = _default_state(symbol)
    if not os.path.exists(path):
        return state
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
        if isinstance(payload, dict):
            state.update(payload)
    except Exception:
        pass
    state["mode"] = _normalize_mode(state.get("mode"))
    state["decision_style"] = _normalize_decision_style(state.get("decision_style"))
    state["strategy_key"] = normalize_strategy_key(state.get("strategy_key"))
    state["symbol"] = str(symbol or DEFAULT_INSTRUMENT).upper()
    return state


def _save_state(symbol: str, state: dict) -> dict:
    payload = dict(state or {})
    payload["symbol"] = str(symbol or payload.get("symbol") or DEFAULT_INSTRUMENT).upper()
    payload["mode"] = _normalize_mode(payload.get("mode"))
    payload["decision_style"] = _normalize_decision_style(payload.get("decision_style"))
    payload["strategy_key"] = normalize_strategy_key(payload.get("strategy_key"))
    payload["updated_at"] = _utc_now().isoformat()
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_state_path(symbol), "w") as handle:
            json.dump(_json_safe(payload), handle, indent=2, sort_keys=True)
    except Exception:
        pass
    return payload


def _write_audit(symbol: str, event: dict) -> None:
    payload = dict(event or {})
    payload.setdefault("logged_at", _utc_now().isoformat())
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_audit_path(symbol), "a") as handle:
            handle.write(json.dumps(_json_safe(payload), sort_keys=True))
            handle.write("\n")
    except Exception:
        pass


def _parse_ts(value) -> pd.Timestamp | None:
    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _session_context(symbol: str, now_utc: pd.Timestamp | None = None) -> dict:
    meta = _instrument_meta(symbol)
    now = _parse_ts(now_utc) or _utc_now()
    tz = str(meta.get("tz") or "America/New_York")
    local = now.tz_convert(tz)
    open_minute = int(meta.get("session_open_minute", 9 * 60 + 30))
    close_minute = int(meta.get("session_close_minute", 16 * 60))
    minutes = int(local.hour * 60 + local.minute)
    session_date = pd.Timestamp(local.date())
    if minutes < open_minute:
        session_date = session_date - pd.Timedelta(days=1)
    until_close = close_minute - minutes
    return {
        "now_utc": now,
        "now_local": local,
        "tz": tz,
        "open_minute": open_minute,
        "close_minute": close_minute,
        "minutes": minutes,
        "session_date": session_date.strftime("%Y-%m-%d"),
        "minutes_to_close": int(until_close),
        "within_session": bool(open_minute <= minutes < close_minute),
        "entry_window_open": bool((open_minute + AUTONOMY_OPEN_DELAY_MIN) <= minutes < (close_minute - AUTONOMY_FLAT_BEFORE_CLOSE_MIN)),
        "flat_window": bool(minutes >= (close_minute - AUTONOMY_FLAT_BEFORE_CLOSE_MIN)),
    }


def _mode_stat_keys(mode: str) -> tuple[str, str]:
    mode_key = _normalize_mode(mode)
    if mode_key == "shadow":
        return "shadow_trades_today", "shadow_realized_pnl_today"
    return "trades_today", "realized_pnl_today"


def _ensure_session_rollover(state: dict, symbol: str, now_utc: pd.Timestamp) -> dict:
    out = dict(state or {})
    session_date = _session_context(symbol, now_utc).get("session_date")
    if out.get("session_date") == session_date:
        return out
    out["session_date"] = session_date
    out["trades_today"] = 0
    out["shadow_trades_today"] = 0
    out["realized_pnl_today"] = 0.0
    out["shadow_realized_pnl_today"] = 0.0
    out["cooldown_until"] = None
    out["last_signal_key"] = None
    return out


def set_autonomy_mode(symbol: str, mode: str) -> dict:
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    state = _load_state(sym)
    state["mode"] = _normalize_mode(mode)
    state["last_action"] = "mode_change"
    state["last_action_detail"] = f"Controller mode set to {_mode_label(state['mode'])}"
    _save_state(sym, state)
    _write_audit(sym, {"kind": "mode_change", "mode": state["mode"], "detail": state["last_action_detail"]})
    return get_autonomy_snapshot(sym)


def set_autonomy_decision_style(symbol: str, decision_style: str, strategy_key: str | None = None) -> dict:
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    state = _load_state(sym)
    state["decision_style"] = _normalize_decision_style(decision_style)
    if strategy_key is not None:
        state["strategy_key"] = normalize_strategy_key(strategy_key)
    style_label = _decision_style_label(state.get("decision_style"))
    if state["decision_style"] == "strategy_autonomy":
        style_label = f"{style_label}: {strategy_label(state.get('strategy_key'))}"
    state["last_action"] = "decision_style_change"
    state["last_action_detail"] = f"Decision style set to {style_label}"
    _save_state(sym, state)
    _write_audit(
        sym,
        {
            "kind": "decision_style_change",
            "decision_style": state["decision_style"],
            "strategy_key": state.get("strategy_key"),
            "detail": state["last_action_detail"],
        },
    )
    return get_autonomy_snapshot(sym)


def set_autonomy_loss_limit(
    symbol: str,
    *,
    dollars: float | None = None,
    pct: float | None = None,
    clear_halt: bool = False,
) -> dict:
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    state = _load_state(sym)
    dollars_val = _safe_float(dollars)
    pct_val = _safe_float(pct)
    state["loss_limit_dollars"] = float(dollars_val) if dollars_val is not None and dollars_val > 0.0 else None
    state["loss_limit_pct"] = float(max(0.0001, pct_val)) if pct_val is not None and pct_val > 0.0 else None
    if clear_halt:
        state["kill_switch"] = False
    limit_desc = _loss_limit_detail(state, equity=100000.0)
    state["last_action"] = "loss_limit_update"
    state["last_action_detail"] = f"Controller stop-loss set: {limit_desc}"
    _save_state(sym, state)
    _write_audit(
        sym,
        {
            "kind": "loss_limit_update",
            "loss_limit_dollars": state.get("loss_limit_dollars"),
            "loss_limit_pct": state.get("loss_limit_pct"),
            "clear_halt": bool(clear_halt),
            "detail": state["last_action_detail"],
        },
    )
    return get_autonomy_snapshot(sym)


def set_autonomy_kill_switch(symbol: str, enabled: bool, reason: str | None = None) -> dict:
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    state = _load_state(sym)
    state["kill_switch"] = bool(enabled)
    state["last_action"] = "kill_switch_on" if enabled else "kill_switch_off"
    state["last_action_detail"] = str(reason or ("Autonomy halted by operator" if enabled else "Autonomy halt cleared"))
    _save_state(sym, state)
    _write_audit(sym, {"kind": "kill_switch", "enabled": bool(enabled), "reason": state["last_action_detail"]})
    return get_autonomy_snapshot(sym)


def _current_position(execution_snapshot: dict | None, symbol: str, route_mode: str | None) -> dict | None:
    snap = dict(execution_snapshot or {})
    symbol = str(symbol or "").upper()
    route = str(route_mode or "").lower()
    if route in {"alpaca_paper", "alpaca_live"}:
        live_snapshot = dict((snap.get("runtime") or {}).get("live_snapshot") or snap.get("live_snapshot") or {})
        for pos in list(live_snapshot.get("positions") or []):
            if str(getattr(pos, "symbol", "")).upper() != symbol:
                continue
            qty = _safe_float(getattr(pos, "qty", None)) or 0.0
            if qty == 0.0:
                return None
            side = str(getattr(pos, "side", "") or "").lower()
            signed_qty = -abs(qty) if side in {"short", "sell"} else abs(qty)
            return {
                "symbol": symbol,
                "qty": float(signed_qty),
                "avg_price": float(_safe_float(getattr(pos, "avg_entry_price", None)) or 0.0),
                "side": "short" if signed_qty < 0 else "long",
            }
        return None
    state = dict(snap.get("state") or {})
    pos = dict((state.get("positions") or {}).get(symbol) or {})
    qty = _safe_float(pos.get("qty"))
    if qty in (None, 0.0):
        return None
    return {
        "symbol": symbol,
        "qty": float(qty),
        "avg_price": float(_safe_float(pos.get("avg_price")) or 0.0),
        "side": "short" if float(qty) < 0 else "long",
        "opened_at": pos.get("opened_at"),
        "stop_price": _safe_float(pos.get("stop_price")),
        "target_price": _safe_float(pos.get("target_price")),
    }


def _market_state(df: pd.DataFrame | None, symbol: str, current_price: float | None, now_utc: pd.Timestamp | None = None) -> dict:
    session = _session_context(symbol, now_utc=now_utc)
    now_utc = session.get("now_utc") or _parse_ts(now_utc) or _utc_now()
    last_ts = None
    source = "-"
    delayed = False
    spread_bps = None
    if df is not None and not getattr(df, "empty", True):
        try:
            last_ts = _parse_ts(df.index[-1])
        except Exception:
            last_ts = None
        try:
            source = str(getattr(df, "attrs", {}).get("data_source") or "-")
            delayed = bool(getattr(df, "attrs", {}).get("data_error"))
        except Exception:
            delayed = False
        try:
            if "ask" in df.columns and "bid" in df.columns and current_price not in (None, 0.0):
                ask = _safe_float(df["ask"].iloc[-1])
                bid = _safe_float(df["bid"].iloc[-1])
                if ask is not None and bid is not None and ask > bid:
                    spread_bps = 10000.0 * (float(ask - bid) / float(current_price))
        except Exception:
            spread_bps = None
    age_sec = None
    if last_ts is not None:
        age_sec = max(0.0, float((now_utc - last_ts).total_seconds()))
    return {
        "session": session,
        "last_bar_ts": last_ts.isoformat() if last_ts is not None else None,
        "age_sec": age_sec,
        "stale": bool(age_sec is None or age_sec > float(AUTONOMY_STALE_MARKET_MAX_SEC)),
        "delayed": bool(delayed),
        "source": source,
        "spread_bps": spread_bps,
    }


def _broker_age_sec(execution_snapshot: dict | None, now_utc: pd.Timestamp | None = None) -> float | None:
    snap = dict(execution_snapshot or {})
    diagnostics = dict(snap.get("diagnostics") or {})
    runtime = dict(snap.get("runtime") or {})
    live_snapshot = dict(runtime.get("live_snapshot") or snap.get("live_snapshot") or {})
    now_ref = _parse_ts(now_utc) or _utc_now()
    for candidate in (diagnostics.get("checked_at"), live_snapshot.get("fetched_at")):
        ts = _parse_ts(candidate)
        if ts is not None:
            return max(0.0, float((now_ref - ts).total_seconds()))
    return None


def _build_client_order_id(symbol: str, mode: str, side: str, ts_value: pd.Timestamp | None = None) -> str:
    ts = _parse_ts(ts_value) or _utc_now()
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    return f"lt4k-{sym.lower()}-{_normalize_mode(mode)}-{str(side or '').lower()}-{ts.strftime('%m%d%H%M%S')}"[:48]


def _reconcile_tracking(
    state: dict,
    symbol: str,
    route_mode: str | None,
    execution_snapshot: dict | None,
    current_price: float | None,
    now_utc: pd.Timestamp | None = None,
) -> dict:
    out = dict(state or {})
    now_ref = _parse_ts(now_utc) or _utc_now()
    actual_position = _current_position(execution_snapshot, symbol, route_mode)
    tracked = dict(out.get("open_trade") or {}) if isinstance(out.get("open_trade"), dict) else None
    pending = dict(out.get("pending_order") or {}) if isinstance(out.get("pending_order"), dict) else None
    if pending:
        created_at = _parse_ts(pending.get("created_at"))
        age_sec = max(0.0, float((now_ref - created_at).total_seconds())) if created_at is not None else None
        if actual_position and str(pending.get("kind")) == "entry":
            out["open_trade"] = {
                "side": "buy" if float(actual_position.get("qty") or 0.0) > 0 else "sell",
                "qty": int(abs(float(actual_position.get("qty") or 0.0))),
                "entry_price": float(actual_position.get("avg_price") or current_price or 0.0),
                "opened_at": now_ref.isoformat(),
                "route_mode": route_mode,
                "client_order_id": pending.get("client_order_id"),
                "order_id": pending.get("order_id"),
            }
            out["pending_order"] = None
        elif age_sec is not None and age_sec > 600:
            out["pending_order"] = None
    if tracked and not actual_position and not out.get("pending_order"):
        out["open_trade"] = None
    return out


def _record_pnl(state: dict, mode: str, pnl: float | None, now_utc: pd.Timestamp | None = None) -> dict:
    out = dict(state or {})
    trades_key, pnl_key = _mode_stat_keys(mode)
    out[trades_key] = int(out.get(trades_key) or 0) + 1
    out[pnl_key] = float(_safe_float(out.get(pnl_key)) or 0.0) + float(_safe_float(pnl) or 0.0)
    if (float(_safe_float(pnl) or 0.0) < 0.0) and _normalize_mode(mode) in AUTONOMY_MODES:
        now_ref = _parse_ts(now_utc) or _utc_now()
        out["cooldown_until"] = (now_ref + pd.Timedelta(minutes=int(AUTONOMY_LOSS_COOLDOWN_MIN))).isoformat()
    return out


def _loss_limit_amount(state: dict | None, equity: float | None) -> float:
    payload = dict(state or {})
    fixed = _safe_float(payload.get("loss_limit_dollars"))
    if fixed is not None and fixed > 0.0:
        return float(fixed)
    pct = _safe_float(payload.get("loss_limit_pct"))
    if pct is None or pct <= 0.0:
        pct = float(AUTONOMY_MAX_DAILY_LOSS_PCT)
    eq = max(1.0, float(_safe_float(equity) or 1.0))
    return float(abs(pct) * eq)


def _loss_limit_detail(state: dict | None, equity: float | None) -> str:
    payload = dict(state or {})
    fixed = _safe_float(payload.get("loss_limit_dollars"))
    pct = _safe_float(payload.get("loss_limit_pct"))
    amount = _loss_limit_amount(payload, equity)
    if fixed is not None and fixed > 0.0:
        pct_text = f" / {100.0 * pct:.2f}%" if pct is not None and pct > 0.0 else ""
        return f"${amount:,.2f}{pct_text}"
    return f"{100.0 * (pct if pct is not None and pct > 0.0 else AUTONOMY_MAX_DAILY_LOSS_PCT):.2f}% (${amount:,.2f})"


def _open_unrealized_pnl(state: dict | None, current_price: float | None) -> float:
    trade = dict((state or {}).get("open_trade") or {}) if isinstance((state or {}).get("open_trade"), dict) else {}
    if not trade:
        return 0.0
    entry = _safe_float(trade.get("entry_price"))
    px = _safe_float(current_price)
    qty = abs(_safe_float(trade.get("qty")) or 0.0)
    side = str(trade.get("side") or "buy").lower()
    if entry is None or px is None or qty <= 0.0:
        return 0.0
    return float((px - entry) * qty if side == "buy" else (entry - px) * qty)


def _close_shadow_position(state: dict, current_price: float | None, reason: str, now_utc: pd.Timestamp | None = None) -> tuple[dict, dict]:
    out = dict(state or {})
    pos = dict(out.get("shadow_position") or {})
    entry_price = _safe_float(pos.get("entry_price")) or _safe_float(current_price) or 0.0
    exit_price = _safe_float(current_price) or entry_price
    qty = int(abs(_safe_float(pos.get("qty")) or 0))
    side = str(pos.get("side") or "buy").lower()
    pnl = (float(exit_price) - float(entry_price)) * qty if side == "buy" else (float(entry_price) - float(exit_price)) * qty
    out["shadow_position"] = None
    out = _record_pnl(out, "shadow", pnl, now_utc=now_utc)
    out["last_action"] = "shadow_exit"
    out["last_action_detail"] = f"{reason} | {qty} sh @ ${float(exit_price):.2f} | pnl ${float(pnl):+.2f}"
    event = {
        "kind": "shadow_exit",
        "reason": reason,
        "qty": qty,
        "side": side,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_dollars": pnl,
    }
    return out, event


def _enter_shadow_position(
    state: dict,
    side: str,
    qty: int,
    current_price: float | None,
    signal_key: str,
    detail: str,
    now_utc: pd.Timestamp | None = None,
) -> tuple[dict, dict]:
    out = dict(state or {})
    px = _safe_float(current_price) or 0.0
    now_ref = _parse_ts(now_utc) or _utc_now()
    out["shadow_position"] = {
        "side": str(side).lower(),
        "qty": int(max(0, qty)),
        "entry_price": float(px),
        "opened_at": now_ref.isoformat(),
        "signal_key": signal_key,
    }
    out["last_signal_key"] = signal_key
    out["last_action"] = "shadow_entry"
    out["last_action_detail"] = detail
    event = {
        "kind": "shadow_entry",
        "side": side,
        "qty": qty,
        "entry_price": px,
        "signal_key": signal_key,
        "detail": detail,
    }
    return out, event


def _stock_path_status(
    symbol: str,
    mode: str,
    df: pd.DataFrame | None,
    advisory: dict | None,
    execution_snapshot: dict | None,
    replay_snapshot: dict | None = None,
    now_utc: pd.Timestamp | None = None,
    readiness_override: dict | None = None,
    controller_state: dict | None = None,
) -> dict:
    advisory = dict(advisory or {})
    controller_state = dict(controller_state or {})
    execution_snapshot = dict(execution_snapshot or {})
    market = _market_state(
        df,
        symbol,
        _safe_float(df["close"].iloc[-1]) if df is not None and not getattr(df, "empty", True) and "close" in df.columns else None,
        now_utc=now_utc,
    )
    quality = dict(advisory.get("quality_meta") or {})
    pred_state = dict(advisory.get("pred_state") or {})
    intraday = dict(pred_state.get("intraday") or {})
    news_meta = dict(advisory.get("news_meta") or {})
    top_news = dict(news_meta.get("top_item") or {})
    liquidity_meta = dict(advisory.get("liquidity_meta") or {})
    macro_meta = dict(advisory.get("macro_meta") or {})
    catalyst_meta = dict(advisory.get("catalyst_meta") or {})
    expression_meta = dict((advisory.get("expression_meta") or {}).get("stock") or {})
    signal_translation = dict(advisory.get("signal_translation") or {})
    readiness = dict(readiness_override or load_readiness_state(symbol) or {})
    deployment = summarize_deployment_readiness(symbol, state=readiness)
    validation_evidence = dict(deployment.get("validation_evidence") or {})
    meta = _instrument_meta(symbol)
    display_name = str(meta.get("display_name") or symbol or "this stock").strip() or str(symbol or "this stock")
    supported_symbols = {str(item or "").strip().upper() for item in AUTONOMY_SUPPORTED_SYMBOLS if str(item or "").strip()}
    actual_supported = str(symbol).upper() in supported_symbols
    route_mode = str((execution_snapshot.get("runtime") or {}).get("routing_mode") or execution_snapshot.get("broker_mode") or "local_paper").lower()
    broker_age_sec = _broker_age_sec(execution_snapshot, now_utc=now_utc)
    current_price = _safe_float(df["close"].iloc[-1]) if df is not None and not getattr(df, "empty", True) and "close" in df.columns else None
    base_rec = dict(execution_snapshot.get("recommendation") or {})
    signal_key = None
    decision_style = _normalize_decision_style(controller_state.get("decision_style"))
    selected_strategy = normalize_strategy_key(controller_state.get("strategy_key"))
    strategy_signal = {}
    advisory_for_risk = advisory
    signal_metrics = dict(intraday)
    side = str(signal_metrics.get("direction") or "-").upper()
    if decision_style == "strategy_autonomy":
        pair_symbol = str(controller_state.get("pair_symbol") or "").strip().upper() or None
        strategy_signal = evaluate_strategy_signal(selected_strategy, df, symbol=symbol, pair_symbol=pair_symbol)
        strategy_side = str(strategy_signal.get("side") or "").lower()
        if strategy_side in {"buy", "sell"}:
            side = "UP" if strategy_side == "buy" else "DOWN"
            advisory_for_risk = build_strategy_advisory(advisory, strategy_signal, current_price=current_price)
            signal_metrics = dict(((advisory_for_risk.get("pred_state") or {}).get("intraday") or signal_metrics))
        else:
            side = ""
    if side == "UP":
        side = "buy"
    elif side == "DOWN":
        side = "sell"
    else:
        side = ""
    if current_price is not None and side:
        strategy_key = f":{selected_strategy}" if decision_style == "strategy_autonomy" else ""
        signal_key = f"{_session_context(symbol).get('session_date')}{strategy_key}:{side}:{round(float(current_price), 2)}"
    reasons = []
    hard_reasons = []
    soft_reasons = []

    def _add_reason(reason: str, hard: bool = False) -> None:
        text = str(reason or "").strip()
        if not text:
            return
        reasons.append(text)
        if hard:
            hard_reasons.append(text)
        else:
            soft_reasons.append(text)

    validation_label = "Validated"
    validation_detail = f"{display_name} is on the validated autonomy rollout list."
    if not actual_supported:
        validation_label = "Shadow only"
        validation_detail = f"{display_name} is not validated for actual autonomy in this rollout; advisory and shadow mode are the honest path right now."
        if mode in _ACTUAL_MODES:
            _add_reason(f"actual autonomy is not validated for {display_name} in this rollout", hard=True)
    deployment_label = str(deployment.get("stage_label") or "Trained / not validated")
    deployment_detail = str(deployment.get("summary") or deployment.get("detail") or "-")
    validation_evidence_label = str(
        validation_evidence.get("validation_status_label")
        or deployment.get("validation_status_label")
        or "Validation missing"
    )
    validation_evidence_detail = str(
        validation_evidence.get("summary")
        or deployment.get("validation_summary")
        or "Validation evidence has not been captured."
    )
    validation_autonomy_blockers = list(validation_evidence.get("autonomy_blockers") or validation_evidence.get("blockers") or [])
    if mode == "shadow" and not bool(deployment.get("shadow_ready")):
        _add_reason(f"{display_name} is not shadow-ready yet ({deployment_label.lower()})", hard=True)
    if mode == "shadow" and not bool(validation_evidence.get("supports_shadow")):
        _add_reason("validation evidence does not support shadow mode yet", hard=True)
    if mode == "paper_auto" and not bool(deployment.get("paper_ready")):
        _add_reason(f"{display_name} is below the paper-autonomy deployment stage ({deployment_label.lower()})", hard=True)
    if mode == "paper_auto" and not bool(validation_evidence.get("supports_paper_candidate")):
        detail = str(validation_autonomy_blockers[0]) if validation_autonomy_blockers else "paper-candidate validation evidence is missing"
        _add_reason(f"validation evidence blocks paper autonomy: {detail}", hard=True)
    if mode == "live_auto" and not bool(deployment.get("live_ready_candidate")):
        _add_reason(f"{display_name} is below the controlled real-world deployment stage ({deployment_label.lower()})", hard=True)
    if mode == "live_auto" and not bool(validation_evidence.get("supports_autonomy_candidate")):
        detail = str(validation_autonomy_blockers[0]) if validation_autonomy_blockers else "autonomy-candidate validation evidence is missing"
        _add_reason(f"validation evidence blocks live autonomy: {detail}", hard=True)
    if bool((replay_snapshot or {}).get("active")):
        _add_reason("replay mode is active", hard=True)
    if not market.get("session", {}).get("within_session"):
        _add_reason("market session is closed", hard=True)
    if market.get("stale"):
        _add_reason("market data is stale", hard=True)
    if market.get("delayed"):
        _add_reason("market data is degraded or delayed", hard=True)
    if not market.get("session", {}).get("entry_window_open"):
        if market.get("session", {}).get("flat_window"):
            _add_reason("inside flat-by-close window", hard=True)
        else:
            _add_reason("still inside the opening no-trade window")
    if bool(quality.get("no_trade")):
        _add_reason("advisory stack marked this setup as no-trade")
    if (_safe_float(quality.get("score")) or 0.0) < float(AUTONOMY_MIN_QUALITY_SCORE):
        _add_reason("quality score is below the autonomy threshold")
    if decision_style == "strategy_autonomy" and not bool(strategy_signal.get("allowed")):
        _add_reason(f"{strategy_label(selected_strategy)} strategy has no executable signal: {strategy_signal.get('reason') or 'no signal'}")
    min_confidence = min(float(AUTONOMY_MIN_CONFIDENCE), 54.0) if decision_style == "strategy_autonomy" else float(AUTONOMY_MIN_CONFIDENCE)
    min_position_size = min(float(AUTONOMY_MIN_POSITION_SIZE), 0.20) if decision_style == "strategy_autonomy" else float(AUTONOMY_MIN_POSITION_SIZE)
    min_edge_bps = min(float(AUTONOMY_MIN_EDGE_BPS), 2.0) if decision_style == "strategy_autonomy" else float(AUTONOMY_MIN_EDGE_BPS)
    if (_safe_float(signal_metrics.get("confidence")) or 0.0) < min_confidence:
        _add_reason("strategy confidence is too low" if decision_style == "strategy_autonomy" else "intraday confidence is too low")
    if (_safe_float(signal_metrics.get("position_size")) or 0.0) < min_position_size:
        _add_reason("strategy size signal is too small" if decision_style == "strategy_autonomy" else "model size signal is too small")
    if (_safe_float(signal_metrics.get("edge_bps")) or 0.0) < min_edge_bps:
        _add_reason("strategy edge is too small" if decision_style == "strategy_autonomy" else "cost-adjusted edge is too small")
    if str(signal_metrics.get("direction_stage") or "").lower() == "tentative":
        if str(signal_metrics.get("horizon_support_stage") or "").lower() in {"supportive", "confirmed"}:
            _add_reason("finished direction is still tentative even though 10m/30m stock support agrees")
        else:
            _add_reason("finished direction is still tentative")
    if side not in {"buy", "sell"}:
        _add_reason("no directional bias is available")
    if (_safe_float(liquidity_meta.get("tradability_score")) or 0.0) < 0.55:
        _add_reason("liquidity is too weak for autonomy")
    if market.get("spread_bps") is not None and float(market.get("spread_bps")) > float(AUTONOMY_MAX_SPREAD_BPS):
        _add_reason("quoted spread is too wide")
    if not bool(readiness.get("live_ready")) or not bool(readiness.get("fresh")):
        _add_reason("readiness gate is not green", hard=True)
    if str(news_meta.get("headline_state") or "").lower() in {"failed", "stale"}:
        _add_reason("news pipeline is degraded", hard=True)
    if (
        str(news_meta.get("alignment") or "").lower() == "conflict"
        and (
            int(news_meta.get("fresh_high_impact_count") or 0) >= 1
            or (_safe_float(top_news.get("impact_score")) or 0.0) >= float(AUTONOMY_NEWS_CONFLICT_IMPACT)
        )
    ):
        _add_reason("fresh high-impact news conflicts with the model bias", hard=True)
    if (_safe_float(macro_meta.get("risk_score")) or 0.0) >= 0.75:
        _add_reason("macro event risk is too high", hard=True)
    if (_safe_float(catalyst_meta.get("catalyst_risk_score")) or 0.0) >= 0.45:
        _add_reason("catalyst risk is elevated", hard=True)
    if mode == "live_auto" and route_mode != "alpaca_live":
        _add_reason("live autonomy requires a live Alpaca route", hard=True)
    elif mode == "paper_auto" and route_mode not in {"local_paper", "alpaca_paper"}:
        _add_reason("paper autonomy requires a paper-capable route", hard=True)
    if mode in _ACTUAL_MODES and route_mode in {"alpaca_paper", "alpaca_live"}:
        if broker_age_sec is None or broker_age_sec > float(AUTONOMY_STALE_BROKER_MAX_SEC):
            _add_reason("broker snapshot is stale", hard=True)
    if mode == "live_auto" and side == "sell":
        _add_reason("live short autonomy stays disabled until broker short inventory is validated", hard=True)
    risk_summary = risk_summary_for_symbol(
        symbol,
        df=df,
        advisory=advisory_for_risk,
        execution_snapshot=execution_snapshot,
        current_price=current_price,
        require_portfolio=True,
        requested_qty=int(base_rec.get("qty") or 0) if base_rec.get("allowed") else None,
    )
    if bool(risk_summary.get("blocked")):
        _add_reason(str(risk_summary.get("primary_blocker") or "risk engine blocked the trade"), hard=True)
    detail = str(expression_meta.get("detail") or f"Direct {display_name} stock is the primary autonomy path.")
    if decision_style == "strategy_autonomy":
        detail = f"{strategy_label(selected_strategy)} on {display_name}: {strategy_signal.get('reason') or detail}"
    qty = int(base_rec.get("qty") or 0) if base_rec.get("allowed") else int(risk_summary.get("suggested_qty") or 0)
    allowed = (not reasons) and qty > 0
    if not allowed and qty <= 0 and not reasons:
        _add_reason(str(base_rec.get("reason") or "risk budget is too small"))
    signal_summary = str(signal_translation.get("headline") or "No finished stock signal")
    signal_detail = str(signal_translation.get("detail") or "")
    if decision_style == "strategy_autonomy":
        signal_summary = f"Strategy Autonomy: {strategy_label(selected_strategy)}"
        signal_detail = str(strategy_signal.get("reason") or signal_detail or "Strategy signal unavailable")
    watch_side = str(signal_translation.get("operator_side") or side).lower()
    interesting = bool(signal_translation.get("watchlist"))
    label = "Eligible Long" if allowed and side == "buy" else "Eligible Short" if allowed and side == "sell" else "Blocked"
    detail_out = detail if allowed else "; ".join(reasons[:3])
    if not allowed and interesting and watch_side in {"buy", "sell"}:
        label = "Watchlist Long" if watch_side == "buy" else "Watchlist Short"
        blocker_bits = list(hard_reasons[:1]) + list(soft_reasons[:2])
        detail_out = signal_detail or signal_summary
        if blocker_bits:
            detail_out = f"{detail_out} | blocked by {'; '.join(blocker_bits)}"
    return {
        "allowed": bool(allowed),
        "label": label,
        "detail": detail if allowed else detail_out,
        "side": side,
        "qty": qty,
        "route_mode": route_mode,
        "signal_key": signal_key,
        "decision_style": decision_style,
        "decision_style_label": _decision_style_label(decision_style),
        "strategy_key": selected_strategy,
        "strategy_label": strategy_label(selected_strategy),
        "strategy_signal": strategy_signal,
        "execution_advisory": advisory_for_risk if decision_style == "strategy_autonomy" else advisory,
        "reasons": reasons,
        "hard_reasons": hard_reasons,
        "soft_reasons": soft_reasons,
        "signal_summary": signal_summary,
        "signal_detail": signal_detail or detail_out,
        "watchlist": interesting,
        "market": market,
        "broker_age_sec": broker_age_sec,
        "validation_label": validation_label,
        "validation_detail": validation_detail,
        "validation_evidence": validation_evidence,
        "validation_evidence_label": validation_evidence_label,
        "validation_evidence_detail": validation_evidence_detail,
        "validation_blockers": validation_autonomy_blockers,
        "deployment_label": deployment_label,
        "deployment_detail": deployment_detail,
        "deployment_stage_key": str(deployment.get("stage_key") or ""),
        "shadow_ready": bool(deployment.get("shadow_ready")),
        "paper_ready": bool(deployment.get("paper_ready")),
        "live_ready_candidate": bool(deployment.get("live_ready_candidate")),
        "readiness_live_ready": bool(readiness.get("live_ready")),
        "readiness_state_fresh": bool(readiness.get("fresh")),
        "headline_state": str(news_meta.get("headline_state") or ""),
        "quality_no_trade": bool(quality.get("no_trade")),
        "signal_present": bool(side in {"buy", "sell"} or interesting),
        "rollout_label": str(deployment.get("rollout_label") or ""),
        "rollout_detail": str(deployment.get("rollout_detail") or ""),
        "actual_supported": actual_supported,
        "risk_summary": risk_summary,
        "news_conflict": bool(
            str(news_meta.get("alignment") or "").lower() == "conflict"
            and (
                (_safe_float(news_meta.get("fresh_high_impact_count")) or 0.0) >= 1.0
                or (_safe_float(top_news.get("impact_score")) or 0.0) >= float(AUTONOMY_NEWS_CONFLICT_IMPACT)
            )
        ),
    }


def _shadow_observation_event(
    symbol: str,
    stock_status: dict | None,
    decision: dict | None,
    now_utc: pd.Timestamp | None = None,
) -> dict:
    status = dict(stock_status or {})
    decision = dict(decision or {})
    market = dict(status.get("market") or {})
    session = dict(market.get("session") or {})
    headline_state = str(status.get("headline_state") or "").lower()
    eligible_environment = bool(
        session.get("within_session")
        and session.get("entry_window_open")
        and not market.get("stale")
        and not market.get("delayed")
        and headline_state not in {"failed", "stale"}
    )
    clean_cycle = bool(eligible_environment and status.get("signal_present"))
    hard_reasons = list(status.get("hard_reasons") or [])
    soft_reasons = list(status.get("soft_reasons") or [])
    return {
        "kind": "shadow_observation",
        "observed_at": (_parse_ts(now_utc) or _utc_now()).isoformat(),
        "symbol": str(symbol or DEFAULT_INSTRUMENT).upper(),
        "decision_status": str(decision.get("status") or ""),
        "decision_detail": str(decision.get("detail") or ""),
        "allowed": bool(status.get("allowed")),
        "clean_cycle": bool(clean_cycle),
        "eligible_environment": bool(eligible_environment),
        "signal_present": bool(status.get("signal_present")),
        "side": str(status.get("side") or ""),
        "deployment_stage_key": str(status.get("deployment_stage_key") or ""),
        "shadow_ready": bool(status.get("shadow_ready")),
        "readiness_live_ready": bool(status.get("readiness_live_ready")),
        "readiness_state_fresh": bool(status.get("readiness_state_fresh")),
        "headline_state": str(status.get("headline_state") or ""),
        "market_stale": bool(market.get("stale")),
        "market_delayed": bool(market.get("delayed")),
        "within_session": bool(session.get("within_session")),
        "entry_window_open": bool(session.get("entry_window_open")),
        "news_conflict": bool(status.get("news_conflict")),
        "quality_no_trade": bool(status.get("quality_no_trade")),
        "blocked_by_market": bool(
            market.get("stale")
            or market.get("delayed")
            or not session.get("within_session")
            or not session.get("entry_window_open")
        ),
        "blocked_by_deployment": bool(
            (not bool(status.get("shadow_ready")))
            or (not bool(status.get("readiness_live_ready")))
            or (not bool(status.get("readiness_state_fresh")))
        ),
        "blocked_by_news": bool(headline_state in {"failed", "stale"} or status.get("news_conflict")),
        "blocked_by_quality": bool(status.get("quality_no_trade") or not status.get("signal_present")),
        "hard_reasons": hard_reasons[:6],
        "soft_reasons": soft_reasons[:6],
    }


def _options_path_status(advisory: dict | None = None, options_snapshot: dict | None = None) -> dict:
    advisory = dict(advisory or {})
    expr = dict((advisory.get("expression_meta") or {}).get("options") or {})
    options_snapshot = dict(options_snapshot or {})
    label = str(expr.get("label") or "Context only")
    detail = str(expr.get("detail") or "Options remain advisory-only until contract selection, spread, and Greeks controls are wired.")
    if str(options_snapshot.get("status") or "").lower() not in {"ok"}:
        label = "Blocked"
    return {"allowed": False, "label": label, "detail": detail}


def _build_snapshot(
    symbol: str,
    state: dict,
    mode: str,
    route_label: str,
    stock_status: dict | None = None,
    options_status: dict | None = None,
    decision: dict | None = None,
    now_utc: pd.Timestamp | None = None,
    events: list[dict] | None = None,
) -> dict:
    state = dict(state or {})
    stock_status = dict(stock_status or {})
    options_status = dict(options_status or {})
    decision = dict(decision or {})
    trades_key, pnl_key = _mode_stat_keys(mode)
    trades_today = int(state.get(trades_key) or 0)
    pnl_today = float(_safe_float(state.get(pnl_key)) or 0.0)
    halt = bool(state.get("kill_switch"))
    mode_label = _mode_label(mode)
    decision_style = _normalize_decision_style(state.get("decision_style"))
    decision_label = _decision_style_label(decision_style)
    strategy_key = normalize_strategy_key(state.get("strategy_key"))
    strategy_name = strategy_label(strategy_key)
    if halt:
        card = f"Autonomy: HALTED ({mode_label})"
    else:
        card = f"Autonomy: {mode_label} / {decision_label}"
    summary = str(decision.get("detail") or "").strip()
    if not summary or str(decision.get("status") or "") in {"idle", "off"}:
        summary = str(stock_status.get("detail") or state.get("last_action_detail") or "Controller idle")
    strategy_bit = f" | strategy {strategy_name}" if decision_style == "strategy_autonomy" else ""
    detail = f"{summary} | {route_label}{strategy_bit} | trades {trades_today}/{int(AUTONOMY_MAX_TRADES_PER_DAY)} | pnl ${pnl_today:+.2f}"
    cooldown_until = _parse_ts(state.get("cooldown_until"))
    now_ref = _parse_ts(now_utc) or _utc_now()
    equity_hint = _safe_float(((stock_status.get("risk_summary") or {}).get("risk_budget") or {}).get("equity")) or 100000.0
    risk_detail = f"Stop-loss {_loss_limit_detail(state, equity_hint)} | kill {'ON' if halt else 'OFF'}"
    if cooldown_until is not None and cooldown_until > now_ref:
        risk_detail = f"Cooldown until {cooldown_until.strftime('%H:%M UTC')} | {risk_detail}"
    rows = [
        ("Controller", mode_label, "Human-guided" if mode == "off" else f"{route_label} | {'halted' if halt else 'active'}"),
        ("Decision Mode", decision_label, strategy_name if decision_style == "strategy_autonomy" else "Model, regime, news, risk, and readiness consensus"),
        ("Validation", str(stock_status.get("validation_label") or "-"), str(stock_status.get("validation_detail") or "-")),
        ("Validation Evidence", str(stock_status.get("validation_evidence_label") or "-"), str(stock_status.get("validation_evidence_detail") or "-")),
        ("Deployment", str(stock_status.get("deployment_label") or "-"), str(stock_status.get("deployment_detail") or stock_status.get("rollout_detail") or "-")),
        ("Signal Lens", str(stock_status.get("signal_summary") or "-"), str(stock_status.get("signal_detail") or stock_status.get("detail") or "-")),
        ("Stock Path", str(stock_status.get("label") or "-"), str(stock_status.get("detail") or "-")),
        ("Options Path", str(options_status.get("label") or "-"), str(options_status.get("detail") or "-")),
        ("Risk", "Guarded", risk_detail),
        ("Last Action", str((decision.get("status") or state.get("last_action") or "idle")).replace("_", " ").title(), summary),
    ]
    operator_label = "Operator: Advisory Only" if mode == "off" else f"Operator: {mode_label} / {decision_label} / {route_label}"
    if halt:
        operator_label = f"{operator_label} / HALTED"
    return {
        "symbol": str(symbol or DEFAULT_INSTRUMENT).upper(),
        "mode": _normalize_mode(mode),
        "mode_label": mode_label,
        "decision_style": decision_style,
        "decision_style_label": decision_label,
        "strategy_key": strategy_key,
        "strategy_label": strategy_name,
        "loss_limit_dollars": _safe_float(state.get("loss_limit_dollars")),
        "loss_limit_pct": _safe_float(state.get("loss_limit_pct")),
        "loss_limit_detail": _loss_limit_detail(state, equity_hint),
        "operator_label": operator_label,
        "manual_trading_locked": bool(_normalize_mode(mode) in {"paper_auto", "live_auto"}),
        "kill_switch": halt,
        "card": card,
        "detail": detail,
        "rows": rows,
        "state": state,
        "decision": decision,
        "stock_path": stock_status,
        "options_path": options_status,
        "status_message": decision.get("status_message"),
        "events": list(events or []),
    }


def get_autonomy_snapshot(symbol: str, advisory: dict | None = None, execution_snapshot: dict | None = None) -> dict:
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    state = _load_state(sym)
    mode = _normalize_mode(state.get("mode"))
    route_label = str(((execution_snapshot or {}).get("route_label")) or (((execution_snapshot or {}).get("runtime") or {}).get("route_label") or "Advisory"))
    return _build_snapshot(sym, state, mode, route_label)


def evaluate_autonomy_cycle(
    symbol: str,
    df: pd.DataFrame | None,
    advisory: dict | None = None,
    execution_snapshot: dict | None = None,
    profile_name: str | None = None,
    replay_snapshot: dict | None = None,
    options_snapshot: dict | None = None,
    now_utc: pd.Timestamp | None = None,
    state_override: dict | None = None,
    persist: bool = True,
    readiness_override: dict | None = None,
) -> dict:
    sym = str(symbol or DEFAULT_INSTRUMENT).upper()
    state = dict(state_override or _load_state(sym))
    now_utc = _parse_ts(now_utc) or _utc_now()
    state = _ensure_session_rollover(state, sym, now_utc)
    mode = _normalize_mode(state.get("mode"))
    execution_snapshot = dict(execution_snapshot or {})
    route_label = str(execution_snapshot.get("route_label") or ((execution_snapshot.get("runtime") or {}).get("route_label") or "Advisory"))
    route_mode = str((execution_snapshot.get("runtime") or {}).get("routing_mode") or execution_snapshot.get("broker_mode") or "local_paper").lower()
    current_price = _safe_float(df["close"].iloc[-1]) if df is not None and not getattr(df, "empty", True) and "close" in df.columns else None
    emitted_events: list[dict] = []

    def _record_event(event: dict | None) -> None:
        payload = dict(event or {})
        if not payload:
            return
        emitted_events.append(payload)
        if persist:
            _write_audit(sym, payload)

    state = _reconcile_tracking(state, sym, route_mode, execution_snapshot, current_price, now_utc=now_utc)
    stock_status = _stock_path_status(
        sym,
        mode,
        df,
        advisory,
        execution_snapshot,
        replay_snapshot=replay_snapshot,
        now_utc=now_utc,
        readiness_override=readiness_override,
        controller_state=state,
    )
    options_status = _options_path_status(advisory, options_snapshot=options_snapshot)
    decision = {"status": "idle", "detail": "Controller idle"}
    open_trade = dict(state.get("open_trade") or {}) if isinstance(state.get("open_trade"), dict) else None
    shadow_position = dict(state.get("shadow_position") or {}) if isinstance(state.get("shadow_position"), dict) else None
    actual_position = _current_position(execution_snapshot, sym, route_mode)
    cooldown_until = _parse_ts(state.get("cooldown_until"))
    equity = _safe_float(((execution_snapshot.get("portfolio") or {}).get("equity")))
    if equity is None:
        live_account = dict((((execution_snapshot.get("runtime") or {}).get("live_snapshot") or {}).get("account") or {}))
        equity = _safe_float(live_account.get("equity")) or _safe_float(live_account.get("cash")) or 100000.0
    if cooldown_until is not None and cooldown_until > now_utc and mode != "off":
        stock_status["allowed"] = False
        stock_status["label"] = "Cooldown"
        stock_status["detail"] = f"Loss cooldown is active until {cooldown_until.strftime('%H:%M UTC')}"
    trades_key, pnl_key = _mode_stat_keys(mode)
    controller_pnl = float(_safe_float(state.get(pnl_key)) or 0.0) + _open_unrealized_pnl(state, current_price)
    loss_limit_amount = _loss_limit_amount(state, equity)
    if controller_pnl <= -abs(loss_limit_amount):
        if not state.get("kill_switch"):
            state["kill_switch"] = True
            state["last_action"] = "auto_stop_loss"
            state["last_action_detail"] = f"Controller stop-loss reached: pnl ${controller_pnl:+.2f} vs limit ${loss_limit_amount:,.2f}"
            _record_event(
                {
                    "kind": "auto_stop_loss",
                    "symbol": sym,
                    "controller_pnl": controller_pnl,
                    "loss_limit": loss_limit_amount,
                }
            )
        stock_status["allowed"] = False
        stock_status["label"] = "Loss Limit"
        stock_status["detail"] = f"Controller stop-loss reached: pnl ${controller_pnl:+.2f} vs limit ${loss_limit_amount:,.2f}"
    if int(state.get(trades_key) or 0) >= int(AUTONOMY_MAX_TRADES_PER_DAY):
        stock_status["allowed"] = False
        stock_status["label"] = "Trade Limit"
        stock_status["detail"] = "Daily autonomy trade limit has been reached"
    if state.get("last_signal_key") and state.get("last_signal_key") == stock_status.get("signal_key"):
        last_decision_at = _parse_ts(state.get("last_decision_at"))
        if last_decision_at is not None and (now_utc - last_decision_at).total_seconds() < (60.0 * float(AUTONOMY_ENTRY_COOLDOWN_MIN)):
            stock_status["allowed"] = False
            stock_status["label"] = "Duplicate"
            stock_status["detail"] = "Duplicate signal suppressed during the entry cooldown window"
    if state.get("kill_switch") and mode in AUTONOMY_MODES:
        stock_status["allowed"] = False
        stock_status["label"] = "Halted"
        stock_status["detail"] = str(state.get("last_action_detail") or "Autonomy halted")

    if mode == "off":
        decision = {"status": "off", "detail": "Controller is off; advisory mode only"}
    elif state.get("kill_switch"):
        if shadow_position:
            state, event = _close_shadow_position(state, current_price, "kill switch", now_utc=now_utc)
            decision = {"status": "shadow_exit", "detail": state.get("last_action_detail"), "status_message": f"Shadow position closed for {sym}"}
            _record_event(event)
        elif open_trade and route_mode:
            result = flatten_routed_position(
                sym,
                current_price=current_price,
                profile_name=profile_name,
                client_order_id=_build_client_order_id(sym, mode, "flat", ts_value=now_utc),
                force=True,
            )
            decision = {
                "status": str(result.get("status") or "blocked"),
                "detail": f"Kill switch flatten: {result.get('reason') or result.get('status') or '-'}",
                "status_message": f"Autonomy halt requested flatten for {sym}",
            }
            _record_event({"kind": "kill_switch_flatten", "result": result, "symbol": sym})
            if str(result.get("status") or "") in {"filled", "submitted"}:
                state["open_trade"] = None if str(result.get("status")) == "filled" else state.get("open_trade")
        else:
            decision = {"status": "halted", "detail": str(state.get("last_action_detail") or "Autonomy halted")}
    elif shadow_position:
        hold_minutes = None
        opened_at = _parse_ts(shadow_position.get("opened_at"))
        if opened_at is not None:
            hold_minutes = max(0.0, float((now_utc - opened_at).total_seconds()) / 60.0)
        side = str(shadow_position.get("side") or "buy").lower()
        stop_price = _safe_float((advisory or {}).get("plan_meta", {}).get("stop"))
        target_price = _safe_float((advisory or {}).get("plan_meta", {}).get("target1"))
        exit_reason = None
        if stock_status.get("market", {}).get("session", {}).get("flat_window"):
            exit_reason = "flat by close"
        elif hold_minutes is not None and hold_minutes >= float(AUTONOMY_MAX_HOLD_MIN):
            exit_reason = "max hold reached"
        elif side == "buy" and stop_price is not None and current_price is not None and current_price <= stop_price:
            exit_reason = "stop hit"
        elif side == "buy" and target_price is not None and current_price is not None and current_price >= target_price:
            exit_reason = "target hit"
        elif side == "sell" and stop_price is not None and current_price is not None and current_price >= stop_price:
            exit_reason = "stop hit"
        elif side == "sell" and target_price is not None and current_price is not None and current_price <= target_price:
            exit_reason = "target hit"
        elif str((advisory or {}).get("quality_meta", {}).get("no_trade")).lower() == "true":
            exit_reason = "quality degraded"
        if exit_reason:
            state, event = _close_shadow_position(state, current_price, exit_reason, now_utc=now_utc)
            decision = {"status": "shadow_exit", "detail": state.get("last_action_detail"), "status_message": f"Shadow exit for {sym}: {exit_reason}"}
            _record_event(event)
        else:
            decision = {"status": "shadow_hold", "detail": "Shadow position remains open"}
    elif open_trade:
        hold_minutes = None
        opened_at = _parse_ts(open_trade.get("opened_at"))
        if opened_at is not None:
            hold_minutes = max(0.0, float((now_utc - opened_at).total_seconds()) / 60.0)
        side = str(open_trade.get("side") or "buy").lower()
        stop_price = _safe_float((actual_position or {}).get("stop_price")) or _safe_float(open_trade.get("stop_price")) or _safe_float((advisory or {}).get("plan_meta", {}).get("stop"))
        target_price = _safe_float((actual_position or {}).get("target_price")) or _safe_float(open_trade.get("target_price")) or _safe_float((advisory or {}).get("plan_meta", {}).get("target1"))
        exit_reason = None
        if stock_status.get("market", {}).get("session", {}).get("flat_window"):
            exit_reason = "flat by close"
        elif hold_minutes is not None and hold_minutes >= float(AUTONOMY_MAX_HOLD_MIN):
            exit_reason = "max hold reached"
        elif stock_status.get("market", {}).get("stale") or stock_status.get("market", {}).get("delayed"):
            exit_reason = "market data degraded"
        elif side == "buy" and stop_price is not None and current_price is not None and current_price <= stop_price:
            exit_reason = "stop hit"
        elif side == "buy" and target_price is not None and current_price is not None and current_price >= target_price:
            exit_reason = "target hit"
        elif side == "sell" and stop_price is not None and current_price is not None and current_price >= stop_price:
            exit_reason = "stop hit"
        elif side == "sell" and target_price is not None and current_price is not None and current_price <= target_price:
            exit_reason = "target hit"
        elif bool((advisory or {}).get("quality_meta", {}).get("no_trade")):
            exit_reason = "quality degraded"
        elif stock_status.get("news_conflict"):
            exit_reason = "news conflict"
        if exit_reason:
            result = flatten_routed_position(
                sym,
                current_price=current_price,
                profile_name=profile_name,
                client_order_id=_build_client_order_id(sym, mode, "flat", ts_value=now_utc),
                force=True,
            )
            decision = {
                "status": str(result.get("status") or "blocked"),
                "detail": f"{exit_reason} | {result.get('reason') or result.get('status') or '-'}",
                "status_message": f"Autonomy exit for {sym}: {exit_reason}",
            }
            _record_event({"kind": "actual_exit", "reason": exit_reason, "result": result, "symbol": sym})
            if str(result.get("status") or "") == "filled":
                entry_price = _safe_float(open_trade.get("entry_price")) or _safe_float((actual_position or {}).get("avg_price")) or _safe_float(current_price) or 0.0
                qty = int(abs(_safe_float(open_trade.get("qty")) or _safe_float((actual_position or {}).get("qty")) or 0))
                pnl = ((float(current_price or entry_price) - float(entry_price)) * qty) if side == "buy" else ((float(entry_price) - float(current_price or entry_price)) * qty)
                state = _record_pnl(state, mode, pnl, now_utc=now_utc)
                state["open_trade"] = None
            elif str(result.get("status") or "") == "submitted":
                state["pending_order"] = {
                    "kind": "exit",
                    "created_at": now_utc.isoformat(),
                    "order_id": result.get("order_id"),
                    "client_order_id": result.get("client_order_id"),
                }
        else:
            decision = {"status": "monitoring", "detail": "Autonomous position remains open"}
    else:
        current_external_position = actual_position if mode in _ACTUAL_MODES else None
        if current_external_position is not None:
            stock_status["allowed"] = False
            stock_status["label"] = "External Position"
            stock_status["detail"] = "A manual position is already open; the controller will not take it over"
            decision = {"status": "blocked", "detail": stock_status["detail"]}
        elif mode == "shadow" and stock_status.get("allowed"):
            qty = max(1, int(stock_status.get("qty") or 0))
            signal_key = str(stock_status.get("signal_key") or "")
            detail = f"Shadow {str(stock_status.get('side') or '').upper()} {qty} sh | {stock_status.get('detail') or '-'}"
            state, event = _enter_shadow_position(state, stock_status.get("side"), qty, current_price, signal_key, detail, now_utc=now_utc)
            decision = {"status": "shadow_entry", "detail": detail, "status_message": f"Shadow entry for {sym}: {stock_status.get('side')}"}
            _record_event(event)
        elif mode in _ACTUAL_MODES and stock_status.get("allowed"):
            qty = max(1, int(stock_status.get("qty") or 0))
            client_order_id = _build_client_order_id(sym, mode, stock_status.get("side"), ts_value=now_utc)
            entry_advisory = stock_status.get("execution_advisory") if isinstance(stock_status.get("execution_advisory"), dict) else advisory
            result = place_routed_order(
                sym,
                stock_status.get("side"),
                qty=qty,
                current_price=current_price,
                advisory=entry_advisory,
                profile_name=profile_name,
                client_order_id=client_order_id,
                force=False,
            )
            decision = {
                "status": str(result.get("status") or "blocked"),
                "detail": f"{stock_status.get('side')} {qty} sh | {result.get('reason') or result.get('status') or '-'}",
                "status_message": f"Autonomy order for {sym}: {stock_status.get('side')} {qty} sh" if str(result.get('status') or '') in {'filled', 'submitted'} else None,
            }
            _record_event({"kind": "actual_entry", "symbol": sym, "side": stock_status.get("side"), "qty": qty, "result": result, "route_mode": route_mode})
            if str(result.get("status") or "") == "filled":
                state["open_trade"] = {
                    "side": stock_status.get("side"),
                    "qty": qty,
                    "entry_price": float(current_price or 0.0),
                    "opened_at": now_utc.isoformat(),
                    "route_mode": route_mode,
                    "client_order_id": client_order_id,
                    "order_id": result.get("order_id"),
                    "stop_price": _safe_float((entry_advisory or {}).get("plan_meta", {}).get("stop")),
                    "target_price": _safe_float((entry_advisory or {}).get("plan_meta", {}).get("target1")),
                }
            elif str(result.get("status") or "") == "submitted":
                state["pending_order"] = {
                    "kind": "entry",
                    "created_at": now_utc.isoformat(),
                    "order_id": result.get("order_id"),
                    "client_order_id": client_order_id,
                    "side": stock_status.get("side"),
                    "qty": qty,
                }
            state["last_signal_key"] = stock_status.get("signal_key")
        else:
            decision = {"status": "blocked", "detail": str(stock_status.get("detail") or "No autonomy action")}

    state["last_decision_at"] = now_utc.isoformat()
    state["last_block_reason"] = None if stock_status.get("allowed") else str(stock_status.get("detail") or decision.get("detail") or "")
    state["last_action"] = decision.get("status") or state.get("last_action")
    state["last_action_detail"] = decision.get("detail") or state.get("last_action_detail")
    if mode == "shadow":
        _record_event(_shadow_observation_event(sym, stock_status, decision, now_utc=now_utc))
    if persist:
        state = _save_state(sym, state)

    snapshot = _build_snapshot(
        sym,
        state,
        mode,
        route_label,
        stock_status=stock_status,
        options_status=options_status,
        decision=decision,
        now_utc=now_utc,
        events=emitted_events,
    )
    if mode in _ACTUAL_MODES and str(decision.get("status") or "") in {"filled", "submitted"}:
        try:
            snapshot["execution_override"] = get_execution_dashboard_snapshot(sym, current_price=current_price, advisory=advisory, profile_name=profile_name)
        except Exception:
            pass
    return snapshot
