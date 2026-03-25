from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from config import DATA_DIR, PAPER_STARTING_CASH
from execution_engine import get_portfolio_snapshot_from_state, new_paper_state, recommend_order_for_state
from performance_analytics import get_performance_snapshot_from_trades
from trader_profiles import classify_confidence_bucket


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type {type(obj)!r}")


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _session_path() -> str:
    return os.path.join(DATA_DIR, "active_replay_session.json")


def _bar_to_record(ts_value, row: pd.Series) -> dict:
    ts = pd.Timestamp(ts_value)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return {
        "timestamp": ts.isoformat(),
        "open": float(pd.to_numeric(row.get("open"), errors="coerce")),
        "high": float(pd.to_numeric(row.get("high"), errors="coerce")),
        "low": float(pd.to_numeric(row.get("low"), errors="coerce")),
        "close": float(pd.to_numeric(row.get("close"), errors="coerce")),
        "volume": float(pd.to_numeric(row.get("volume"), errors="coerce") if "volume" in row else 0.0),
    }


def _session_summary(session: dict | None) -> dict:
    session = dict(session or {})
    bars = list(session.get("bars") or [])
    current = dict(session.get("current_bar") or {})
    state = dict(session.get("paper_state") or {})
    price = _safe_float(current.get("close"))
    portfolio = get_portfolio_snapshot_from_state(
        state if state else new_paper_state(),
        current_prices={str(session.get("symbol") or "").upper(): price} if price is not None else None,
    )
    perf = get_performance_snapshot_from_trades(state.get("closed_trades") or [], symbol=session.get("symbol"))
    status = str(session.get("status") or "inactive")
    active = status in {"paused", "active", "completed"} and bool(bars)
    if not active:
        return {
            "active": False,
            "status": status if status != "inactive" else "inactive",
            "card": "Replay: off",
            "detail": "Simulation inactive",
            "rows": [("Replay", "Off", "Simulation inactive")],
            "performance_rows": [],
            "portfolio_rows": [],
            "session": session,
        }
    cursor = int(session.get("cursor") or 0)
    total = int(len(bars))
    status_label = "Completed" if status == "completed" else "Running" if status == "active" else "Paused"
    current_price = _safe_float(current.get("close"))
    detail = f"{str(session.get('symbol') or '-').upper()} | {status_label.lower()} | bar {cursor + 1}/{total}"
    if current.get("timestamp"):
        detail = f"{detail} | {current.get('timestamp')}"
    pos = next((p for p in portfolio.get("positions", []) if p.get("symbol") == str(session.get("symbol") or "").upper()), None)
    trade_count = int(perf.get("trade_count") or 0)
    rows = [
        ("Replay", status_label, "Simulation only"),
        ("Progress", f"{cursor + 1}/{total}", f"Start {int(session.get('start_index') or 0) + 1}/{total}"),
        ("Bar Time", str(current.get("timestamp") or "-"), "-" if current_price is None else f"Close ${current_price:,.2f}"),
        ("Route", "Replay Paper", "Orders stay inside replay"),
        (
            "Position",
            "-" if not pos else f"{pos.get('side')} {abs(int(_safe_float(pos.get('qty')) or 0))}",
            "-" if not pos else f"Avg ${(_safe_float(pos.get('avg_price')) or 0.0):.2f} | UPL ${(_safe_float(pos.get('unrealized_pnl')) or 0.0):+.2f}",
        ),
        (
            "Replay PnL",
            f"${(_safe_float(portfolio.get('unrealized_pnl')) or 0.0):+.2f}",
            perf.get("summary", "No closed replay trades yet"),
        ),
        ("Replay Trades", str(trade_count), "Paper simulation only"),
    ]
    portfolio_rows = [
        ("Replay Equity", f"${(_safe_float(portfolio.get('equity')) or 0.0):,.2f}", f"Cash ${(_safe_float(portfolio.get('cash')) or 0.0):,.2f}"),
        ("Gross / Net", f"{100.0 * (_safe_float(portfolio.get('gross_exposure_pct')) or 0.0):.1f}% / {100.0 * (_safe_float(portfolio.get('net_exposure_pct')) or 0.0):+.1f}%", f"{int(portfolio.get('open_positions') or 0)} open replay positions"),
    ]
    return {
        "active": True,
        "status": status,
        "paused": bool(session.get("paused", True)),
        "completed": bool(session.get("done", False)),
        "card": f"Replay: {status_label}",
        "detail": detail,
        "rows": rows,
        "performance_rows": perf.get("rows", []),
        "portfolio_rows": portfolio_rows,
        "session": session,
    }


def _normalize_session(session: dict | None) -> dict:
    session = dict(session or {})
    bars = list(session.get("bars") or [])
    cursor = int(session.get("cursor") or 0)
    cursor = min(max(0, cursor), max(0, len(bars) - 1)) if bars else 0
    session["bars"] = bars
    session["cursor"] = cursor
    session["current_bar"] = bars[cursor] if bars else None
    session["paused"] = bool(session.get("paused", True))
    session["done"] = bool(session.get("done", False))
    if "paper_state" not in session or not isinstance(session.get("paper_state"), dict):
        session["paper_state"] = new_paper_state(broker_mode="replay_paper")
    return session


def _load_active_session() -> dict:
    path = _session_path()
    if not os.path.exists(path):
        return {"status": "inactive", "bars": []}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return _normalize_session(payload)
    except Exception:
        return {"status": "inactive", "bars": []}


def _save_active_session(session: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_session_path(), "w") as f:
            json.dump(_normalize_session(session), f, default=_json_default, indent=2, sort_keys=True)
    except Exception:
        pass


def clear_replay_session() -> dict:
    try:
        os.remove(_session_path())
    except FileNotFoundError:
        pass
    except Exception:
        pass
    return _session_summary({"status": "inactive", "bars": []})


def build_replay_session(
    df: pd.DataFrame,
    symbol: str,
    starting_cash: float | None = None,
    profile_name: str | None = None,
    start_index: int | None = None,
) -> dict:
    if df is None or df.empty:
        return {
            "status": "empty",
            "mode": "replay",
            "symbol": str(symbol or "").upper(),
            "bars": [],
            "cursor": 0,
            "start_index": 0,
            "current_bar": None,
            "starting_cash": float(starting_cash if starting_cash is not None else PAPER_STARTING_CASH),
            "profile": profile_name,
            "paper_state": new_paper_state(starting_cash=starting_cash, broker_mode="replay_paper"),
            "created_at": _utc_now(),
        }
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.sort_index()
    bars = [_bar_to_record(idx, row) for idx, row in frame.iterrows()]
    cursor = int(start_index if start_index is not None else 0)
    cursor = min(max(0, cursor), len(bars) - 1)
    return {
        "status": "ready",
        "mode": "replay",
        "symbol": str(symbol or "").upper(),
        "bars": bars,
        "cursor": cursor,
        "start_index": cursor,
        "current_bar": bars[cursor] if bars else None,
        "done": False if cursor < len(bars) - 1 else True,
        "paused": True,
        "starting_cash": float(starting_cash if starting_cash is not None else PAPER_STARTING_CASH),
        "profile": profile_name,
        "paper_state": new_paper_state(starting_cash=starting_cash, broker_mode="replay_paper"),
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }


def _time_bucket(ts_value) -> str:
    try:
        ts = pd.Timestamp(ts_value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        hour = int(ts.tz_convert("America/New_York").hour)
    except Exception:
        return "Unknown"
    if hour < 11:
        return "AM"
    if hour < 14:
        return "Midday"
    return "PM"


def _record_replay_trade(state: dict, symbol: str, side: str, qty: float, entry_price: float, exit_price: float, pos_meta: dict, confidence) -> None:
    qty_abs = abs(float(qty))
    pnl = (exit_price - entry_price) * qty_abs if side == "long" else (entry_price - exit_price) * qty_abs
    risk_per_share = _safe_float(pos_meta.get("risk_per_share"))
    risk_dollars = (risk_per_share or 0.0) * qty_abs
    opened_at = pos_meta.get("opened_at")
    closed_at = _utc_now()
    hold_minutes = None
    try:
        if opened_at:
            hold_minutes = max(0.0, (pd.Timestamp(closed_at) - pd.Timestamp(opened_at)).total_seconds() / 60.0)
    except Exception:
        hold_minutes = None
    trade = {
        "symbol": str(symbol).upper(),
        "side": side,
        "qty": qty_abs,
        "entry_price": float(entry_price),
        "exit_price": float(exit_price),
        "pnl_dollars": float(pnl),
        "pnl_pct": float((pnl / max(0.01, entry_price * qty_abs))),
        "risk_multiple": float(pnl / risk_dollars) if risk_dollars not in (None, 0.0) else None,
        "opened_at": opened_at,
        "closed_at": closed_at,
        "hold_minutes": hold_minutes,
        "setup_type": pos_meta.get("setup_type"),
        "regime": pos_meta.get("regime"),
        "style": pos_meta.get("style"),
        "confidence": confidence,
        "confidence_bucket": classify_confidence_bucket(confidence),
        "quality_score": _safe_float(pos_meta.get("quality_score")),
        "time_bucket": _time_bucket(closed_at),
    }
    state.setdefault("closed_trades", []).append(trade)


def _apply_replay_fill(state: dict, symbol: str, side: str, qty: int, price: float, advisory: dict | None, rec: dict | None, profile_name: str | None) -> dict:
    positions = state.setdefault("positions", {})
    signed_qty = int(qty) if str(side).lower() == "buy" else -int(qty)
    cost = float(signed_qty * price)
    state["cash"] = float((_safe_float(state.get("cash")) or 0.0) - cost)
    pos = dict(positions.get(symbol) or {})
    old_qty = _safe_float(pos.get("qty")) or 0.0
    old_avg = _safe_float(pos.get("avg_price")) or 0.0
    new_qty = old_qty + signed_qty
    confidence = _safe_float((((advisory or {}).get("pred_state") or {}).get("intraday") or {}).get("p_up"))
    order = {
        "order_id": uuid.uuid4().hex[:12],
        "broker_mode": "replay_paper",
        "symbol": symbol,
        "side": str(side).lower(),
        "qty": int(qty),
        "status": "filled",
        "created_at": _utc_now(),
        "filled_at": _utc_now(),
        "fill_price": float(price),
        "profile": profile_name,
        "setup_type": ((advisory or {}).get("plan_meta") or {}).get("action") if isinstance(advisory, dict) else None,
        "regime": ((advisory or {}).get("quality_meta") or {}).get("regime_label") if isinstance(advisory, dict) else None,
        "confidence": confidence,
        "quality_score": _safe_float((rec or {}).get("quality_score")),
        "stop_price": _safe_float((rec or {}).get("stop_price")),
        "take_profit": _safe_float((rec or {}).get("target_price")),
        "risk_per_share": _safe_float((rec or {}).get("risk_per_share")),
    }
    state.setdefault("orders", []).append(order)
    state.setdefault("fills", []).append(
        {
            "fill_id": uuid.uuid4().hex[:12],
            "order_id": order["order_id"],
            "symbol": symbol,
            "side": side,
            "qty": int(qty),
            "fill_price": float(price),
            "filled_at": order["filled_at"],
            "status": "filled",
        }
    )
    if old_qty == 0.0 or (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
        total_qty = abs(old_qty) + abs(signed_qty)
        avg_price = ((abs(old_qty) * old_avg) + (abs(signed_qty) * price)) / max(1.0, total_qty)
        positions[symbol] = {
            "qty": float(new_qty),
            "avg_price": float(avg_price),
            "opened_at": pos.get("opened_at") or order["filled_at"],
            "setup_type": order.get("setup_type"),
            "regime": order.get("regime"),
            "style": profile_name,
            "confidence": confidence,
            "quality_score": order.get("quality_score"),
            "stop_price": order.get("stop_price"),
            "target_price": order.get("take_profit"),
            "risk_per_share": order.get("risk_per_share"),
        }
    else:
        close_qty = min(abs(old_qty), abs(signed_qty))
        closing_side = "long" if old_qty > 0 else "short"
        _record_replay_trade(state, symbol, closing_side, close_qty, old_avg, price, pos, confidence)
        if new_qty == 0:
            positions.pop(symbol, None)
        elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
            positions[symbol] = {
                "qty": float(new_qty),
                "avg_price": float(price),
                "opened_at": order["filled_at"],
                "setup_type": order.get("setup_type"),
                "regime": order.get("regime"),
                "style": profile_name,
                "confidence": confidence,
                "quality_score": order.get("quality_score"),
                "stop_price": order.get("stop_price"),
                "target_price": order.get("take_profit"),
                "risk_per_share": order.get("risk_per_share"),
            }
        else:
            pos["qty"] = float(new_qty)
            positions[symbol] = pos
    state["updated_at"] = _utc_now()
    return order


def replay_frame_from_session(session: dict | None) -> pd.DataFrame:
    session = _normalize_session(session)
    bars = list(session.get("bars") or [])
    if not bars:
        return pd.DataFrame()
    cursor = int(session.get("cursor") or 0)
    rows = bars[: cursor + 1]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.set_index("timestamp")
    return df.sort_index()


def get_active_replay_frame(symbol: str | None = None) -> pd.DataFrame:
    session = _load_active_session()
    if symbol and str(session.get("symbol") or "").upper() != str(symbol).upper():
        return pd.DataFrame()
    return replay_frame_from_session(session)


def get_replay_dashboard_snapshot(symbol: str | None = None) -> dict:
    session = _load_active_session()
    if symbol and session.get("symbol") and str(session.get("symbol")).upper() != str(symbol).upper():
        return {
            "active": False,
            "status": "other_symbol",
            "card": "Replay: other symbol",
            "detail": f"Active replay on {str(session.get('symbol')).upper()}",
            "rows": [("Replay", "Other Symbol", f"Active on {str(session.get('symbol')).upper()}")],
            "performance_rows": [],
            "portfolio_rows": [],
            "session": session,
        }
    return _session_summary(session)


def start_active_replay(
    df: pd.DataFrame,
    symbol: str,
    starting_cash: float | None = None,
    profile_name: str | None = None,
    warmup_bars: int = 120,
) -> dict:
    if df is None or df.empty:
        session = build_replay_session(df, symbol, starting_cash=starting_cash, profile_name=profile_name)
        _save_active_session(session)
        return _session_summary(session)
    frame = df.copy()
    frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
    frame = frame.sort_index()
    max_start = max(0, len(frame) - 2)
    start_index = min(max_start, max(0, int(warmup_bars) - 1))
    session = build_replay_session(frame, symbol, starting_cash=starting_cash, profile_name=profile_name, start_index=start_index)
    session["status"] = "paused" if not session.get("done") else "completed"
    session["paused"] = True
    session["updated_at"] = _utc_now()
    _save_active_session(session)
    return _session_summary(session)


def reset_active_replay() -> dict:
    session = _load_active_session()
    bars = list(session.get("bars") or [])
    if not bars:
        return _session_summary(session)
    start_index = min(max(0, int(session.get("start_index") or 0)), len(bars) - 1)
    starting_cash = _safe_float(session.get("starting_cash"))
    session["cursor"] = start_index
    session["current_bar"] = bars[start_index]
    session["done"] = bool(start_index >= len(bars) - 1)
    session["paused"] = True
    session["status"] = "completed" if session["done"] else "paused"
    session["paper_state"] = new_paper_state(starting_cash=starting_cash, broker_mode="replay_paper")
    session["updated_at"] = _utc_now()
    _save_active_session(session)
    return _session_summary(session)


def step_replay_session(session: dict, steps: int = 1) -> dict:
    out = _normalize_session(session)
    bars = list(out.get("bars") or [])
    if not bars:
        out["status"] = "empty"
        out["current_bar"] = None
        out["done"] = True
        return out
    cur = int(out.get("cursor") or 0)
    nxt = min(max(0, cur + max(1, int(steps))), len(bars) - 1)
    out["cursor"] = nxt
    out["current_bar"] = bars[nxt]
    out["done"] = bool(nxt >= len(bars) - 1)
    out["status"] = "completed" if out["done"] else ("paused" if out.get("paused", True) else "active")
    out["updated_at"] = _utc_now()
    return out


def step_active_replay(steps: int = 1) -> dict:
    session = _load_active_session()
    if not list(session.get("bars") or []):
        return _session_summary(session)
    session = step_replay_session(session, steps=steps)
    session["paused"] = True if not session.get("done") else True
    session["status"] = "completed" if session.get("done") else "paused"
    _save_active_session(session)
    return _session_summary(session)


def set_replay_paused(paused: bool) -> dict:
    session = _load_active_session()
    if not list(session.get("bars") or []):
        return _session_summary(session)
    session["paused"] = bool(paused)
    if session.get("done"):
        session["status"] = "completed"
    else:
        session["status"] = "paused" if paused else "active"
    session["updated_at"] = _utc_now()
    _save_active_session(session)
    return _session_summary(session)


def stop_active_replay() -> dict:
    session = _load_active_session()
    session["status"] = "stopped"
    session["paused"] = True
    session["done"] = True
    session["updated_at"] = _utc_now()
    _save_active_session(session)
    clear_replay_session()
    return _session_summary({"status": "inactive", "bars": []})


def place_replay_order(side: str, advisory: dict | None = None, qty: int | None = None) -> dict:
    session = _load_active_session()
    if not list(session.get("bars") or []):
        return {"status": "blocked", "reason": "replay inactive"}
    if session.get("done"):
        return {"status": "blocked", "reason": "replay completed"}
    symbol = str(session.get("symbol") or "").upper()
    current_price = _safe_float((session.get("current_bar") or {}).get("close"))
    state = dict(session.get("paper_state") or new_paper_state(broker_mode="replay_paper"))
    rec = recommend_order_for_state(
        state,
        symbol,
        current_price=current_price,
        advisory=advisory,
        profile_name=session.get("profile"),
        side=side,
    )
    qty_final = int(qty if qty is not None else rec.get("qty") or 0)
    if qty_final <= 0:
        return {"status": "blocked", "reason": rec.get("reason") or "size unavailable", "recommendation": rec}
    order = _apply_replay_fill(
        state,
        symbol,
        side,
        qty_final,
        current_price if current_price is not None else 0.0,
        advisory=advisory,
        rec=rec,
        profile_name=session.get("profile"),
    )
    session["paper_state"] = state
    session["updated_at"] = _utc_now()
    _save_active_session(session)
    return {
        "status": "filled",
        "routing_mode": "replay_paper",
        "order": order,
        "recommendation": rec,
        "replay": _session_summary(session),
    }


def flatten_replay_position(advisory: dict | None = None) -> dict:
    session = _load_active_session()
    if not list(session.get("bars") or []):
        return {"status": "blocked", "reason": "replay inactive"}
    symbol = str(session.get("symbol") or "").upper()
    state = dict(session.get("paper_state") or {})
    pos = (state.get("positions") or {}).get(symbol)
    qty = int(abs(_safe_float((pos or {}).get("qty")) or 0))
    if qty <= 0:
        return {"status": "blocked", "reason": "no open replay position"}
    side = "sell" if (_safe_float((pos or {}).get("qty")) or 0.0) > 0 else "buy"
    return place_replay_order(side, advisory=advisory, qty=qty)
