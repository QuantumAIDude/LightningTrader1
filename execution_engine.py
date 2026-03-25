from __future__ import annotations

import json
import math
import os
import uuid
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from credential_store import (
    clear_saved_alpaca_credentials,
    get_runtime_broker_settings,
    save_alpaca_credentials,
)
from config import (
    DATA_DIR,
    PAPER_STARTING_CASH,
    PAPER_TRADING_ENABLED,
)
from trader_profiles import classify_confidence_bucket, get_trader_profile

try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest, StopLossRequest, TakeProfitRequest

    _ALPACA_TRADING_AVAILABLE = True
except Exception:
    TradingClient = None
    OrderSide = None
    QueryOrderStatus = None
    TimeInForce = None
    GetOrdersRequest = None
    MarketOrderRequest = None
    StopLossRequest = None
    TakeProfitRequest = None
    _ALPACA_TRADING_AVAILABLE = False


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _broker_settings(settings_override: dict | None = None) -> dict:
    return dict(get_runtime_broker_settings(settings_override))


def _classify_broker_exception(exc: Exception) -> tuple[str, str]:
    text = str(exc or "").strip()
    lowered = text.lower()
    if any(token in lowered for token in ("401", "403", "forbidden", "unauthorized", "authentication", "invalid api", "credentials")):
        return "invalid_credentials", "Alpaca rejected the API key or secret."
    if any(token in lowered for token in ("dns", "name or service not known", "failed to resolve", "temporary failure", "connection", "network", "timed out", "timeout", "ssl")):
        return "network_error", "The app could not reach Alpaca right now. Check your internet connection and try again."
    return "error", text or "Broker connection failed"


def _call_with_optional_settings(func, *args, settings_override: dict | None = None, **kwargs):
    try:
        return func(*args, settings_override=settings_override, **kwargs)
    except TypeError as exc:
        if "settings_override" not in str(exc):
            raise
        return func(*args, **kwargs)


def _state_path() -> str:
    return os.path.join(DATA_DIR, "paper_broker_state.json")


def _broker_verification_path() -> str:
    return os.path.join(DATA_DIR, "broker_verification_status.json")


def _json_default(obj):
    if isinstance(obj, (datetime, pd.Timestamp)):
        return pd.Timestamp(obj).isoformat()
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Unsupported type {type(obj)!r}")


def _json_safe(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "__dict__"):
        try:
            return {str(k): _json_safe(v) for k, v in vars(value).items() if not str(k).startswith("_")}
        except Exception:
            return repr(value)
    return repr(value)


def _default_state() -> dict:
    return {
        "broker_mode": "paper",
        "starting_cash": float(PAPER_STARTING_CASH),
        "cash": float(PAPER_STARTING_CASH),
        "orders": [],
        "fills": [],
        "positions": {},
        "closed_trades": [],
        "updated_at": _utc_now(),
    }


def _load_state() -> dict:
    path = _state_path()
    if not os.path.exists(path):
        return _default_state()
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            base = _default_state()
            base.update(payload)
            base["orders"] = list(base.get("orders") or [])
            base["fills"] = list(base.get("fills") or [])
            base["closed_trades"] = list(base.get("closed_trades") or [])
            base["positions"] = dict(base.get("positions") or {})
            return base
    except Exception:
        pass
    return _default_state()


def _save_state(state: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_state_path(), "w") as f:
            json.dump(state, f, default=_json_default, indent=2, sort_keys=True)
    except Exception:
        pass


def reset_paper_state(starting_cash: float | None = None) -> dict:
    state = _default_state()
    if starting_cash is not None:
        state["starting_cash"] = float(starting_cash)
        state["cash"] = float(starting_cash)
    _save_state(state)
    return state


def _load_broker_verification() -> dict:
    path = _broker_verification_path()
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def _save_broker_verification(payload: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_broker_verification_path(), "w") as f:
            json.dump(_json_safe(dict(payload or {})), f, default=_json_default, indent=2, sort_keys=True)
    except Exception:
        pass


def new_paper_state(starting_cash: float | None = None, broker_mode: str = "paper") -> dict:
    state = _default_state()
    if starting_cash is not None:
        state["starting_cash"] = float(starting_cash)
        state["cash"] = float(starting_cash)
    state["broker_mode"] = str(broker_mode or "paper")
    state["updated_at"] = _utc_now()
    return state


def _normalize_broker_mode(mode_value: str | None = None) -> str:
    if mode_value is None:
        mode_value = _broker_settings().get("broker_mode")
    mode = str(mode_value or "local_paper").strip().lower()
    if mode in {"paper", "local_paper", "paper_local"}:
        return "local_paper"
    if mode in {"alpaca_paper", "broker_paper", "paper_broker"}:
        return "alpaca_paper"
    if mode in {"alpaca_live", "live", "broker_live"}:
        return "alpaca_live"
    if mode in {"disabled", "off", "none"}:
        return "disabled"
    return "local_paper"


def _broker_mode_label(mode_key: str | None) -> str:
    mode = str(mode_key or "").strip().lower()
    mapping = {
        "local_paper": "Local Paper",
        "alpaca_paper": "Alpaca Paper",
        "alpaca_live": "Alpaca Live",
        "disabled": "Disabled",
        "unavailable": "Unavailable",
    }
    return mapping.get(mode, str(mode_key or "Unavailable").replace("_", " ").title())


def _fmt_bool(value: bool | None) -> str:
    if value is None:
        return "-"
    return "Yes" if bool(value) else "No"


def _broker_operator_hint(
    *,
    provider: str,
    requested_mode: str,
    runtime_status: str,
    sdk_installed: bool | None,
    credentials_present: bool | None,
    connected: bool,
    paper_mode: bool = True,
) -> str:
    provider = str(provider or "alpaca").lower()
    requested_mode = str(requested_mode or "local_paper").lower()
    runtime_status = str(runtime_status or "unavailable").lower()
    if requested_mode == "disabled":
        return "Broker routing is disabled. Orders stay off the broker path until you change BROKER_MODE_DEFAULT."
    if requested_mode == "local_paper":
        return "Orders stay in Local Paper right now. Open Alpaca Setup when you want broker-routed paper trading."
    if provider != "alpaca":
        return f"Provider '{provider}' is not wired for live routing here. Use Local Paper or wire a supported broker adapter."
    if not sdk_installed:
        return "The Alpaca connection component is missing in this app runtime. Install alpaca-py, then try again."
    if not credentials_present:
        return "Open Alpaca Setup and enter your API key and secret. Paper mode is recommended."
    if requested_mode == "alpaca_paper" and not paper_mode:
        return "Turn Paper Mode on in Alpaca Setup to use the safe Alpaca paper route."
    if requested_mode == "alpaca_live" and paper_mode:
        return "Live trading is still set to paper-safe mode. Only switch this off if you intentionally want live routing."
    if runtime_status == "invalid_credentials":
        return "Alpaca rejected the saved credentials. Re-open Alpaca Setup and check the key and secret."
    if runtime_status == "network_error":
        return "The app could not reach Alpaca right now. Check your internet connection and try Refresh Broker again."
    if connected and requested_mode == "alpaca_paper":
        return "Alpaca Paper looks connected. Use 'Verify Alpaca Paper' to run the acceptance check."
    if connected and requested_mode == "alpaca_live":
        return "Alpaca Live appears connected. Keep paper-safe defaults unless you intentionally want live routing."
    if runtime_status == "fallback_local_paper":
        return "Alpaca is unavailable right now, so orders route to Local Paper until the broker path is healthy."
    return "Use 'Refresh Broker' after fixing SDK, credentials, or network access to re-check the broker path."


def _live_adapter_status(settings_override: dict | None = None) -> dict:
    settings = _broker_settings(settings_override)
    provider = str(settings.get("provider") or "alpaca").lower()
    if provider != "alpaca":
        return {
            "status": "unavailable",
            "provider": provider,
            "mode": "live",
            "detail": "Only Alpaca live adapter is wired",
            "connected": False,
        }
    if not _ALPACA_TRADING_AVAILABLE:
        return {
            "status": "sdk_missing",
            "provider": "alpaca",
            "mode": "live",
            "detail": "alpaca.trading SDK not installed",
            "connected": False,
        }
    if not settings.get("credentials_present"):
        return {
            "status": "credentials_missing",
            "provider": "alpaca",
            "mode": "paper" if settings.get("paper", True) else "live",
            "detail": "Alpaca API key / secret missing",
            "connected": False,
        }
    return {
        "status": "adapter_ready",
        "provider": "alpaca",
        "mode": "paper" if settings.get("paper", True) else "live",
        "detail": "Adapter present; attempting account connection",
        "connected": False,
    }


def _alpaca_client(settings_override: dict | None = None):
    settings = _broker_settings(settings_override)
    if not _ALPACA_TRADING_AVAILABLE or not settings.get("credentials_present"):
        return None
    try:
        return TradingClient(
            settings.get("api_key"),
            settings.get("secret_key"),
            paper=bool(settings.get("paper", True)),
        )
    except Exception:
        return None


def _latest_cached_price(symbol: str) -> float | None:
    path = os.path.join(DATA_DIR, f"prices_{str(symbol or '').upper()}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if "close" not in df.columns or df.empty:
            return None
        return _safe_float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
    except Exception:
        return None


def _current_prices(symbol: str | None = None, current_price: float | None = None) -> dict[str, float]:
    out = {}
    if symbol and current_price not in (None,):
        px = _safe_float(current_price)
        if px is not None:
            out[str(symbol).upper()] = px
    return out


def _position_market_value(pos: dict, market_price: float | None) -> tuple[float, float]:
    qty = _safe_float(pos.get("qty")) or 0.0
    avg = _safe_float(pos.get("avg_price")) or 0.0
    px = _safe_float(market_price)
    if qty == 0.0 or px is None:
        return 0.0, 0.0
    if qty > 0:
        pnl = (px - avg) * qty
    else:
        pnl = (avg - px) * abs(qty)
    return float(abs(qty) * px), float(pnl)


def _portfolio_snapshot(state: dict, current_prices: dict[str, float] | None = None) -> dict:
    current_prices = {str(k).upper(): float(v) for k, v in dict(current_prices or {}).items() if _safe_float(v) is not None}
    positions = dict(state.get("positions") or {})
    gross = 0.0
    net = 0.0
    unrealized = 0.0
    position_rows = []
    largest = None
    for symbol, pos in positions.items():
        qty = _safe_float(pos.get("qty")) or 0.0
        if qty == 0.0:
            continue
        sym = str(symbol).upper()
        px = current_prices.get(sym)
        if px is None:
            px = _latest_cached_price(sym)
        notional, pnl = _position_market_value(pos, px)
        gross += abs(notional)
        net += (qty * (px or (_safe_float(pos.get("avg_price")) or 0.0)))
        unrealized += pnl
        exposure = {
            "symbol": sym,
            "qty": qty,
            "side": "Long" if qty > 0 else "Short",
            "avg_price": _safe_float(pos.get("avg_price")),
            "market_price": px,
            "notional": notional,
            "unrealized_pnl": pnl,
            "setup_type": pos.get("setup_type"),
            "regime": pos.get("regime"),
            "confidence": _safe_float(pos.get("confidence")),
            "style": pos.get("style"),
            "stop_price": _safe_float(pos.get("stop_price")),
            "target_price": _safe_float(pos.get("target_price")),
        }
        position_rows.append(exposure)
        if largest is None or notional > largest.get("notional", 0.0):
            largest = exposure
    cash = _safe_float(state.get("cash")) or 0.0
    equity = float(cash + sum(((_safe_float(p.get("market_price")) or (_safe_float(p.get("avg_price")) or 0.0)) * (_safe_float(p.get("qty")) or 0.0)) for p in position_rows))
    gross_pct = float(gross / max(1.0, abs(equity))) if equity not in (None, 0.0) else 0.0
    net_pct = float(net / max(1.0, abs(equity))) if equity not in (None, 0.0) else 0.0
    return {
        "cash": cash,
        "equity": equity,
        "gross_exposure": gross,
        "net_exposure": net,
        "gross_exposure_pct": gross_pct,
        "net_exposure_pct": net_pct,
        "unrealized_pnl": unrealized,
        "positions": position_rows,
        "open_positions": len(position_rows),
        "largest_position": largest,
    }


def get_portfolio_snapshot_from_state(state: dict, current_prices: dict[str, float] | None = None) -> dict:
    return _portfolio_snapshot(state or _default_state(), current_prices=current_prices)


def _quality_meta(advisory: dict | None) -> dict:
    if not isinstance(advisory, dict):
        return {}
    return dict(advisory.get("quality_meta") or {})


def _plan_meta(advisory: dict | None) -> dict:
    if not isinstance(advisory, dict):
        return {}
    return dict(advisory.get("plan_meta") or {})


def recommend_order_for_state(
    state: dict | None,
    symbol: str,
    current_price: float | None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    side: str | None = None,
) -> dict:
    symbol = str(symbol or "").upper()
    px = _safe_float(current_price) or _latest_cached_price(symbol)
    profile = get_trader_profile(profile_name)
    quality = _quality_meta(advisory)
    plan = _plan_meta(advisory)
    state = dict(state or _load_state())
    portfolio = _portfolio_snapshot(state, current_prices=_current_prices(symbol, px))
    direction = str(plan.get("bias") or "-").lower()
    requested_side = str(side or "").strip().lower()
    if requested_side not in {"buy", "sell"}:
        if direction == "bullish":
            requested_side = "buy"
        elif direction == "bearish":
            requested_side = "sell"
    tradability = _safe_float(((advisory or {}).get("liquidity_meta") or {}).get("tradability_score"))
    quality_score = _safe_float(quality.get("score")) or 0.0
    no_trade = bool(quality.get("no_trade"))
    position_size = _safe_float(((advisory or {}).get("pred_state") or {}).get("intraday", {}).get("position_size"))
    if position_size is None:
        position_size = _safe_float(((advisory or {}).get("predictions_summary") or {}).get("intraday_position_size"))
    entry = _safe_float(plan.get("entry")) or px
    stop = _safe_float(plan.get("stop"))
    if entry is None:
        return {
            "symbol": symbol,
            "allowed": False,
            "reason": "price unavailable",
            "qty": 0,
            "profile": profile,
        }
    if requested_side not in {"buy", "sell"}:
        return {
            "symbol": symbol,
            "allowed": False,
            "reason": "no directional setup",
            "qty": 0,
            "profile": profile,
        }
    if no_trade:
        return {
            "symbol": symbol,
            "allowed": False,
            "reason": "advisory marked no-trade",
            "qty": 0,
            "profile": profile,
        }
    if tradability is not None and tradability < 0.45:
        return {
            "symbol": symbol,
            "allowed": False,
            "reason": "tradability too weak",
            "qty": 0,
            "profile": profile,
        }
    if quality_score < float(profile.get("min_quality_score") or 0.58):
        return {
            "symbol": symbol,
            "allowed": False,
            "reason": "quality below profile threshold",
            "qty": 0,
            "profile": profile,
        }
    if stop is None:
        stop = float(entry * (0.9975 if requested_side == "buy" else 1.0025))
    risk_per_share = abs(float(entry) - float(stop))
    risk_per_share = max(0.01, risk_per_share)
    equity = max(1.0, _safe_float(portfolio.get("equity")) or float(PAPER_STARTING_CASH))
    gross_room = max(0.0, (float(profile.get("max_gross_exposure") or 0.5) * equity) - (_safe_float(portfolio.get("gross_exposure")) or 0.0))
    base_risk_budget = float(profile.get("risk_per_trade") or 0.004) * equity
    quality_mult = max(0.35, min(1.20, quality_score / max(0.01, float(profile.get("min_quality_score") or 0.58))))
    model_mult = max(0.20, min(1.00, float(position_size if position_size is not None else 0.5)))
    qty_risk = int(math.floor(base_risk_budget / risk_per_share))
    qty_notional = int(math.floor(min(float(profile.get("max_single_position_pct") or 0.12) * equity, gross_room) / max(0.01, float(entry))))
    qty = max(0, int(math.floor(min(qty_risk, qty_notional) * min(1.0, quality_mult * model_mult))))
    target = _safe_float(plan.get("target1")) or _safe_float(plan.get("target2"))
    return {
        "symbol": symbol,
        "allowed": qty > 0,
        "reason": "ok" if qty > 0 else "risk budget too small",
        "qty": qty,
        "side": requested_side,
        "entry_price": float(entry),
        "stop_price": float(stop),
        "target_price": float(target) if target is not None else None,
        "risk_per_share": float(risk_per_share),
        "risk_budget": float(base_risk_budget),
        "quality_score": float(quality_score),
        "profile": profile,
    }


def recommend_paper_order(
    symbol: str,
    current_price: float | None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    side: str | None = None,
) -> dict:
    return recommend_order_for_state(
        _load_state(),
        symbol,
        current_price=current_price,
        advisory=advisory,
        profile_name=profile_name,
        side=side,
    )


def stage_paper_order(
    symbol: str,
    side: str,
    qty: int,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    order_type: str = "market",
) -> dict:
    state = _load_state()
    profile = get_trader_profile(profile_name)
    rec = recommend_paper_order(symbol, current_price=current_price, advisory=advisory, profile_name=profile_name, side=side)
    qty_final = int(qty if int(qty) > 0 else rec.get("qty") or 0)
    order = {
        "order_id": uuid.uuid4().hex[:12],
        "broker_mode": "paper",
        "symbol": str(symbol).upper(),
        "side": str(side).lower(),
        "qty": qty_final,
        "order_type": str(order_type).lower(),
        "status": "staged",
        "created_at": _utc_now(),
        "entry_price_hint": rec.get("entry_price"),
        "stop_price": rec.get("stop_price"),
        "take_profit": rec.get("target_price"),
        "risk_per_share": rec.get("risk_per_share"),
        "quality_score": rec.get("quality_score"),
        "profile": profile.get("key"),
        "setup_type": (_plan_meta(advisory).get("action") if isinstance(advisory, dict) else None),
        "regime": ((_quality_meta(advisory).get("regime_label")) if isinstance(advisory, dict) else None),
        "confidence": _safe_float((((advisory or {}).get("pred_state") or {}).get("intraday") or {}).get("p_up")),
        "notes": rec.get("reason"),
    }
    state.setdefault("orders", []).append(order)
    state["updated_at"] = _utc_now()
    _save_state(state)
    return order


def _record_closed_trade(state: dict, symbol: str, side: str, qty: float, entry_price: float, exit_price: float, pos_meta: dict, order: dict):
    qty_abs = abs(float(qty))
    pnl = (exit_price - entry_price) * qty_abs if side == "long" else (entry_price - exit_price) * qty_abs
    opened_at = pos_meta.get("opened_at")
    closed_at = _utc_now()
    risk_per_share = _safe_float(pos_meta.get("risk_per_share")) or _safe_float(order.get("risk_per_share"))
    risk_dollars = (risk_per_share or 0.0) * qty_abs
    try:
        hold_minutes = max(0.0, (pd.Timestamp(closed_at) - pd.Timestamp(opened_at)).total_seconds() / 60.0) if opened_at else None
    except Exception:
        hold_minutes = None
    confidence = _safe_float(pos_meta.get("confidence"))
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
        "setup_type": pos_meta.get("setup_type") or order.get("setup_type"),
        "regime": pos_meta.get("regime") or order.get("regime"),
        "style": pos_meta.get("style") or order.get("profile"),
        "confidence": confidence,
        "confidence_bucket": classify_confidence_bucket(confidence),
        "quality_score": _safe_float(pos_meta.get("quality_score")) or _safe_float(order.get("quality_score")),
        "time_bucket": _time_bucket(closed_at),
    }
    state.setdefault("closed_trades", []).append(trade)


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


def submit_paper_order(order_id: str, fill_price: float | None = None, current_price: float | None = None) -> dict:
    state = _load_state()
    orders = state.setdefault("orders", [])
    order = next((o for o in orders if str(o.get("order_id")) == str(order_id)), None)
    if not isinstance(order, dict):
        return {"status": "error", "reason": "order not found"}
    if str(order.get("status")) != "staged":
        return {"status": "error", "reason": "order not staged"}
    qty = int(_safe_float(order.get("qty")) or 0)
    if qty <= 0:
        return {"status": "error", "reason": "qty must be positive"}
    symbol = str(order.get("symbol") or "").upper()
    px = _safe_float(fill_price) or _safe_float(current_price) or _latest_cached_price(symbol) or _safe_float(order.get("entry_price_hint"))
    if px is None:
        return {"status": "error", "reason": "fill price unavailable"}

    signed_qty = qty if str(order.get("side")).lower() == "buy" else -qty
    cost = float(signed_qty * px)
    state["cash"] = float((_safe_float(state.get("cash")) or 0.0) - cost)
    positions = state.setdefault("positions", {})
    pos = dict(positions.get(symbol) or {})
    old_qty = _safe_float(pos.get("qty")) or 0.0
    old_avg = _safe_float(pos.get("avg_price")) or 0.0
    new_qty = old_qty + signed_qty
    fill = {
        "fill_id": uuid.uuid4().hex[:12],
        "order_id": order_id,
        "symbol": symbol,
        "side": order.get("side"),
        "qty": qty,
        "fill_price": float(px),
        "filled_at": _utc_now(),
        "status": "filled",
        "profile": order.get("profile"),
        "setup_type": order.get("setup_type"),
        "regime": order.get("regime"),
        "confidence": order.get("confidence"),
    }
    state.setdefault("fills", []).append(fill)

    if old_qty == 0.0 or (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
        total_qty = abs(old_qty) + abs(signed_qty)
        avg_price = ((abs(old_qty) * old_avg) + (abs(signed_qty) * px)) / max(1.0, total_qty)
        pos.update(
            {
                "qty": float(new_qty),
                "avg_price": float(avg_price),
                "opened_at": pos.get("opened_at") or fill["filled_at"],
                "setup_type": order.get("setup_type"),
                "regime": order.get("regime"),
                "style": order.get("profile"),
                "confidence": order.get("confidence"),
                "quality_score": order.get("quality_score"),
                "stop_price": order.get("stop_price"),
                "target_price": order.get("take_profit"),
                "risk_per_share": order.get("risk_per_share"),
            }
        )
        positions[symbol] = pos
    else:
        close_qty = min(abs(old_qty), abs(signed_qty))
        closing_side = "long" if old_qty > 0 else "short"
        _record_closed_trade(state, symbol, closing_side, close_qty, old_avg, px, pos, order)
        if new_qty == 0:
            positions.pop(symbol, None)
        elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
            positions[symbol] = {
                "qty": float(new_qty),
                "avg_price": float(px),
                "opened_at": fill["filled_at"],
                "setup_type": order.get("setup_type"),
                "regime": order.get("regime"),
                "style": order.get("profile"),
                "confidence": order.get("confidence"),
                "quality_score": order.get("quality_score"),
                "stop_price": order.get("stop_price"),
                "target_price": order.get("take_profit"),
                "risk_per_share": order.get("risk_per_share"),
            }
        else:
            pos["qty"] = float(new_qty)
            positions[symbol] = pos

    order["status"] = "filled"
    order["filled_at"] = fill["filled_at"]
    order["fill_price"] = float(px)
    state["updated_at"] = _utc_now()
    _save_state(state)
    return {"status": "filled", "order": order, "fill": fill}


def place_paper_order(
    symbol: str,
    side: str,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    qty: int | None = None,
) -> dict:
    if not PAPER_TRADING_ENABLED:
        return {"status": "disabled", "reason": "paper trading disabled"}
    rec = recommend_paper_order(symbol, current_price=current_price, advisory=advisory, profile_name=profile_name, side=side)
    qty_final = int(qty if qty is not None else rec.get("qty") or 0)
    if qty_final <= 0:
        return {"status": "blocked", "reason": rec.get("reason"), "recommendation": rec}
    order = stage_paper_order(symbol, side, qty_final, current_price=current_price, advisory=advisory, profile_name=profile_name)
    submit = submit_paper_order(order.get("order_id"), current_price=current_price)
    submit["recommendation"] = rec
    return submit


def flatten_paper_position(symbol: str, current_price: float | None = None, profile_name: str | None = None) -> dict:
    state = _load_state()
    pos = (state.get("positions") or {}).get(str(symbol).upper())
    qty = int(abs(_safe_float((pos or {}).get("qty")) or 0))
    if qty <= 0:
        return {"status": "blocked", "reason": "no open position"}
    side = "sell" if (_safe_float((pos or {}).get("qty")) or 0.0) > 0 else "buy"
    order = stage_paper_order(symbol, side, qty, current_price=current_price, advisory=None, profile_name=profile_name)
    return submit_paper_order(order.get("order_id"), current_price=current_price)


def get_broker_runtime_status(symbol: str | None = None, settings_override: dict | None = None) -> dict:
    settings = _broker_settings(settings_override)
    requested = _normalize_broker_mode(settings.get("broker_mode"))
    live_snapshot = _call_with_optional_settings(get_live_broker_snapshot, symbol, settings_override=settings_override)
    local_paper_ok = bool(PAPER_TRADING_ENABLED)
    connected_live = bool(live_snapshot.get("connected"))
    live_mode = str(live_snapshot.get("mode") or "live").strip().lower()
    route_mode = None
    effective_mode = "unavailable"
    status = "unavailable"
    detail = ""

    if requested == "disabled":
        effective_mode = "disabled"
        status = "disabled"
        detail = "Broker routing disabled by configuration"
    elif requested == "local_paper":
        if local_paper_ok:
            effective_mode = "local_paper"
            route_mode = "local_paper"
            status = "operational"
            detail = "Orders route to the local paper engine"
        else:
            detail = "Local paper engine disabled"
    elif requested == "alpaca_paper":
        if connected_live and live_mode == "paper":
            effective_mode = "alpaca_paper"
            route_mode = "alpaca_paper"
            status = "connected"
            detail = "Orders route to Alpaca paper"
        elif local_paper_ok:
            effective_mode = "local_paper"
            route_mode = "local_paper"
            status = "fallback_local_paper"
            detail = f"Alpaca paper unavailable; using local paper fallback ({live_snapshot.get('detail') or live_snapshot.get('status')})"
        else:
            detail = f"Alpaca paper unavailable ({live_snapshot.get('detail') or live_snapshot.get('status')})"
    elif requested == "alpaca_live":
        if connected_live and live_mode == "live":
            effective_mode = "alpaca_live"
            route_mode = "alpaca_live"
            status = "connected"
            detail = "Orders route to Alpaca live"
        elif local_paper_ok:
            effective_mode = "local_paper"
            route_mode = "local_paper"
            status = "fallback_local_paper"
            detail = f"Alpaca live unavailable; using local paper fallback ({live_snapshot.get('detail') or live_snapshot.get('status')})"
        else:
            detail = f"Alpaca live unavailable ({live_snapshot.get('detail') or live_snapshot.get('status')})"

    can_connect = str(live_snapshot.get("status") or "") in {
        "adapter_ready",
        "connected",
        "error",
        "invalid_credentials",
        "network_error",
    }
    out = {
        "requested_mode": requested,
        "effective_mode": effective_mode,
        "routing_mode": route_mode,
        "status": status,
        "detail": detail,
        "provider": live_snapshot.get("provider"),
        "connected": connected_live and route_mode in {"alpaca_paper", "alpaca_live"},
        "local_paper_active": bool(route_mode == "local_paper"),
        "can_connect": bool(can_connect),
        "can_place_orders": bool(route_mode in {"local_paper", "alpaca_paper", "alpaca_live"}),
        "can_cancel": bool(route_mode in {"alpaca_paper", "alpaca_live"} and connected_live),
        "can_read_positions": bool(route_mode == "local_paper" or connected_live),
        "live_snapshot": live_snapshot,
        "route_label": _broker_mode_label(route_mode if route_mode else effective_mode),
        "requested_label": _broker_mode_label(requested),
        "settings": settings,
    }
    out["capability_summary"] = (
        f"connect {'Y' if out['can_connect'] else 'N'} | "
        f"place {'Y' if out['can_place_orders'] else 'N'} | "
        f"cancel {'Y' if out['can_cancel'] else 'N'} | "
        f"read {'Y' if out['can_read_positions'] else 'N'}"
    )
    return out


def get_broker_diagnostics_snapshot(symbol: str | None = None, settings_override: dict | None = None) -> dict:
    symbol = str(symbol or "").upper()
    checked_at = _utc_now()
    settings = _broker_settings(settings_override)
    runtime = _call_with_optional_settings(get_broker_runtime_status, symbol, settings_override=settings_override)
    live_snapshot = runtime.get("live_snapshot") or {}
    provider = str(live_snapshot.get("provider") or settings.get("provider") or "alpaca").lower()
    requested_mode = str(runtime.get("requested_mode") or "disabled")
    effective_mode = str(runtime.get("effective_mode") or "unavailable")
    sdk_installed = _ALPACA_TRADING_AVAILABLE if provider == "alpaca" else None
    credentials_present = bool(settings.get("credentials_present")) if provider == "alpaca" else None
    connected = bool(runtime.get("connected"))
    account = dict(live_snapshot.get("account") or {})
    orders = list(live_snapshot.get("orders") or [])
    positions = list(live_snapshot.get("positions") or [])
    route_target = runtime.get("route_label") or _broker_mode_label(effective_mode)
    config_state = "ready"
    if provider == "alpaca":
        if not sdk_installed:
            config_state = "sdk_missing"
        elif not credentials_present:
            config_state = "credentials_missing"
        elif requested_mode == "local_paper" and str(settings.get("broker_mode_source") or "") == "default":
            config_state = "mode_default_local_paper"
        elif requested_mode == "alpaca_paper" and not settings.get("paper", True):
            config_state = "paper_flag_off"
        elif requested_mode == "alpaca_live" and settings.get("paper", True):
            config_state = "live_flag_mismatch"
        elif str(live_snapshot.get("status") or "") in {"invalid_credentials", "network_error"}:
            config_state = str(live_snapshot.get("status"))
        elif requested_mode == "disabled":
            config_state = "disabled"
        elif connected:
            config_state = "connected"
        else:
            config_state = str(live_snapshot.get("status") or "adapter_ready")
    operator_hint = _broker_operator_hint(
        provider=provider,
        requested_mode=requested_mode,
        runtime_status=str(runtime.get("status") or ""),
        sdk_installed=sdk_installed,
        credentials_present=credentials_present,
        connected=connected,
        paper_mode=bool(settings.get("paper", True)),
    )
    verification_ready = bool(
        requested_mode == "alpaca_paper"
        and provider == "alpaca"
        and sdk_installed
        and credentials_present
        and bool(settings.get("paper", True))
        and connected
    )
    credentials_source = str(settings.get("credentials_source") or "missing")
    if credentials_source == "env":
        credentials_label = "Environment"
        credentials_detail = "Using runtime environment variables"
    elif credentials_source == "local_store":
        credentials_label = "Saved Locally"
        credentials_detail = settings.get("storage_note") or "Stored on this computer"
    elif credentials_source == "override":
        credentials_label = "Setup Dialog"
        credentials_detail = "Testing the values entered in the setup window"
    elif credentials_source == "env_partial":
        credentials_label = "Incomplete"
        credentials_detail = "Only part of the Alpaca credential pair is available"
    else:
        credentials_label = "Missing"
        credentials_detail = "Open Alpaca Setup to add your API key and secret"
    rows = [
        ("Requested Mode", runtime.get("requested_label") or "-", f"Provider {provider}"),
        ("Effective Route", route_target, str(runtime.get("detail") or "-")),
        ("SDK / Creds", f"{_fmt_bool(sdk_installed)} / {_fmt_bool(credentials_present)}", f"Config {config_state.replace('_', ' ')}"),
        ("Credential Source", credentials_label, credentials_detail),
        ("Broker Mode", "Paper" if settings.get("paper", True) else "Live", f"Mode source {str(settings.get('paper_source') or '-').replace('_', ' ')}"),
        ("Connected", _fmt_bool(connected), f"Can place {_fmt_bool(runtime.get('can_place_orders'))} | cancel {_fmt_bool(runtime.get('can_cancel'))}"),
        ("Account / Positions", "-" if _safe_float(account.get("equity")) is None else f"${float(account.get('equity')):,.2f}", f"{len(positions)} positions | {len(orders)} open orders"),
        ("Verify Ready", _fmt_bool(verification_ready), operator_hint),
        ("Checked", checked_at, f"Orders go to {route_target}" if runtime.get("can_place_orders") else "Routing unavailable"),
    ]
    return {
        "provider": provider,
        "requested_mode": requested_mode,
        "effective_mode": effective_mode,
        "requested_label": runtime.get("requested_label"),
        "route_label": route_target,
        "status": str(runtime.get("status") or "unavailable"),
        "config_state": config_state,
        "connected": connected,
        "sdk_installed": sdk_installed,
        "credentials_present": credentials_present,
        "credentials_source": credentials_source,
        "storage_kind": settings.get("storage_kind"),
        "storage_path": settings.get("storage_path"),
        "storage_note": settings.get("storage_note"),
        "alpaca_paper_flag": bool(settings.get("paper", True)) if provider == "alpaca" else None,
        "can_connect": bool(runtime.get("can_connect")),
        "can_place_orders": bool(runtime.get("can_place_orders")),
        "can_cancel": bool(runtime.get("can_cancel")),
        "can_read_positions": bool(runtime.get("can_read_positions")),
        "verification_ready": verification_ready,
        "route_target": route_target,
        "checked_at": checked_at,
        "detail": str(runtime.get("detail") or live_snapshot.get("detail") or "-"),
        "operator_hint": operator_hint,
        "orders_count": len(orders),
        "positions_count": len(positions),
        "account": account,
        "rows": rows,
        "runtime": runtime,
        "live_snapshot": live_snapshot,
        "settings": settings,
    }


def refresh_broker_connection(symbol: str | None = None, settings_override: dict | None = None) -> dict:
    return get_broker_diagnostics_snapshot(symbol, settings_override=settings_override)


def save_local_broker_credentials(
    api_key: str,
    secret_key: str,
    *,
    paper: bool = True,
    broker_mode: str = "alpaca_paper",
) -> dict:
    return save_alpaca_credentials(
        api_key,
        secret_key,
        paper=paper,
        broker_mode=broker_mode,
        saved_at=_utc_now(),
    )


def clear_local_broker_credentials() -> bool:
    return clear_saved_alpaca_credentials()


def get_broker_verification_snapshot(symbol: str | None = None) -> dict:
    symbol = str(symbol or "").upper()
    snap = dict(_load_broker_verification() or {})
    if symbol and snap.get("symbol") and str(snap.get("symbol")).upper() != symbol:
        return {
            "status": "other_symbol",
            "symbol": symbol,
            "rows": [("Broker Verify", "Other Symbol", f"Latest verification stored for {str(snap.get('symbol')).upper()}")],
            "detail": f"Latest verification is for {str(snap.get('symbol')).upper()}",
        }
    return snap


def get_live_broker_snapshot(symbol: str | None = None, settings_override: dict | None = None) -> dict:
    settings = _broker_settings(settings_override)
    status = _live_adapter_status(settings_override=settings_override)
    base = {
        "provider": status.get("provider"),
        "status": status.get("status"),
        "mode": status.get("mode"),
        "connected": False,
        "rows": [
            ("Live Adapter", str(status.get("status") or "-").replace("_", " ").title(), str(status.get("detail") or "-")),
        ],
        "account": {},
        "orders": [],
        "positions": [],
        "detail": status.get("detail"),
    }
    client = _call_with_optional_settings(_alpaca_client, settings_override=settings_override)
    if client is None:
        return base
    try:
        account = client.get_account()
        try:
            order_req = GetOrdersRequest(status=QueryOrderStatus.OPEN) if GetOrdersRequest and QueryOrderStatus else None
            open_orders = list(client.get_orders(filter=order_req) if order_req is not None else client.get_orders())
        except Exception:
            open_orders = []
        try:
            live_positions = list(client.get_all_positions())
        except Exception:
            live_positions = []
        cash = _safe_float(getattr(account, "cash", None))
        buying_power = _safe_float(getattr(account, "buying_power", None))
        equity = _safe_float(getattr(account, "equity", None))
        rows = [
            ("Live Adapter", "Connected", f"Alpaca {'paper' if settings.get('paper', True) else 'live'} account"),
            ("Buying Power", "-" if buying_power is None else f"${buying_power:,.2f}", "-" if cash is None else f"Cash ${cash:,.2f}"),
            ("Equity", "-" if equity is None else f"${equity:,.2f}", f"{len(live_positions)} positions | {len(open_orders)} open orders"),
        ]
        if symbol:
            sym = str(symbol).upper()
            sym_pos = next((p for p in live_positions if str(getattr(p, "symbol", "")).upper() == sym), None)
            if sym_pos is not None:
                rows.append(
                    (
                        "Live Position",
                        f"{getattr(sym_pos, 'side', '-')}".title(),
                        f"{getattr(sym_pos, 'qty', '-')} @ ${_safe_float(getattr(sym_pos, 'avg_entry_price', None)) or 0.0:.2f}",
                    )
                )
        base.update(
            {
                "status": "connected",
                "connected": True,
                "rows": rows,
                "account": {
                    "cash": cash,
                    "buying_power": buying_power,
                    "equity": equity,
                    "status": getattr(account, "status", None),
                },
                "orders": open_orders,
                "positions": live_positions,
                "detail": f"Connected to Alpaca {'paper' if settings.get('paper', True) else 'live'}",
            }
        )
        return base
    except Exception as exc:
        err_state, detail = _classify_broker_exception(exc)
        base["status"] = err_state
        base["rows"] = [("Live Adapter", err_state.replace("_", " ").title(), detail)]
        base["detail"] = detail
        return base


def place_live_order(
    symbol: str,
    side: str,
    qty: int | None = None,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    settings_override: dict | None = None,
) -> dict:
    snapshot = _call_with_optional_settings(get_live_broker_snapshot, symbol, settings_override=settings_override)
    if not snapshot.get("connected"):
        return {"status": "blocked", "reason": snapshot.get("detail") or snapshot.get("status"), "live_snapshot": snapshot}
    rec = recommend_paper_order(symbol, current_price=current_price, advisory=advisory, profile_name=profile_name, side=side)
    qty_final = int(qty if qty is not None else rec.get("qty") or 0)
    if qty_final <= 0:
        return {"status": "blocked", "reason": rec.get("reason"), "recommendation": rec, "live_snapshot": snapshot}
    client = _call_with_optional_settings(_alpaca_client, settings_override=settings_override)
    if client is None or MarketOrderRequest is None or OrderSide is None or TimeInForce is None:
        return {"status": "blocked", "reason": "live order request classes unavailable", "live_snapshot": snapshot}
    try:
        order_side = OrderSide.BUY if str(side).lower() == "buy" else OrderSide.SELL
        kwargs = {
            "symbol": str(symbol).upper(),
            "qty": qty_final,
            "side": order_side,
            "time_in_force": TimeInForce.DAY,
        }
        stop_price = _safe_float(rec.get("stop_price"))
        target_price = _safe_float(rec.get("target_price"))
        if stop_price is not None and StopLossRequest is not None:
            kwargs["stop_loss"] = StopLossRequest(stop_price=round(float(stop_price), 2))
        if target_price is not None and TakeProfitRequest is not None:
            kwargs["take_profit"] = TakeProfitRequest(limit_price=round(float(target_price), 2))
        order_data = MarketOrderRequest(**kwargs)
        order = client.submit_order(order_data=order_data)
        return {
            "status": "submitted",
            "broker_mode": "live",
            "order_id": getattr(order, "id", None),
            "symbol": getattr(order, "symbol", str(symbol).upper()),
            "qty": getattr(order, "qty", qty_final),
            "side": side,
            "broker_order_status": str(getattr(order, "status", "") or "").lower() or None,
            "filled_qty": _safe_float(getattr(order, "filled_qty", None)),
            "filled_avg_price": _safe_float(getattr(order, "filled_avg_price", None)),
            "submitted_at": getattr(order, "submitted_at", None),
            "recommendation": rec,
        }
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "recommendation": rec}


def cancel_live_order(order_id: str, settings_override: dict | None = None) -> dict:
    snapshot = _call_with_optional_settings(get_live_broker_snapshot, settings_override=settings_override)
    if not snapshot.get("connected"):
        return {"status": "blocked", "reason": snapshot.get("detail") or snapshot.get("status"), "live_snapshot": snapshot}
    client = _call_with_optional_settings(_alpaca_client, settings_override=settings_override)
    if client is None:
        return {"status": "blocked", "reason": "live broker client unavailable", "live_snapshot": snapshot}
    try:
        client.cancel_order_by_id(order_id)
        return {"status": "canceled", "order_id": order_id}
    except Exception as exc:
        return {"status": "error", "reason": str(exc), "order_id": order_id}


def place_routed_order(
    symbol: str,
    side: str,
    qty: int | None = None,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    settings_override: dict | None = None,
) -> dict:
    runtime = _call_with_optional_settings(get_broker_runtime_status, symbol, settings_override=settings_override)
    route_mode = runtime.get("routing_mode")
    if route_mode == "local_paper":
        result = place_paper_order(
            symbol,
            side,
            current_price=current_price,
            advisory=advisory,
            profile_name=profile_name,
            qty=qty,
        )
        result["routing_mode"] = "local_paper"
        result["routing_label"] = runtime.get("route_label")
        result["runtime_status"] = runtime
        return result
    if route_mode in {"alpaca_paper", "alpaca_live"}:
        result = _call_with_optional_settings(
            place_live_order,
            symbol,
            side,
            qty=qty,
            current_price=current_price,
            advisory=advisory,
            profile_name=profile_name,
            settings_override=settings_override,
        )
        result["routing_mode"] = route_mode
        result["routing_label"] = runtime.get("route_label")
        result["runtime_status"] = runtime
        return result
    return {
        "status": "blocked",
        "reason": runtime.get("detail") or runtime.get("status") or "routing unavailable",
        "routing_mode": route_mode,
        "routing_label": runtime.get("route_label"),
        "runtime_status": runtime,
    }


def cancel_routed_order(order_id: str | None = None, symbol: str | None = None, settings_override: dict | None = None) -> dict:
    runtime = _call_with_optional_settings(get_broker_runtime_status, symbol, settings_override=settings_override)
    route_mode = runtime.get("routing_mode")
    if route_mode == "local_paper":
        return {
            "status": "blocked",
            "reason": "Local paper orders fill immediately; nothing to cancel",
            "routing_mode": route_mode,
            "runtime_status": runtime,
        }
    if route_mode not in {"alpaca_paper", "alpaca_live"}:
        return {
            "status": "blocked",
            "reason": runtime.get("detail") or runtime.get("status") or "routing unavailable",
            "routing_mode": route_mode,
            "runtime_status": runtime,
        }
    live_snapshot = runtime.get("live_snapshot") or {}
    chosen = order_id
    if not chosen:
        open_orders = list(live_snapshot.get("orders") or [])
        if symbol:
            sym = str(symbol).upper()
            open_orders = [o for o in open_orders if str(getattr(o, "symbol", "")).upper() == sym]
        if open_orders:
            chosen = getattr(open_orders[0], "id", None)
    if not chosen:
        return {
            "status": "blocked",
            "reason": "No open routed order to cancel",
            "routing_mode": route_mode,
            "runtime_status": runtime,
        }
        result = _call_with_optional_settings(cancel_live_order, str(chosen), settings_override=settings_override)
    result["routing_mode"] = route_mode
    result["runtime_status"] = runtime
    return result


def run_broker_verification(
    symbol: str,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    qty: int = 1,
    settings_override: dict | None = None,
) -> dict:
    symbol = str(symbol or "").upper()
    diagnostics = _call_with_optional_settings(refresh_broker_connection, symbol, settings_override=settings_override)
    runtime = diagnostics.get("runtime") or {}
    route_mode = str(runtime.get("routing_mode") or "")
    result = {
        "symbol": symbol,
        "requested_mode": diagnostics.get("requested_mode"),
        "effective_mode": diagnostics.get("effective_mode"),
        "route_label": diagnostics.get("route_label"),
        "checked_at": diagnostics.get("checked_at"),
        "diagnostics": diagnostics,
    }
    if route_mode != "alpaca_paper":
        detail = str(diagnostics.get("operator_hint") or "Verification only runs against a connected Alpaca paper route")
        if runtime.get("status") == "fallback_local_paper":
            detail = str(runtime.get("detail") or detail)
        result.update(
            {
                "status": "blocked",
                "detail": detail,
                "rows": [
                    ("Broker Verify", "Blocked", detail),
                    ("Requested / Effective", f"{diagnostics.get('requested_label') or '-'} / {diagnostics.get('route_label') or '-'}", "Paper-safe fallback preserved"),
                ],
            }
        )
        _save_broker_verification(result)
        return result
    if not diagnostics.get("connected"):
        detail = str(diagnostics.get("operator_hint") or diagnostics.get("detail") or "Broker not connected")
        result.update(
            {
                "status": "blocked",
                "detail": detail,
                "rows": [
                    ("Broker Verify", "Not Connected", detail),
                    ("Capabilities", runtime.get("capability_summary") or "-", "Connection required before test order"),
                ],
            }
        )
        _save_broker_verification(result)
        return result

    before = _call_with_optional_settings(get_live_broker_snapshot, symbol, settings_override=settings_override)
    submit = _call_with_optional_settings(
        place_live_order,
        symbol,
        "buy",
        qty=max(1, int(qty or 1)),
        current_price=current_price,
        advisory=advisory,
        profile_name=profile_name,
        settings_override=settings_override,
    )
    after_submit = _call_with_optional_settings(get_live_broker_snapshot, symbol, settings_override=settings_override)
    submitted = str(submit.get("status") or "") == "submitted"
    cleanup = {"status": "skipped", "reason": "no cleanup needed"}
    submit_order_id = submit.get("order_id")
    symbol_position = next(
        (p for p in list(after_submit.get("positions") or []) if str(getattr(p, "symbol", "")).upper() == symbol),
        None,
    )
    open_order = next(
        (o for o in list(after_submit.get("orders") or []) if str(getattr(o, "symbol", "")).upper() == symbol),
        None,
    )
    if submitted and symbol_position is not None:
        qty_out = int(abs(_safe_float(getattr(symbol_position, "qty", None)) or 0))
        if qty_out > 0:
            side_out = "sell" if str(getattr(symbol_position, "side", "")).lower() == "long" else "buy"
            cleanup = _call_with_optional_settings(
                place_live_order,
                symbol,
                side_out,
                qty=qty_out,
                current_price=current_price,
                advisory=None,
                profile_name=profile_name,
                settings_override=settings_override,
            )
    elif submitted and submit_order_id:
        cleanup = _call_with_optional_settings(cancel_live_order, str(submit_order_id), settings_override=settings_override)
    elif open_order is not None:
        cleanup = _call_with_optional_settings(cancel_live_order, str(getattr(open_order, "id", "")), settings_override=settings_override)

    after_cleanup = _call_with_optional_settings(get_live_broker_snapshot, symbol, settings_override=settings_override)
    status = "verified" if submitted else str(submit.get("status") or "error")
    detail = "Alpaca paper verification completed"
    if status != "verified":
        detail = str(submit.get("reason") or submit.get("status") or "verification failed")
    elif str(cleanup.get("status") or "") == "submitted":
        detail = "Paper test order sent and cleanup order submitted"
    elif str(cleanup.get("status") or "") == "canceled":
        detail = "Paper test order submitted and canceled before fill"
    rows = [
        ("Broker Verify", str(status).replace("_", " ").title(), detail),
        ("Account", "-" if _safe_float((before.get("account") or {}).get("equity")) is None else f"${float((before.get('account') or {}).get('equity')):,.2f}", f"{len(before.get('positions') or [])} positions before check"),
        ("Test Order", str(submit.get("status") or "-").title(), f"ID {submit_order_id or '-'} | qty {int(submit.get('qty') or qty or 1)}"),
        ("Cleanup", str(cleanup.get("status") or "-").title(), str(cleanup.get("reason") or cleanup.get("status") or "-")),
        ("After Check", f"{len(after_cleanup.get('positions') or [])} positions", f"{len(after_cleanup.get('orders') or [])} open orders"),
    ]
    result.update(
        {
            "status": status,
            "detail": detail,
            "submit": submit,
            "cleanup": cleanup,
            "before": before,
            "after_submit": after_submit,
            "after_cleanup": after_cleanup,
            "rows": rows,
        }
    )
    _save_broker_verification(result)
    return result


def get_execution_dashboard_snapshot(
    symbol: str,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
) -> dict:
    symbol = str(symbol or "").upper()
    profile = get_trader_profile(profile_name)
    state = _load_state()
    prices = _current_prices(symbol, current_price)
    portfolio = _portfolio_snapshot(state, prices)
    pos = next((p for p in portfolio.get("positions", []) if p.get("symbol") == symbol), None)
    rec = recommend_paper_order(symbol, current_price=current_price, advisory=advisory, profile_name=profile_name)
    runtime = get_broker_runtime_status(symbol)
    live_snapshot = runtime.get("live_snapshot") or get_live_broker_snapshot(symbol)
    diagnostics = get_broker_diagnostics_snapshot(symbol)
    verification = get_broker_verification_snapshot(symbol)
    staged = [o for o in state.get("orders", []) if str(o.get("status")) == "staged"]
    recent_fill = (state.get("fills") or [])[-1] if (state.get("fills") or []) else None
    broker_mode = runtime.get("effective_mode") or "unavailable"
    rows = [
        ("Route", runtime.get("route_label") or "Unavailable", str(runtime.get("detail") or "-")),
        ("Requested", runtime.get("requested_label") or "-", f"Status {str(runtime.get('status') or '-').replace('_', ' ')}"),
        ("Broker Check", "Connected" if diagnostics.get("connected") else "Not connected", f"Checked {str(diagnostics.get('checked_at') or '-')}"),
        ("SDK / Creds", f"{_fmt_bool(diagnostics.get('sdk_installed'))} / {_fmt_bool(diagnostics.get('credentials_present'))}", f"Provider {str(diagnostics.get('provider') or '-')}"),
        (
            "Credentials",
            str(diagnostics.get("credentials_source") or "missing").replace("_", " ").title(),
            str(diagnostics.get("storage_note") or "Open Alpaca Setup to save keys on this computer"),
        ),
        ("Capabilities", runtime.get("capability_summary") or "-", f"Live adapter {str(live_snapshot.get('status') or '-').replace('_', ' ')}"),
        ("Profile", profile.get("label", "Balanced"), f"Risk/trade {100.0 * float(profile.get('risk_per_trade') or 0.0):.2f}% | gross cap {100.0 * float(profile.get('max_gross_exposure') or 0.0):.0f}%"),
        (
            "Local Paper Pos",
            "-" if not pos else f"{pos.get('side')} {abs(int(_safe_float(pos.get('qty')) or 0))}",
            "-" if not pos else f"Avg ${(_safe_float(pos.get('avg_price')) or 0.0):.2f} | UPL ${(_safe_float(pos.get('unrealized_pnl')) or 0.0):+.2f}",
        ),
        (
            "Next Size",
            "-" if not rec.get("allowed") else f"{int(rec.get('qty') or 0)} sh",
            f"Stop ${(_safe_float(rec.get('stop_price')) or 0.0):.2f} | target ${(_safe_float(rec.get('target_price')) or 0.0):.2f}" if rec.get("allowed") else str(rec.get("reason") or "-"),
        ),
        (
            "Orders / Fills",
            f"{len(staged)} / {len(state.get('fills') or [])}",
            "-" if not recent_fill else f"{recent_fill.get('symbol')} {recent_fill.get('side')} {int(_safe_float(recent_fill.get('qty')) or 0)} @ ${(_safe_float(recent_fill.get('fill_price')) or 0.0):.2f}",
        ),
    ]
    verify_status = str(verification.get("status") or "")
    if verify_status:
        rows.append(
            (
                "Verify",
                verify_status.replace("_", " ").title(),
                str(verification.get("detail") or "-"),
            )
        )
    for live_row in list(live_snapshot.get("rows") or [])[1:3]:
        rows.append(live_row)
    if runtime.get("routing_mode") in {"alpaca_paper", "alpaca_live"} and live_snapshot.get("connected"):
        account = live_snapshot.get("account") or {}
        positions = list(live_snapshot.get("positions") or [])
        orders = list(live_snapshot.get("orders") or [])
        sym_pos = next((p for p in positions if str(getattr(p, "symbol", "")).upper() == symbol), None)
        portfolio_rows = [
            ("Account Equity", "-" if _safe_float(account.get("equity")) is None else f"${float(account.get('equity')):,.2f}", "-" if _safe_float(account.get("cash")) is None else f"Cash ${float(account.get('cash')):,.2f}"),
            ("Buying Power", "-" if _safe_float(account.get("buying_power")) is None else f"${float(account.get('buying_power')):,.2f}", f"{len(positions)} live positions"),
            ("Open Orders", str(len(orders)), f"{runtime.get('route_label')} connected"),
            (
                "Symbol Position",
                "-" if sym_pos is None else f"{getattr(sym_pos, 'side', '-')} {getattr(sym_pos, 'qty', '-')}",
                "-" if sym_pos is None else f"Avg ${_safe_float(getattr(sym_pos, 'avg_entry_price', None)) or 0.0:.2f}",
            ),
        ]
    else:
        portfolio_rows = [
            ("Equity", f"${(_safe_float(portfolio.get('equity')) or 0.0):,.2f}", f"Cash ${(_safe_float(portfolio.get('cash')) or 0.0):,.2f}"),
            ("Gross / Net", f"{100.0 * (_safe_float(portfolio.get('gross_exposure_pct')) or 0.0):.1f}% / {100.0 * (_safe_float(portfolio.get('net_exposure_pct')) or 0.0):+.1f}%", f"{int(portfolio.get('open_positions') or 0)} open positions"),
            ("Unrealized", f"${(_safe_float(portfolio.get('unrealized_pnl')) or 0.0):+.2f}", "Marked to current cached prices"),
            (
                "Largest Pos",
                "-" if not portfolio.get("largest_position") else str((portfolio.get("largest_position") or {}).get("symbol") or "-"),
                "-" if not portfolio.get("largest_position") else f"${(_safe_float((portfolio.get('largest_position') or {}).get('notional')) or 0.0):,.0f} notionally",
            ),
        ]
    return {
        "broker_mode": str(broker_mode).lower(),
        "live_status": live_snapshot.get("status"),
        "live_detail": live_snapshot.get("detail"),
        "rows": rows[:11],
        "portfolio_rows": portfolio_rows[:5],
        "portfolio": portfolio,
        "recommendation": rec,
        "state": state,
        "live_snapshot": live_snapshot,
        "runtime": runtime,
        "diagnostics": diagnostics,
        "verification": verification,
        "route_label": runtime.get("route_label"),
        "route_buy_label": f"Buy ({runtime.get('route_label')})",
        "route_sell_label": f"Sell ({runtime.get('route_label')})",
        "route_cancel_label": "Cancel Routed",
        "route_refresh_label": "Refresh Broker",
        "route_verify_label": "Verify Alpaca Paper",
        "route_setup_label": "Alpaca Setup",
        "route_clear_label": "Remove Saved Keys",
    }
