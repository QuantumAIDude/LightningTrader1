from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_EQUITY = 100000.0
DEFAULT_RISK_PER_TRADE = 0.004
DEFAULT_MAX_SYMBOL_PCT = 0.12
DEFAULT_MAX_GROSS_PCT = 0.50
DEFAULT_MAX_NET_PCT = 0.30
DEFAULT_DAILY_DRAWDOWN_PCT = 0.03
DEFAULT_MIN_STOP_PCT = 0.0025


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _clean_label_list(values: Any) -> list[str]:
    clean: list[str] = []
    seen: set[str] = set()
    for item in list(values or []):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return clean


def _close_series(df: pd.DataFrame | None) -> pd.Series:
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return pd.Series(dtype=float)
    series = pd.to_numeric(df.get("close"), errors="coerce").dropna()
    return series.astype(float) if not series.empty else pd.Series(dtype=float)


def _infer_side(advisory: dict | None, side: str | None = None) -> str:
    explicit = str(side or "").strip().lower()
    if explicit in {"buy", "sell"}:
        return explicit
    plan = dict((advisory or {}).get("plan_meta") or {})
    bias = str(plan.get("bias") or "").strip().lower()
    if bias == "bullish":
        return "buy"
    if bias == "bearish":
        return "sell"
    intraday = dict(((advisory or {}).get("pred_state") or {}).get("intraday") or {})
    direction = str(intraday.get("direction") or "").strip().upper()
    if direction == "UP":
        return "buy"
    if direction == "DOWN":
        return "sell"
    return ""


def classify_market_regime(
    df: pd.DataFrame | None,
    *,
    predictions: dict | None = None,
    news_meta: dict | None = None,
    min_rows: int = 20,
) -> dict:
    news_meta = dict(news_meta or {})
    headline_state = str(news_meta.get("headline_state") or "").strip().lower()
    alignment = str(news_meta.get("alignment") or "").strip().lower()
    high_impact_count = int(news_meta.get("fresh_high_impact_count") or 0)
    if headline_state in {"failed", "stale", "error"} or (alignment == "conflict" and high_impact_count >= 1):
        return {
            "label": "event_risk_or_news_disturbed",
            "confidence": 0.90,
            "reason": "Fresh conflicting or degraded news context is dominating the current environment.",
            "evidence_strength": "high",
            "trend_score": None,
            "volatility_bps": None,
            "unknown": False,
        }

    closes = _close_series(df)
    if len(closes) < int(min_rows):
        return {
            "label": "unknown_or_insufficient_evidence",
            "confidence": 0.0,
            "reason": "Too few recent bars are available to classify the current market regime safely.",
            "evidence_strength": "low",
            "trend_score": None,
            "volatility_bps": None,
            "unknown": True,
        }

    returns = closes.pct_change().dropna()
    if len(returns) < max(10, int(min_rows // 2)):
        return {
            "label": "unknown_or_insufficient_evidence",
            "confidence": 0.0,
            "reason": "Recent return history is too thin to classify the market regime safely.",
            "evidence_strength": "low",
            "trend_score": None,
            "volatility_bps": None,
            "unknown": True,
        }

    window = min(len(closes), 40)
    recent_close = closes.iloc[-window:]
    recent_ret = returns.iloc[-min(len(returns), window - 1) :]
    last_price = max(0.01, float(recent_close.iloc[-1]))
    volatility_bps = float(recent_ret.std(ddof=0) * 10000.0) if not recent_ret.empty else 0.0
    trend_score = float((recent_close.iloc[-1] / recent_close.iloc[0]) - 1.0) if len(recent_close) >= 2 else 0.0

    regime_block = {}
    if isinstance(predictions, dict):
        intraday = dict(predictions.get("intraday") or {})
        regime_block = dict(intraday.get("regime") or {})
    bull = _safe_float(regime_block.get("Bullish")) or 0.0
    bear = _safe_float(regime_block.get("Bearish")) or 0.0
    neutral = _safe_float(regime_block.get("Neutral")) or 0.0
    regime_conf = _safe_float(regime_block.get("regime_conf"))
    directional_bias = (bull - bear) / 100.0 if any((bull, bear, neutral)) else 0.0
    combined_trend = float(trend_score + (0.5 * directional_bias))
    high_vol = volatility_bps >= 18.0
    low_vol = volatility_bps <= 7.0

    if combined_trend >= 0.012:
        label = "trend_up"
    elif combined_trend <= -0.012:
        label = "trend_down"
    elif high_vol:
        label = "high_volatility"
    elif low_vol:
        label = "low_volatility"
    else:
        label = "range_bound"

    score_parts = [min(1.0, abs(combined_trend) / 0.03), min(1.0, volatility_bps / 30.0)]
    if regime_conf is not None:
        score_parts.append(max(0.0, min(1.0, float(regime_conf))))
    confidence = float(np.mean(score_parts)) if score_parts else 0.0
    reason = (
        f"Trend {combined_trend:+.2%} | vol {volatility_bps:.1f} bps"
        f" | regime conf {regime_conf:.2f}" if regime_conf is not None else f"Trend {combined_trend:+.2%} | vol {volatility_bps:.1f} bps"
    )
    return {
        "label": label,
        "confidence": round(confidence, 3),
        "reason": reason,
        "evidence_strength": "high" if confidence >= 0.7 else "medium" if confidence >= 0.45 else "low",
        "trend_score": round(combined_trend, 5),
        "volatility_bps": round(volatility_bps, 3),
        "unknown": False,
        "last_price": round(last_price, 4),
    }


def compute_risk_budget(
    *,
    equity: float | None,
    risk_fraction: float = DEFAULT_RISK_PER_TRADE,
    quality_score: float | None = None,
    position_size_signal: float | None = None,
) -> dict:
    eq = _safe_float(equity)
    if eq is None or eq <= 0.0:
        return {"allowed": False, "risk_budget": 0.0, "reason": "equity is unavailable"}
    risk_fraction = max(0.0001, float(_safe_float(risk_fraction) or DEFAULT_RISK_PER_TRADE))
    quality_mult = max(0.35, min(1.20, float(_safe_float(quality_score) or 0.6) / 0.60))
    signal_mult = max(0.20, min(1.00, float(_safe_float(position_size_signal) or 0.5)))
    risk_budget = float(eq * risk_fraction * min(1.20, quality_mult * signal_mult))
    return {
        "allowed": risk_budget > 0.0,
        "risk_budget": round(risk_budget, 4),
        "equity": float(eq),
        "risk_fraction": float(risk_fraction),
        "quality_mult": round(quality_mult, 4),
        "signal_mult": round(signal_mult, 4),
    }


def compute_stop_distance(
    *,
    current_price: float | None,
    atr14: float | None = None,
    volatility_bps: float | None = None,
    min_stop_pct: float = DEFAULT_MIN_STOP_PCT,
) -> dict:
    px = _safe_float(current_price)
    if px is None or px <= 0.0:
        return {"allowed": False, "stop_distance": None, "reason": "current price is unavailable"}
    base = float(px) * float(min_stop_pct)
    atr = _safe_float(atr14)
    if atr is not None and atr > 0.0:
        base = max(base, 0.35 * float(atr))
    vol_bps = _safe_float(volatility_bps)
    if vol_bps is not None and vol_bps > 0.0:
        base = max(base, float(px) * (float(vol_bps) / 10000.0) * 1.5)
    return {
        "allowed": True,
        "stop_distance": round(max(0.01, base), 4),
        "volatility_bps": vol_bps,
        "atr14": atr,
    }


def compute_stop_price(side: str | None, entry_price: float | None, stop_distance: float | None) -> float | None:
    entry = _safe_float(entry_price)
    distance = _safe_float(stop_distance)
    side_key = str(side or "").strip().lower()
    if entry is None or distance is None or distance <= 0.0 or side_key not in {"buy", "sell"}:
        return None
    if side_key == "buy":
        return round(float(entry - distance), 4)
    return round(float(entry + distance), 4)


def portfolio_exposure_check(
    *,
    portfolio: dict | None,
    symbol: str,
    proposed_notional: float | None,
    side: str | None,
    max_symbol_pct: float = DEFAULT_MAX_SYMBOL_PCT,
    max_gross_pct: float = DEFAULT_MAX_GROSS_PCT,
    max_net_pct: float = DEFAULT_MAX_NET_PCT,
) -> dict:
    snapshot = dict(portfolio or {})
    eq = _safe_float(snapshot.get("equity"))
    notional = max(0.0, float(_safe_float(proposed_notional) or 0.0))
    if eq is None or eq <= 0.0:
        return {"allowed": False, "reason": "portfolio equity is unavailable"}
    positions = list(snapshot.get("positions") or [])
    symbol_key = str(symbol or "").upper()
    symbol_notional = 0.0
    for pos in positions:
        if str((pos or {}).get("symbol") or "").upper() != symbol_key:
            continue
        symbol_notional += abs(float(_safe_float((pos or {}).get("notional")) or 0.0))
    gross = abs(float(_safe_float(snapshot.get("gross_exposure")) or 0.0))
    net = float(_safe_float(snapshot.get("net_exposure")) or 0.0)
    signed_notional = notional if str(side or "").strip().lower() == "buy" else -notional
    symbol_limit = float(eq) * float(max_symbol_pct)
    gross_limit = float(eq) * float(max_gross_pct)
    net_limit = float(eq) * float(max_net_pct)
    new_symbol = symbol_notional + notional
    new_gross = gross + notional
    new_net = net + signed_notional
    reasons: list[str] = []
    if new_symbol > symbol_limit:
        reasons.append("symbol concentration cap breached")
    if new_gross > gross_limit:
        reasons.append("gross exposure cap breached")
    if abs(new_net) > net_limit:
        reasons.append("net exposure cap breached")
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "symbol_notional": round(symbol_notional, 4),
        "gross_exposure": round(gross, 4),
        "net_exposure": round(net, 4),
        "new_symbol_notional": round(new_symbol, 4),
        "new_gross_exposure": round(new_gross, 4),
        "new_net_exposure": round(new_net, 4),
        "symbol_limit": round(symbol_limit, 4),
        "gross_limit": round(gross_limit, 4),
        "net_limit": round(net_limit, 4),
    }


def should_pause_trading(
    *,
    portfolio: dict | None,
    max_daily_drawdown_pct: float = DEFAULT_DAILY_DRAWDOWN_PCT,
) -> dict:
    snapshot = dict(portfolio or {})
    eq = _safe_float(snapshot.get("equity"))
    if eq is None or eq <= 0.0:
        return {"allowed": False, "reason": "portfolio equity is unavailable"}
    daily_pnl = _safe_float(snapshot.get("daily_pnl"))
    if daily_pnl is None:
        daily_pnl = _safe_float(snapshot.get("realized_pnl_today"))
    if daily_pnl is None:
        return {"allowed": True, "paused": False, "reason": "daily pnl unavailable"}
    limit = abs(float(eq) * float(max_daily_drawdown_pct))
    paused = float(daily_pnl) <= -limit
    return {
        "allowed": not paused,
        "paused": paused,
        "daily_pnl": float(daily_pnl),
        "drawdown_limit": round(limit, 4),
        "reason": "daily drawdown pause triggered" if paused else "ok",
    }


def risk_gate_guidance(risk_summary: dict | None) -> dict:
    risk = dict(risk_summary or {})
    if not bool(risk.get("blocked")):
        return {}
    primary = str(risk.get("primary_blocker") or "").strip()
    if not primary:
        return {}
    lower = primary.lower()
    if primary == "regime is unknown or evidence is insufficient":
        return {
            "key": "unknown_regime",
            "primary_blocker": primary,
            "detail": "The current regime cannot be classified safely, so stronger execution postures should remain blocked.",
            "next_requirement": "Collect enough recent price action and stable context to classify the current regime safely.",
        }
    if "portfolio snapshot is unavailable" in lower or ("portfolio" in lower and "unavailable" in lower):
        return {
            "key": "portfolio_unavailable",
            "primary_blocker": primary,
            "detail": "Portfolio-aware autonomy cannot be proven because no current portfolio snapshot is available.",
            "next_requirement": "Refresh broker or portfolio state before treating this setup as autonomy-eligible; advisory review can continue separately.",
        }
    if "directional side is unavailable" in lower:
        return {
            "key": "side_unavailable",
            "primary_blocker": primary,
            "detail": "The model path has not resolved a buy/sell side, so risk cannot size the trade.",
            "next_requirement": "Wait for a directional side that survives signal translation before escalating beyond advisory review.",
        }
    if "current price is unavailable" in lower:
        return {
            "key": "price_unavailable",
            "primary_blocker": primary,
            "detail": "Risk cannot size or validate the setup without a current executable price.",
            "next_requirement": "Refresh market data until a current price is available before evaluating execution risk.",
        }
    if "risk budget" in lower or "requested size exceeds risk budget" in lower:
        return {
            "key": "risk_budget_blocked",
            "primary_blocker": primary,
            "detail": "The setup may have signal value, but the available risk budget is still too small for safe execution.",
            "next_requirement": "Wait for a cleaner setup with a safer stop distance and enough edge to support non-trivial size.",
        }
    if "exposure cap" in lower or "concentration cap" in lower:
        return {
            "key": "exposure_blocked",
            "primary_blocker": primary,
            "detail": "The setup may have signal value, but symbol or portfolio exposure caps are already too tight.",
            "next_requirement": "Reduce existing exposure or wait for gross and net exposure to clear before escalating beyond advisory review.",
        }
    if "drawdown pause" in lower:
        return {
            "key": "drawdown_pause",
            "primary_blocker": primary,
            "detail": "The setup may have signal value, but the risk engine is still pausing trading after losses.",
            "next_requirement": "Wait for the drawdown pause to clear before treating the setup as autonomy-eligible.",
        }
    if "stop logic" in lower:
        return {
            "key": "stop_unavailable",
            "primary_blocker": primary,
            "detail": "A safe invalidation distance is still missing, so the setup should not be promoted into stronger operating modes.",
            "next_requirement": "Require a concrete stop and invalidation distance before escalating beyond advisory review.",
        }
    return {
        "key": "risk_blocked",
        "primary_blocker": primary,
        "detail": f"The setup is still blocked by the risk engine: {primary}.",
        "next_requirement": "Resolve the active risk blocker before treating the setup as autonomy-eligible.",
    }


def risk_summary_for_symbol(
    symbol: str,
    *,
    df: pd.DataFrame | None = None,
    advisory: dict | None = None,
    predictions: dict | None = None,
    execution_snapshot: dict | None = None,
    portfolio: dict | None = None,
    current_price: float | None = None,
    atr14: float | None = None,
    profile: dict | None = None,
    require_portfolio: bool = False,
    requested_qty: int | None = None,
) -> dict:
    advisory = dict(advisory or {})
    profile = dict(profile or {})
    execution_snapshot = dict(execution_snapshot or {})
    pred_state = dict(advisory.get("pred_state") or {})
    intraday = dict(pred_state.get("intraday") or {})
    quality = dict(advisory.get("quality_meta") or {})
    plan = dict(advisory.get("plan_meta") or {})
    news_meta = dict(advisory.get("news_meta") or {})

    regime = classify_market_regime(df, predictions=predictions, news_meta=news_meta)
    side = _infer_side(advisory)
    px = _safe_float(current_price)
    if px is None:
        px = _safe_float(plan.get("entry"))
    if px is None and df is not None and not getattr(df, "empty", True) and "close" in df.columns:
        px = _safe_float(pd.to_numeric(df.get("close"), errors="coerce").iloc[-1])

    active_portfolio = dict(portfolio or execution_snapshot.get("portfolio") or {})
    portfolio_context = {
        "portfolio_context_available": bool(active_portfolio),
        "portfolio_context_required": bool(require_portfolio),
        "portfolio_context_detail": (
            "Portfolio snapshot is available."
            if active_portfolio
            else "Portfolio snapshot is unavailable; portfolio-aware autonomy cannot be proven."
        ),
    }
    if require_portfolio and not active_portfolio:
        return {
            "blocked": True,
            "primary_blocker": "portfolio snapshot is unavailable",
            "blockers": ["portfolio snapshot is unavailable"],
            "regime": regime,
            **portfolio_context,
        }

    if regime.get("unknown"):
        return {
            "blocked": True,
            "primary_blocker": "regime is unknown or evidence is insufficient",
            "blockers": ["regime is unknown or evidence is insufficient"],
            "regime": regime,
            **portfolio_context,
        }

    if side not in {"buy", "sell"}:
        return {
            "blocked": True,
            "primary_blocker": "directional side is unavailable",
            "blockers": ["directional side is unavailable"],
            "regime": regime,
            **portfolio_context,
        }

    if px is None or px <= 0.0:
        return {
            "blocked": True,
            "primary_blocker": "current price is unavailable",
            "blockers": ["current price is unavailable"],
            "regime": regime,
            **portfolio_context,
        }

    stop_price = _safe_float(plan.get("stop"))
    stop_info = compute_stop_distance(
        current_price=px,
        atr14=atr14,
        volatility_bps=_safe_float(regime.get("volatility_bps")),
        min_stop_pct=float(_safe_float(profile.get("min_stop_pct")) or DEFAULT_MIN_STOP_PCT),
    )
    if stop_price is None:
        stop_price = compute_stop_price(side, px, stop_info.get("stop_distance"))
    if stop_price is None:
        return {
            "blocked": True,
            "primary_blocker": "stop logic is unavailable",
            "blockers": ["stop logic is unavailable"],
            "regime": regime,
            **portfolio_context,
        }

    equity = _safe_float(active_portfolio.get("equity")) or _safe_float(profile.get("equity")) or DEFAULT_EQUITY
    budget = compute_risk_budget(
        equity=equity,
        risk_fraction=float(_safe_float(profile.get("risk_per_trade")) or DEFAULT_RISK_PER_TRADE),
        quality_score=_safe_float(quality.get("score")),
        position_size_signal=_safe_float(intraday.get("position_size")),
    )
    risk_per_share = max(0.01, abs(float(px) - float(stop_price)))
    max_qty = int(math.floor(float(budget.get("risk_budget") or 0.0) / risk_per_share))
    inferred_qty = 0
    rec = dict(execution_snapshot.get("recommendation") or {})
    rec_qty = _safe_float(rec.get("qty"))
    if rec_qty is not None and rec_qty > 0:
        inferred_qty = int(rec_qty)
    req_qty = int(_safe_float(requested_qty) or 0)
    if req_qty <= 0 and inferred_qty > 0:
        req_qty = int(inferred_qty)
    qty = req_qty if req_qty > 0 else max_qty
    quantity_source = "requested" if requested_qty is not None and req_qty > 0 else "execution_recommendation" if inferred_qty > 0 else "risk_budget_max"
    proposed_notional = float(qty) * float(px)
    exposure = portfolio_exposure_check(
        portfolio=active_portfolio,
        symbol=symbol,
        proposed_notional=proposed_notional,
        side=side,
        max_symbol_pct=float(_safe_float(profile.get("max_symbol_pct")) or DEFAULT_MAX_SYMBOL_PCT),
        max_gross_pct=float(_safe_float(profile.get("max_gross_pct")) or DEFAULT_MAX_GROSS_PCT),
        max_net_pct=float(_safe_float(profile.get("max_net_pct")) or DEFAULT_MAX_NET_PCT),
    ) if active_portfolio else {"allowed": not require_portfolio, "reasons": ["portfolio snapshot is unavailable"] if require_portfolio else []}
    pause = should_pause_trading(
        portfolio=active_portfolio,
        max_daily_drawdown_pct=float(_safe_float(profile.get("daily_drawdown_pct")) or DEFAULT_DAILY_DRAWDOWN_PCT),
    ) if active_portfolio else {"allowed": not require_portfolio, "paused": False, "reason": "daily pnl unavailable"}

    blockers = []
    if not bool(budget.get("allowed")):
        blockers.append(str(budget.get("reason") or "risk budget is unavailable"))
    if req_qty > 0 and req_qty > max_qty:
        blockers.append("requested size exceeds risk budget")
    if qty <= 0:
        blockers.append("risk budget is too small")
    if not bool(exposure.get("allowed")):
        blockers.extend(_clean_label_list(exposure.get("reasons") or []))
    if not bool(pause.get("allowed")):
        blockers.append(str(pause.get("reason") or "trading pause is active"))

    allowed = not blockers
    return {
        "blocked": not allowed,
        "primary_blocker": blockers[0] if blockers else "",
        "blockers": blockers,
        "regime": regime,
        "risk_budget": budget,
        "stop_distance": _safe_float(stop_info.get("stop_distance")),
        "stop_price": stop_price,
        "entry_price": round(float(px), 4),
        "target_price": _safe_float(plan.get("target1")) or _safe_float(plan.get("target2")),
        "side": side,
        "suggested_qty": int(max(0, qty)),
        "max_qty_by_risk": int(max(0, max_qty)),
        "proposed_notional": round(proposed_notional, 4),
        "risk_per_share": round(risk_per_share, 4),
        "exposure_check": exposure,
        "pause_check": pause,
        "requested_qty": int(max(0, req_qty)),
        "quantity_source": quantity_source,
        **portfolio_context,
        "detail": blockers[0] if blockers else f"Risk budget ${float(budget.get('risk_budget') or 0.0):.2f} | qty {int(max(0, qty))} | stop ${float(stop_price):.2f}",
    }
