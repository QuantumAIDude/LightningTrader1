from __future__ import annotations

import numpy as np
import pandas as pd


STRATEGY_CATALOG = {
    "trend_following": {
        "label": "Trend Following",
        "family": "trend",
        "description": "EMA alignment with slope confirmation.",
    },
    "momentum": {
        "label": "Momentum",
        "family": "momentum",
        "description": "Recent relative move with volume participation.",
    },
    "mean_reversion": {
        "label": "Mean Reversion",
        "family": "reversion",
        "description": "Oversold/overbought move back toward a rolling mean.",
    },
    "breakout": {
        "label": "Breakout",
        "family": "breakout",
        "description": "Price clears a recent range with participation.",
    },
    "vwap_reversion": {
        "label": "VWAP Reversion",
        "family": "reversion",
        "description": "Intraday stretch away from VWAP fades back toward VWAP.",
    },
    "opening_range_breakout": {
        "label": "Opening Range Breakout",
        "family": "breakout",
        "description": "Price breaks the first 30-minute range.",
    },
    "volatility_breakout": {
        "label": "Volatility Breakout",
        "family": "breakout",
        "description": "ATR-adjusted range expansion.",
    },
    "pairs_stat_arb": {
        "label": "Pairs / Stat Arb",
        "family": "stat_arb",
        "description": "Pairs spread mean reversion; requires a configured comparison symbol.",
        "requires_pair": True,
    },
}

DEFAULT_STRATEGY_KEY = "trend_following"


def _safe_float(value) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _close(df: pd.DataFrame | None) -> pd.Series:
    if df is None or getattr(df, "empty", True) or "close" not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df["close"], errors="coerce").dropna().astype(float)


def _volume(df: pd.DataFrame | None, index: pd.Index) -> pd.Series:
    if df is None or getattr(df, "empty", True) or "volume" not in df.columns:
        return pd.Series(0.0, index=index, dtype=float)
    return pd.to_numeric(df["volume"], errors="coerce").reindex(index).fillna(0.0).astype(float)


def _atr(df: pd.DataFrame | None, lookback: int = 14) -> float | None:
    if df is None or getattr(df, "empty", True):
        return None
    if not all(col in df.columns for col in ("high", "low", "close")):
        close = _close(df)
        if len(close) < 3:
            return None
        return _safe_float(close.diff().abs().tail(lookback).mean())
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return _safe_float(tr.tail(max(3, lookback)).mean())


def _vwap(df: pd.DataFrame | None, close: pd.Series) -> float | None:
    if close.empty:
        return None
    vol = _volume(df, close.index)
    if float(vol.sum()) <= 0.0:
        return _safe_float(close.tail(min(20, len(close))).mean())
    return _safe_float((close * vol).sum() / max(1e-12, float(vol.sum())))


def _confidence_from_strength(strength: float, floor: float = 0.50, span: float = 0.35) -> float:
    return float(max(floor, min(0.88, floor + span * max(0.0, min(1.0, strength)))))


def _base_signal(strategy_key: str, side: str = "", reason: str = "No signal", confidence: float = 0.0, edge_bps: float = 0.0) -> dict:
    meta = get_strategy(strategy_key)
    side = str(side or "").lower()
    if side not in {"buy", "sell"}:
        side = ""
    confidence = max(0.0, min(1.0, float(confidence or 0.0)))
    edge_bps = float(edge_bps or 0.0)
    return {
        "strategy_key": normalize_strategy_key(strategy_key),
        "strategy_label": str(meta.get("label") or strategy_key),
        "strategy_family": str(meta.get("family") or ""),
        "side": side,
        "allowed": bool(side and confidence >= 0.54 and edge_bps > 0.0),
        "confidence": confidence,
        "confidence_pct": round(confidence * 100.0, 2),
        "edge_bps": round(edge_bps, 3),
        "position_size": round(max(0.0, min(1.0, (confidence - 0.50) / 0.28)), 4),
        "reason": str(reason or "No signal"),
    }


def normalize_strategy_key(value: str | None) -> str:
    raw = str(value or DEFAULT_STRATEGY_KEY).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "trend": "trend_following",
        "moving_average": "trend_following",
        "ma_cross": "trend_following",
        "mean": "mean_reversion",
        "reversion": "mean_reversion",
        "vwap": "vwap_reversion",
        "orb": "opening_range_breakout",
        "opening_range": "opening_range_breakout",
        "vol_breakout": "volatility_breakout",
        "stat_arb": "pairs_stat_arb",
        "pairs": "pairs_stat_arb",
        "pairs_trading": "pairs_stat_arb",
    }
    key = aliases.get(raw, raw)
    return key if key in STRATEGY_CATALOG else DEFAULT_STRATEGY_KEY


def get_strategy(strategy_key: str | None) -> dict:
    key = normalize_strategy_key(strategy_key)
    payload = dict(STRATEGY_CATALOG.get(key) or STRATEGY_CATALOG[DEFAULT_STRATEGY_KEY])
    payload["key"] = key
    return payload


def list_strategy_catalog() -> list[dict]:
    return [get_strategy(key) for key in STRATEGY_CATALOG]


def strategy_label(strategy_key: str | None) -> str:
    return str(get_strategy(strategy_key).get("label") or normalize_strategy_key(strategy_key))


def _trend_following(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if len(close) < 34:
        return _base_signal("trend_following", reason="Need at least 34 bars for EMA trend confirmation")
    fast = close.ewm(span=9, adjust=False).mean()
    slow = close.ewm(span=21, adjust=False).mean()
    slope = (slow.iloc[-1] / slow.iloc[-8] - 1.0) if len(slow) >= 8 and slow.iloc[-8] else 0.0
    spread = (fast.iloc[-1] - slow.iloc[-1]) / max(0.01, close.iloc[-1])
    side = "buy" if spread > 0 and slope > 0 else "sell" if spread < 0 and slope < 0 else ""
    strength = min(1.0, abs(spread) / 0.004 + abs(slope) / 0.012)
    edge_bps = max(0.0, abs(spread) * 10000.0 * 0.65)
    return _base_signal(
        "trend_following",
        side=side,
        confidence=_confidence_from_strength(strength),
        edge_bps=edge_bps,
        reason=f"EMA9 {'above' if spread > 0 else 'below'} EMA21; slope {slope:+.2%}",
    )


def _momentum(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if len(close) < 31:
        return _base_signal("momentum", reason="Need at least 31 bars for momentum confirmation")
    ret_fast = close.iloc[-1] / close.iloc[-6] - 1.0
    ret_slow = close.iloc[-1] / close.iloc[-21] - 1.0
    vol = _volume(df, close.index)
    vol_ratio = _safe_float(vol.tail(10).mean() / max(1.0, vol.tail(60).median())) if len(vol) >= 20 else 1.0
    side = "buy" if ret_fast > 0 and ret_slow > 0 else "sell" if ret_fast < 0 and ret_slow < 0 else ""
    strength = min(1.0, (abs(ret_fast) * 90.0) + (abs(ret_slow) * 35.0) + max(0.0, (vol_ratio or 1.0) - 1.0) * 0.15)
    edge_bps = max(0.0, abs(0.65 * ret_fast + 0.35 * ret_slow) * 10000.0)
    return _base_signal(
        "momentum",
        side=side,
        confidence=_confidence_from_strength(strength),
        edge_bps=edge_bps,
        reason=f"Fast {ret_fast:+.2%}; slow {ret_slow:+.2%}; volume {vol_ratio or 1.0:.2f}x",
    )


def _mean_reversion(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if len(close) < 31:
        return _base_signal("mean_reversion", reason="Need at least 31 bars for z-score reversion")
    roll = close.rolling(30, min_periods=20)
    mean = roll.mean().iloc[-1]
    std = roll.std(ddof=0).iloc[-1]
    if not std or not np.isfinite(std):
        return _base_signal("mean_reversion", reason="Rolling volatility is too small")
    z = float((close.iloc[-1] - mean) / std)
    side = "buy" if z <= -1.05 else "sell" if z >= 1.05 else ""
    strength = min(1.0, max(0.0, (abs(z) - 0.80) / 1.50))
    edge_bps = max(0.0, min(80.0, abs(close.iloc[-1] - mean) / max(0.01, close.iloc[-1]) * 10000.0 * 0.55))
    return _base_signal(
        "mean_reversion",
        side=side,
        confidence=_confidence_from_strength(strength, floor=0.51),
        edge_bps=edge_bps,
        reason=f"Price z-score {z:+.2f} vs 30-bar mean",
    )


def _breakout(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if len(close) < 42:
        return _base_signal("breakout", reason="Need at least 42 bars for range breakout")
    prior = close.iloc[:-1].tail(40)
    high = float(prior.max())
    low = float(prior.min())
    last = float(close.iloc[-1])
    side = "buy" if last > high else "sell" if last < low else ""
    rng = max(0.01, high - low)
    strength = min(1.0, abs(last - (high if side == "buy" else low if side == "sell" else last)) / rng * 6.0)
    edge_bps = max(0.0, abs(last - (high if side == "buy" else low if side == "sell" else last)) / max(0.01, last) * 10000.0)
    return _base_signal(
        "breakout",
        side=side,
        confidence=_confidence_from_strength(strength),
        edge_bps=edge_bps,
        reason=f"Last ${last:.2f}; 40-bar range ${low:.2f}-${high:.2f}",
    )


def _vwap_reversion(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if len(close) < 20:
        return _base_signal("vwap_reversion", reason="Need at least 20 bars for VWAP reversion")
    vwap = _vwap(df, close)
    last = _safe_float(close.iloc[-1])
    if vwap is None or last in (None, 0.0):
        return _base_signal("vwap_reversion", reason="VWAP is unavailable")
    dist = (last - vwap) / max(0.01, last)
    threshold = 0.0022
    side = "buy" if dist <= -threshold else "sell" if dist >= threshold else ""
    strength = min(1.0, max(0.0, abs(dist) / 0.006))
    edge_bps = max(0.0, (abs(dist) - threshold * 0.50) * 10000.0)
    return _base_signal(
        "vwap_reversion",
        side=side,
        confidence=_confidence_from_strength(strength, floor=0.51),
        edge_bps=edge_bps,
        reason=f"Price is {dist:+.2%} from session VWAP",
    )


def _opening_range_breakout(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if df is None or getattr(df, "empty", True) or len(close) < 24:
        return _base_signal("opening_range_breakout", reason="Need at least 24 bars for opening range")
    window = min(15, max(4, len(df) // 5))
    head = df.iloc[:window]
    hi = _safe_float(pd.to_numeric(head.get("high", head.get("close")), errors="coerce").max())
    lo = _safe_float(pd.to_numeric(head.get("low", head.get("close")), errors="coerce").min())
    last = _safe_float(close.iloc[-1])
    if hi is None or lo is None or last is None:
        return _base_signal("opening_range_breakout", reason="Opening range levels are unavailable")
    side = "buy" if last > hi else "sell" if last < lo else ""
    rng = max(0.01, hi - lo)
    strength = min(1.0, abs(last - (hi if side == "buy" else lo if side == "sell" else last)) / rng * 5.0)
    edge_bps = max(0.0, abs(last - (hi if side == "buy" else lo if side == "sell" else last)) / max(0.01, last) * 10000.0)
    return _base_signal(
        "opening_range_breakout",
        side=side,
        confidence=_confidence_from_strength(strength),
        edge_bps=edge_bps,
        reason=f"Opening range ${lo:.2f}-${hi:.2f}; last ${last:.2f}",
    )


def _volatility_breakout(df: pd.DataFrame | None) -> dict:
    close = _close(df)
    if len(close) < 42:
        return _base_signal("volatility_breakout", reason="Need at least 42 bars for volatility breakout")
    atr = _atr(df, lookback=20)
    if atr is None or atr <= 0:
        return _base_signal("volatility_breakout", reason="ATR is unavailable")
    anchor = _safe_float(close.iloc[-21])
    last = _safe_float(close.iloc[-1])
    if anchor is None or last is None:
        return _base_signal("volatility_breakout", reason="Price anchor is unavailable")
    move = last - anchor
    threshold = max(0.01, 0.75 * atr)
    side = "buy" if move > threshold else "sell" if move < -threshold else ""
    strength = min(1.0, abs(move) / max(0.01, 1.8 * atr))
    edge_bps = max(0.0, (abs(move) - 0.35 * atr) / max(0.01, last) * 10000.0)
    return _base_signal(
        "volatility_breakout",
        side=side,
        confidence=_confidence_from_strength(strength),
        edge_bps=edge_bps,
        reason=f"20-bar move ${move:+.2f}; ATR ${atr:.2f}",
    )


def _pairs_stat_arb(df: pd.DataFrame | None, pair_symbol: str | None = None) -> dict:
    if not str(pair_symbol or "").strip():
        return _base_signal("pairs_stat_arb", reason="Pairs / stat arb requires a configured pair symbol before automation")
    return _base_signal("pairs_stat_arb", reason="Pair spread feed is not connected in this runtime")


def evaluate_strategy_signal(
    strategy_key: str | None,
    df: pd.DataFrame | None,
    *,
    symbol: str | None = None,
    pair_symbol: str | None = None,
) -> dict:
    key = normalize_strategy_key(strategy_key)
    try:
        if key == "trend_following":
            signal = _trend_following(df)
        elif key == "momentum":
            signal = _momentum(df)
        elif key == "mean_reversion":
            signal = _mean_reversion(df)
        elif key == "breakout":
            signal = _breakout(df)
        elif key == "vwap_reversion":
            signal = _vwap_reversion(df)
        elif key == "opening_range_breakout":
            signal = _opening_range_breakout(df)
        elif key == "volatility_breakout":
            signal = _volatility_breakout(df)
        elif key == "pairs_stat_arb":
            signal = _pairs_stat_arb(df, pair_symbol=pair_symbol)
        else:
            signal = _trend_following(df)
    except Exception as exc:
        signal = _base_signal(key, reason=f"Strategy evaluation failed: {exc}")
    signal["symbol"] = str(symbol or "").upper()
    if signal.get("allowed") and not signal.get("side"):
        signal["allowed"] = False
    if not signal.get("allowed") and signal.get("side"):
        signal["reason"] = f"{signal.get('reason')}; signal below strategy confidence/edge threshold"
    return signal


def build_strategy_advisory(advisory: dict | None, signal: dict | None, current_price: float | None = None) -> dict:
    payload = dict(advisory or {})
    signal = dict(signal or {})
    side = str(signal.get("side") or "").lower()
    price = _safe_float(current_price)
    if price is None:
        plan = dict(payload.get("plan_meta") or {})
        price = _safe_float(plan.get("entry"))
    if side not in {"buy", "sell"} or price is None:
        return payload
    confidence_pct = _safe_float(signal.get("confidence_pct")) or 0.0
    edge_bps = _safe_float(signal.get("edge_bps")) or 0.0
    position_size = _safe_float(signal.get("position_size")) or 0.0
    stop_pct = max(0.0025, min(0.012, (edge_bps / 10000.0) * 1.35 if edge_bps > 0 else 0.0035))
    target_pct = max(stop_pct * 1.35, min(0.025, stop_pct * 2.0))
    stop = price * (1.0 - stop_pct) if side == "buy" else price * (1.0 + stop_pct)
    target = price * (1.0 + target_pct) if side == "buy" else price * (1.0 - target_pct)
    payload["plan_meta"] = {
        **dict(payload.get("plan_meta") or {}),
        "bias": "Bullish" if side == "buy" else "Bearish",
        "action": f"{signal.get('strategy_label') or 'Strategy'} setup",
        "entry": float(price),
        "stop": round(float(stop), 4),
        "target1": round(float(target), 4),
        "target2": None,
        "strategy_key": signal.get("strategy_key"),
        "strategy_label": signal.get("strategy_label"),
    }
    pred_state = dict(payload.get("pred_state") or {})
    intraday = dict(pred_state.get("intraday") or {})
    intraday.update(
        {
            "direction": "UP" if side == "buy" else "DOWN",
            "raw_direction": "UP" if side == "buy" else "DOWN",
            "direction_stage": "strategy",
            "confidence": confidence_pct,
            "raw_confidence": confidence_pct,
            "edge_bps": edge_bps,
            "raw_edge_bps": edge_bps,
            "position_size": position_size,
            "raw_position_size": position_size,
            "note": f"{signal.get('strategy_label')}: {signal.get('reason')}",
        }
    )
    pred_state["intraday"] = intraday
    payload["pred_state"] = pred_state
    quality = dict(payload.get("quality_meta") or {})
    quality["score"] = max(_safe_float(quality.get("score")) or 0.0, _safe_float(signal.get("confidence")) or 0.0)
    quality["no_trade"] = False
    quality["strategy"] = str(signal.get("strategy_label") or "")
    payload["quality_meta"] = quality
    return payload
