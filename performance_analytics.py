from __future__ import annotations

import math

import numpy as np
import pandas as pd

from execution_engine import _load_state


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    if not np.isfinite(out):
        return None
    return float(out)


def _fmt_pct(value):
    val = _safe_float(value)
    return "-" if val is None else f"{100.0 * val:+.1f}%"


def _fmt_dollar(value):
    val = _safe_float(value)
    return "-" if val is None else f"${val:,.2f}"


def _trades_frame(symbol: str | None = None) -> pd.DataFrame:
    trades = list((_load_state().get("closed_trades") or []))
    return _trades_frame_from_records(trades, symbol=symbol)


def _trades_frame_from_records(trades: list[dict] | None, symbol: str | None = None) -> pd.DataFrame:
    trades = list(trades or [])
    if not trades:
        return pd.DataFrame()
    df = pd.DataFrame(trades)
    if symbol:
        df = df[df["symbol"].astype(str).str.upper() == str(symbol).upper()].copy()
    if df.empty:
        return df
    for col in ("pnl_dollars", "pnl_pct", "risk_multiple", "quality_score", "confidence", "hold_minutes"):
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _max_drawdown(values: pd.Series) -> float | None:
    if values is None or values.empty:
        return None
    equity = values.cumsum()
    peak = equity.cummax()
    dd = equity - peak
    return _safe_float(dd.min())


def _bucket_best(df: pd.DataFrame, col: str) -> tuple[str, str]:
    if df.empty or col not in df.columns:
        return "-", "-"
    grouped = df.groupby(df[col].fillna("Unknown"))["pnl_dollars"].agg(["mean", "count"])
    grouped = grouped[grouped["count"] >= 1].sort_values(["mean", "count"], ascending=[False, False])
    if grouped.empty:
        return "-", "-"
    best = grouped.iloc[0]
    return str(grouped.index[0]), f"{int(best['count'])} trades | avg {_fmt_dollar(best['mean'])}"


def _performance_snapshot_from_frame(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "summary": "No closed paper trades yet",
            "rows": [
                ("Win Rate", "-", "Need closed trades"),
                ("Expectancy", "-", "No realized paper results"),
                ("Drawdown", "-", "No realized paper results"),
            ],
            "trade_count": 0,
        }
    pnl = df["pnl_dollars"].fillna(0.0)
    wins = df[pnl > 0]
    losses = df[pnl < 0]
    win_rate = float((pnl > 0).mean()) if len(df) else None
    expectancy = _safe_float(pnl.mean())
    avg_win = _safe_float(wins["pnl_dollars"].mean()) if not wins.empty else None
    avg_loss = _safe_float(losses["pnl_dollars"].mean()) if not losses.empty else None
    profit_factor = None
    if not wins.empty and not losses.empty:
        profit_factor = _safe_float(wins["pnl_dollars"].sum() / abs(losses["pnl_dollars"].sum()))
    sharpe_like = None
    if len(df) >= 2 and float(pnl.std(ddof=0) or 0.0) > 1e-9:
        sharpe_like = _safe_float((pnl.mean() / pnl.std(ddof=0)) * math.sqrt(len(df)))
    mdd = _max_drawdown(pnl)
    best_setup, best_setup_detail = _bucket_best(df, "setup_type")
    best_conf, best_conf_detail = _bucket_best(df, "confidence_bucket")
    best_regime, best_regime_detail = _bucket_best(df, "regime")
    best_time, best_time_detail = _bucket_best(df, "time_bucket")
    summary = f"{len(df)} closed trades | win {_fmt_pct(win_rate)} | exp {_fmt_dollar(expectancy)}"
    rows = [
        ("Win Rate", _fmt_pct(win_rate), f"Avg win {_fmt_dollar(avg_win)} | avg loss {_fmt_dollar(avg_loss)}"),
        ("Expectancy", _fmt_dollar(expectancy), f"Profit factor {'-' if profit_factor is None else f'{profit_factor:.2f}x'}"),
        ("Max Drawdown", _fmt_dollar(mdd), f"Sharpe-like {'-' if sharpe_like is None else f'{sharpe_like:.2f}'}"),
        ("Best Setup", best_setup, best_setup_detail),
        ("Best Confidence", best_conf, best_conf_detail),
        ("Best Regime", best_regime, best_regime_detail),
        ("Best Time", best_time, best_time_detail),
    ]
    return {
        "summary": summary,
        "rows": rows[:7],
        "trade_count": int(len(df)),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "max_drawdown": mdd,
        "sharpe_like": sharpe_like,
    }


def get_performance_snapshot_from_trades(trades: list[dict] | None, symbol: str | None = None) -> dict:
    return _performance_snapshot_from_frame(_trades_frame_from_records(trades, symbol=symbol))


def get_performance_snapshot(symbol: str | None = None) -> dict:
    return _performance_snapshot_from_frame(_trades_frame(symbol))
