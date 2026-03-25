from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from config import ALERT_EVENT_COOLDOWN_SEC, DATA_DIR
from execution_engine import get_execution_dashboard_snapshot


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _alerts_path() -> str:
    return os.path.join(DATA_DIR, "trader_alerts.jsonl")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_alerts(limit: int = 200) -> list[dict]:
    path = _alerts_path()
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except Exception:
        return []
    return rows[-int(limit):]


def _append_alert(alert: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(_alerts_path(), "a") as f:
            f.write(json.dumps(alert, sort_keys=True) + "\n")
    except Exception:
        pass


def _should_persist(alert: dict, recent: list[dict], cooldown_sec: int) -> bool:
    fp = str(alert.get("fingerprint") or "")
    if not fp:
        return True
    now = pd.Timestamp.now(tz="UTC")
    for item in reversed(recent):
        if str(item.get("fingerprint") or "") != fp:
            continue
        try:
            ts = pd.Timestamp(item.get("generated_at"))
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            age = (now - ts).total_seconds()
            return age > float(cooldown_sec)
        except Exception:
            return False
    return True


def get_alert_snapshot(
    symbol: str,
    current_price: float | None = None,
    advisory: dict | None = None,
    profile_name: str | None = None,
    persist: bool = True,
    cooldown_sec: int | None = None,
) -> dict:
    symbol = str(symbol or "").upper()
    advisory = advisory or {}
    quality = dict(advisory.get("quality_meta") or {})
    macro = dict(advisory.get("macro_meta") or {})
    catalyst = dict(advisory.get("catalyst_meta") or {})
    liquidity = dict(advisory.get("liquidity_meta") or {})
    plan = dict(advisory.get("plan_meta") or {})
    execution = get_execution_dashboard_snapshot(symbol, current_price=current_price, advisory=advisory, profile_name=profile_name)
    portfolio = execution.get("portfolio") or {}
    pos = next((p for p in portfolio.get("positions", []) if str(p.get("symbol") or "").upper() == symbol), None)
    alerts = []
    quality_score = _safe_float(quality.get("score")) or 0.0
    agreement = _safe_float(quality.get("agreement")) or 0.0
    tradability = _safe_float(liquidity.get("tradability_score"))
    macro_risk = _safe_float(macro.get("risk_score")) or 0.0
    catalyst_risk = _safe_float(catalyst.get("catalyst_risk_score")) or 0.0
    no_trade = bool(quality.get("no_trade"))
    reasons = ", ".join((quality.get("reasons") or [])[:2]) or "edge below threshold"
    now = _utc_now()

    if no_trade:
        alerts.append(
            {
                "symbol": symbol,
                "kind": "no_trade",
                "severity": "medium",
                "title": "No-trade condition",
                "detail": reasons,
                "generated_at": now,
                "fingerprint": f"{symbol}|no_trade|{reasons}",
            }
        )
    elif quality_score >= 0.72 and agreement >= 0.72 and (tradability is None or tradability >= 0.60):
        alerts.append(
            {
                "symbol": symbol,
                "kind": "setup",
                "severity": "high",
                "title": str(plan.get("action") or "High-quality setup"),
                "detail": f"score {int(round(100.0 * quality_score))}/100 | agreement {int(round(100.0 * agreement))}%",
                "generated_at": now,
                "fingerprint": f"{symbol}|setup|{str(plan.get('action') or '-')}",
            }
        )

    if macro_risk >= 0.75 or bool(macro.get("high_risk_24h")):
        next_event = dict(macro.get("next_event") or {})
        title = str(next_event.get("title") or "Macro event risk")
        detail = str(next_event.get("time_to_event_text") or "Within 24h")
        alerts.append(
            {
                "symbol": symbol,
                "kind": "macro",
                "severity": "high",
                "title": title,
                "detail": f"{detail} | risk overlay only",
                "generated_at": now,
                "fingerprint": f"{symbol}|macro|{title}|{detail}",
            }
        )

    active_catalyst = dict(catalyst.get("active_catalyst") or {})
    if catalyst_risk >= 0.45 and active_catalyst:
        alerts.append(
            {
                "symbol": symbol,
                "kind": "catalyst",
                "severity": "medium",
                "title": str(active_catalyst.get("title") or "Fresh catalyst"),
                "detail": f"{active_catalyst.get('risk_kind') or 'event risk'} | ambiguity {active_catalyst.get('ambiguity') or 'high'}",
                "generated_at": now,
                "fingerprint": f"{symbol}|catalyst|{str(active_catalyst.get('title') or '-')}",
            }
        )

    if pos and current_price is not None:
        side = str(pos.get("side") or "").lower()
        stop = _safe_float(pos.get("stop_price"))
        target = _safe_float(pos.get("target_price"))
        px = _safe_float(current_price)
        if side == "long" and stop is not None and px is not None and px <= stop:
            alerts.append(
                {
                    "symbol": symbol,
                    "kind": "invalidation",
                    "severity": "high",
                    "title": "Long invalidation hit",
                    "detail": f"spot ${px:.2f} <= stop ${stop:.2f}",
                    "generated_at": now,
                    "fingerprint": f"{symbol}|long_stop|{stop:.2f}",
                }
            )
        elif side == "short" and stop is not None and px is not None and px >= stop:
            alerts.append(
                {
                    "symbol": symbol,
                    "kind": "invalidation",
                    "severity": "high",
                    "title": "Short invalidation hit",
                    "detail": f"spot ${px:.2f} >= stop ${stop:.2f}",
                    "generated_at": now,
                    "fingerprint": f"{symbol}|short_stop|{stop:.2f}",
                }
            )
        if target is not None and px is not None:
            hit = (side == "long" and px >= target) or (side == "short" and px <= target)
            if hit:
                alerts.append(
                    {
                        "symbol": symbol,
                        "kind": "target",
                        "severity": "low",
                        "title": "Target reached",
                        "detail": f"spot ${px:.2f} vs target ${target:.2f}",
                        "generated_at": now,
                        "fingerprint": f"{symbol}|target|{target:.2f}",
                    }
                )

    recent = _load_alerts(limit=200)
    cooldown = int(cooldown_sec if cooldown_sec is not None else ALERT_EVENT_COOLDOWN_SEC)
    if persist:
        for item in alerts:
            if _should_persist(item, recent, cooldown):
                _append_alert(item)
                recent.append(item)

    visible = list(reversed(recent[-6:])) if recent else list(reversed(alerts[-6:]))
    rows = [(str(item.get("title") or "-")[:28], str(item.get("severity") or "-").title(), str(item.get("detail") or "-")[:120]) for item in visible]
    summary = "No active trader alerts" if not rows else f"{len(rows)} recent alerts"
    return {
        "summary": summary,
        "rows": rows[:6],
        "alerts": alerts,
        "recent": visible[:6],
    }
