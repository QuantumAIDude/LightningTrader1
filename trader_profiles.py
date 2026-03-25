from __future__ import annotations

from copy import deepcopy

from config import TRADER_PROFILE_DEFAULT


_PROFILES = {
    "scalper": {
        "label": "Scalper",
        "risk_per_trade": 0.0030,
        "max_single_position_pct": 0.10,
        "max_gross_exposure": 0.35,
        "min_quality_score": 0.62,
        "preferred_horizon": "intraday",
        "preferred_strategies": {"Scalp", "Breakout"},
        "prefer_options": False,
        "scan_weights": {"quality": 0.42, "tradability": 0.24, "agreement": 0.20, "catalyst": 0.04, "style": 0.10},
    },
    "day_trader": {
        "label": "Day Trader",
        "risk_per_trade": 0.0045,
        "max_single_position_pct": 0.14,
        "max_gross_exposure": 0.55,
        "min_quality_score": 0.58,
        "preferred_horizon": "intraday",
        "preferred_strategies": {"Breakout", "Trend", "Mean Reversion"},
        "prefer_options": False,
        "scan_weights": {"quality": 0.40, "tradability": 0.20, "agreement": 0.18, "catalyst": 0.08, "style": 0.14},
    },
    "swing": {
        "label": "Swing Trader",
        "risk_per_trade": 0.0065,
        "max_single_position_pct": 0.18,
        "max_gross_exposure": 0.75,
        "min_quality_score": 0.55,
        "preferred_horizon": "swing",
        "preferred_strategies": {"Trend", "Swing", "Breakout"},
        "prefer_options": False,
        "scan_weights": {"quality": 0.34, "tradability": 0.14, "agreement": 0.14, "catalyst": 0.18, "style": 0.20},
    },
    "options": {
        "label": "Options Trader",
        "risk_per_trade": 0.0040,
        "max_single_position_pct": 0.10,
        "max_gross_exposure": 0.45,
        "min_quality_score": 0.57,
        "preferred_horizon": "mixed",
        "preferred_strategies": {"Breakout", "Swing", "Catalyst"},
        "prefer_options": True,
        "scan_weights": {"quality": 0.30, "tradability": 0.12, "agreement": 0.12, "catalyst": 0.18, "style": 0.28},
    },
    "conservative": {
        "label": "Conservative",
        "risk_per_trade": 0.0025,
        "max_single_position_pct": 0.08,
        "max_gross_exposure": 0.28,
        "min_quality_score": 0.64,
        "preferred_horizon": "mixed",
        "preferred_strategies": {"Trend", "Swing"},
        "prefer_options": False,
        "scan_weights": {"quality": 0.44, "tradability": 0.20, "agreement": 0.18, "catalyst": 0.06, "style": 0.12},
    },
    "balanced": {
        "label": "Balanced",
        "risk_per_trade": 0.0040,
        "max_single_position_pct": 0.12,
        "max_gross_exposure": 0.50,
        "min_quality_score": 0.58,
        "preferred_horizon": "mixed",
        "preferred_strategies": {"Breakout", "Trend", "Swing", "Mean Reversion"},
        "prefer_options": False,
        "scan_weights": {"quality": 0.38, "tradability": 0.18, "agreement": 0.16, "catalyst": 0.10, "style": 0.18},
    },
    "aggressive": {
        "label": "Aggressive",
        "risk_per_trade": 0.0080,
        "max_single_position_pct": 0.22,
        "max_gross_exposure": 0.95,
        "min_quality_score": 0.52,
        "preferred_horizon": "mixed",
        "preferred_strategies": {"Breakout", "Trend", "Swing", "Catalyst"},
        "prefer_options": True,
        "scan_weights": {"quality": 0.32, "tradability": 0.12, "agreement": 0.14, "catalyst": 0.16, "style": 0.26},
    },
}


def list_trader_profiles() -> list[str]:
    return list(_PROFILES.keys())


def get_trader_profile(name: str | None = None) -> dict:
    key = str(name or TRADER_PROFILE_DEFAULT or "balanced").strip().lower()
    payload = deepcopy(_PROFILES.get(key, _PROFILES["balanced"]))
    payload["key"] = key if key in _PROFILES else "balanced"
    return payload


def classify_confidence_bucket(value: float | None) -> str:
    try:
        val = float(value)
    except Exception:
        return "Unknown"
    if val >= 0.70:
        return "High"
    if val >= 0.58:
        return "Medium"
    return "Low"


def score_style_fit(profile_name: str | None, strategy: str | None, horizon_key: str | None, options_bias: bool = False) -> float:
    profile = get_trader_profile(profile_name)
    score = 0.5
    strat = str(strategy or "").strip()
    horizon = str(horizon_key or "").strip().lower()
    if strat and strat in set(profile.get("preferred_strategies") or set()):
        score += 0.25
    pref_horizon = str(profile.get("preferred_horizon") or "mixed").lower()
    if pref_horizon == "mixed":
        score += 0.10
    elif pref_horizon == "intraday" and horizon in ("intraday", "4m"):
        score += 0.20
    elif pref_horizon == "swing" and horizon in ("daily", "weekend", "1d", "1w"):
        score += 0.20
    if bool(profile.get("prefer_options")) == bool(options_bias):
        score += 0.10
    return max(0.0, min(1.0, score))
