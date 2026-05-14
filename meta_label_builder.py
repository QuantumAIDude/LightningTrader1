from __future__ import annotations

import math
from typing import Any


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return float(out)


def _sign(value: float | None) -> int | None:
    val = _safe_float(value)
    if val is None or val == 0.0:
        return None
    return 1 if val > 0.0 else -1


def _action_direction(action: str | None) -> int | None:
    text = str(action or "").strip().lower()
    if text in {"buy", "long", "long_candidate", "watchlist_long"}:
        return 1
    if text in {"sell", "short", "short_candidate", "watchlist_short"}:
        return -1
    return None


def _future_return(record: dict, horizon: str) -> float | None:
    return _safe_float((record.get("future_return_by_horizon") or {}).get(horizon))


def _net_after_costs(record: dict, horizon: str) -> float | None:
    return _safe_float((record.get("net_after_costs_by_horizon") or {}).get(horizon))


def base_model_direction_correctness(record: dict, horizon: str = "1_step") -> dict:
    realized = _future_return(record, horizon)
    real_sign = _sign(realized)
    if real_sign is None:
        return {}
    out = {}
    for name, item in dict(record.get("base_predictions") or {}).items():
        pred_sign = _sign((item or {}).get("pred_ret"))
        out[name] = None if pred_sign is None else bool(pred_sign == real_sign)
    return out


def best_base_model(record: dict, horizon: str = "1_step") -> str | None:
    stored = (record.get("best_base_model_by_horizon") or {}).get(horizon)
    if stored:
        return str(stored)
    realized = _future_return(record, horizon)
    if realized is None:
        return None
    best_name = None
    best_error = None
    for name, item in dict(record.get("base_predictions") or {}).items():
        pred = _safe_float((item or {}).get("pred_ret"))
        if pred is None:
            continue
        err = abs(pred - realized)
        if best_error is None or err < best_error:
            best_error = err
            best_name = str(name)
    return best_name


def regime_edge_bucket(record: dict) -> str:
    regime = str(record.get("regime_label") or "unknown").lower().replace(" ", "_")
    edge = dict(record.get("economic_edge") or {})
    state = str(edge.get("state_key") or edge.get("state_label") or "unknown").lower().replace(" ", "_")
    return f"{regime}|{state}"


def meta_target_action_class(record: dict, horizon: str = "1_step", cost_floor: float | None = None) -> str:
    realized = _future_return(record, horizon)
    if realized is None:
        return "no_label"
    net = _net_after_costs(record, horizon)
    if net is not None:
        if net <= 0.0:
            return "stand_aside"
        return "long_candidate" if realized > 0.0 else "short_candidate" if realized < 0.0 else "stand_aside"
    floor = _safe_float(cost_floor)
    if floor is not None and abs(realized) <= floor:
        return "stand_aside"
    if realized > 0.0:
        return "long_candidate"
    if realized < 0.0:
        return "short_candidate"
    return "no_label"


def build_meta_labels(record: dict, horizon: str = "1_step", cost_floor: float | None = None) -> dict:
    """
    Derive simple supervised labels from already-resolved experience records.

    Assumptions:
    - Future returns are labels only and must not be fed into same-timestamp decisions.
    - A signal survives costs only when the signed net-after-cost outcome is positive.
    - Stand-aside is reasonable when the realized opportunity did not survive costs,
      or when the realized absolute move is smaller than the supplied cost floor.
    """
    record = dict(record or {})
    realized = _future_return(record, horizon)
    realized_sign = _sign(realized)
    avg_sign = _sign(record.get("average_prediction"))
    action = str(record.get("advisory_action") or "").lower()
    action_sign = _action_direction(action)
    net = _net_after_costs(record, horizon)
    survived = (record.get("signal_survived_costs_by_horizon") or {}).get(horizon)
    if survived is None and net is not None:
        survived = bool(net > 0.0)
    floor = _safe_float(cost_floor)
    stand_aside_reasonable = None
    if action == "stand_aside" and realized is not None:
        if net is not None:
            stand_aside_reasonable = bool(net <= 0.0)
        elif floor is not None:
            stand_aside_reasonable = bool(abs(realized) <= floor)
        else:
            stand_aside_reasonable = bool(realized == 0.0)
    return {
        "horizon": horizon,
        "average_prediction_was_directionally_correct": (
            None if avg_sign is None or realized_sign is None else bool(avg_sign == realized_sign)
        ),
        "advisory_action_was_directionally_correct": (
            None if action_sign is None or realized_sign is None else bool(action_sign == realized_sign)
        ),
        "signal_survived_costs": None if survived is None else bool(survived),
        "stand_aside_was_reasonable": stand_aside_reasonable,
        "best_base_model": best_base_model(record, horizon=horizon),
        "base_model_direction_correctness": base_model_direction_correctness(record, horizon=horizon),
        "regime_edge_bucket": regime_edge_bucket(record),
        "meta_target_action_class": meta_target_action_class(record, horizon=horizon, cost_floor=cost_floor),
    }
