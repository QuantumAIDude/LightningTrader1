from __future__ import annotations

import json
import math
import os
import uuid
import copy
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import DATA_DIR, INTERVAL


DEFAULT_LEDGER_PATH = os.path.join(DATA_DIR, "model_experience_ledger.jsonl")
MODEL_KEYS = ("base", "forest", "boost", "extra_trees", "meta")


def _safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except Exception:
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return float(out)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    try:
        import pandas as pd

        ts = pd.Timestamp(value)
        if pd.isna(ts):
            return None
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.tz_convert("UTC").isoformat()
    except Exception:
        if isinstance(value, datetime):
            ts = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc).isoformat()
    text = str(value).strip()
    return text or None


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return _iso(value)
    try:
        import numpy as np
        import pandas as pd

        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return _safe_float(value)
        if isinstance(value, (pd.Timestamp,)):
            return _iso(value)
        if isinstance(value, (pd.Timedelta,)):
            return str(value)
    except Exception:
        pass
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _prediction_block(predictions: dict | None) -> dict:
    predictions = dict(predictions or {})
    intraday = dict(predictions.get("intraday") or predictions)
    out: dict[str, dict] = {}
    for key in MODEL_KEYS:
        item = intraday.get(key)
        if not isinstance(item, dict):
            continue
        out[key] = {
            "direction": str(item.get("direction") or "").upper() or None,
            "pred_ret": _safe_float(item.get("pred_ret")),
            "raw_pred_ret": _safe_float(item.get("raw_pred_ret")),
            "pred_norm": _safe_float(item.get("pred_norm")),
            "confidence": _safe_float(item.get("confidence")),
            "sig_bucket": item.get("sig_bucket"),
        }
    return out


def _average_prediction(predictions: dict | None, base_predictions: dict | None = None) -> float | None:
    predictions = dict(predictions or {})
    intraday = dict(predictions.get("intraday") or predictions)
    consensus = dict(intraday.get("consensus") or {})
    consensus_ret = _safe_float(consensus.get("pred_ret"))
    if consensus_ret is not None:
        return consensus_ret
    vals = []
    for item in dict(base_predictions or {}).values():
        pred = _safe_float((item or {}).get("pred_ret"))
        if pred is not None:
            vals.append(pred)
    return float(sum(vals) / len(vals)) if vals else None


def _model_confidence(base_predictions: dict | None, predictions: dict | None = None) -> dict:
    conf = {}
    for key, item in dict(base_predictions or {}).items():
        value = _safe_float((item or {}).get("confidence"))
        if value is not None:
            conf[key] = value
    intraday = dict((predictions or {}).get("intraday") or predictions or {})
    consensus_conf = _safe_float(dict(intraday.get("consensus") or {}).get("confidence"))
    if consensus_conf is not None:
        conf["consensus"] = consensus_conf
    return conf


def _model_disagreement(predictions: dict | None) -> float | None:
    intraday = dict((predictions or {}).get("intraday") or predictions or {})
    consensus = dict(intraday.get("consensus") or {})
    disagreement_bps = _safe_float(consensus.get("disagreement_bps"))
    if disagreement_bps is not None:
        return disagreement_bps
    diversity = dict(dict(intraday.get("_meta") or {}).get("diversity") or {})
    return _safe_float(diversity.get("dispersion_bps") or diversity.get("disagreement_bps"))


def _regime_from(predictions: dict | None, advisory: dict | None) -> tuple[str | None, float | None]:
    advisory = dict(advisory or {})
    regime_meta = dict(advisory.get("regime_meta") or {})
    label = regime_meta.get("label")
    conf = _safe_float(regime_meta.get("confidence"))
    if label or conf is not None:
        return (str(label) if label else None, conf)
    intraday = dict((predictions or {}).get("intraday") or predictions or {})
    regime = dict(intraday.get("regime") or {})
    if not regime:
        return None, None
    scores = {
        "bullish": _safe_float(regime.get("Bullish")) or 0.0,
        "neutral": _safe_float(regime.get("Neutral")) or 0.0,
        "bearish": _safe_float(regime.get("Bearish")) or 0.0,
    }
    label = max(scores.items(), key=lambda item: item[1])[0] if scores else None
    return label, _safe_float(regime.get("regime_conf"))


def _best_base_model(base_predictions: dict, realized_return: float | None) -> str | None:
    if realized_return is None:
        return None
    best_name = None
    best_error = None
    for key, item in dict(base_predictions or {}).items():
        pred = _safe_float((item or {}).get("pred_ret"))
        if pred is None:
            continue
        error = abs(float(pred) - float(realized_return))
        if best_error is None or error < best_error:
            best_error = error
            best_name = key
    return best_name


def build_experience_record(
    *,
    symbol: str,
    as_of: Any,
    source: str,
    horizon: str | None = None,
    interval: str | None = None,
    predictions: dict | None = None,
    advisory: dict | None = None,
    risk_summary: dict | None = None,
    validation_evidence: dict | None = None,
    calibration_state: dict | None = None,
    market_context: dict | None = None,
    news_context: dict | None = None,
    event_context: dict | None = None,
    macro_context: dict | None = None,
    model_runtime: dict | None = None,
    record_id: str | None = None,
    created_at: Any = None,
) -> dict:
    advisory = dict(advisory or {})
    decision_payload = dict(advisory.get("decision_payload") or {})
    quality = dict(advisory.get("quality_meta") or {})
    deployment = dict(advisory.get("deployment_meta") or {})
    validation = dict(validation_evidence or deployment.get("validation_evidence") or {})
    base_predictions = _prediction_block(predictions)
    avg_pred = _average_prediction(predictions, base_predictions)
    regime_label, regime_conf = _regime_from(predictions, advisory)
    risk = dict(risk_summary or advisory.get("risk_meta") or decision_payload.get("risk_context") or {})
    blockers = list(decision_payload.get("blockers") or [])
    for item in list(decision_payload.get("validation_blockers") or []):
        if item not in blockers:
            blockers.append(item)
    return _json_safe(
        {
            "record_type": "experience",
            "schema_version": 1,
            "record_id": record_id or uuid.uuid4().hex,
            "created_at": _iso(created_at) or datetime.now(timezone.utc).isoformat(),
            "symbol": str(symbol or "").upper(),
            "as_of": _iso(as_of),
            "source": str(source or "unknown"),
            "horizon": horizon,
            "interval": interval or INTERVAL,
            "base_predictions": base_predictions,
            "average_prediction": avg_pred,
            "prediction_units": "return_fraction",
            "model_disagreement": _model_disagreement(predictions),
            "model_confidence": _model_confidence(base_predictions, predictions),
            "trained_modes": (model_runtime or {}).get("trained_modes") if isinstance(model_runtime, dict) else None,
            "model_runtime_status": (model_runtime or {}).get("status") if isinstance(model_runtime, dict) else None,
            "advisory_action": decision_payload.get("action") or signal_action_from_advisory(advisory),
            "advisory_confidence": _safe_float(
                decision_payload.get("model_edge", {}).get("confidence") if isinstance(decision_payload.get("model_edge"), dict) else None
            )
            or _safe_float(quality.get("score")),
            "advisory_thesis": decision_payload.get("thesis"),
            "economic_edge": decision_payload.get("economic_edge"),
            "model_edge": decision_payload.get("model_edge"),
            "blockers": blockers,
            "contradictory_evidence": list(decision_payload.get("contradictory_evidence") or []),
            "next_best_action": decision_payload.get("next_best_action"),
            "autonomy_posture": decision_payload.get("autonomy_posture"),
            "regime_label": regime_label,
            "regime_confidence": regime_conf,
            "risk_summary": risk,
            "validation_status": validation.get("validation_status") or decision_payload.get("validation_status") or deployment.get("validation_status"),
            "validation_method": validation.get("validation_method") or decision_payload.get("validation_method") or deployment.get("validation_method"),
            "validation_blockers": list(validation.get("blockers") or validation.get("autonomy_blockers") or decision_payload.get("validation_blockers") or []),
            "calibration_state": calibration_state,
            "drift_flags": {
                "calibration_drift": bool((calibration_state or {}).get("drift_flag")) if isinstance(calibration_state, dict) else None,
                "validation_drift": "drift_detected" in list(validation.get("blockers") or []),
            },
            "market_context": market_context,
            "news_context": news_context,
            "event_context": event_context,
            "macro_context": macro_context,
            "future_return_by_horizon": {},
            "direction_correct_by_horizon": {},
            "net_after_costs_by_horizon": {},
            "best_base_model_by_horizon": {},
            "signal_survived_costs_by_horizon": {},
            "outcome_resolved_at": None,
        }
    )


def signal_action_from_advisory(advisory: dict | None) -> str | None:
    advisory = dict(advisory or {})
    trans = dict(advisory.get("signal_translation") or {})
    side = str(trans.get("operator_side") or "").lower()
    if side in {"buy", "sell"}:
        return side
    if bool(trans.get("watchlist")):
        direction = str(trans.get("bias") or trans.get("action_label") or "").lower()
        if "short" in direction or "sell" in direction:
            return "watchlist_short"
        if "long" in direction or "buy" in direction:
            return "watchlist_long"
        return "watchlist"
    return "stand_aside"


def _record_direction(record: dict) -> int | None:
    action = str(record.get("advisory_action") or "").lower()
    if action in {"buy", "long", "long_candidate", "watchlist_long"}:
        return 1
    if action in {"sell", "short", "short_candidate", "watchlist_short"}:
        return -1
    edge = dict(record.get("model_edge") or {})
    direction = str(edge.get("direction") or "").upper()
    if direction == "UP":
        return 1
    if direction == "DOWN":
        return -1
    avg = _safe_float(record.get("average_prediction"))
    if avg is not None and avg != 0.0:
        return 1 if avg > 0 else -1
    return None


def resolve_one_step_outcome(
    record: dict,
    *,
    current_price: float | None,
    future_price: float | None,
    future_as_of: Any = None,
    cost_pct: float | None = None,
    horizon: str = "1_step",
) -> dict:
    out = copy.deepcopy(dict(record or {}))
    current = _safe_float(current_price)
    future = _safe_float(future_price)
    if current is None or current <= 0 or future is None:
        return out
    realized = float((future / current) - 1.0)
    direction = _record_direction(out)
    signed = None if direction is None else float(realized * direction)
    cost = _safe_float(cost_pct)
    net = None if signed is None or cost is None else float(signed - cost)
    out.setdefault("future_return_by_horizon", {})[horizon] = realized
    out.setdefault("direction_correct_by_horizon", {})[horizon] = None if signed is None else bool(signed > 0.0)
    out.setdefault("net_after_costs_by_horizon", {})[horizon] = net
    out.setdefault("signal_survived_costs_by_horizon", {})[horizon] = None if net is None else bool(net > 0.0)
    out.setdefault("best_base_model_by_horizon", {})[horizon] = _best_base_model(dict(out.get("base_predictions") or {}), realized)
    out["outcome_resolved_at"] = _iso(future_as_of) or datetime.now(timezone.utc).isoformat()
    return _json_safe(out)


def build_outcome_record(
    *,
    record_id: str,
    source: str,
    outcome: dict,
    as_of: Any = None,
    created_at: Any = None,
) -> dict:
    return _json_safe(
        {
            "record_type": "outcome",
            "schema_version": 1,
            "record_id": uuid.uuid4().hex,
            "parent_record_id": str(record_id),
            "source": str(source or "unknown"),
            "as_of": _iso(as_of),
            "created_at": _iso(created_at) or datetime.now(timezone.utc).isoformat(),
            "outcome": outcome or {},
        }
    )


def append_record(record: dict, path: str | os.PathLike[str] = DEFAULT_LEDGER_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(_json_safe(record), sort_keys=True, separators=(",", ":")) + "\n")


def append_records(records: list[dict] | tuple[dict, ...], path: str | os.PathLike[str] = DEFAULT_LEDGER_PATH) -> int:
    count = 0
    for record in list(records or []):
        append_record(record, path=path)
        count += 1
    return count


def read_records(path: str | os.PathLike[str] = DEFAULT_LEDGER_PATH) -> list[dict]:
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def summarize_ledger(path: str | os.PathLike[str] = DEFAULT_LEDGER_PATH) -> dict:
    records = read_records(path)
    experiences = [r for r in records if str(r.get("record_type") or "experience") == "experience"]
    outcomes = [r for r in records if str(r.get("record_type") or "") == "outcome"]
    outcome_parent_ids = {str(r.get("parent_record_id")) for r in outcomes if r.get("parent_record_id")}
    resolved = [
        r
        for r in experiences
        if r.get("outcome_resolved_at")
        or any((r.get("future_return_by_horizon") or {}).values())
        or str(r.get("record_id")) in outcome_parent_ids
    ]
    symbols = sorted({str(r.get("symbol") or "").upper() for r in experiences if r.get("symbol")})
    horizons = sorted(
        {
            str(h)
            for r in experiences
            for h in list((r.get("future_return_by_horizon") or {}).keys()) + ([r.get("horizon")] if r.get("horizon") else [])
            if h
        }
    )
    missing = []
    if not experiences:
        missing.append("no experience records")
    if any(not r.get("base_predictions") for r in experiences):
        missing.append("base_predictions incomplete")
    if any(not r.get("future_return_by_horizon") for r in experiences):
        missing.append("unresolved future outcomes")
    if any(not r.get("validation_status") for r in experiences):
        missing.append("validation status missing on some rows")
    if any(not r.get("regime_label") for r in experiences):
        missing.append("regime label missing on some rows")
    return {
        "path": str(path),
        "exists": bool(os.path.exists(path)),
        "records": len(records),
        "experience_records": len(experiences),
        "outcome_records": len(outcomes),
        "resolved_experience_records": len(resolved),
        "unresolved_experience_records": max(0, len(experiences) - len(resolved)),
        "symbols": symbols,
        "horizons": horizons,
        "missing_fields": missing,
    }


OUTCOME_KEYS = {
    "future_return_by_horizon",
    "direction_correct_by_horizon",
    "net_after_costs_by_horizon",
    "best_base_model_by_horizon",
    "signal_survived_costs_by_horizon",
    "outcome_resolved_at",
    "outcome",
}

REQUIRED_EXPERIENCE_FIELDS = ("record_id", "symbol", "as_of", "source")
RECOMMENDED_EXPERIENCE_FIELDS = (
    "advisory_action",
    "base_predictions",
    "validation_status",
    "regime_label",
    "risk_summary",
)
INPUT_CONTEXT_FIELDS = (
    "base_predictions",
    "model_confidence",
    "risk_summary",
    "calibration_state",
    "market_context",
    "news_context",
    "event_context",
    "macro_context",
    "economic_edge",
    "model_edge",
    "autonomy_posture",
)


def _read_records_with_malformed_count(path: str | os.PathLike[str]) -> tuple[list[dict], int]:
    if not os.path.exists(path):
        return [], 0
    records = []
    malformed = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if isinstance(item, dict):
                records.append(item)
            else:
                malformed += 1
    return records, malformed


def _contains_outcome_keys(value: Any) -> bool:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            if key_text in OUTCOME_KEYS or key_text.startswith("future_") or "outcome" in key_text:
                return True
            if _contains_outcome_keys(item):
                return True
    elif isinstance(value, list):
        return any(_contains_outcome_keys(item) for item in value)
    return False


def validate_experience_record(record: dict | None) -> dict:
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(record, dict):
        return {"status": "invalid", "errors": ["record is not an object"], "warnings": []}
    if str(record.get("record_type") or "experience") != "experience":
        errors.append("record_type is not experience")
    for field in REQUIRED_EXPERIENCE_FIELDS:
        if not record.get(field):
            errors.append(f"missing required field: {field}")
    for field in RECOMMENDED_EXPERIENCE_FIELDS:
        if not record.get(field):
            warnings.append(f"missing recommended field: {field}")
    if record.get("average_prediction") is None and not (record.get("model_edge") or {}).get("raw_edge_bps"):
        warnings.append("missing recommended field: average_prediction_or_model_edge")
    if not any(
        key in record
        for key in (
            "future_return_by_horizon",
            "direction_correct_by_horizon",
            "net_after_costs_by_horizon",
            "best_base_model_by_horizon",
            "signal_survived_costs_by_horizon",
        )
    ):
        warnings.append("missing outcome placeholder fields")
    input_payload = {field: record.get(field) for field in INPUT_CONTEXT_FIELDS}
    if _contains_outcome_keys(input_payload):
        errors.append("future outcome field appears inside input/context fields")
    status = "invalid" if errors else "warning" if warnings else "valid"
    return {"status": status, "errors": errors, "warnings": warnings}


def ledger_quality_summary(path: str | os.PathLike[str] = DEFAULT_LEDGER_PATH) -> dict:
    records, malformed = _read_records_with_malformed_count(path)
    experiences = [r for r in records if str(r.get("record_type") or "experience") == "experience"]
    outcomes = [r for r in records if str(r.get("record_type") or "") == "outcome"]
    outcome_parent_ids = {str(r.get("parent_record_id")) for r in outcomes if r.get("parent_record_id")}
    ids = [str(r.get("record_id") or "") for r in records if r.get("record_id")]
    duplicate_ids = sorted([key for key, count in Counter(ids).items() if count > 1])
    existing_experience_ids = {str(r.get("record_id")) for r in experiences if r.get("record_id")}
    broken_outcome_links = [
        str(r.get("parent_record_id") or "")
        for r in outcomes
        if r.get("parent_record_id") and str(r.get("parent_record_id")) not in existing_experience_ids
    ]
    missing_required = [
        str(r.get("record_id") or f"row_{idx}")
        for idx, r in enumerate(experiences)
        if any(not r.get(field) for field in REQUIRED_EXPERIENCE_FIELDS)
    ]
    resolved = [
        r
        for r in experiences
        if r.get("outcome_resolved_at")
        or bool(r.get("future_return_by_horizon"))
        or str(r.get("record_id")) in outcome_parent_ids
    ]
    as_of_values = sorted([str(r.get("as_of")) for r in experiences if r.get("as_of")])
    future_leak_records = []
    for r in experiences:
        input_payload = {field: r.get(field) for field in INPUT_CONTEXT_FIELDS}
        if _contains_outcome_keys(input_payload):
            future_leak_records.append(str(r.get("record_id") or "unknown"))
    base_model_names = sorted(
        {
            str(name)
            for r in experiences
            for name in dict(r.get("base_predictions") or {}).keys()
            if name
        }
    )
    def _count_missing(field: str) -> int:
        return int(sum(1 for r in experiences if not r.get(field)))
    validations = [validate_experience_record(r) for r in experiences]
    validation_statuses = Counter(str(v.get("status") or "invalid") for v in validations)
    warning_counts = Counter(
        warning
        for validation in validations
        for warning in list(validation.get("warnings") or [])
    )
    error_counts = Counter(
        error
        for validation in validations
        for error in list(validation.get("errors") or [])
    )

    return {
        "path": str(path),
        "exists": bool(os.path.exists(path)),
        "file_size_bytes": int(os.path.getsize(path)) if os.path.exists(path) else 0,
        "total_records": len(records),
        "experience_records": len(experiences),
        "outcome_records": len(outcomes),
        "malformed_records": int(malformed),
        "missing_required_identity_count": len(missing_required),
        "missing_required_identity_record_ids": missing_required[:20],
        "missing_base_predictions_count": _count_missing("base_predictions"),
        "missing_advisory_action_count": _count_missing("advisory_action"),
        "missing_validation_status_count": _count_missing("validation_status"),
        "missing_regime_count": _count_missing("regime_label"),
        "missing_risk_context_count": int(sum(1 for r in experiences if not r.get("risk_summary"))),
        "resolved_outcome_count": len(resolved),
        "unresolved_outcome_count": max(0, len(experiences) - len(resolved)),
        "symbols": sorted({str(r.get("symbol") or "").upper() for r in experiences if r.get("symbol")}),
        "horizons": sorted(
            {
                str(h)
                for r in experiences
                for h in list((r.get("future_return_by_horizon") or {}).keys())
                + ([r.get("horizon")] if r.get("horizon") else [])
                if h
            }
        ),
        "date_start": as_of_values[0] if as_of_values else None,
        "date_end": as_of_values[-1] if as_of_values else None,
        "source_distribution": dict(Counter(str(r.get("source") or "unknown") for r in experiences)),
        "advisory_action_distribution": dict(Counter(str(r.get("advisory_action") or "missing") for r in experiences)),
        "regime_label_distribution": dict(Counter(str(r.get("regime_label") or "missing") for r in experiences)),
        "validation_status_distribution": dict(Counter(str(r.get("validation_status") or "missing") for r in experiences)),
        "autonomy_posture_distribution": dict(
            Counter(str((r.get("autonomy_posture") or {}).get("stage") or (r.get("autonomy_posture") or {}).get("operator_state") or "missing") for r in experiences)
        ),
        "future_outcome_input_leak_count": len(future_leak_records),
        "future_outcome_input_leak_record_ids": future_leak_records[:20],
        "duplicate_record_id_count": len(duplicate_ids),
        "duplicate_record_ids": duplicate_ids[:20],
        "broken_outcome_link_count": len(broken_outcome_links),
        "broken_outcome_links": broken_outcome_links[:20],
        "base_model_names": base_model_names,
        "future_return_1_step_count": int(sum(1 for r in experiences if "1_step" in dict(r.get("future_return_by_horizon") or {}))),
        "direction_correct_1_step_count": int(sum(1 for r in experiences if "1_step" in dict(r.get("direction_correct_by_horizon") or {}))),
        "net_after_costs_1_step_count": int(sum(1 for r in experiences if "1_step" in dict(r.get("net_after_costs_by_horizon") or {}))),
        "best_base_model_1_step_count": int(sum(1 for r in experiences if "1_step" in dict(r.get("best_base_model_by_horizon") or {}))),
        "schema_valid_record_count": int(validation_statuses.get("valid", 0)),
        "schema_warning_record_count": int(validation_statuses.get("warning", 0)),
        "schema_invalid_record_count": int(validation_statuses.get("invalid", 0) + malformed),
        "top_schema_warnings": dict(warning_counts.most_common(10)),
        "top_schema_errors": dict(error_counts.most_common(10)),
    }


def meta_model_readiness_verdict(summary: dict) -> str:
    summary = dict(summary or {})
    total = int(summary.get("experience_records") or 0)
    resolved = int(summary.get("resolved_outcome_count") or 0)
    invalid = int(summary.get("schema_invalid_record_count") or 0)
    if total <= 0 or resolved <= 0 or invalid > 0:
        return "not_ready"
    if total < 100:
        return "inspection_ready"
    resolved_share = float(resolved / max(1, total))
    validation_dist = dict(summary.get("validation_status_distribution") or {})
    all_validation_missing = bool(validation_dist) and set(validation_dist.keys()) <= {"missing"}
    regime_diversity = len([k for k in dict(summary.get("regime_label_distribution") or {}).keys() if k and k != "missing"])
    action_diversity = len([k for k in dict(summary.get("advisory_action_distribution") or {}).keys() if k and k != "missing"])
    if (
        total < 5000
        or int(summary.get("missing_base_predictions_count") or 0) > 0
        or int(summary.get("missing_validation_status_count") or 0) > 0
        or int(summary.get("future_outcome_input_leak_count") or 0) > 0
        or int(summary.get("duplicate_record_id_count") or 0) > 0
    ):
        return "research_ready_small_sample"
    if total < 10000 or resolved_share < 0.95:
        return "research_ready_large_sample"
    if all_validation_missing or regime_diversity < 2 or action_diversity < 3:
        return "research_ready_large_sample"
    return "training_ready"


def format_quality_report(summary: dict, generated_by: str = "manual replay command", sample: bool = True) -> str:
    summary = dict(summary or {})
    verdict = meta_model_readiness_verdict(summary)
    missing_fields = [
        ("base predictions", summary.get("missing_base_predictions_count")),
        ("advisory action", summary.get("missing_advisory_action_count")),
        ("validation status", summary.get("missing_validation_status_count")),
        ("regime label", summary.get("missing_regime_count")),
        ("risk context", summary.get("missing_risk_context_count")),
    ]
    return "\n".join(
        [
            "# Model Experience Ledger Quality Report: QQQ Sample",
            "",
            "## 1. Executive Summary",
            "",
            (
                "The sample ledger is useful for inspecting the experience-record contract, but it is not enough to train a meta-gating model."
                if verdict != "training_ready"
                else "The ledger passes basic quality checks for training readiness, subject to separate validation and leakage review."
            ),
            "",
            "## 2. Ledger File",
            "",
            f"- Path: `{summary.get('path')}`",
            f"- Record count: {int(summary.get('total_records') or 0)}",
            f"- File size: {int(summary.get('file_size_bytes') or 0)} bytes",
            f"- Generated by: `{generated_by}`",
            f"- Ledger type: {'sample' if sample else 'production'}",
            f"- Schema valid / warning / invalid: {int(summary.get('schema_valid_record_count') or 0)} / {int(summary.get('schema_warning_record_count') or 0)} / {int(summary.get('schema_invalid_record_count') or 0)}",
            "",
            "## 3. Coverage",
            "",
            f"- Symbols: {', '.join(summary.get('symbols') or []) or '-'}",
            f"- Date range: {summary.get('date_start') or '-'} to {summary.get('date_end') or '-'}",
            f"- Horizons: {', '.join(summary.get('horizons') or []) or '-'}",
            f"- Sources: {json.dumps(summary.get('source_distribution') or {}, sort_keys=True)}",
            f"- Advisory actions: {json.dumps(summary.get('advisory_action_distribution') or {}, sort_keys=True)}",
            f"- Regime labels: {json.dumps(summary.get('regime_label_distribution') or {}, sort_keys=True)}",
            f"- Validation statuses: {json.dumps(summary.get('validation_status_distribution') or {}, sort_keys=True)}",
            f"- Autonomy postures: {json.dumps(summary.get('autonomy_posture_distribution') or {}, sort_keys=True)}",
            f"- Base model names observed: {', '.join(summary.get('base_model_names') or []) or '-'}",
            "",
            "## 4. Outcome Resolution",
            "",
            f"- Resolved experience records: {int(summary.get('resolved_outcome_count') or 0)}",
            f"- Unresolved experience records: {int(summary.get('unresolved_outcome_count') or 0)}",
            f"- 1-step future return coverage: {int(summary.get('future_return_1_step_count') or 0)}",
            f"- Direction correctness coverage: {int(summary.get('direction_correct_1_step_count') or 0)}",
            f"- Net-after-costs coverage: {int(summary.get('net_after_costs_1_step_count') or 0)}",
            f"- Best-base-model coverage: {int(summary.get('best_base_model_1_step_count') or 0)}",
            "",
            "## 5. Missing Field Analysis",
            "",
            *[f"- Missing {name}: {int(value or 0)}" for name, value in missing_fields],
            f"- Malformed JSONL rows: {int(summary.get('malformed_records') or 0)}",
            f"- Missing required identity fields: {int(summary.get('missing_required_identity_count') or 0)}",
            f"- Top schema warnings: {json.dumps(summary.get('top_schema_warnings') or {}, sort_keys=True)}",
            f"- Top schema errors: {json.dumps(summary.get('top_schema_errors') or {}, sort_keys=True)}",
            "",
            "Important current limitations: validation status may still be missing or static, calibration is not fully reconstructed as-of, macro/breadth context may be placeholder-only, and paper/live execution outcomes are absent.",
            "",
            "## 6. Leakage / Lookahead Checks",
            "",
            f"- Future outcome fields found inside input/context fields: {int(summary.get('future_outcome_input_leak_count') or 0)}",
            f"- Duplicate record IDs: {int(summary.get('duplicate_record_id_count') or 0)}",
            f"- Broken linked outcome records: {int(summary.get('broken_outcome_link_count') or 0)}",
            "",
            "Outcomes are resolved from the next replay row after the decision timestamp. They are stored in outcome fields, not in input/context fields.",
            "",
            "## 7. Meta-Model Readiness Verdict",
            "",
            verdict,
            "",
            f"- Suitable for inspection: {verdict in {'inspection_ready', 'research_ready_small_sample', 'research_ready_large_sample', 'training_ready'}}",
            f"- Suitable for small research: {verdict in {'research_ready_small_sample', 'research_ready_large_sample', 'training_ready'}}",
            f"- Suitable for training: {verdict == 'training_ready'}",
            "",
            "## 8. Recommended Next Steps",
            "",
            "1. Keep this as a sample ledger and do not train from it yet.",
            "2. Generate a larger strict walk-forward ledger with stable validation and calibration evidence.",
            "3. Add schema enforcement before training any future contextual expert router.",
            "4. Keep advisory explanation separate from readiness/risk/autonomy permissions.",
            "",
        ]
    )


def format_status_report(summary: dict) -> str:
    summary = dict(summary or {})
    return "\n".join(
        [
            "# Model Experience Ledger Status",
            "",
            "| Field | Value |",
            "| --- | --- |",
            f"| Ledger path | `{summary.get('path')}` |",
            f"| Exists | {bool(summary.get('exists'))} |",
            f"| Total records | {int(summary.get('records') or 0)} |",
            f"| Experience records | {int(summary.get('experience_records') or 0)} |",
            f"| Outcome records | {int(summary.get('outcome_records') or 0)} |",
            f"| Resolved experience records | {int(summary.get('resolved_experience_records') or 0)} |",
            f"| Unresolved experience records | {int(summary.get('unresolved_experience_records') or 0)} |",
            f"| Symbols | {', '.join(summary.get('symbols') or []) or '-'} |",
            f"| Horizons | {', '.join(summary.get('horizons') or []) or '-'} |",
            "",
            "## Missing Fields Blocking Meta-Model Training",
            "",
            *[f"- {item}" for item in (summary.get("missing_fields") or ["none"])],
            "",
            "## Recommended Next Step",
            "",
            "Run replay/backtest with `experience_ledger_path` enabled, then require enough resolved records before training any meta-gating model.",
            "",
        ]
    )
