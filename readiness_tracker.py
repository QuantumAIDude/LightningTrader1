#!/usr/bin/env python3
"""
readiness_tracker.py

Tracks deployment readiness over time and computes pass streaks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    DATA_DIR,
    INSTRUMENTS,
    AUTONOMY_SUPPORTED_SYMBOLS,
    READINESS_MIN_PASS_STREAK_DAYS,
    READINESS_MAX_AGE_HOURS,
    READINESS_BACKTEST_SPLITS,
    READINESS_MIN_DIRECTIONAL_ACC,
    READINESS_MIN_NET_RETURN,
    READINESS_MIN_SHARPE_LIKE,
    READINESS_MIN_ACTIVE_FRACTION,
    READINESS_MIN_FOLDS,
    READINESS_REQUIRE_DIRECTIONAL,
    READINESS_REQUIRE_PROFITABILITY,
    READINESS_REQUIRE_SHARPE,
    READINESS_REQUIRE_ACTIVE,
    READINESS_USE_HORIZON_PRIORITY,
    READINESS_MIN_HORIZON_SCORE,
    READINESS_WEIGHT_WEEKEND,
    READINESS_WEIGHT_DAILY,
    READINESS_WEIGHT_INTRADAY,
    READINESS_INTRADAY_WINDOW_DOLLARS,
    READINESS_DAILY_WINDOW_DOLLARS,
    READINESS_WEEKEND_WINDOW_DOLLARS,
    READINESS_INTRADAY_WINDOW_PCT,
    READINESS_DAILY_WINDOW_PCT,
    READINESS_WEEKEND_WINDOW_PCT,
    READINESS_MIN_INTRADAY_WINDOW_HIT_RATE,
    READINESS_MIN_DAILY_WINDOW_HIT_RATE,
    READINESS_MIN_WEEKEND_WINDOW_HIT_RATE,
    READINESS_MAX_WEEKEND_ERR_PCT,
    READINESS_MAX_DAILY_ERR_PCT,
    READINESS_MAX_INTRADAY_ERR_BPS,
    READINESS_MIN_WEEKEND_SAMPLES,
    READINESS_MIN_DAILY_SAMPLES,
    READINESS_MIN_INTRADAY_SAMPLES,
    CALIBRATION_MAX_AGE_DAYS,
    CALIBRATION_EVIDENCE_MAX_AGE_DAYS,
    TS_SPLIT_GAP_BARS,
)
from data_fetch import update_price_data
from features import build_features_for_training, build_next_session_dataset, build_weekend_session_dataset, DEFAULT_FEATURES
from horizon_overlay import load_horizon_overlay
from model_trainers import walk_forward_validate
from model_utils import _models_dir_signature, load_models, predict_horizon
from risk_engine import classify_market_regime, risk_summary_for_symbol, risk_gate_guidance
from accuracy_windows import (
    majority_direction_vote,
    price_window_threshold_dollars,
    summarize_price_window_errors,
    min_window_hit_rate,
    window_ratio_score,
)


def _history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"readiness_history_{instrument_key}.jsonl")


def _state_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"readiness_state_{instrument_key}.json")


def _evidence_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"deployment_evidence_{instrument_key}.json")


def _validation_evidence_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"validation_evidence_{instrument_key}.json")


def _stage_history_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"deployment_stage_history_{instrument_key}.jsonl")


def _wf_cache_path(instrument_key: str) -> str:
    return os.path.join(DATA_DIR, f"walk_forward_cache_{instrument_key}.json")


def _safe_mean(values):
    try:
        vals = [float(v) for v in (values or []) if v is not None and np.isfinite(float(v))]
    except Exception:
        vals = []
    if not vals:
        return None
    return float(np.mean(vals))


def _safe_float(v):
    try:
        x = float(v)
    except Exception:
        return None
    if not np.isfinite(x):
        return None
    return float(x)


def _artifact_age_hours(ts_value) -> float | None:
    ts = pd.to_datetime(ts_value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    try:
        return float(max(0.0, (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0))
    except Exception:
        return None


def _load_price_df(instrument_key: str, online: bool = False) -> pd.DataFrame:
    if online:
        df = update_price_data(instrument_key)
    else:
        p = os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")
        if not os.path.exists(p):
            return pd.DataFrame()
        df = pd.read_csv(p, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.sort_index()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _load_json(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _save_json(path: str, payload: dict):
    tmp = f"{path}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass


def _file_signature(path: str):
    try:
        st = os.stat(path)
        return [int(st.st_mtime_ns), int(st.st_size)]
    except Exception:
        return None


def _append_jsonl(path: str, row: dict):
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, sort_keys=True))
            f.write("\n")
    except Exception:
        pass


def _count_jsonl_rows(path: str) -> int:
    if not path or not os.path.exists(path):
        return 0
    count = 0
    try:
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
    except Exception:
        return 0
    return int(count)


def _normalized_strings(values) -> list[str]:
    clean = []
    seen = set()
    for item in list(values or []):
        text = str(item or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return clean


def _join_labels(values) -> str:
    return ", ".join(_normalized_strings(values))


VALIDATION_STATUS_ORDER = {
    "missing": 0,
    "insufficient": 1,
    "stale": 1,
    "failed": 1,
    "passed_advisory_only": 2,
    "passed_shadow_only": 3,
    "passed_paper_candidate": 4,
    "passed_autonomy_candidate": 5,
}
VALIDATION_METHODS = {
    "none",
    "simple_holdout",
    "walk_forward",
    "purged_walk_forward",
    "paper_trading_ledger",
    "replay_backtest",
}
VALIDATION_BLOCKERS = {
    "insufficient_sample_count",
    "stale_validation",
    "no_strict_oos",
    "no_cost_model",
    "negative_net_edge",
    "excessive_drawdown",
    "drift_detected",
    "no_paper_trading_record",
    "missing_holdout",
    "unknown",
}
VALIDATION_EVIDENCE_MAX_AGE_DAYS = 30.0


def _validation_status_label(status: str) -> str:
    return {
        "missing": "Validation missing",
        "insufficient": "Validation insufficient",
        "stale": "Validation stale",
        "failed": "Validation failed",
        "passed_advisory_only": "Validation supports advisory only",
        "passed_shadow_only": "Validation supports shadow only",
        "passed_paper_candidate": "Validation supports paper candidate",
        "passed_autonomy_candidate": "Validation supports autonomy candidate",
    }.get(str(status or "missing"), "Validation missing")


def _empty_validation_window() -> dict:
    return {
        "train_start": None,
        "train_end": None,
        "validation_start": None,
        "validation_end": None,
        "holdout_start": None,
        "holdout_end": None,
    }


def _empty_leakage_controls() -> dict:
    return {
        "uses_strict_oos": False,
        "uses_purging": False,
        "uses_embargo": False,
        "uses_untouched_holdout": False,
        "notes": "",
    }


def _empty_economic_metrics() -> dict:
    return {
        "gross_return": None,
        "net_return_after_costs": None,
        "estimated_spread_cost": None,
        "estimated_slippage_cost": None,
        "transaction_cost_assumption": "",
        "trade_count": None,
        "average_trade_return": None,
        "max_drawdown": None,
        "win_rate": None,
        "profit_factor": None,
    }


def _empty_calibration_metrics() -> dict:
    return {
        "prediction_error": None,
        "directional_accuracy": None,
        "confidence_calibration_error": None,
        "sample_count": None,
    }


def _empty_drift_metrics() -> dict:
    return {
        "feature_drift_score": None,
        "prediction_drift_score": None,
        "realized_error_drift": None,
        "regime_shift_flag": None,
    }


def _validation_next_action(blockers: list[str], status: str, method: str) -> str:
    blocker_set = {str(item or "") for item in blockers}
    if "stale_validation" in blocker_set or status == "stale":
        return "refresh stale validation evidence"
    if "insufficient_sample_count" in blocker_set:
        return "collect more bars"
    if "no_strict_oos" in blocker_set:
        return "run walk-forward validation"
    if "missing_holdout" in blocker_set:
        return "run walk-forward validation with an untouched holdout"
    if "no_cost_model" in blocker_set:
        return "run replay backtest with transaction costs"
    if "negative_net_edge" in blocker_set:
        return "do not advance autonomy"
    if "excessive_drawdown" in blocker_set:
        return "do not advance autonomy"
    if "drift_detected" in blocker_set:
        return "refresh drift diagnostics before deployment"
    if "no_paper_trading_record" in blocker_set:
        return "start shadow-mode paper ledger"
    if status in {"missing", "insufficient", "failed"} or method == "none":
        return "run walk-forward validation"
    if status == "passed_advisory_only":
        return "start shadow-mode paper ledger"
    if status == "passed_shadow_only":
        return "continue paper ledger before paper autonomy"
    return "do not advance autonomy without all deployment gates"


def _validation_summary_text(evidence: dict) -> str:
    status = str(evidence.get("validation_status") or "missing")
    method = str(evidence.get("validation_method") or "none")
    econ = dict(evidence.get("economic_metrics") or {})
    cal = dict(evidence.get("calibration_metrics") or {})
    parts = [_validation_status_label(status), method.replace("_", " ")]
    trade_count = _safe_float(econ.get("trade_count"))
    net = _safe_float(econ.get("net_return_after_costs"))
    gross = _safe_float(econ.get("gross_return"))
    sample_count = _safe_float(cal.get("sample_count"))
    if trade_count is not None:
        parts.append(f"trades {int(trade_count)}")
    if sample_count is not None:
        parts.append(f"samples {int(sample_count)}")
    if gross is not None or net is not None:
        parts.append(f"gross {gross if gross is not None else '-'} / net {net if net is not None else '-'}")
    blockers = list(evidence.get("blockers") or [])
    if blockers:
        parts.append(f"blockers {_join_labels(blockers[:3])}")
    return " | ".join(str(part) for part in parts if str(part).strip())


def _coerce_validation_evidence(
    instrument_key: str,
    raw: dict | None,
    *,
    calibration_summary: dict | None = None,
) -> dict:
    sym = str(instrument_key or "").upper()
    payload = dict(raw or {})
    artifact_available = bool(payload.get("artifact_available")) if "artifact_available" in payload else bool(payload)
    method = str(payload.get("validation_method") or payload.get("method") or "none").strip().lower()
    if method not in VALIDATION_METHODS:
        method = "none"
    status = str(payload.get("validation_status") or payload.get("status") or "").strip().lower()
    if status not in VALIDATION_STATUS_ORDER:
        status = "missing" if not payload else "insufficient"

    generated_at = payload.get("generated_at") or payload.get("updated_at") or payload.get("created_at")
    validation_age_hours = _safe_float(payload.get("validation_age_hours") or payload.get("age_hours"))
    if validation_age_hours is None:
        validation_age_hours = _artifact_age_hours(generated_at)
    stale = bool(payload.get("stale") or payload.get("validation_stale"))
    if validation_age_hours is not None and validation_age_hours > 24.0 * float(VALIDATION_EVIDENCE_MAX_AGE_DAYS):
        stale = True
    if stale and status not in {"missing", "failed"}:
        status = "stale"

    data_window = {**_empty_validation_window(), **dict(payload.get("data_window") or {})}
    leakage = {**_empty_leakage_controls(), **dict(payload.get("leakage_controls") or {})}
    econ = {**_empty_economic_metrics(), **dict(payload.get("economic_metrics") or {})}
    cal = {**_empty_calibration_metrics(), **dict(payload.get("calibration_metrics") or {})}
    drift = {**_empty_drift_metrics(), **dict(payload.get("drift_metrics") or {})}

    calibration = dict(calibration_summary or {})
    if drift.get("realized_error_drift") is None and calibration:
        drift["realized_error_drift"] = bool(
            calibration.get("drift_flag")
            or calibration.get("recent_drift_flag")
            or calibration.get("recent_price_drift_flag")
        )
    if drift.get("prediction_drift_score") is None and calibration.get("drift_score") is not None:
        drift["prediction_drift_score"] = _safe_float(calibration.get("drift_score"))

    blockers = _normalized_strings(payload.get("blockers") or [])
    if status == "missing":
        blockers.extend(["missing_holdout", "no_strict_oos", "no_cost_model"])
    elif status in {"insufficient", "failed"}:
        blockers.append("insufficient_sample_count" if status == "insufficient" else "unknown")
    if status == "stale":
        blockers.append("stale_validation")
    if not bool(leakage.get("uses_strict_oos")):
        blockers.append("no_strict_oos")
    if not bool(leakage.get("uses_untouched_holdout")) and status in {
        "passed_paper_candidate",
        "passed_autonomy_candidate",
    }:
        blockers.append("missing_holdout")
    tx_assumption = str(econ.get("transaction_cost_assumption") or "").strip()
    has_cost_model = bool(
        tx_assumption
        or econ.get("estimated_spread_cost") is not None
        or econ.get("estimated_slippage_cost") is not None
    )
    if not has_cost_model:
        blockers.append("no_cost_model")
    gross = _safe_float(econ.get("gross_return"))
    net = _safe_float(econ.get("net_return_after_costs"))
    if gross is not None and gross > 0.0 and net is not None and net <= 0.0:
        blockers.append("negative_net_edge")
    max_dd = _safe_float(econ.get("max_drawdown"))
    if max_dd is not None and max_dd < -0.20:
        blockers.append("excessive_drawdown")
    if bool(drift.get("regime_shift_flag")) or bool(drift.get("realized_error_drift")):
        blockers.append("drift_detected")
    sample_count = _safe_float(cal.get("sample_count"))
    trade_count = _safe_float(econ.get("trade_count"))
    if sample_count is not None and sample_count < 30:
        blockers.append("insufficient_sample_count")
    if trade_count is not None and trade_count < 5:
        blockers.append("insufficient_sample_count")
    blockers = [item for item in _normalized_strings(blockers) if item in VALIDATION_BLOCKERS]

    supports_advisory = VALIDATION_STATUS_ORDER.get(status, 0) >= VALIDATION_STATUS_ORDER["passed_advisory_only"]
    supports_shadow = VALIDATION_STATUS_ORDER.get(status, 0) >= VALIDATION_STATUS_ORDER["passed_shadow_only"]
    supports_paper = VALIDATION_STATUS_ORDER.get(status, 0) >= VALIDATION_STATUS_ORDER["passed_paper_candidate"]
    supports_autonomy = VALIDATION_STATUS_ORDER.get(status, 0) >= VALIDATION_STATUS_ORDER["passed_autonomy_candidate"]
    blocks_autonomy = bool(
        blockers
        or status in {"missing", "insufficient", "stale", "failed", "passed_advisory_only", "passed_shadow_only"}
    )
    autonomy_blockers = list(blockers)
    if status in {"passed_advisory_only", "passed_shadow_only"} and "no_paper_trading_record" not in autonomy_blockers:
        autonomy_blockers.append("no_paper_trading_record")
    autonomy_blockers = _normalized_strings(autonomy_blockers)
    next_action = str(payload.get("next_best_action") or "").strip() or _validation_next_action(autonomy_blockers or blockers, status, method)
    out = {
        "instrument": sym,
        "generated_at": generated_at,
        "validation_age_hours": round(float(validation_age_hours), 2) if validation_age_hours is not None else None,
        "validation_status": status,
        "validation_status_label": _validation_status_label(status),
        "validation_method": method,
        "data_window": data_window,
        "leakage_controls": leakage,
        "economic_metrics": econ,
        "calibration_metrics": cal,
        "drift_metrics": drift,
        "blockers": blockers,
        "autonomy_blockers": autonomy_blockers,
        "next_best_action": next_action,
        "stale": bool(stale),
        "artifact_available": bool(artifact_available),
        "supports_advisory": bool(supports_advisory),
        "supports_shadow": bool(supports_shadow and not blockers),
        "supports_paper_candidate": bool(supports_paper and not blockers),
        "supports_autonomy_candidate": bool(supports_autonomy and not blockers),
        "blocks_autonomy": bool(blocks_autonomy),
    }
    out["summary"] = str(payload.get("summary") or _validation_summary_text(out))
    out["leakage_controls_summary"] = (
        f"strict_oos={'yes' if leakage.get('uses_strict_oos') else 'no'}"
        f" | purging={'yes' if leakage.get('uses_purging') else 'no'}"
        f" | embargo={'yes' if leakage.get('uses_embargo') else 'no'}"
        f" | holdout={'yes' if leakage.get('uses_untouched_holdout') else 'no'}"
    )
    return out


def _validation_stage_cap(validation_evidence: dict | None) -> str:
    evidence = dict(validation_evidence or {})
    if bool(evidence.get("supports_autonomy_candidate")):
        return "controlled_real_world_candidate"
    if bool(evidence.get("supports_paper_candidate")):
        return "paper_autonomy_candidate"
    if bool(evidence.get("supports_shadow")):
        return "advisory_shadow_ready"
    if bool(evidence.get("supports_advisory")):
        return "advisory_only"
    return "advisory_only"


def _validation_blocker_messages(validation_evidence: dict | None) -> list[str]:
    evidence = dict(validation_evidence or {})
    status = str(evidence.get("validation_status") or "missing")
    blockers = set(str(item or "").strip() for item in list(evidence.get("blockers") or []))
    messages = []
    if not bool(evidence.get("artifact_available")):
        messages.append("validation evidence has not been captured")
    elif status == "stale" or bool(evidence.get("stale")):
        messages.append("validation evidence is stale")
    elif status == "insufficient":
        messages.append("validation sample count is insufficient")
    elif status == "failed":
        messages.append("validation evidence failed")
    blocker_labels = {
        "insufficient_sample_count": "validation sample count is insufficient",
        "stale_validation": "validation evidence is stale",
        "no_strict_oos": "validation evidence lacks strict out-of-sample controls",
        "no_cost_model": "validation evidence lacks explicit transaction-cost assumptions",
        "negative_net_edge": "validation gross edge is positive but net edge is negative after costs",
        "excessive_drawdown": "validation drawdown is excessive",
        "drift_detected": "validation or calibration drift is active",
        "no_paper_trading_record": "paper-trading evidence ledger is not available",
        "missing_holdout": "validation evidence lacks an untouched holdout",
        "unknown": "validation blocker is unknown",
    }
    for blocker in blockers:
        text = blocker_labels.get(blocker)
        if text:
            messages.append(text)
    return _normalized_strings(messages)


def _validation_evidence_from_sources(
    instrument_key: str,
    *,
    state: dict | None = None,
    onboarding_summary: dict | None = None,
    calibration_summary: dict | None = None,
) -> dict:
    payload = dict(state or {})
    onboarding = dict(onboarding_summary or {})
    artifacts = dict(onboarding.get("artifacts") or {})
    candidates = [
        payload.get("validation_evidence"),
        (payload.get("deployment_evidence") or {}).get("validation_evidence") if isinstance(payload.get("deployment_evidence"), dict) else None,
        artifacts.get("validation_evidence"),
        (artifacts.get("deployment_evidence") or {}).get("validation_evidence") if isinstance(artifacts.get("deployment_evidence"), dict) else None,
        load_validation_evidence(instrument_key),
    ]
    for candidate in candidates:
        if isinstance(candidate, dict) and candidate:
            return _coerce_validation_evidence(
                instrument_key,
                candidate,
                calibration_summary=calibration_summary,
            )
    return _coerce_validation_evidence(
        instrument_key,
        {},
        calibration_summary=calibration_summary,
    )


def load_validation_evidence(instrument_key: str) -> dict:
    return _load_json(_validation_evidence_path(str(instrument_key or "").upper()))


def save_validation_evidence(instrument_key: str, evidence: dict) -> dict:
    normalized = _coerce_validation_evidence(str(instrument_key or "").upper(), dict(evidence or {}))
    _save_json(_validation_evidence_path(str(instrument_key or "").upper()), normalized)
    return normalized


def _price_artifact_summary(instrument_key: str, system_report: dict | None = None) -> dict:
    sym = str(instrument_key or "").upper()
    path = os.path.join(DATA_DIR, f"prices_{sym}.csv")
    long_path = os.path.join(DATA_DIR, f"prices_{sym}_daily_long.csv")
    out = {
        "path": path,
        "daily_long_path": long_path,
        "available": False,
        "rows": 0,
        "last_ts": None,
        "age_hours": None,
        "stale": None,
        "stale_severity": None,
        "error": None,
        "daily_long_available": False,
        "daily_long_rows": 0,
        "daily_long_last_ts": None,
    }
    data = dict((system_report or {}).get("data") or {})
    if data:
        out["rows"] = int(data.get("rows") or 0)
        out["last_ts"] = data.get("last_ts")
        out["stale_severity"] = data.get("stale_severity")
        out["error"] = data.get("error")
        staleness = dict(data.get("staleness") or {})
        if staleness:
            out["stale"] = bool(staleness.get("is_stale"))
            age_sec = _safe_float(staleness.get("age_sec"))
            out["age_hours"] = round(age_sec / 3600.0, 2) if age_sec is not None else None
        out["available"] = bool(out["rows"] > 0 and not out["error"])
    if not os.path.exists(path):
        out["error"] = out.get("error") or "missing price file"
    else:
        try:
            df = pd.read_csv(path, usecols=[0], parse_dates=[0])
            rows = int(len(df))
            out["rows"] = rows
            if rows > 0:
                ts = pd.to_datetime(df.iloc[:, 0], utc=True, errors="coerce").max()
                if pd.notna(ts):
                    out["last_ts"] = ts.isoformat()
                    out["age_hours"] = round(float((pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0), 2)
            out["available"] = bool(rows > 0)
        except Exception:
            out["error"] = "failed reading price file"
    if os.path.exists(long_path):
        try:
            long_df = pd.read_csv(long_path, usecols=[0], parse_dates=[0])
            long_rows = int(len(long_df))
            out["daily_long_rows"] = long_rows
            out["daily_long_available"] = bool(long_rows > 0)
            if long_rows > 0:
                long_ts = pd.to_datetime(long_df.iloc[:, 0], utc=True, errors="coerce").max()
                if pd.notna(long_ts):
                    out["daily_long_last_ts"] = long_ts.isoformat()
        except Exception:
            pass
    return out


def _model_artifact_summary(instrument_key: str) -> dict:
    sym = str(instrument_key or "").upper()
    try:
        payload = load_models(include_modes=True, instrument_key=sym) or {}
    except Exception:
        payload = {}
    core_keys = ("base", "forest", "boost", "extra_trees")
    core_present = sorted([key for key in core_keys if isinstance(payload.get(key), dict)])
    modes = dict(payload.get("modes") or {})
    mode_present = {}
    for mode in ("daily_high", "daily_low", "weekend"):
        group = dict(modes.get(mode) or {})
        mode_present[mode] = sorted([key for key, val in group.items() if isinstance(val, dict)])
    return {
        "has_any_models": bool(core_present or any(mode_present.values()) or isinstance(payload.get("meta"), dict) or payload.get("regime") is not None),
        "core_4min_models": core_present,
        "core_4min_ready": bool(len(core_present) == len(core_keys)),
        "meta_present": bool(isinstance(payload.get("meta"), dict)),
        "regime_present": bool(payload.get("regime") is not None),
        "horizon_models": mode_present,
        "horizon_ready": {mode: bool(items) for mode, items in mode_present.items()},
    }


def _news_artifact_summary(instrument_key: str, news_health: dict | None = None) -> dict:
    sym = str(instrument_key or "").upper()
    health = dict(news_health or {})
    path = os.path.join(DATA_DIR, f"news_{sym}.csv")
    out = {
        "path": path,
        "file_present": bool(os.path.exists(path)),
        "status": str(health.get("status") or "unknown"),
        "headline_state": str(health.get("headline_state") or health.get("status") or "unknown"),
        "history_state": str(health.get("history_state") or "unknown"),
        "rows": int(health.get("rows") or 0),
        "coverage_ratio": _safe_float(((health.get("coverage") or {}).get("coverage_ratio"))),
        "signal_ready": bool(health.get("signal_ready")),
        "last_published_at": health.get("last_published_at"),
        "age_hours": _safe_float(health.get("age_hours")),
    }
    if out["rows"] <= 0 and os.path.exists(path):
        try:
            df = pd.read_csv(path, usecols=[0])
            out["rows"] = int(len(df))
        except Exception:
            pass
    return out


def _history_summary(instrument_key: str) -> dict:
    rows = _read_history(instrument_key)
    unique_dates = sorted({str(row.get("date") or "").strip() for row in rows if str(row.get("date") or "").strip()})
    pass_rows = [row for row in rows if bool(row.get("passed"))]
    last_row = rows[-1] if rows else {}
    return {
        "entries": int(len(rows)),
        "unique_dates": int(len(unique_dates)),
        "pass_count": int(len(pass_rows)),
        "last_date": last_row.get("date"),
        "last_passed": bool(last_row.get("passed")) if rows else False,
    }


def _stage_history_summary(instrument_key: str, limit: int = 20) -> dict:
    path = _stage_history_path(instrument_key)
    rows = []
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                rows = [json.loads(line) for line in f if line.strip()]
        except Exception:
            rows = []
    recent = rows[-max(1, int(limit)) :]
    blocker_counts = {}
    stage_transitions = 0
    stable_stage_cycles = 0
    last_stage = None
    last_stage_change_ts = None
    for row in recent:
        stage_key = str(row.get("stage_key") or "").strip()
        blocker = str(row.get("primary_blocker") or "").strip()
        if blocker:
            blocker_counts[blocker] = blocker_counts.get(blocker, 0) + 1
        if last_stage is not None and stage_key and stage_key != last_stage:
            stage_transitions += 1
            last_stage_change_ts = row.get("ts")
        last_stage = stage_key or last_stage
    trailing_stage = str(recent[-1].get("stage_key") or "").strip() if recent else ""
    for row in reversed(recent):
        if str(row.get("stage_key") or "").strip() != trailing_stage:
            break
        stable_stage_cycles += 1
    top_blockers = sorted(blocker_counts.items(), key=lambda item: (-int(item[1]), item[0]))[:5]
    return {
        "entries": int(len(rows)),
        "recent_entries": int(len(recent)),
        "stage_transitions": int(stage_transitions),
        "stable_stage_cycles": int(stable_stage_cycles),
        "last_stage_change_ts": last_stage_change_ts,
        "top_primary_blockers": [{"blocker": blocker, "count": int(count)} for blocker, count in top_blockers],
    }


def _observation_summary(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"autonomy_audit_{str(instrument_key or '').upper()}.jsonl")
    out = {
        "path": path,
        "events": 0,
        "last_observed_at": None,
        "last_clean_observed_at": None,
        "shadow_entries": 0,
        "shadow_exits": 0,
        "shadow_observations": 0,
        "shadow_clean_observations": 0,
        "shadow_fresh_observations": 0,
        "shadow_allowed_observations": 0,
        "shadow_blocked_observations": 0,
        "actual_entries": 0,
        "actual_exits": 0,
        "kill_switch_events": 0,
        "filled_events": 0,
        "submitted_events": 0,
    }
    if not os.path.exists(path):
        return out
    try:
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                out["events"] += 1
                kind = str(row.get("kind") or "").strip().lower()
                if kind == "shadow_entry":
                    out["shadow_entries"] += 1
                elif kind == "shadow_exit":
                    out["shadow_exits"] += 1
                elif kind == "shadow_observation":
                    out["shadow_observations"] += 1
                    observed_at = row.get("observed_at") or row.get("logged_at")
                    if observed_at:
                        out["last_observed_at"] = observed_at
                    if bool(row.get("eligible_environment")):
                        out["shadow_fresh_observations"] += 1
                    if bool(row.get("clean_cycle")):
                        out["shadow_clean_observations"] += 1
                        out["last_clean_observed_at"] = observed_at or out["last_clean_observed_at"]
                    if bool(row.get("allowed")):
                        out["shadow_allowed_observations"] += 1
                    else:
                        out["shadow_blocked_observations"] += 1
                elif kind == "actual_entry":
                    out["actual_entries"] += 1
                elif kind == "actual_exit":
                    out["actual_exits"] += 1
                elif kind.startswith("kill_switch"):
                    out["kill_switch_events"] += 1
                result = dict(row.get("result") or {})
                status = str(result.get("status") or row.get("status") or "").strip().lower()
                if status == "filled":
                    out["filled_events"] += 1
                elif status == "submitted":
                    out["submitted_events"] += 1
    except Exception:
        return out
    return out


def _calibration_summary(instrument_key: str, system_report: dict | None = None) -> dict:
    sym = str(instrument_key or "").upper()
    path = os.path.join(DATA_DIR, f"calibration_live_{sym}.json")
    payload = _load_json(path)
    system_cal = dict((system_report or {}).get("calibration") or {})
    last_update = payload.get("last_update") or system_cal.get("last_update")
    last_check = payload.get("last_check") or system_cal.get("last_check")
    last_processed = payload.get("last_processed") or system_cal.get("last_processed")
    last_resolved_target_ts = payload.get("last_resolved_target_ts") or system_cal.get("last_resolved_target_ts")
    age_days = _safe_float(system_cal.get("age_days"))
    evidence_age_days = _safe_float(system_cal.get("evidence_age_days"))
    if evidence_age_days is None:
        anchor = pd.to_datetime(last_resolved_target_ts or last_processed, utc=True, errors="coerce")
        if pd.notna(anchor):
            try:
                evidence_age_days = float((pd.Timestamp.now(tz="UTC") - anchor).total_seconds() / 86400.0)
            except Exception:
                evidence_age_days = None
    recomputed_evidence_stale = bool(
        evidence_age_days is None or evidence_age_days > float(CALIBRATION_EVIDENCE_MAX_AGE_DAYS)
    )
    evidence_stale = bool(
        system_cal.get(
            "evidence_stale",
            payload.get("evidence_stale", recomputed_evidence_stale),
        )
    )
    if recomputed_evidence_stale:
        evidence_stale = True
    return {
        "path": path,
        "available": bool(payload),
        "last_update": last_update,
        "last_check": last_check,
        "last_processed": last_processed,
        "last_resolved_target_ts": last_resolved_target_ts,
        "age_days": age_days,
        "evidence_age_days": round(float(evidence_age_days), 2) if evidence_age_days is not None else None,
        "evidence_stale": bool(evidence_stale),
        "drift_flag": bool(system_cal.get("drift_flag", payload.get("drift_flag", False))),
        "recent_drift_flag": bool(system_cal.get("recent_drift_flag", payload.get("recent_drift_flag", False))),
        "recent_price_drift_flag": bool(system_cal.get("recent_price_drift_flag", payload.get("recent_price_drift_flag", False))),
        "drift_score": _safe_float(system_cal.get("drift_score", payload.get("drift_score"))),
        "recent_acc": _safe_float(system_cal.get("recent_acc", payload.get("recent_acc"))),
        "recent_total": int(system_cal.get("recent_total", payload.get("recent_total", 0)) or 0),
        "recent_price_window_hit_rate": _safe_float(system_cal.get("recent_price_window_hit_rate", payload.get("recent_price_window_hit_rate"))),
        "recent_price_window_total": int(system_cal.get("recent_price_window_total", payload.get("recent_price_window_total", 0)) or 0),
        "recent_recovery_streak": int(payload.get("recent_recovery_streak", 0) or 0),
        "recent_price_recovery_streak": int(payload.get("recent_price_recovery_streak", 0) or 0),
        "pending_unresolved_rows": int(system_cal.get("pending_unresolved_rows", payload.get("pending_unresolved_rows", 0)) or 0),
        "pending_unresolved_newest_prediction_ts": system_cal.get("pending_unresolved_newest_prediction_ts", payload.get("pending_unresolved_newest_prediction_ts")),
        "drift_components": dict(system_cal.get("drift_components") or payload.get("drift_components") or {}),
        "recovery_ready": bool(payload.get("recovery_ready", False)),
    }


def _categorize_blocker(text: str) -> str:
    low = str(text or "").strip().lower()
    if not low:
        return "other"
    if any(bit in low for bit in ("unknown instrument", "no price data", "missing models", "model artifacts", "readiness metrics are not available yet")):
        return "onboarding"
    if any(bit in low for bit in ("stale", "news pipeline is degraded", "news context is thin", "thin or degraded")):
        return "freshness"
    if any(bit in low for bit in ("horizon score", "window hit rate", "cost-adjusted edge", "finished direction", "too weak")):
        return "signal_quality"
    if any(bit in low for bit in ("readiness lock", "system check", "drift flag", "degenerate")):
        return "controls_validation"
    return "other"


def _next_stage_key(stage_key: str) -> str:
    order = [
        "supported_not_trained",
        "trained_not_validated",
        "advisory_only",
        "advisory_shadow_ready",
        "paper_autonomy_candidate",
        "controlled_real_world_candidate",
    ]
    current = str(stage_key or "supported_not_trained").strip().lower()
    try:
        idx = order.index(current)
    except ValueError:
        idx = 0
    return order[min(idx + 1, len(order) - 1)]


def _build_deployment_checklist(
    instrument_key: str,
    stage_key: str,
    payload: dict,
    metrics: dict,
    thresholds: dict,
    system: dict,
    news: dict,
    blockers: list[str],
    monitor_flags: list[str],
) -> dict:
    artifact = dict(payload.get("artifact_summary") or {})
    history = dict(payload.get("history_summary") or {})
    observations = dict(payload.get("observation_summary") or {})
    calibration_summary = dict(payload.get("calibration_summary") or {})
    validation_evidence = dict(payload.get("validation_evidence") or {})
    freshness_summary = dict(payload.get("freshness_summary") or {})
    onboarding_summary = dict(payload.get("onboarding_summary") or {})
    onboarding_artifacts = dict(onboarding_summary.get("artifacts") or {})
    training_plan_state = _onboarding_training_plan_state(onboarding_summary)
    model_runtime_state = _onboarding_model_runtime_state(onboarding_summary)
    onboarding_flags = dict(onboarding_summary.get("completion_flags") or {})
    onboarding_detail_bits = [str(onboarding_summary.get("summary") or onboarding_summary.get("status_label") or "not started")]
    readiness_stage_label = str(onboarding_summary.get("readiness_stage_label") or "").strip()
    if readiness_stage_label:
        onboarding_detail_bits.append(readiness_stage_label)
    if onboarding_artifacts:
        onboarding_detail_bits.append(f"artifacts {len(onboarding_artifacts)}")
    if list(training_plan_state.get("eligible_modes") or []):
        onboarding_detail_bits.append(f"eligible {_join_labels(training_plan_state.get('eligible_modes'))}")
    if list(training_plan_state.get("deferred_modes") or []):
        onboarding_detail_bits.append(f"deferred {_join_labels(training_plan_state.get('deferred_modes'))}")
    if list(training_plan_state.get("blocked_modes") or []):
        onboarding_detail_bits.append(f"blocked {_join_labels(training_plan_state.get('blocked_modes'))}")
    if bool(training_plan_state.get("retained_from_prior_run")):
        onboarding_detail_bits.append("plan retained")
    if list(model_runtime_state.get("trained_modes") or []):
        onboarding_detail_bits.append(f"trained {_join_labels(model_runtime_state.get('trained_modes'))}")
    if list(model_runtime_state.get("deferred_modes") or []):
        onboarding_detail_bits.append(f"runtime deferred {_join_labels(model_runtime_state.get('deferred_modes'))}")
    if list(model_runtime_state.get("blocked_modes") or []):
        onboarding_detail_bits.append(f"runtime blocked {_join_labels(model_runtime_state.get('blocked_modes'))}")
    if list(model_runtime_state.get("failed_modes") or []):
        onboarding_detail_bits.append(f"runtime failed {_join_labels(model_runtime_state.get('failed_modes'))}")
    if bool(model_runtime_state.get("stale")):
        onboarding_detail_bits.append("runtime stale")
    if bool(model_runtime_state.get("retained_from_prior_run")):
        onboarding_detail_bits.append("runtime retained")
    if onboarding_flags:
        onboarding_detail_bits.append("honest" if bool(onboarding_flags.get("honest_completion")) else "incomplete")
    fresh = bool(payload.get("fresh"))
    live_ready = bool(payload.get("live_ready"))
    required_streak = int(payload.get("required_streak_days") or READINESS_MIN_PASS_STREAK_DAYS)
    pass_streak = int(payload.get("pass_streak_days") or 0)
    hscore = _safe_float(metrics.get("horizon_weighted_score"))
    intr_hit = _safe_float(metrics.get("intraday_window_hit_rate"))
    day_hit = _safe_float(metrics.get("daily_window_hit_rate"))
    week_hit = _safe_float(metrics.get("weekend_window_hit_rate"))
    min_hscore = float(thresholds.get("min_horizon_score") or READINESS_MIN_HORIZON_SCORE)
    min_intr_hit = float(thresholds.get("min_intraday_window_hit_rate") or READINESS_MIN_INTRADAY_WINDOW_HIT_RATE)
    min_day_hit = float(thresholds.get("min_daily_window_hit_rate") or READINESS_MIN_DAILY_WINDOW_HIT_RATE)
    min_week_hit = float(thresholds.get("min_weekend_window_hit_rate") or READINESS_MIN_WEEKEND_WINDOW_HIT_RATE)
    system_status = str(system.get("status") or "").strip().lower()
    news_status = str(news.get("headline_state") or news.get("status") or "").strip().lower()
    diag = dict((((system.get("predictions") or {}).get("4min") or {}).get("diagnostics") or {}))
    critical_stale = bool(diag.get("critical_stale"))
    readiness_locked = bool(diag.get("readiness_locked"))
    calibration = dict(system.get("calibration") or {})
    horizon_accuracy = dict(system.get("horizon_accuracy") or {})
    checklist = {
        "price_data": {
            "ok": bool((artifact.get("prices") or {}).get("available")),
            "detail": f"rows {(artifact.get('prices') or {}).get('rows') or 0} | age {(artifact.get('prices') or {}).get('age_hours')}",
        },
        "core_models": {
            "ok": bool((artifact.get("models") or {}).get("core_4min_ready")),
            "detail": f"core 4m {','.join((artifact.get('models') or {}).get('core_4min_models') or []) or '-'}",
        },
        "readiness_metrics": {
            "ok": bool(metrics),
            "detail": f"history {history.get('entries') or 0} entries | unique dates {history.get('unique_dates') or 0}",
        },
        "onboarding_pipeline": {
            "ok": str(onboarding_summary.get("status") or "not_started") not in {"failed"},
            "detail": " | ".join([item for item in onboarding_detail_bits if str(item).strip()]),
        },
        "system_ok": {
            "ok": system_status in {"", "ok"},
            "detail": system_status or "unknown",
        },
        "data_fresh": {
            "ok": not critical_stale,
            "detail": "critical stale guard clear" if not critical_stale else "critical stale guard active",
        },
        "news_ready": {
            "ok": news_status not in {"stale", "failed", "error", "missing"},
            "detail": news_status or "unknown",
        },
        "readiness_streak": {
            "ok": bool(live_ready and pass_streak >= required_streak),
            "detail": f"{pass_streak}/{required_streak} passing days",
        },
        "readiness_state_fresh": {
            "ok": fresh,
            "detail": "fresh" if fresh else "stale",
        },
        "calibration_evidence": {
            "ok": not bool(calibration_summary.get("evidence_stale")),
            "detail": "fresh" if not bool(calibration_summary.get("evidence_stale")) else f"stale ({calibration_summary.get('evidence_age_days')})",
        },
        "validation_evidence": {
            "ok": bool(validation_evidence.get("supports_shadow") or validation_evidence.get("supports_paper_candidate") or validation_evidence.get("supports_autonomy_candidate")),
            "detail": str(validation_evidence.get("summary") or validation_evidence.get("validation_status_label") or "validation missing"),
        },
        "calibration_ok": {
            "ok": not bool(calibration.get("drift_flag")),
            "detail": "drift clear" if not bool(calibration.get("drift_flag")) else "drift flagged",
        },
        "horizon_drift_ok": {
            "ok": not bool(horizon_accuracy.get("drift_flag")),
            "detail": "clear" if not bool(horizon_accuracy.get("drift_flag")) else "drift flagged",
        },
        "horizon_score": {
            "ok": bool(hscore is not None and hscore >= min_hscore),
            "detail": "-" if hscore is None else f"{hscore:.2f} / {min_hscore:.2f}",
        },
        "intraday_hit_rate": {
            "ok": bool(intr_hit is not None and intr_hit >= min_intr_hit),
            "detail": "-" if intr_hit is None else f"{intr_hit:.0%} / {min_intr_hit:.0%}",
        },
        "daily_hit_rate": {
            "ok": bool(day_hit is not None and day_hit >= min_day_hit),
            "detail": "-" if day_hit is None else f"{day_hit:.0%} / {min_day_hit:.0%}",
        },
        "weekend_hit_rate": {
            "ok": bool(week_hit is not None and week_hit >= min_week_hit),
            "detail": "-" if week_hit is None else f"{week_hit:.0%} / {min_week_hit:.0%}",
        },
        "shadow_observations": {
            "ok": (int(observations.get("shadow_clean_observations") or 0) >= 1) or (int(observations.get("shadow_entries") or 0) >= 1),
            "detail": (
                f"clean {int(observations.get('shadow_clean_observations') or 0)}"
                f" | obs {int(observations.get('shadow_observations') or 0)}"
                f" | entries {int(observations.get('shadow_entries') or 0)}"
            ),
        },
    }
    passed = sum(1 for item in checklist.values() if bool(item.get("ok")))
    checklist["summary"] = {"passed": int(passed), "total": int(len(checklist) - 1)}
    return checklist


def build_deployment_evidence(
    instrument_key: str,
    state: dict | None = None,
    system_report: dict | None = None,
    news_health: dict | None = None,
    deployment_summary: dict | None = None,
) -> dict:
    sym = str(instrument_key or "").upper()
    payload = dict(state or {})
    system = dict(system_report or payload.get("system_report") or payload.get("system_checks") or {})
    news = dict(news_health or payload.get("news_health") or {})
    metrics = dict((payload.get("evaluation") or {}).get("metrics") or payload.get("metrics") or {})
    thresholds = dict((payload.get("evaluation") or {}).get("thresholds") or payload.get("thresholds") or {})
    summary = dict(deployment_summary or {})
    artifact_summary = {
        "prices": _price_artifact_summary(sym, system_report=system),
        "models": _model_artifact_summary(sym),
        "news": _news_artifact_summary(sym, news_health=news),
    }
    history_summary = _history_summary(sym)
    stage_history_summary = _stage_history_summary(sym)
    observation_summary = _observation_summary(sym)
    calibration_summary = _calibration_summary(sym, system_report=system)
    onboarding_summary = _onboarding_summary(sym)
    validation_evidence = _validation_evidence_from_sources(
        sym,
        state=payload,
        onboarding_summary=onboarding_summary,
        calibration_summary=calibration_summary,
    )
    onboarding_artifacts = dict(onboarding_summary.get("artifacts") or {})
    training_plan_guidance = _training_plan_guidance(
        onboarding_summary,
        core_models_ready=bool((artifact_summary.get("models") or {}).get("core_4min_ready")),
    )
    model_runtime_guidance = _model_runtime_guidance(
        onboarding_summary,
        core_models_ready=bool((artifact_summary.get("models") or {}).get("core_4min_ready")),
    )
    runtime_regime_risk = _runtime_regime_risk_state(sym, state=payload, system_report=system, news_health=news)
    decision_state = _decision_quality_state(summary, runtime_regime_risk)
    economics_state = dict(runtime_regime_risk.get("economics") or {})
    risk_guidance = _risk_gate_guidance((runtime_regime_risk.get("risk") or {}))
    portfolio_context = dict(runtime_regime_risk.get("portfolio_context") or {})
    if not portfolio_context:
        risk_meta = dict(runtime_regime_risk.get("risk") or {})
        portfolio_context = {
            "available": bool(risk_meta.get("portfolio_context_available")),
            "required": bool(risk_meta.get("portfolio_context_required")),
            "detail": str(
                risk_meta.get("portfolio_context_detail")
                or "Portfolio snapshot is unavailable; portfolio-aware autonomy cannot be proven."
            ),
        }
    onboarding_flags = dict(onboarding_summary.get("completion_flags") or {})
    blockers = list(summary.get("blockers") or [])
    monitor_flags = list(summary.get("monitor_flags") or [])
    blocker_groups = {"onboarding": [], "freshness": [], "signal_quality": [], "controls_validation": [], "other": []}
    for item in blockers + monitor_flags:
        blocker_groups.setdefault(_categorize_blocker(item), []).append(str(item))
    freshness_summary = {
        "state_fresh": bool(payload.get("fresh")),
        "price_stale": bool((artifact_summary.get("prices") or {}).get("stale")),
        "price_stale_severity": (artifact_summary.get("prices") or {}).get("stale_severity"),
        "price_age_hours": _safe_float((artifact_summary.get("prices") or {}).get("age_hours")),
        "critical_stale": bool(summary.get("critical_stale")),
        "news_headline_state": str((artifact_summary.get("news") or {}).get("headline_state") or news.get("headline_state") or ""),
        "news_age_hours": _safe_float((artifact_summary.get("news") or {}).get("age_hours")),
        "calibration_evidence_stale": bool(calibration_summary.get("evidence_stale")),
        "calibration_evidence_age_days": _safe_float(calibration_summary.get("evidence_age_days")),
    }
    freshness_summary["inputs_fresh"] = bool(
        not freshness_summary["critical_stale"]
        and not freshness_summary["price_stale"]
        and freshness_summary["news_headline_state"] not in {"failed", "stale", "error", "missing"}
        and not freshness_summary["calibration_evidence_stale"]
    )
    checklist_payload = {
        "artifact_summary": artifact_summary,
        "history_summary": history_summary,
        "stage_history_summary": stage_history_summary,
        "observation_summary": observation_summary,
        "calibration_summary": calibration_summary,
        "validation_evidence": validation_evidence,
        "onboarding_summary": onboarding_summary,
        "onboarding_artifacts": onboarding_artifacts,
        "freshness_summary": freshness_summary,
        "fresh": bool(payload.get("fresh")),
        "live_ready": bool(payload.get("live_ready")),
        "pass_streak_days": int(payload.get("pass_streak_days") or 0),
        "required_streak_days": int(payload.get("required_streak_days") or READINESS_MIN_PASS_STREAK_DAYS),
    }
    checklist = _build_deployment_checklist(sym, str(summary.get("stage_key") or ""), checklist_payload, metrics, thresholds, system, news, blockers, monitor_flags)
    missing_evidence = []
    if not bool((artifact_summary.get("prices") or {}).get("available")):
        missing_evidence.append("canonical price data has not been staged")
    if not bool((artifact_summary.get("models") or {}).get("core_4min_ready")):
        missing_evidence.append(
            str(
                training_plan_guidance.get("missing_evidence")
                or model_runtime_guidance.get("missing_evidence")
                or "core 4-minute model artifacts are missing"
            )
        )
    runtime_missing = str(model_runtime_guidance.get("missing_evidence") or "").strip()
    if runtime_missing and runtime_missing not in missing_evidence:
        missing_evidence.append(runtime_missing)
    if not bool(metrics):
        missing_evidence.append("readiness metrics have not been generated")
    if str(onboarding_summary.get("status") or "") == "failed":
        missing_evidence.append(f"onboarding pipeline failed: {str(onboarding_summary.get('last_error') or onboarding_summary.get('summary') or 'unknown reason')}")
    elif str(onboarding_summary.get("status") or "") == "completed" and not bool(onboarding_flags.get("honest_completion")):
        missing_evidence.append("onboarding finished without all expected milestone artifacts")
    if int(history_summary.get("entries") or 0) < 1:
        missing_evidence.append("readiness history has not started yet")
    if bool((runtime_regime_risk.get("regime") or {}).get("unknown")):
        missing_evidence.append("current regime evidence is too thin to classify safely")
    if not bool(portfolio_context.get("available")):
        missing_evidence.append("portfolio snapshot is unavailable; portfolio-aware autonomy cannot be proven")
    shadow_obs = int(observation_summary.get("shadow_observations") or 0)
    shadow_clean = int(observation_summary.get("shadow_clean_observations") or 0)
    shadow_entries = int(observation_summary.get("shadow_entries") or 0)
    if shadow_obs < 1 and shadow_entries < 1:
        missing_evidence.append("no shadow observation cycles have been recorded yet")
    elif shadow_clean < 1 and shadow_entries < 1:
        missing_evidence.append("shadow observation cycles exist, but none were clean enough to support stage advancement")
    next_requirements = []
    prices = artifact_summary.get("prices") or {}
    models = artifact_summary.get("models") or {}
    if not bool(prices.get("available")):
        next_requirements.append("Stage canonical price history for this symbol and rerun readiness refresh.")
    if bool(prices.get("available")) and not bool(models.get("core_4min_ready")):
        next_requirements.append(
            str(
                training_plan_guidance.get("primary_next_requirement")
                or model_runtime_guidance.get("primary_next_requirement")
                or "Train the core 4-minute model set for this symbol before treating it as trained."
            )
        )
    if not bool(metrics):
        next_requirements.append("Generate a readiness evaluation so the symbol has actual deployment metrics.")
    if str(decision_state.get("key") or "") == "unknown_regime":
        next_requirements.append("Collect enough recent price action and stable context to classify the current regime safely.")
    if str(economics_state.get("key") or "") in {"cost_crushed", "uncertainty_crushed"}:
        next_requirements.append("Wait for a cleaner finished edge that stays positive after estimated costs and uncertainty.")
    if str(risk_guidance.get("next_requirement") or "").strip():
        next_requirements.append(str(risk_guidance.get("next_requirement") or "").strip())
    if not bool(portfolio_context.get("available")):
        next_requirements.append(
            "Refresh broker or portfolio state before treating this setup as autonomy-eligible; advisory review can continue separately."
        )
    if str(onboarding_summary.get("status") or "") == "failed":
        next_requirements.append(f"Resolve the onboarding failure and rerun symbol setup: {str(onboarding_summary.get('last_error') or onboarding_summary.get('summary') or 'unknown reason')}")
    elif str(onboarding_summary.get("status") or "") == "completed" and not bool(onboarding_flags.get("honest_completion")):
        next_requirements.append("Rebuild the missing onboarding milestone artifacts before treating this workspace as fully staged.")
    onboarding_next_requirement = str(onboarding_summary.get("primary_next_requirement") or "").strip()
    if onboarding_next_requirement:
        next_requirements.append(onboarding_next_requirement)
    elif str(training_plan_guidance.get("primary_next_requirement") or "").strip():
        next_requirements.append(str(training_plan_guidance.get("primary_next_requirement") or "").strip())
    elif str(model_runtime_guidance.get("primary_next_requirement") or "").strip():
        next_requirements.append(str(model_runtime_guidance.get("primary_next_requirement") or "").strip())
    if bool(summary.get("critical_stale")):
        next_requirements.append("Refresh market data until the critical stale-data guard clears.")
    if "news pipeline is degraded" in blockers:
        next_requirements.append("Refresh and extend the symbol news history until the headline state is no longer stale or missing.")
    if "calibration drift flag is active" in blockers:
        next_requirements.append("Rebuild live calibration and confirm the drift flag clears on the next system check.")
    if bool(calibration_summary.get("evidence_stale")):
        next_requirements.append("Resolve fresh intraday predictions so calibration evidence is current again.")
    if int(calibration_summary.get("pending_unresolved_rows") or 0) > 0:
        next_requirements.append(
            f"Advance price history enough to resolve {int(calibration_summary.get('pending_unresolved_rows') or 0)} pending calibration snapshot(s)."
        )
    if "daily/weekend drift flag is active" in blockers:
        next_requirements.append("Refresh horizon accuracy history until the daily/weekend drift flag clears.")
    validation_messages = _validation_blocker_messages(validation_evidence)
    missing_evidence.extend(validation_messages)
    for validation_message in validation_messages:
        blocker_groups.setdefault("controls_validation", []).append(validation_message)
    validation_next = str(validation_evidence.get("next_best_action") or "").strip()
    if validation_next:
        next_requirements.append(validation_next)
    if not bool(payload.get("live_ready")):
        needed = max(0, int(payload.get("required_streak_days") or READINESS_MIN_PASS_STREAK_DAYS) - int(payload.get("pass_streak_days") or 0))
        if needed > 0:
            next_requirements.append(f"Accumulate {needed} more passing readiness day(s) before the pass-streak gate can turn green.")
    if "horizon score is below the deployment floor" in " | ".join(blockers):
        next_requirements.append("Improve weighted horizon accuracy enough to clear the deployment floor.")
    if shadow_obs < 1 and shadow_entries < 1 and bool(summary.get("advisory_ready")):
        next_requirements.append("Record at least one shadow observation cycle before claiming shadow-readiness confidence.")
    elif shadow_clean < 1 and shadow_entries < 1 and bool(summary.get("advisory_ready")):
        next_requirements.append("Collect at least one clean shadow observation cycle under fresh, non-conflicted conditions.")
    next_stage_key = _next_stage_key(str(summary.get("stage_key") or "supported_not_trained"))
    deduped_requirements = []
    seen_requirements = set()
    for item in next_requirements:
        text = str(item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen_requirements:
            continue
        seen_requirements.add(key)
        deduped_requirements.append(text)
    prioritized_missing = _prioritize_requirements(missing_evidence)
    prioritized_requirements = _prioritize_requirements(deduped_requirements)
    primary_next_requirement = str(
        summary.get("primary_next_requirement")
        or training_plan_guidance.get("primary_next_requirement")
        or (prioritized_requirements[0] if prioritized_requirements else "")
        or model_runtime_guidance.get("primary_next_requirement")
    )
    next_requirement_category = str(summary.get("next_requirement_category") or _requirement_category(primary_next_requirement))
    return {
        "instrument": sym,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "stage_key": str(summary.get("stage_key") or "supported_not_trained"),
        "stage_label": str(summary.get("stage_label") or _deployment_stage_label("supported_not_trained")),
        "status_flags": dict(summary.get("status_flags") or {}),
        "symbol_maturity_key": str(summary.get("symbol_maturity_key") or ""),
        "symbol_maturity_label": str(summary.get("symbol_maturity_label") or ""),
        "symbol_maturity_detail": str(summary.get("symbol_maturity_detail") or ""),
        "evidence_state_key": str(summary.get("evidence_state_key") or ""),
        "evidence_state_label": str(summary.get("evidence_state_label") or ""),
        "evidence_state_detail": str(summary.get("evidence_state_detail") or ""),
        "operator_state_key": str(summary.get("operator_state_key") or ""),
        "operator_state_label": str(summary.get("operator_state_label") or ""),
        "operator_state_detail": str(summary.get("operator_state_detail") or ""),
        "next_stage_key": next_stage_key,
        "next_stage_label": _deployment_stage_label(next_stage_key),
        "artifact_summary": artifact_summary,
        "calibration_summary": calibration_summary,
        "validation_evidence": validation_evidence,
        "validation_status": str(validation_evidence.get("validation_status") or "missing"),
        "validation_status_label": str(validation_evidence.get("validation_status_label") or ""),
        "validation_method": str(validation_evidence.get("validation_method") or "none"),
        "validation_summary": str(validation_evidence.get("summary") or ""),
        "validation_blockers": list(validation_evidence.get("blockers") or []),
        "validation_next_best_action": str(validation_evidence.get("next_best_action") or ""),
        "freshness_summary": freshness_summary,
        "history_summary": history_summary,
        "stage_history_summary": stage_history_summary,
        "observation_summary": observation_summary,
        "onboarding_summary": onboarding_summary,
        "onboarding_artifacts": onboarding_artifacts,
        "model_runtime_guidance": model_runtime_guidance,
        "model_runtime_state_key": str(model_runtime_guidance.get("state_key") or ""),
        "model_runtime_state_label": str(model_runtime_guidance.get("state_label") or ""),
        "model_runtime_state_detail": str(model_runtime_guidance.get("detail") or ""),
        "regime_risk_summary": runtime_regime_risk,
        "portfolio_context": portfolio_context,
        "risk_guidance": risk_guidance,
        "decision_state_key": str(decision_state.get("key") or ""),
        "decision_state_label": str(decision_state.get("label") or ""),
        "decision_state_detail": str(decision_state.get("detail") or ""),
        "economics_state_key": str(economics_state.get("key") or ""),
        "economics_state_label": str(economics_state.get("label") or ""),
        "economics_state_detail": str(economics_state.get("detail") or ""),
        "blocker_groups": blocker_groups,
        "missing_evidence": prioritized_missing,
        "next_requirements": prioritized_requirements,
        "primary_next_requirement": primary_next_requirement,
        "next_requirement_category": next_requirement_category,
        "checklist": checklist,
        "primary_blocker": str(summary.get("primary_blocker") or ""),
        "summary": str(summary.get("summary") or ""),
    }


def load_deployment_evidence(instrument_key: str) -> dict:
    return _load_json(_evidence_path(instrument_key))


def _persist_deployment_evidence(instrument_key: str, state: dict):
    evidence = build_deployment_evidence(instrument_key, state=state, deployment_summary=state)
    _save_json(_evidence_path(instrument_key), evidence)
    history_row = {
        "ts": state.get("last_update") or datetime.now(timezone.utc).isoformat(),
        "instrument": str(instrument_key or "").upper(),
        "stage_key": state.get("stage_key"),
        "stage_label": state.get("stage_label"),
        "symbol_maturity_label": state.get("symbol_maturity_label"),
        "evidence_state_label": state.get("evidence_state_label"),
        "operator_state_label": state.get("operator_state_label"),
        "next_stage_key": evidence.get("next_stage_key"),
        "next_stage_label": evidence.get("next_stage_label"),
        "primary_blocker": state.get("primary_blocker"),
        "primary_next_requirement": state.get("primary_next_requirement"),
        "summary": state.get("summary"),
        "fresh": bool(state.get("fresh")),
        "live_ready": bool(state.get("live_ready")),
        "pass_streak_days": int(state.get("pass_streak_days") or 0),
        "required_streak_days": int(state.get("required_streak_days") or READINESS_MIN_PASS_STREAK_DAYS),
        "checklist_summary": dict((evidence.get("checklist") or {}).get("summary") or {}),
    }
    _append_jsonl(_stage_history_path(instrument_key), history_row)


def _walk_forward_signature(instrument_key: str, df: pd.DataFrame, n_splits: int) -> dict:
    root = os.path.dirname(__file__)
    try:
        rows = int(len(df))
        last_ts = pd.to_datetime(df.index.max(), utc=True, errors="coerce")
        first_ts = pd.to_datetime(df.index.min(), utc=True, errors="coerce")
    except Exception:
        rows = 0
        last_ts = pd.NaT
        first_ts = pd.NaT
    return {
        "instrument": instrument_key,
        "n_splits": int(n_splits),
        "prices": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")),
        "models": [list(x) for x in _models_dir_signature()],
        "rows": rows,
        "first_ts": first_ts.isoformat() if pd.notna(first_ts) else None,
        "last_ts": last_ts.isoformat() if pd.notna(last_ts) else None,
        "code": {
            "readiness_tracker": _file_signature(__file__),
            "model_trainers": _file_signature(os.path.join(root, "model_trainers.py")),
            "accuracy_windows": _file_signature(os.path.join(root, "accuracy_windows.py")),
            "forecast_calibration": _file_signature(os.path.join(root, "forecast_calibration.py")),
        },
    }


def _horizon_backfill_signature(instrument_key: str) -> dict:
    root = os.path.dirname(__file__)
    return {
        "models": [list(x) for x in _models_dir_signature()],
        "prices": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}.csv")),
        "prices_daily_long": _file_signature(os.path.join(DATA_DIR, f"prices_{instrument_key}_daily_long.csv")),
        "structural_snapshots": _file_signature(os.path.join(DATA_DIR, f"horizon_structural_snapshots_{instrument_key}.csv")),
        "structural_outcomes": _file_signature(os.path.join(DATA_DIR, f"horizon_structural_outcomes_{instrument_key}.csv")),
        "overlay": _file_signature(os.path.join(DATA_DIR, f"horizon_overlay_{instrument_key}.json")),
        "code": {
            "horizon_accuracy_tracker": _file_signature(os.path.join(root, "horizon_accuracy_tracker.py")),
            "model_utils": _file_signature(os.path.join(root, "model_utils.py")),
            "horizon_overlay": _file_signature(os.path.join(root, "horizon_overlay.py")),
        },
    }


def _ts_split(n_splits: int):
    gap = max(0, int(TS_SPLIT_GAP_BARS))
    try:
        return TimeSeriesSplit(n_splits=n_splits, gap=gap)
    except TypeError:
        return TimeSeriesSplit(n_splits=n_splits)


def _ridge_pipeline():
    return Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])


def _cv_abs_ret_metrics(dataset: pd.DataFrame, target_norm_col: str, target_ret_col: str,
                        horizon: str, instrument_key: str, min_rows: int = 80) -> dict:
    if dataset is None or dataset.empty or len(dataset) < max(6, int(min_rows)):
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    if target_norm_col not in dataset.columns or target_ret_col not in dataset.columns:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    feats = [f for f in DEFAULT_FEATURES if f in dataset.columns]
    if not feats:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    X = dataset[feats].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_norm = pd.to_numeric(dataset[target_norm_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_ret = pd.to_numeric(dataset[target_ret_col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    close_base = pd.to_numeric(dataset.get("close"), errors="coerce").replace([np.inf, -np.inf], np.nan)
    vol = pd.to_numeric(dataset.get("ewm_vol"), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.001).clip(lower=1e-6)
    n = len(X)
    scale = max(6, int(max(8, min_rows) // 2))
    n_splits = min(max(2, int(READINESS_BACKTEST_SPLITS)), max(2, n // scale))
    if n_splits < 2:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    tss = _ts_split(n_splits)
    maes_ret = []
    maes_dollars = []
    ratio_vals = []
    hit_vals = []
    threshold_vals = []
    for tr_idx, te_idx in tss.split(X):
        try:
            Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
            ytr = y_norm.iloc[tr_idx]
            yte_ret = y_ret.iloc[te_idx].values.astype(float)
            if len(Xtr) < max(6, int(min_rows // 4)) or len(Xte) < 2:
                continue
            model = _ridge_pipeline()
            model.fit(Xtr, ytr)
            pred_norm = np.asarray(model.predict(Xte), dtype=float)
            pred_ret = pred_norm * vol.iloc[te_idx].values.astype(float)
            err = np.abs(pred_ret - yte_ret)
            maes_ret.append(float(np.nanmean(err)))
            if "close" in dataset.columns:
                base_prices = close_base.iloc[te_idx].values.astype(float)
                actual_prices = base_prices * (1.0 + yte_ret)
                pred_prices = base_prices * (1.0 + pred_ret)
                window_metrics = summarize_price_window_errors(
                    np.abs(pred_prices - actual_prices),
                    actual_prices=actual_prices,
                    horizon=horizon,
                    instrument_key=instrument_key,
                )
                if window_metrics.get("mae_dollars") is not None:
                    maes_dollars.append(float(window_metrics["mae_dollars"]))
                if window_metrics.get("window_ratio_mean") is not None:
                    ratio_vals.append(float(window_metrics["window_ratio_mean"]))
                if window_metrics.get("window_hit_rate") is not None:
                    hit_vals.append(float(window_metrics["window_hit_rate"]))
                if window_metrics.get("window_threshold_mean_dollars") is not None:
                    threshold_vals.append(float(window_metrics["window_threshold_mean_dollars"]))
        except Exception:
            continue
    if not maes_ret:
        return {"folds": 0, "mae_ret": None, "mae_pct": None, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
    mae_ret = float(np.nanmean(maes_ret))
    return {
        "folds": int(len(maes_ret)),
        "mae_ret": mae_ret,
        "mae_pct": float(mae_ret * 100.0),
        "mae_dollars": float(np.nanmean(maes_dollars)) if maes_dollars else None,
        "window_ratio_mean": float(np.nanmean(ratio_vals)) if ratio_vals else None,
        "window_hit_rate": float(np.nanmean(hit_vals)) if hit_vals else None,
        "window_threshold_mean_dollars": float(np.nanmean(threshold_vals)) if threshold_vals else None,
    }


def _aggregate_horizon_payload(payload: dict) -> dict:
    by_h = payload.get("by_horizon", {}) if isinstance(payload, dict) else {}

    def _agg_h(h_name: str):
        block = by_h.get(h_name, {}) if isinstance(by_h, dict) else {}
        models = block.get("models", {}) if isinstance(block, dict) else {}
        preferred = []
        for name in ("consensus", "meta"):
            mv = models.get(name)
            if isinstance(mv, dict) and int(mv.get("n", 0) or 0) > 0:
                preferred.append(mv)
                break
        if not preferred:
            preferred = [v for v in models.values() if isinstance(v, dict) and int(v.get("n", 0) or 0) > 0]
        if not preferred:
            return {"n": 0, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
        if len(preferred) == 1:
            block_out = preferred[0]
            return {
                "n": int(block_out.get("n", 0) or 0),
                "mae_dollars": _safe_float(block_out.get("mae_dollars", block_out.get("mae"))),
                "window_ratio_mean": _safe_float(block_out.get("window_ratio_mean")),
                "window_hit_rate": _safe_float(block_out.get("window_hit_rate")),
            }
        maes = [_safe_float(v.get("mae_dollars", v.get("mae"))) for v in preferred]
        ratios = [_safe_float(v.get("window_ratio_mean")) for v in preferred]
        hits = [_safe_float(v.get("window_hit_rate")) for v in preferred]
        ns = [int(v.get("n", 0) or 0) for v in preferred]
        maes = [v for v in maes if v is not None]
        ratios = [v for v in ratios if v is not None]
        hits = [v for v in hits if v is not None]
        return {
            "n": int(max(ns) if ns else 0),
            "mae_dollars": float(np.nanmedian(maes)) if maes else None,
            "window_ratio_mean": float(np.nanmedian(ratios)) if ratios else None,
            "window_hit_rate": float(np.nanmedian(hits)) if hits else None,
        }

    week = _agg_h("weekend")
    dh = _agg_h("daily_high")
    dl = _agg_h("daily_low")
    daily_maes = [v for v in (dh.get("mae_dollars"), dl.get("mae_dollars")) if v is not None]
    daily_ratios = [v for v in (dh.get("window_ratio_mean"), dl.get("window_ratio_mean")) if v is not None]
    daily_hits = [v for v in (dh.get("window_hit_rate"), dl.get("window_hit_rate")) if v is not None]
    daily_n = min([n for n in (dh.get("n", 0), dl.get("n", 0)) if n > 0], default=max(int(dh.get("n", 0) or 0), int(dl.get("n", 0) or 0)))
    return {
        "weekend_mae_dollars": week.get("mae_dollars"),
        "weekend_window_ratio": week.get("window_ratio_mean"),
        "weekend_window_hit_rate": week.get("window_hit_rate"),
        "weekend_n": int(week.get("n", 0) or 0),
        "daily_mae_dollars": float(np.mean(daily_maes)) if daily_maes else None,
        "daily_window_ratio": float(np.mean(daily_ratios)) if daily_ratios else None,
        "daily_window_hit_rate": float(np.mean(daily_hits)) if daily_hits else None,
        "daily_n": int(daily_n or 0),
    }


def _horizon_live_errors(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"horizon_accuracy_{instrument_key}.json")
    payload = _load_json(path)
    return _aggregate_horizon_payload(payload)


def _overlay_backtest_errors(instrument_key: str) -> dict:
    payload = load_horizon_overlay(instrument_key)
    horizons = payload.get("horizons", {}) if isinstance(payload, dict) else {}
    daily_pair = payload.get("daily_pair", {}) if isinstance(payload, dict) else {}
    structural_pair = payload.get("daily_pair_structural", {}) if isinstance(payload, dict) else {}
    structural_weekend = payload.get("weekend_structural", {}) if isinstance(payload, dict) else {}

    def _obj(block: dict, horizon_name: str):
        if not isinstance(block, dict):
            return None
        ratio = _safe_float(block.get("window_ratio_mean"))
        mae = _safe_float(block.get("mae_dollars"))
        hit = _safe_float(block.get("window_hit_rate"))
        if ratio is None or mae is None or hit is None:
            return None
        hit_req = float(min_window_hit_rate("daily" if "daily" in horizon_name else horizon_name, instrument_key=instrument_key))
        return float(ratio + (1.25 * max(0.0, hit_req - hit)) + (0.05 * mae))

    def _agg(h_name: str):
        models = horizons.get(h_name, {}) if isinstance(horizons, dict) else {}
        rows = []
        for entry in models.values():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("selected") or "").strip().lower() == "insufficient_week_context":
                continue
            selected = entry.get("selected_metrics", {})
            rows.append(
                {
                    "n": int(entry.get("n", 0) or 0),
                    "mae_dollars": _safe_float(selected.get("mae_dollars")),
                    "window_ratio_mean": _safe_float(selected.get("window_ratio_mean")),
                    "window_hit_rate": _safe_float(selected.get("window_hit_rate")),
                }
            )
        rows = [r for r in rows if r["mae_dollars"] is not None]
        if not rows:
            return {"n": 0, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None}
        return {
            "n": int(max(r["n"] for r in rows)),
            "mae_dollars": float(np.mean([r["mae_dollars"] for r in rows])),
            "window_ratio_mean": float(np.mean([r["window_ratio_mean"] for r in rows if r["window_ratio_mean"] is not None])),
            "window_hit_rate": float(np.mean([r["window_hit_rate"] for r in rows if r["window_hit_rate"] is not None])),
        }

    week = _agg("weekend")
    if isinstance(structural_weekend, dict):
        agg = structural_weekend.get("aggregate_metrics", {})
        candidate = {
            "n": int(structural_weekend.get("n", 0) or 0),
            "mae_dollars": _safe_float(agg.get("mae_dollars")),
            "window_ratio_mean": _safe_float(agg.get("window_ratio_mean")),
            "window_hit_rate": _safe_float(agg.get("window_hit_rate")),
        }
        if _obj(candidate, "weekend") is not None:
            if _obj(week, "weekend") is None or float(_obj(candidate, "weekend")) + 1e-9 < float(_obj(week, "weekend")):
                week = candidate
    dh = _agg("daily_high")
    dl = _agg("daily_low")
    pair_rows = []
    structural_rows = []
    if isinstance(structural_pair, dict):
        agg = structural_pair.get("aggregate_metrics", {})
        if _safe_float(agg.get("mae_dollars")) is not None:
            structural_rows.append(
                {
                    "n": int(structural_pair.get("n", 0) or 0),
                    "mae_dollars": _safe_float(agg.get("mae_dollars")),
                    "window_ratio_mean": _safe_float(agg.get("window_ratio_mean")),
                    "window_hit_rate": _safe_float(agg.get("window_hit_rate")),
                }
            )
    for entry in daily_pair.values() if isinstance(daily_pair, dict) else []:
        if not isinstance(entry, dict):
            continue
        hm = entry.get("high_metrics", {})
        lm = entry.get("low_metrics", {})
        hi_mae = _safe_float(hm.get("mae_dollars"))
        lo_mae = _safe_float(lm.get("mae_dollars"))
        hi_ratio = _safe_float(hm.get("window_ratio_mean"))
        lo_ratio = _safe_float(lm.get("window_ratio_mean"))
        hi_hit = _safe_float(hm.get("window_hit_rate"))
        lo_hit = _safe_float(lm.get("window_hit_rate"))
        if hi_mae is None or lo_mae is None:
            continue
        pair_rows.append(
            {
                "n": int(entry.get("n", 0) or 0),
                "mae_dollars": 0.5 * (hi_mae + lo_mae),
                "window_ratio_mean": 0.5 * (hi_ratio + lo_ratio) if hi_ratio is not None and lo_ratio is not None else None,
                "window_hit_rate": 0.5 * (hi_hit + lo_hit) if hi_hit is not None and lo_hit is not None else None,
            }
        )
    if structural_rows:
        best_struct = min(
            structural_rows,
            key=lambda row: _obj(row, "daily") if _obj(row, "daily") is not None else float("inf"),
        )
        if not pair_rows:
            pair_rows = [best_struct]
        else:
            existing_pair = {
                "n": int(max(int(r.get("n", 0) or 0) for r in pair_rows)),
                "mae_dollars": float(np.mean([r["mae_dollars"] for r in pair_rows if r["mae_dollars"] is not None])),
                "window_ratio_mean": float(np.mean([r["window_ratio_mean"] for r in pair_rows if r["window_ratio_mean"] is not None])),
                "window_hit_rate": float(np.mean([r["window_hit_rate"] for r in pair_rows if r["window_hit_rate"] is not None])),
            }
            pair_obj = _obj(existing_pair, "daily")
            struct_obj = _obj(best_struct, "daily")
            if struct_obj is not None and (pair_obj is None or float(struct_obj) + 1e-9 < float(pair_obj)):
                pair_rows = [best_struct]
    if pair_rows:
        daily_maes = [r["mae_dollars"] for r in pair_rows if r["mae_dollars"] is not None]
        daily_ratios = [r["window_ratio_mean"] for r in pair_rows if r["window_ratio_mean"] is not None]
        daily_hits = [r["window_hit_rate"] for r in pair_rows if r["window_hit_rate"] is not None]
        daily_n = max(int(r["n"] or 0) for r in pair_rows)
    else:
        daily_maes = [v for v in (dh.get("mae_dollars"), dl.get("mae_dollars")) if v is not None]
        daily_ratios = [v for v in (dh.get("window_ratio_mean"), dl.get("window_ratio_mean")) if v is not None]
        daily_hits = [v for v in (dh.get("window_hit_rate"), dl.get("window_hit_rate")) if v is not None]
        daily_n = min([n for n in (dh.get("n", 0), dl.get("n", 0)) if n > 0], default=max(int(dh.get("n", 0) or 0), int(dl.get("n", 0) or 0)))
    return {
        "daily_mae_dollars": float(np.mean(daily_maes)) if daily_maes else None,
        "daily_window_ratio": float(np.mean(daily_ratios)) if daily_ratios else None,
        "daily_window_hit_rate": float(np.mean(daily_hits)) if daily_hits else None,
        "daily_samples": int(daily_n or 0),
        "weekend_mae_dollars": week.get("mae_dollars"),
        "weekend_window_ratio": week.get("window_ratio_mean"),
        "weekend_window_hit_rate": week.get("window_hit_rate"),
        "weekend_samples": int(week.get("n", 0) or 0),
    }


def _cached_horizon_backtest_errors(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"horizon_accuracy_backfill_{instrument_key}.json")
    payload = _load_json(path)
    out = {}
    if payload and payload.get("signature") == _horizon_backfill_signature(instrument_key):
        agg = _aggregate_horizon_payload(payload)
        out = {
            "daily_mae_dollars": agg.get("daily_mae_dollars"),
            "daily_window_ratio": agg.get("daily_window_ratio"),
            "daily_window_hit_rate": agg.get("daily_window_hit_rate"),
            "daily_samples": int(agg.get("daily_n", 0) or 0),
            "weekend_mae_dollars": agg.get("weekend_mae_dollars"),
            "weekend_window_ratio": agg.get("weekend_window_ratio"),
            "weekend_window_hit_rate": agg.get("weekend_window_hit_rate"),
            "weekend_samples": int(agg.get("weekend_n", 0) or 0),
        }
    overlay = _overlay_backtest_errors(instrument_key)
    if overlay.get("daily_mae_dollars") is not None:
        out["daily_mae_dollars"] = overlay.get("daily_mae_dollars")
        out["daily_window_ratio"] = overlay.get("daily_window_ratio")
        out["daily_window_hit_rate"] = overlay.get("daily_window_hit_rate")
        out["daily_samples"] = int(overlay.get("daily_samples", 0) or 0)
    if overlay.get("weekend_mae_dollars") is not None:
        out["weekend_mae_dollars"] = overlay.get("weekend_mae_dollars")
        out["weekend_window_ratio"] = overlay.get("weekend_window_ratio")
        out["weekend_window_hit_rate"] = overlay.get("weekend_window_hit_rate")
        out["weekend_samples"] = int(overlay.get("weekend_samples", 0) or 0)
    return out


def _onboarding_summary(instrument_key: str) -> dict:
    try:
        from onboarding_jobs import summarize_onboarding_state

        return dict(summarize_onboarding_state(instrument_key) or {})
    except Exception:
        return {
            "symbol": str(instrument_key or "").upper(),
            "status": "unknown",
            "status_label": "Unknown",
            "summary": "Onboarding summary unavailable",
        }


def _onboarding_training_plan_state(onboarding_summary: dict) -> dict:
    summary = dict(onboarding_summary or {})
    artifacts = dict(summary.get("artifacts") or {})
    plan = dict(artifacts.get("training_plan") or {})
    feature = dict(artifacts.get("feature_readiness") or {})
    history = dict(artifacts.get("history_assessment") or {})
    return {
        "status": str(plan.get("status") or "").strip(),
        "summary": str(plan.get("summary") or feature.get("summary") or history.get("detail") or "").strip(),
        "primary_next_requirement": str(
            plan.get("primary_next_requirement")
            or feature.get("primary_next_requirement")
            or history.get("primary_next_requirement")
            or ""
        ).strip(),
        "eligible_modes": _normalized_strings(plan.get("eligible_modes") or feature.get("eligible_modes")),
        "deferred_modes": _normalized_strings(plan.get("deferred_modes")),
        "blocked_modes": _normalized_strings(plan.get("blocked_modes") or history.get("blocked_modes")),
        "retained_prior_fields": _normalized_strings(plan.get("retained_prior_fields")),
        "retained_from_prior_run": bool(plan.get("retained_from_prior_run")),
    }


def _onboarding_model_runtime_state(onboarding_summary: dict) -> dict:
    summary = dict(onboarding_summary or {})
    artifacts = dict(summary.get("artifacts") or {})
    runtime = dict(artifacts.get("model_runtime") or {})
    trained_at = runtime.get("trained_at") or runtime.get("generated_at") or runtime.get("updated_at") or runtime.get("last_update")
    age_hours = _safe_float(runtime.get("age_hours"))
    if age_hours is None:
        age_hours = _artifact_age_hours(trained_at)
    explicit_stale = bool(runtime.get("stale") or runtime.get("runtime_stale") or runtime.get("evidence_stale"))
    age_stale = bool(age_hours is not None and age_hours > float(CALIBRATION_MAX_AGE_DAYS) * 24.0)
    return {
        "artifact_available": bool(runtime),
        "status": str(runtime.get("status") or "").strip(),
        "detail": str(runtime.get("detail") or "").strip(),
        "primary_next_requirement": str(runtime.get("primary_next_requirement") or "").strip(),
        "trained_at": trained_at,
        "age_hours": round(float(age_hours), 2) if age_hours is not None else None,
        "stale": bool(explicit_stale or age_stale),
        "trained_modes": _normalized_strings(runtime.get("trained_modes")),
        "available_modes": _normalized_strings(runtime.get("available_modes")),
        "attempted_modes": _normalized_strings(runtime.get("attempted_modes")),
        "failed_modes": _normalized_strings(runtime.get("failed_modes")),
        "deferred_modes": _normalized_strings(runtime.get("deferred_modes")),
        "blocked_modes": _normalized_strings(runtime.get("blocked_modes")),
        "retained_prior_modes": _normalized_strings(runtime.get("retained_prior_modes")),
        "retained_from_prior_run": bool(runtime.get("retained_from_prior_run")),
        "mode_outcomes": dict(runtime.get("mode_outcomes") or {}),
    }


def _training_plan_guidance(onboarding_summary: dict, *, core_models_ready: bool) -> dict:
    state = _onboarding_training_plan_state(onboarding_summary)
    eligible_modes = list(state.get("eligible_modes") or [])
    deferred_modes = list(state.get("deferred_modes") or [])
    blocked_modes = list(state.get("blocked_modes") or [])
    primary_next_requirement = str(state.get("primary_next_requirement") or "").strip()
    if not (eligible_modes or deferred_modes or blocked_modes or primary_next_requirement):
        return {}

    if not core_models_ready:
        if "4min" in eligible_modes:
            return {
                "primary_blocker": "core 4-minute path is trainable now but not yet trained",
                "missing_evidence": "the core 4-minute path is trainable now, but its model artifacts are still missing",
                "primary_next_requirement": primary_next_requirement
                or f"Train the eligible model modes for this symbol: {_join_labels(eligible_modes)}.",
            }
        if "4min" in deferred_modes:
            return {
                "primary_blocker": "core 4-minute path is still below its training floor",
                "missing_evidence": "the core 4-minute path can build features, but it is still below its training floor",
                "primary_next_requirement": primary_next_requirement
                or "Advance intraday history until the 4-minute path reaches 100 usable feature rows.",
            }
        if "4min" in blocked_modes:
            return {
                "primary_blocker": "core 4-minute path cannot build usable rows yet",
                "missing_evidence": "the core 4-minute path cannot yet build usable rows from staged history",
                "primary_next_requirement": primary_next_requirement
                or "Stage additional recent price history until the 4-minute feature path can build usable rows.",
            }
        if eligible_modes:
            return {
                "primary_blocker": "some non-core model paths are trainable, but the core 4-minute path remains unavailable",
                "missing_evidence": "some non-core model paths are trainable, but the core advisory path is still unavailable",
                "primary_next_requirement": primary_next_requirement
                or f"Train the currently eligible non-core modes for this symbol: {_join_labels(eligible_modes)}.",
            }
        if deferred_modes:
            return {
                "primary_blocker": "some model paths have feature coverage but remain below their training floors",
                "missing_evidence": "some model paths can build features, but none have yet reached their training floors",
                "primary_next_requirement": primary_next_requirement
                or f"Advance history until deferred modes become trainable: {_join_labels(deferred_modes)}.",
            }
        if blocked_modes:
            return {
                "primary_blocker": "staged history still blocks all supported model paths",
                "missing_evidence": "staged history still blocks all supported model paths",
                "primary_next_requirement": primary_next_requirement
                or f"Extend symbol history until at least one blocked mode becomes trainable: {_join_labels(blocked_modes)}.",
            }
        return {"primary_next_requirement": primary_next_requirement}

    longer_horizon_partial = [mode for mode in ("DailyHigh", "WeekEnd") if mode in deferred_modes or mode in blocked_modes]
    if longer_horizon_partial:
        return {
            "monitor_flag": "daily / weekend horizon coverage remains partial",
            "primary_next_requirement": primary_next_requirement,
        }
    return {"primary_next_requirement": primary_next_requirement}


def _model_runtime_guidance(onboarding_summary: dict, *, core_models_ready: bool) -> dict:
    state = _onboarding_model_runtime_state(onboarding_summary)
    trained_modes = list(state.get("trained_modes") or [])
    deferred_modes = list(state.get("deferred_modes") or [])
    blocked_modes = list(state.get("blocked_modes") or [])
    failed_modes = list(state.get("failed_modes") or [])
    if not bool(state.get("artifact_available")):
        return {
            "state_key": "model_runtime_missing",
            "state_label": "No model runtime artifact",
            "detail": "No persisted model-runtime artifact is available for this symbol yet.",
            "missing_evidence": "model runtime artifact has not been captured",
            "primary_next_requirement": (
                "Train the core 4-minute model set for this symbol before treating it as trained."
                if not core_models_ready
                else "Run onboarding or training long enough to capture a model-runtime artifact for this symbol."
            ),
        }
    if bool(state.get("stale")):
        return {
            "state_key": "model_runtime_stale",
            "state_label": "Stale model runtime artifact",
            "detail": str(state.get("detail") or "The persisted model-runtime artifact is stale or explicitly marked stale."),
            "missing_evidence": "model runtime artifact is stale",
            "monitor_flag": "model runtime artifact is stale",
            "primary_next_requirement": "Refresh the model runtime artifact before using it as current deployment evidence.",
        }
    if core_models_ready:
        if deferred_modes or blocked_modes or failed_modes:
            return {
                "state_key": "partially_trained",
                "state_label": "Partially trained runtime",
                "detail": str(state.get("detail") or f"Trained {_join_labels(trained_modes)} while {_join_labels(deferred_modes + blocked_modes + failed_modes)} remains deferred, blocked, or failed."),
                "monitor_flag": "some model runtime modes remain deferred, blocked, or failed",
                "primary_next_requirement": str(state.get("primary_next_requirement") or ""),
            }
        return {
            "state_key": "trained_runtime_available",
            "state_label": "Trained runtime available",
            "detail": str(state.get("detail") or f"Trained model coverage: {_join_labels(trained_modes)}."),
        }
    if not trained_modes:
        return {
            "state_key": "trainable_but_untrained",
            "state_label": "Trainable but untrained",
            "detail": str(state.get("detail") or "Model runtime exists, but no trained modes are available yet."),
            "missing_evidence": "model runtime artifact exists, but no trained modes are available",
            "primary_next_requirement": "Train the eligible model modes for this symbol before treating it as trained.",
        }
    trained_label = _join_labels(trained_modes)
    retained = bool(state.get("retained_from_prior_run"))
    detail_prefix = "Retained non-core model coverage exists" if retained else "Non-core model coverage exists"
    return {
        "state_key": "partially_trained",
        "state_label": "Partially trained runtime",
        "primary_blocker": "non-core model coverage exists, but the core 4-minute stack is still missing",
        "missing_evidence": "non-core trained model artifacts exist, but the core advisory path is still missing",
        "primary_next_requirement": "Train the core 4-minute model set for this symbol before treating it as trained.",
        "detail": f"{detail_prefix}: {trained_label}.",
    }


def _runtime_intraday_consensus(state: dict | None = None, system_report: dict | None = None) -> dict:
    payload = dict(state or {})
    advisory_snapshot = dict(payload.get("advisory_snapshot") or {})
    pred_state = dict(advisory_snapshot.get("pred_state") or {})
    intraday = dict(pred_state.get("intraday") or {})
    if intraday:
        return intraday
    predictions = dict(payload.get("predictions") or {})
    intraday_block = dict(predictions.get("intraday") or {})
    consensus = dict(intraday_block.get("consensus") or {})
    if consensus:
        return consensus
    system = dict(system_report or {})
    return dict((((system.get("predictions") or {}).get("4min") or {}).get("consensus") or {}))


def _economic_signal_state(consensus: dict | None) -> dict:
    intr = dict(consensus or {})
    direction = str(intr.get("direction") or "").strip().upper()
    raw_direction = str(intr.get("raw_direction") or "").strip().upper()
    raw_edge_bps = _safe_float(intr.get("raw_edge_bps"))
    edge_bps = _safe_float(intr.get("edge_bps"))
    raw_pred_ret_bps = _safe_float(intr.get("raw_pred_ret_bps"))
    cost_bps_est = _safe_float(intr.get("cost_bps_est"))
    uncertainty_penalty_bps = _safe_float(intr.get("uncertainty_penalty_bps"))
    if direction not in {"UP", "DOWN"} and raw_direction not in {"UP", "DOWN"} and raw_edge_bps is None and edge_bps is None and raw_pred_ret_bps is None:
        return {
            "key": "economics_unknown",
            "label": "Economics unknown",
            "detail": "No current intraday edge summary is available to judge whether the setup is economically viable.",
            "model_edge_present": False,
            "finished_edge_positive": False,
        }
    if edge_bps is not None and edge_bps > 0.0:
        return {
            "key": "finished_edge_positive",
            "label": "Economically viable",
            "detail": "The finished edge remains positive after estimated costs and uncertainty.",
            "model_edge_present": True,
            "finished_edge_positive": True,
        }
    if raw_edge_bps is not None and raw_edge_bps > 0.0 and (edge_bps is None or edge_bps <= 0.0):
        if uncertainty_penalty_bps is not None and uncertainty_penalty_bps >= max(0.5, float(cost_bps_est or 0.0)):
            return {
                "key": "uncertainty_crushed",
                "label": "Uncertainty-crushed",
                "detail": "A raw signal exists, but uncertainty still overwhelms the finished trade.",
                "model_edge_present": True,
                "finished_edge_positive": False,
            }
        return {
            "key": "cost_crushed",
            "label": "Cost-crushed",
            "detail": "A raw signal exists, but estimated costs and uncertainty still remove the finished edge.",
            "model_edge_present": True,
            "finished_edge_positive": False,
        }
    if direction in {"UP", "DOWN"} or raw_direction in {"UP", "DOWN"} or (raw_pred_ret_bps is not None and abs(raw_pred_ret_bps) >= 1.0):
        return {
            "key": "weak_model_edge",
            "label": "Weak model edge",
            "detail": "The model still has a directional lean, but the finished edge is not strong enough yet.",
            "model_edge_present": True,
            "finished_edge_positive": False,
        }
    return {
        "key": "no_model_edge",
        "label": "No model edge",
        "detail": "No positive finished edge is currently available from the model output.",
        "model_edge_present": False,
        "finished_edge_positive": False,
    }


def _runtime_regime_risk_state(
    instrument_key: str,
    state: dict | None = None,
    system_report: dict | None = None,
    news_health: dict | None = None,
) -> dict:
    payload = dict(state or {})
    system = dict(system_report or {})
    news = dict(news_health or payload.get("news_health") or {})
    sym = str(instrument_key or "").upper()
    consensus = _runtime_intraday_consensus(payload, system)
    advisory_snapshot = dict(payload.get("advisory_snapshot") or {})
    execution_snapshot = dict(payload.get("execution_snapshot") or payload.get("execution_meta") or {})
    portfolio_payload = dict(payload.get("portfolio") or execution_snapshot.get("portfolio") or {})
    regime_meta = dict(payload.get("regime_meta") or advisory_snapshot.get("regime_meta") or {})
    risk_meta = dict(payload.get("risk_summary") or payload.get("risk_meta") or advisory_snapshot.get("risk_meta") or {})
    if not consensus and not regime_meta and not risk_meta:
        portfolio_context = {
            "available": bool(portfolio_payload),
            "required": False,
            "detail": (
                "Portfolio snapshot is available."
                if portfolio_payload
                else "Portfolio snapshot is unavailable; portfolio-aware autonomy cannot be proven."
            ),
        }
        return {
            "consensus": {},
            "regime": {},
            "risk": {},
            "economics": _economic_signal_state({}),
            "portfolio_context": portfolio_context,
        }
    df = _load_price_df(sym, online=False)
    predictions = {"intraday": {"consensus": consensus}} if consensus else {}
    current_price = None
    if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns:
        try:
            current_price = float(pd.to_numeric(df.get("close"), errors="coerce").dropna().iloc[-1])
        except Exception:
            current_price = None
    if not regime_meta:
        regime_meta = classify_market_regime(df, predictions=predictions, news_meta=news)
    if not risk_meta:
        confidence = _safe_float(consensus.get("confidence"))
        advisory_stub = {
            "pred_state": {"intraday": consensus},
            "quality_meta": {"score": max(0.0, min(1.0, float(confidence or 60.0) / 100.0))},
            "plan_meta": {},
            "news_meta": news,
        }
        risk_meta = risk_summary_for_symbol(
            sym,
            df=df,
            advisory=advisory_stub,
            predictions=predictions,
            execution_snapshot=execution_snapshot,
            portfolio=portfolio_payload,
            current_price=current_price,
            require_portfolio=False,
        )
    portfolio_context = {
        "available": bool(risk_meta.get("portfolio_context_available")) if "portfolio_context_available" in risk_meta else bool(portfolio_payload),
        "required": bool(risk_meta.get("portfolio_context_required")),
        "detail": str(
            risk_meta.get("portfolio_context_detail")
            or (
                "Portfolio snapshot is available."
                if bool(portfolio_payload)
                else "Portfolio snapshot is unavailable; portfolio-aware autonomy cannot be proven."
            )
        ),
    }
    return {
        "consensus": consensus,
        "regime": dict(regime_meta or {}),
        "risk": dict(risk_meta or {}),
        "economics": _economic_signal_state(consensus),
        "portfolio_context": portfolio_context,
    }


def _risk_gate_guidance(risk_summary: dict | None) -> dict:
    return risk_gate_guidance(risk_summary)


def _decision_quality_state(
    deployment_summary: dict | None,
    runtime_state: dict | None,
) -> dict:
    summary = dict(deployment_summary or {})
    runtime = dict(runtime_state or {})
    regime = dict(runtime.get("regime") or {})
    economics = dict(runtime.get("economics") or {})
    risk = dict(runtime.get("risk") or {})
    risk_guidance = _risk_gate_guidance(risk)
    evidence_key = str(summary.get("evidence_state_key") or "")
    if not regime and not risk and str(economics.get("key") or "") == "economics_unknown":
        maturity_key = str(summary.get("symbol_maturity_key") or "")
        if maturity_key == "partially_trained":
            return {
                "key": "partially_trained",
                "label": "Partially trained",
                "detail": str(summary.get("symbol_maturity_detail") or "Some model coverage exists, but the core deployment path is still incomplete."),
            }
        if maturity_key == "onboarded_untrained":
            return {
                "key": "trainable_but_untrained",
                "label": "Trainable but untrained",
                "detail": str(summary.get("symbol_maturity_detail") or "The symbol has staged onboarding evidence, but trained model artifacts are still missing."),
            }
        if bool(summary.get("live_ready_candidate")):
            return {
                "key": "controlled_live_candidate",
                "label": "Controlled-live candidate",
                "detail": "The symbol clears the current controlled-live candidate stage, subject to separate rollout permission.",
            }
        if bool(summary.get("paper_ready")):
            return {
                "key": "paper_plausible_live_blocked",
                "label": "Paper plausible, live blocked",
                "detail": "The setup is strong enough for the paper-autonomy candidate stage, but live autonomy still requires a higher evidence bar.",
            }
        if bool(summary.get("advisory_ready")):
            return {
                "key": "advisory_usable_autonomy_blocked",
                "label": "Advisory-usable / autonomy blocked",
                "detail": "The setup is usable for operator review, but deployment stage and controls still block autonomy.",
            }
        return {
            "key": "validation_pending",
            "label": "Validation pending",
            "detail": "Training, evaluation, or stage evidence still remains below the current advisory and autonomy bar.",
        }
    if evidence_key in {"stale_market_data", "stale_news", "calibration_drift", "stale_readiness", "missing_evidence"}:
        return {
            "key": "stale_or_missing_evidence",
            "label": "Stale / missing evidence",
            "detail": "Evidence freshness or readiness quality is still too degraded to treat the setup as a stronger operating opportunity.",
        }
    if str(risk_guidance.get("key") or "") == "unknown_regime" or bool(regime.get("unknown")):
        return {
            "key": "unknown_regime",
            "label": "Unknown regime",
            "detail": str(risk_guidance.get("detail") or "The current regime is still too unclear to trust stronger deployment decisions."),
        }
    econ_key = str(economics.get("key") or "")
    if econ_key in {"economics_unknown", "no_model_edge", "weak_model_edge"}:
        return {
            "key": "no_predictive_edge",
            "label": "No predictive edge",
            "detail": str(economics.get("detail") or "The model is not currently offering a finished edge worth acting on."),
        }
    if econ_key == "cost_crushed":
        return {
            "key": "economic_fail_costs",
            "label": "Cost-crushed",
            "detail": str(economics.get("detail") or "The raw signal exists, but estimated costs remove the finished edge."),
        }
    if econ_key == "uncertainty_crushed":
        return {
            "key": "economic_fail_uncertainty",
            "label": "Uncertainty-crushed",
            "detail": str(economics.get("detail") or "The raw signal exists, but uncertainty removes the finished edge."),
        }
    if bool(risk.get("blocked")):
        return {
            "key": "risk_blocked",
            "label": "Risk blocked",
            "detail": str(risk_guidance.get("detail") or risk.get("primary_blocker") or "The risk engine is blocking the setup."),
        }
    if bool(summary.get("live_ready_candidate")):
        return {
            "key": "controlled_live_candidate",
            "label": "Controlled-live candidate",
            "detail": "Signal, economics, and risk are aligned enough for the current controlled-live candidate stage, subject to separate rollout permission.",
        }
    if bool(summary.get("paper_ready")):
        return {
            "key": "paper_plausible_live_blocked",
            "label": "Paper plausible, live blocked",
            "detail": "The setup is strong enough for the paper-autonomy candidate stage, but live autonomy still requires a higher evidence bar.",
        }
    if bool(summary.get("advisory_ready")):
        return {
            "key": "advisory_usable_autonomy_blocked",
            "label": "Advisory-usable / autonomy blocked",
            "detail": "The setup is usable for operator review, but deployment stage and controls still block autonomy.",
        }
    return {
        "key": "validation_pending",
        "label": "Validation pending",
        "detail": "Training, evaluation, or stage evidence still remains below the current advisory and autonomy bar.",
    }


def _load_cached_walk_forward(instrument_key: str, df: pd.DataFrame, n_splits: int) -> dict:
    payload = _load_json(_wf_cache_path(instrument_key))
    if not payload:
        return {}
    if payload.get("signature") != _walk_forward_signature(instrument_key, df, n_splits):
        return {}
    metrics = payload.get("metrics", {})
    return metrics if isinstance(metrics, dict) else {}


def _save_cached_walk_forward(instrument_key: str, df: pd.DataFrame, n_splits: int, metrics: dict):
    payload = {
        "signature": _walk_forward_signature(instrument_key, df, n_splits),
        "metrics": metrics if isinstance(metrics, dict) else {},
        "last_update": datetime.now(timezone.utc).isoformat(),
    }
    _save_json(_wf_cache_path(instrument_key), payload)


def _intraday_live_error(instrument_key: str) -> dict:
    path = os.path.join(DATA_DIR, f"calibration_live_{instrument_key}.json")
    payload = _load_json(path)
    errs = payload.get("recent_price_abs_errors", []) if isinstance(payload, dict) else []
    ratios = payload.get("recent_price_window_ratios", []) if isinstance(payload, dict) else []
    hits = payload.get("recent_price_window_hits", []) if isinstance(payload, dict) else []
    thresholds = payload.get("recent_price_window_thresholds", []) if isinstance(payload, dict) else []

    err_vals = [float(v) for v in (_safe_float(e) for e in errs) if v is not None]
    ratio_vals = [float(v) for v in (_safe_float(e) for e in ratios) if v is not None]
    hit_vals = [float(v) for v in (_safe_float(e) for e in hits) if v is not None]
    threshold_vals = [float(v) for v in (_safe_float(e) for e in thresholds) if v is not None]
    if err_vals:
        return {
            "n": int(len(err_vals)),
            "mae_dollars": float(np.mean(err_vals)),
            "window_ratio_mean": float(np.mean(ratio_vals)) if ratio_vals else None,
            "window_hit_rate": float(np.mean(hit_vals)) if hit_vals else None,
            "window_threshold_mean_dollars": float(np.mean(threshold_vals)) if threshold_vals else None,
        }
    return {"n": 0, "mae_dollars": None, "window_ratio_mean": None, "window_hit_rate": None, "window_threshold_mean_dollars": None}


def _horizon_backtest_errors(df: pd.DataFrame, instrument_key: str, wf: dict | None = None) -> dict:
    out = {
        "daily_mae_dollars": None,
        "daily_window_ratio": None,
        "daily_window_hit_rate": None,
        "daily_folds": 0,
        "daily_samples": 0,
        "weekend_mae_dollars": None,
        "weekend_window_ratio": None,
        "weekend_window_hit_rate": None,
        "weekend_folds": 0,
        "weekend_samples": 0,
        "intraday_mae_dollars": None,
        "intraday_window_ratio": None,
        "intraday_window_hit_rate": None,
        "intraday_folds": 0,
    }
    if df is None or df.empty:
        return out
    cached = _cached_horizon_backtest_errors(instrument_key)
    if int(cached.get("daily_samples", 0) or 0) < max(8, int(READINESS_MIN_DAILY_SAMPLES)) or int(cached.get("weekend_samples", 0) or 0) < max(4, int(READINESS_MIN_WEEKEND_SAMPLES)):
        try:
            from horizon_accuracy_tracker import backfill_horizon_accuracy

            backfill_horizon_accuracy(
                instrument_key,
                max_daily_targets=max(48, int(READINESS_MIN_DAILY_SAMPLES) * 6),
                max_week_targets=max(32, int(READINESS_MIN_WEEKEND_SAMPLES) * 8),
            )
            cached = _cached_horizon_backtest_errors(instrument_key)
        except Exception:
            pass
    try:
        if int(cached.get("daily_samples", 0) or 0) >= max(8, int(READINESS_MIN_DAILY_SAMPLES)):
            out["daily_mae_dollars"] = _safe_float(cached.get("daily_mae_dollars"))
            out["daily_window_ratio"] = _safe_float(cached.get("daily_window_ratio"))
            out["daily_window_hit_rate"] = _safe_float(cached.get("daily_window_hit_rate"))
            out["daily_samples"] = int(cached.get("daily_samples", 0) or 0)
            out["daily_folds"] = out["daily_samples"]
        if int(cached.get("weekend_samples", 0) or 0) >= max(4, int(READINESS_MIN_WEEKEND_SAMPLES)):
            out["weekend_mae_dollars"] = _safe_float(cached.get("weekend_mae_dollars"))
            out["weekend_window_ratio"] = _safe_float(cached.get("weekend_window_ratio"))
            out["weekend_window_hit_rate"] = _safe_float(cached.get("weekend_window_hit_rate"))
            out["weekend_samples"] = int(cached.get("weekend_samples", 0) or 0)
            out["weekend_folds"] = out["weekend_samples"]
    except Exception:
        pass
    # Prefer replay against the actual horizon prediction stack.
    try:
        if out["daily_mae_dollars"] is None or out["weekend_mae_dollars"] is None:
            rep = _horizon_replay_errors(df, instrument_key=instrument_key)
        else:
            rep = {}
        if rep.get("daily_mae_dollars") is not None:
            out["daily_mae_dollars"] = float(rep["daily_mae_dollars"])
            out["daily_window_ratio"] = _safe_float(rep.get("daily_window_ratio"))
            out["daily_window_hit_rate"] = _safe_float(rep.get("daily_window_hit_rate"))
            out["daily_samples"] = int(rep.get("daily_samples", 0) or 0)
            out["daily_folds"] = out["daily_samples"]
        if rep.get("weekend_mae_dollars") is not None:
            out["weekend_mae_dollars"] = float(rep["weekend_mae_dollars"])
            out["weekend_window_ratio"] = _safe_float(rep.get("weekend_window_ratio"))
            out["weekend_window_hit_rate"] = _safe_float(rep.get("weekend_window_hit_rate"))
            out["weekend_samples"] = int(rep.get("weekend_samples", 0) or 0)
            out["weekend_folds"] = out["weekend_samples"]
    except Exception:
        pass
    try:
        ds_daily = build_next_session_dataset(df, instrument_key=instrument_key)
        dh = _cv_abs_ret_metrics(ds_daily, "target_high_ret_norm", "target_high_ret", horizon="daily_high", instrument_key=instrument_key, min_rows=20)
        dl = _cv_abs_ret_metrics(ds_daily, "target_low_ret_norm", "target_low_ret", horizon="daily_low", instrument_key=instrument_key, min_rows=20)
        maes = [v for v in (dh.get("mae_dollars"), dl.get("mae_dollars")) if v is not None]
        ratios = [v for v in (dh.get("window_ratio_mean"), dl.get("window_ratio_mean")) if v is not None]
        hits = [v for v in (dh.get("window_hit_rate"), dl.get("window_hit_rate")) if v is not None]
        if maes and out["daily_mae_dollars"] is None:
            out["daily_mae_dollars"] = float(np.mean(maes))
            out["daily_window_ratio"] = float(np.mean(ratios)) if ratios else None
            out["daily_window_hit_rate"] = float(np.mean(hits)) if hits else None
            out["daily_folds"] = int(min(dh.get("folds", 0), dl.get("folds", 0)))
    except Exception:
        pass
    try:
        ds_week = build_weekend_session_dataset(df, instrument_key=instrument_key)
        wk = _cv_abs_ret_metrics(ds_week, "target_weekend_ret_norm", "target_weekend_ret", horizon="weekend", instrument_key=instrument_key, min_rows=20)
        if out["weekend_mae_dollars"] is None:
            out["weekend_mae_dollars"] = wk.get("mae_dollars")
            out["weekend_window_ratio"] = wk.get("window_ratio_mean")
            out["weekend_window_hit_rate"] = wk.get("window_hit_rate")
            out["weekend_folds"] = int(wk.get("folds", 0) or 0)
    except Exception:
        pass
    if isinstance(wf, dict):
        out["intraday_mae_dollars"] = _safe_mean(wf.get("price_mae_dollars"))
        out["intraday_window_ratio"] = _safe_mean(wf.get("price_window_ratio"))
        out["intraday_window_hit_rate"] = _safe_mean(wf.get("price_window_hit_rate"))
        out["intraday_folds"] = int(wf.get("folds", 0) or 0)
    return out


def _pick_metric(live_val, live_n: int, live_min_n: int, bt_val):
    lv = _safe_float(live_val)
    bv = _safe_float(bt_val)
    n = max(0, int(live_n or 0))
    min_n = max(1, int(live_min_n))
    if lv is not None and n >= min_n:
        return lv, "live", n
    if bv is not None:
        return bv, "backtest", 0
    if lv is not None and n > 0:
        return lv, "live_low_sample", n
    return None, "missing", n


def _pick_horizon_block(live_block: dict, live_min_n: int, bt_block: dict) -> tuple[dict, str, int]:
    live_block = live_block if isinstance(live_block, dict) else {}
    bt_block = bt_block if isinstance(bt_block, dict) else {}
    live_n = int(live_block.get("n", 0) or 0)
    min_n = max(1, int(live_min_n))
    live_ready = (
        live_n >= min_n
        and _safe_float(live_block.get("window_ratio_mean")) is not None
        and _safe_float(live_block.get("window_hit_rate")) is not None
    )
    if live_ready:
        return live_block, "live", live_n
    bt_ready = (
        _safe_float(bt_block.get("window_ratio_mean")) is not None
        and _safe_float(bt_block.get("window_hit_rate")) is not None
    )
    if bt_ready:
        return bt_block, "backtest", live_n
    live_partial = (
        live_n > 0
        and (
            _safe_float(live_block.get("window_ratio_mean")) is not None
            or _safe_float(live_block.get("window_hit_rate")) is not None
            or _safe_float(live_block.get("mae_dollars")) is not None
        )
    )
    if live_partial:
        return live_block, "live_low_sample", live_n
    return {
        "n": live_n,
        "mae_dollars": None,
        "window_ratio_mean": None,
        "window_hit_rate": None,
        "window_threshold_mean_dollars": None,
    }, "missing", live_n


def _preferred_pred_price(block: dict) -> float | None:
    if not isinstance(block, dict):
        return None
    for name in ("consensus", "meta"):
        v = block.get(name)
        if isinstance(v, dict):
            pv = _safe_float(v.get("pred_price"))
            if pv is not None:
                return float(pv)
    vals = []
    for v in block.values():
        if not isinstance(v, dict):
            continue
        pv = _safe_float(v.get("pred_price"))
        if pv is not None:
            vals.append(float(pv))
    if not vals:
        return None
    return float(np.mean(vals))


def _model_pred_prices(block: dict) -> list[float]:
    if not isinstance(block, dict):
        return []
    vals = []
    for model_key in ("base", "forest", "boost", "extra_trees"):
        v = block.get(model_key)
        if not isinstance(v, dict):
            continue
        pv = _safe_float(v.get("pred_price"))
        if pv is not None:
            vals.append(float(pv))
    return vals


def _horizon_replay_errors(df: pd.DataFrame, instrument_key: str, max_daily_samples: int = 64, max_week_samples: int = 64) -> dict:
    out = {
        "daily_mae_dollars": None,
        "daily_window_ratio": None,
        "daily_window_hit_rate": None,
        "daily_samples": 0,
        "daily_directional_acc": None,
        "daily_directional_samples": 0,
        "daily_directional_ties": 0,
        "weekend_mae_dollars": None,
        "weekend_window_ratio": None,
        "weekend_window_hit_rate": None,
        "weekend_samples": 0,
        "weekend_directional_acc": None,
        "weekend_directional_samples": 0,
        "weekend_directional_ties": 0,
    }
    if df is None or df.empty:
        return out
    try:
        models = load_models(include_modes=True, instrument_key=instrument_key)
    except Exception:
        return out
    if not isinstance(models, dict) or not models:
        return out

    try:
        ds_daily = build_next_session_dataset(df, instrument_key=instrument_key)
    except Exception:
        ds_daily = pd.DataFrame()
    if ds_daily is not None and not ds_daily.empty:
        ds_daily = ds_daily.sort_index().tail(max(2, int(max_daily_samples)))
        errs = []
        actuals = []
        dir_hits = []
        dir_ties = 0
        for ts, row in ds_daily.iterrows():
            try:
                subset = df[df.index <= ts]
                if subset is None or subset.empty or len(subset) < 300:
                    continue
                res = predict_horizon(
                    models,
                    subset,
                    instrument_key=instrument_key,
                    mode="DailyHigh",
                    log_side_effects=False,
                )
                if not isinstance(res, dict) or res.get("error"):
                    continue
                pred_h = _preferred_pred_price(res.get("high", {}))
                pred_l = _preferred_pred_price(res.get("low", {}))
                close_v = _safe_float(row.get("close"))
                th = _safe_float(row.get("target_high_ret"))
                tl = _safe_float(row.get("target_low_ret"))
                if close_v is None:
                    continue
                if pred_h is not None and th is not None:
                    act_h = float(close_v * (1.0 + th))
                    errs.append(abs(pred_h - act_h))
                    actuals.append(act_h)
                    vote_h = majority_direction_vote(_model_pred_prices(res.get("high", {})), close_v)
                    dir_ties += int(vote_h.get("ties_excluded", 0) or 0)
                    if vote_h.get("direction") is not None and act_h != close_v:
                        dir_hits.append(1.0 if vote_h["direction"] == ("UP" if act_h > close_v else "DOWN") else 0.0)
                if pred_l is not None and tl is not None:
                    act_l = float(close_v * (1.0 + tl))
                    errs.append(abs(pred_l - act_l))
                    actuals.append(act_l)
                    vote_l = majority_direction_vote(_model_pred_prices(res.get("low", {})), close_v)
                    dir_ties += int(vote_l.get("ties_excluded", 0) or 0)
                    if vote_l.get("direction") is not None and act_l != close_v:
                        dir_hits.append(1.0 if vote_l["direction"] == ("UP" if act_l > close_v else "DOWN") else 0.0)
            except Exception:
                continue
        if errs:
            wm = summarize_price_window_errors(errs, actual_prices=actuals, horizon="daily", instrument_key=instrument_key)
            out["daily_mae_dollars"] = wm.get("mae_dollars")
            out["daily_window_ratio"] = wm.get("window_ratio_mean")
            out["daily_window_hit_rate"] = wm.get("window_hit_rate")
            out["daily_samples"] = int(len(errs))
            out["daily_directional_acc"] = float(np.mean(dir_hits)) if dir_hits else None
            out["daily_directional_samples"] = int(len(dir_hits))
            out["daily_directional_ties"] = int(dir_ties)

    try:
        ds_week = build_weekend_session_dataset(df, instrument_key=instrument_key)
    except Exception:
        ds_week = pd.DataFrame()
    if ds_week is not None and not ds_week.empty:
        ds_week = ds_week.sort_index().tail(max(2, int(max_week_samples)))
        errs_w = []
        actuals_w = []
        dir_hits_w = []
        dir_ties_w = 0
        for ts, row in ds_week.iterrows():
            try:
                subset = df[df.index <= ts]
                if subset is None or subset.empty or len(subset) < 300:
                    continue
                res = predict_horizon(
                    models,
                    subset,
                    instrument_key=instrument_key,
                    mode="WeekEnd",
                    log_side_effects=False,
                )
                if not isinstance(res, dict) or res.get("error"):
                    continue
                pred_w = _preferred_pred_price(res.get("week", {}))
                close_v = _safe_float(row.get("close"))
                tw = _safe_float(row.get("target_weekend_ret"))
                if pred_w is None or close_v is None or tw is None:
                    continue
                act_w = float(close_v * (1.0 + tw))
                errs_w.append(abs(pred_w - act_w))
                actuals_w.append(act_w)
                vote_w = majority_direction_vote(_model_pred_prices(res.get("week", {})), close_v)
                dir_ties_w += int(vote_w.get("ties_excluded", 0) or 0)
                if vote_w.get("direction") is not None and act_w != close_v:
                    dir_hits_w.append(1.0 if vote_w["direction"] == ("UP" if act_w > close_v else "DOWN") else 0.0)
            except Exception:
                continue
        if errs_w:
            wm = summarize_price_window_errors(errs_w, actual_prices=actuals_w, horizon="weekend", instrument_key=instrument_key)
            out["weekend_mae_dollars"] = wm.get("mae_dollars")
            out["weekend_window_ratio"] = wm.get("window_ratio_mean")
            out["weekend_window_hit_rate"] = wm.get("window_hit_rate")
            out["weekend_samples"] = int(len(errs_w))
            out["weekend_directional_acc"] = float(np.mean(dir_hits_w)) if dir_hits_w else None
            out["weekend_directional_samples"] = int(len(dir_hits_w))
            out["weekend_directional_ties"] = int(dir_ties_w)

    return out


def evaluate_readiness_metrics(instrument_key: str, online: bool = False) -> dict:
    if instrument_key not in INSTRUMENTS:
        return {"instrument": instrument_key, "error": "unknown instrument"}
    df = _load_price_df(instrument_key, online=online)
    if df is None or df.empty:
        return {"instrument": instrument_key, "error": "no price data"}
    wf_splits = max(3, int(READINESS_BACKTEST_SPLITS))
    if online:
        wf = walk_forward_validate(
            df,
            n_splits=wf_splits,
            include_transaction_costs=False,
            instrument_key=instrument_key,
        )
    else:
        wf = _load_cached_walk_forward(instrument_key, df, wf_splits)
        if not wf:
            wf = walk_forward_validate(
                df,
                n_splits=wf_splits,
                include_transaction_costs=False,
                instrument_key=instrument_key,
            )
            _save_cached_walk_forward(instrument_key, df, wf_splits, wf)
    folds = int(wf.get("folds", 0) or 0)
    mean_dir = _safe_mean(wf.get("directional_acc", []))
    mean_majority_dir = _safe_mean(wf.get("majority_direction_acc", []))
    intraday_majority_samples = int(np.nansum(pd.to_numeric(wf.get("majority_direction_samples", []), errors="coerce"))) if isinstance(wf, dict) else 0
    intraday_majority_ties = int(np.nansum(pd.to_numeric(wf.get("majority_direction_ties", []), errors="coerce"))) if isinstance(wf, dict) else 0
    mean_net = _safe_mean(wf.get("net_returns", []))
    mean_sharpe = _safe_mean(wf.get("sharpe_like", []))
    mean_active = _safe_mean(wf.get("active_fraction", []))
    mean_trade_active = _safe_mean(wf.get("trade_active_fraction", []))
    live_h = _horizon_live_errors(instrument_key)
    live_4m = _intraday_live_error(instrument_key)
    bt_h = _horizon_backtest_errors(df, instrument_key, wf=wf)

    live_week = {
        "n": int(live_h.get("weekend_n", 0) or 0),
        "mae_dollars": live_h.get("weekend_mae_dollars"),
        "window_ratio_mean": live_h.get("weekend_window_ratio"),
        "window_hit_rate": live_h.get("weekend_window_hit_rate"),
    }
    bt_week = {
        "n": int(bt_h.get("weekend_samples", 0) or 0),
        "mae_dollars": bt_h.get("weekend_mae_dollars"),
        "window_ratio_mean": bt_h.get("weekend_window_ratio"),
        "window_hit_rate": bt_h.get("weekend_window_hit_rate"),
    }
    live_daily = {
        "n": int(live_h.get("daily_n", 0) or 0),
        "mae_dollars": live_h.get("daily_mae_dollars"),
        "window_ratio_mean": live_h.get("daily_window_ratio"),
        "window_hit_rate": live_h.get("daily_window_hit_rate"),
    }
    bt_daily = {
        "n": int(bt_h.get("daily_samples", 0) or 0),
        "mae_dollars": bt_h.get("daily_mae_dollars"),
        "window_ratio_mean": bt_h.get("daily_window_ratio"),
        "window_hit_rate": bt_h.get("daily_window_hit_rate"),
    }
    live_intr = {
        "n": int(live_4m.get("n", 0) or 0),
        "mae_dollars": live_4m.get("mae_dollars"),
        "window_ratio_mean": live_4m.get("window_ratio_mean"),
        "window_hit_rate": live_4m.get("window_hit_rate"),
        "window_threshold_mean_dollars": live_4m.get("window_threshold_mean_dollars"),
    }
    bt_intr = {
        "n": int(bt_h.get("intraday_folds", 0) or 0),
        "mae_dollars": bt_h.get("intraday_mae_dollars"),
        "window_ratio_mean": bt_h.get("intraday_window_ratio"),
        "window_hit_rate": bt_h.get("intraday_window_hit_rate"),
    }

    week_block, week_src, week_n = _pick_horizon_block(live_week, READINESS_MIN_WEEKEND_SAMPLES, bt_week)
    daily_block, daily_src, daily_n = _pick_horizon_block(live_daily, READINESS_MIN_DAILY_SAMPLES, bt_daily)
    intr_block, intr_src, intr_n = _pick_horizon_block(live_intr, READINESS_MIN_INTRADAY_SAMPLES, bt_intr)

    week_score = window_ratio_score(week_block.get("window_ratio_mean"))
    daily_score = window_ratio_score(daily_block.get("window_ratio_mean"))
    intr_score = window_ratio_score(intr_block.get("window_ratio_mean"))
    try:
        ref_price = float(pd.to_numeric(df.get("close"), errors="coerce").dropna().iloc[-1])
    except Exception:
        ref_price = None
    intr_target_window = price_window_threshold_dollars("intraday", instrument_key=instrument_key, reference_price=ref_price)
    daily_target_window = price_window_threshold_dollars("daily", instrument_key=instrument_key, reference_price=ref_price)
    weekend_target_window = price_window_threshold_dollars("weekend", instrument_key=instrument_key, reference_price=ref_price)
    w_week = max(0.0, float(READINESS_WEIGHT_WEEKEND))
    w_day = max(0.0, float(READINESS_WEIGHT_DAILY))
    w_intra = max(0.0, float(READINESS_WEIGHT_INTRADAY))
    weighted_num = 0.0
    weighted_den = 0.0
    if week_score is not None:
        weighted_num += w_week * week_score
        weighted_den += w_week
    if daily_score is not None:
        weighted_num += w_day * daily_score
        weighted_den += w_day
    if intr_score is not None:
        weighted_num += w_intra * intr_score
        weighted_den += w_intra
    horizon_score = float(weighted_num / weighted_den) if weighted_den > 0 else 0.0

    thresholds = {
        "min_folds": int(READINESS_MIN_FOLDS),
        "min_directional_acc": float(READINESS_MIN_DIRECTIONAL_ACC),
        "min_net_return": float(READINESS_MIN_NET_RETURN),
        "min_sharpe_like": float(READINESS_MIN_SHARPE_LIKE),
        "min_active_fraction": float(READINESS_MIN_ACTIVE_FRACTION),
        "intraday_window_dollars": float(READINESS_INTRADAY_WINDOW_DOLLARS),
        "intraday_window_pct": float(READINESS_INTRADAY_WINDOW_PCT),
        "daily_window_dollars": float(READINESS_DAILY_WINDOW_DOLLARS),
        "daily_window_pct": float(READINESS_DAILY_WINDOW_PCT),
        "weekend_window_dollars": float(READINESS_WEEKEND_WINDOW_DOLLARS),
        "weekend_window_pct": float(READINESS_WEEKEND_WINDOW_PCT),
        "min_intraday_window_hit_rate": float(READINESS_MIN_INTRADAY_WINDOW_HIT_RATE),
        "min_daily_window_hit_rate": float(READINESS_MIN_DAILY_WINDOW_HIT_RATE),
        "min_weekend_window_hit_rate": float(READINESS_MIN_WEEKEND_WINDOW_HIT_RATE),
        "legacy_max_weekend_err_pct": float(READINESS_MAX_WEEKEND_ERR_PCT),
        "legacy_max_daily_err_pct": float(READINESS_MAX_DAILY_ERR_PCT),
        "legacy_max_intraday_err_bps": float(READINESS_MAX_INTRADAY_ERR_BPS),
        "min_horizon_score": float(READINESS_MIN_HORIZON_SCORE),
    }
    req_active = bool(READINESS_REQUIRE_ACTIVE)
    week_hit_req = float(min_window_hit_rate("weekend", instrument_key=instrument_key))
    daily_hit_req = float(min_window_hit_rate("daily", instrument_key=instrument_key))
    intr_hit_req = float(min_window_hit_rate("intraday", instrument_key=instrument_key))
    week_mae = _safe_float(week_block.get("mae_dollars"))
    day_mae = _safe_float(daily_block.get("mae_dollars"))
    intr_mae = _safe_float(intr_block.get("mae_dollars"))
    checks = {
        "folds_ok": bool(folds >= thresholds["min_folds"]),
        "directional_ok": True,
        "net_ok": True,
        "sharpe_ok": True,
        "active_ok": bool((not req_active) or (mean_active is not None and mean_active >= thresholds["min_active_fraction"])),
        "weekend_discrepancy_ok": bool(week_mae is not None and week_mae <= float(weekend_target_window)),
        "daily_discrepancy_ok": bool(day_mae is not None and day_mae <= float(daily_target_window)),
        "intraday_discrepancy_ok": bool(intr_mae is not None and intr_mae <= float(intr_target_window)),
        "weekend_window_ratio_ok": bool(_safe_float(week_block.get("window_ratio_mean")) is not None and float(week_block["window_ratio_mean"]) <= 1.0),
        "weekend_window_hit_ok": bool(_safe_float(week_block.get("window_hit_rate")) is not None and float(week_block["window_hit_rate"]) >= week_hit_req),
        "daily_window_ratio_ok": bool(_safe_float(daily_block.get("window_ratio_mean")) is not None and float(daily_block["window_ratio_mean"]) <= 1.0),
        "daily_window_hit_ok": bool(_safe_float(daily_block.get("window_hit_rate")) is not None and float(daily_block["window_hit_rate"]) >= daily_hit_req),
        "intraday_window_ratio_ok": bool(_safe_float(intr_block.get("window_ratio_mean")) is not None and float(intr_block["window_ratio_mean"]) <= 1.0),
        "intraday_window_hit_ok": bool(_safe_float(intr_block.get("window_hit_rate")) is not None and float(intr_block["window_hit_rate"]) >= intr_hit_req),
        "horizon_score_ok": bool(horizon_score >= thresholds["min_horizon_score"]),
    }
    checks["weekend_window_ok"] = bool(checks["weekend_discrepancy_ok"])
    checks["daily_window_ok"] = bool(checks["daily_discrepancy_ok"])
    checks["intraday_window_ok"] = bool(checks["intraday_discrepancy_ok"])
    legacy_ok = bool(
        checks["folds_ok"]
        and checks["active_ok"]
    )
    if READINESS_USE_HORIZON_PRIORITY:
        horizon_ok = bool(
            checks["weekend_window_ok"]
            and checks["daily_window_ok"]
            and checks["intraday_window_ok"]
        )
        passed = bool(legacy_ok and horizon_ok)
    else:
        passed = bool(legacy_ok)
    return {
        "instrument": instrument_key,
        "ts": datetime.now(timezone.utc).isoformat(),
        "date": datetime.now(timezone.utc).date().isoformat(),
        "passed": passed,
        "checks": checks,
        "metrics": {
            "folds": folds,
            "directional_acc_mean": mean_dir,
            "intraday_updown_acc": mean_majority_dir,
            "intraday_updown_samples": intraday_majority_samples,
            "intraday_updown_ties_excluded": intraday_majority_ties,
            "net_return_mean": mean_net,
            "sharpe_like_mean": mean_sharpe,
            "active_fraction_mean": mean_active,
            "trade_active_fraction_mean": mean_trade_active,
            "horizon_weighted_score": horizon_score,
            "weekend_mae_dollars": week_mae,
            "weekend_avg_discrepancy_dollars": week_mae,
            "weekend_window_ratio": _safe_float(week_block.get("window_ratio_mean")),
            "weekend_window_hit_rate": _safe_float(week_block.get("window_hit_rate")),
            "weekend_target_window_dollars": float(weekend_target_window),
            "weekend_success": bool(checks["weekend_window_ok"]),
            "weekend_updown_acc": _safe_float(week_block.get("directional_acc")),
            "weekend_updown_samples": int(week_block.get("directional_samples", 0) or 0),
            "weekend_updown_ties_excluded": int(week_block.get("directional_ties", 0) or 0),
            "daily_mae_dollars": day_mae,
            "daily_avg_discrepancy_dollars": day_mae,
            "daily_window_ratio": _safe_float(daily_block.get("window_ratio_mean")),
            "daily_window_hit_rate": _safe_float(daily_block.get("window_hit_rate")),
            "daily_target_window_dollars": float(daily_target_window),
            "daily_success": bool(checks["daily_window_ok"]),
            "daily_updown_acc": _safe_float(daily_block.get("directional_acc")),
            "daily_updown_samples": int(daily_block.get("directional_samples", 0) or 0),
            "daily_updown_ties_excluded": int(daily_block.get("directional_ties", 0) or 0),
            "intraday_mae_dollars": intr_mae,
            "intraday_avg_discrepancy_dollars": intr_mae,
            "intraday_window_ratio": _safe_float(intr_block.get("window_ratio_mean")),
            "intraday_window_hit_rate": _safe_float(intr_block.get("window_hit_rate")),
            "intraday_target_window_dollars": float(intr_target_window),
            "intraday_success": bool(checks["intraday_window_ok"]),
            "intraday_price_mae_backtest": _safe_mean(wf.get("price_mae_dollars")),
            "intraday_price_window_hit_rate_backtest": _safe_mean(wf.get("price_window_hit_rate")),
            "intraday_price_window_ratio_backtest": _safe_mean(wf.get("price_window_ratio")),
        },
        "sources": {
            "weekend": {"source": week_src, "live_samples": int(week_n), "live_min_samples": int(READINESS_MIN_WEEKEND_SAMPLES)},
            "daily": {"source": daily_src, "live_samples": int(daily_n), "live_min_samples": int(READINESS_MIN_DAILY_SAMPLES)},
            "intraday": {"source": intr_src, "live_samples": int(intr_n), "live_min_samples": int(READINESS_MIN_INTRADAY_SAMPLES)},
        },
        "horizon_live": live_h,
        "horizon_backtest": bt_h,
        "thresholds": thresholds,
    }


def _read_history(instrument_key: str) -> list[dict]:
    path = _history_path(instrument_key)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def _write_history(instrument_key: str, rows: list[dict]):
    path = _history_path(instrument_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _write_state(instrument_key: str, state: dict):
    path = _state_path(instrument_key)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)


def _compute_pass_streak(rows: list[dict]) -> int:
    if not rows:
        return 0
    dedup = {}
    for r in rows:
        d = str(r.get("date", "")).strip()
        if not d:
            continue
        prev = dedup.get(d)
        if prev is None:
            dedup[d] = r
            continue
        ts_new = pd.to_datetime(r.get("ts"), utc=True, errors="coerce")
        ts_old = pd.to_datetime(prev.get("ts"), utc=True, errors="coerce")
        if pd.isna(ts_old) or (pd.notna(ts_new) and ts_new >= ts_old):
            dedup[d] = r
    ordered = sorted(dedup.values(), key=lambda x: str(x.get("date", "")))
    if not ordered:
        return 0
    streak = 0
    prev_date = None
    for row in reversed(ordered):
        try:
            d = pd.to_datetime(row.get("date")).date()
        except Exception:
            break
        if not bool(row.get("passed", False)):
            break
        if prev_date is None:
            streak = 1
            prev_date = d
            continue
        if (prev_date - d).days == 1:
            streak += 1
            prev_date = d
        else:
            break
    return int(streak)


def _is_state_fresh(state: dict | None) -> bool:
    if not isinstance(state, dict):
        return False
    ts = pd.to_datetime(state.get("last_update"), utc=True, errors="coerce")
    if pd.isna(ts):
        return False
    age_hours = (pd.Timestamp.now(tz="UTC") - ts).total_seconds() / 3600.0
    return bool(age_hours <= float(READINESS_MAX_AGE_HOURS))


_DEPLOYMENT_STAGE_ORDER = {
    "supported_not_trained": 0,
    "trained_not_validated": 1,
    "advisory_only": 2,
    "advisory_shadow_ready": 3,
    "paper_autonomy_candidate": 4,
    "controlled_real_world_candidate": 5,
}

_DEPLOYMENT_STAGE_LABELS = {
    "supported_not_trained": "Supported / not trained",
    "trained_not_validated": "Trained / not validated",
    "advisory_only": "Advisory only",
    "advisory_shadow_ready": "Advisory + shadow",
    "paper_autonomy_candidate": "Paper-autonomy candidate",
    "controlled_real_world_candidate": "Controlled real-world candidate",
}


def _deployment_stage_label(stage_key: str) -> str:
    key = str(stage_key or "trained_not_validated").strip().lower()
    return _DEPLOYMENT_STAGE_LABELS.get(key, _DEPLOYMENT_STAGE_LABELS["trained_not_validated"])


def _requirement_category(text: str) -> str:
    lower = str(text or "").strip().lower()
    if not lower:
        return "other"
    if any(
        token in lower
        for token in (
            "price history",
            "advance intraday history",
            "extend session-level history",
            "extend daily / weekly history",
            "core 4-minute model",
            "onboarding failure",
            "symbol setup",
            "milestone artifacts",
            "missing onboarding",
        )
    ):
        return "symbol_maturity"
    if any(
        token in lower
        for token in (
            "refresh market data",
            "news history",
            "drift flag clears",
            "intraday predictions so calibration evidence is current",
            "horizon accuracy history",
        )
    ):
        return "freshness"
    if any(
        token in lower
        for token in (
            "readiness evaluation",
            "passing readiness",
            "weighted horizon accuracy",
            "shadow observation",
            "pass-streak gate",
            "validation evidence",
            "walk-forward validation",
            "strict out-of-sample",
            "transaction-cost",
            "transaction costs",
            "untouched holdout",
            "paper ledger",
            "paper-trading evidence",
            "net edge",
            "drawdown",
        )
    ):
        return "validation"
    return "other"


def _prioritize_requirements(items: list[str]) -> list[str]:
    def _priority(text: str) -> tuple[int, int]:
        lower = str(text or "").strip().lower()
        if "stage canonical price history" in lower:
            return (0, 0)
        if "resolve the onboarding failure" in lower or "symbol setup" in lower:
            return (0, 1)
        if "train the core 4-minute model set" in lower:
            return (0, 2)
        if "readiness evaluation" in lower and "generate" in lower:
            return (0, 3)
        if "advance intraday history" in lower:
            return (0, 4)
        if "extend session-level history" in lower:
            return (0, 5)
        if "extend daily / weekly history" in lower:
            return (0, 6)
        if "refresh market data" in lower:
            return (1, 0)
        if "refresh and extend the symbol news history" in lower:
            return (1, 1)
        if "rebuild live calibration" in lower:
            return (1, 2)
        if "advance price history enough" in lower:
            return (1, 3)
        if "resolve fresh intraday predictions" in lower:
            return (1, 4)
        if "refresh horizon accuracy history" in lower:
            return (1, 5)
        if "validation evidence has not been captured" in lower or "run walk-forward validation" in lower:
            return (2, 0)
        if "validation evidence is stale" in lower or "refresh stale validation" in lower:
            return (2, 1)
        if "strict out-of-sample" in lower or "transaction-cost" in lower or "transaction costs" in lower:
            return (2, 2)
        if "untouched holdout" in lower or "paper ledger" in lower or "paper-trading evidence" in lower:
            return (2, 3)
        if "negative after costs" in lower or "drawdown is excessive" in lower:
            return (2, 4)
        if "improve weighted horizon accuracy" in lower:
            return (3, 0)
        if "passing readiness day" in lower:
            return (3, 1)
        if "shadow observation" in lower:
            return (3, 2)
        category_order = {"symbol_maturity": 4, "freshness": 5, "validation": 6, "other": 7}
        return (category_order.get(_requirement_category(text), 9), 9)

    cleaned = []
    seen = set()
    for item in items or []:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return sorted(
        cleaned,
        key=lambda text: (_priority(text), cleaned.index(text)),
    )


def _status_flags(
    instrument_known: bool,
    price_available: bool,
    core_models_ready: bool,
    has_metrics: bool,
    advisory_ready: bool,
    shadow_ready: bool,
    paper_ready: bool,
    live_ready_candidate: bool,
    rollout_actual_allowed: bool,
) -> dict:
    return {
        "supported": bool(instrument_known),
        "onboarded": bool(price_available),
        "trained": bool(core_models_ready),
        "validated": bool(has_metrics),
        "advisory_usable": bool(advisory_ready),
        "shadow_candidate": bool(shadow_ready),
        "paper_candidate": bool(paper_ready),
        "live_candidate": bool(live_ready_candidate),
        "rollout_allowed": bool(rollout_actual_allowed and live_ready_candidate),
    }


def _symbol_maturity_state(
    *,
    instrument_known: bool,
    onboarding_summary: dict,
    price_available: bool,
    any_models: bool,
    core_models_ready: bool,
    has_metrics: bool,
    advisory_ready: bool,
    shadow_ready: bool,
    paper_ready: bool,
    live_ready_candidate: bool,
) -> dict:
    onboarding_status = str((onboarding_summary or {}).get("status") or "not_started").strip().lower()
    onboarding_detail = str((onboarding_summary or {}).get("summary") or (onboarding_summary or {}).get("current_stage_label") or "").strip()
    honest_completion = bool(((onboarding_summary or {}).get("completion_flags") or {}).get("honest_completion"))
    if not instrument_known:
        return {
            "key": "unsupported",
            "label": "Unsupported",
            "detail": "This symbol is not registered in the runtime registry.",
        }
    if live_ready_candidate:
        return {
            "key": "live_candidate",
            "label": "Live candidate",
            "detail": "This symbol has progressed through training, validation, and controlled-live candidate checks.",
        }
    if paper_ready:
        return {
            "key": "paper_candidate",
            "label": "Paper candidate",
            "detail": "This symbol has cleared the current paper-autonomy candidate bar, but live rollout still needs separate evidence and approval.",
        }
    if shadow_ready:
        return {
            "key": "shadow_candidate",
            "label": "Shadow candidate",
            "detail": "This symbol has enough evidence for advisory use plus shadow monitoring, but it is still below the paper-autonomy bar.",
        }
    if advisory_ready:
        return {
            "key": "advisory_usable",
            "label": "Advisory-usable",
            "detail": "This symbol has trained models plus readiness metrics, so the advisory stack is usable even though autonomy remains gated.",
        }
    if onboarding_status == "failed":
        return {
            "key": "onboarding_failed",
            "label": "Onboarding failed",
            "detail": onboarding_detail or "The latest onboarding attempt failed before the symbol reached a reliable workspace state.",
        }
    if onboarding_status == "running":
        return {
            "key": "onboarding_incomplete",
            "label": "Onboarding incomplete",
            "detail": onboarding_detail or "The onboarding pipeline is still staging symbol-specific data, models, or evidence.",
        }
    if price_available and not any_models:
        return {
            "key": "onboarded_untrained",
            "label": "Onboarded, not trained",
            "detail": "Market data exists for this symbol, but no trained model stack is available yet.",
        }
    if price_available and any_models and not core_models_ready:
        return {
            "key": "partially_trained",
            "label": "Partially trained",
            "detail": "Some model artifacts exist, but the core 4-minute stack is still incomplete for this symbol.",
        }
    if core_models_ready and not has_metrics:
        return {
            "key": "trained_validation_pending",
            "label": "Trained, validation pending",
            "detail": "Core models exist, but the symbol still lacks the readiness evidence needed for honest advisory maturity.",
        }
    if price_available and onboarding_status == "completed" and not honest_completion:
        return {
            "key": "onboarding_incomplete",
            "label": "Onboarding incomplete",
            "detail": "The onboarding job finished, but some expected milestone artifacts are still missing.",
        }
    if price_available:
        return {
            "key": "onboarded",
            "label": "Onboarded",
            "detail": "Canonical market data has been staged, but symbol-specific model and readiness maturity is still incomplete.",
        }
    return {
        "key": "supported_only",
        "label": "Supported only",
        "detail": "The platform can host this symbol, but symbol-specific data and evidence have not been staged yet.",
    }


def _evidence_state(
    *,
    price_available: bool,
    core_models_ready: bool,
    has_metrics: bool,
    fresh: bool,
    critical_stale: bool,
    price_stale: bool,
    headline_state: str,
    news_status: str,
    calibration_drift: bool,
    horizon_drift: bool,
    calibration_evidence_stale: bool,
    input_fresh: bool,
    blockers: list[str],
    monitor_flags: list[str],
) -> dict:
    headline = str(headline_state or news_status or "").strip().lower()
    blocker_text = " | ".join(str(item or "").strip().lower() for item in blockers)
    if not price_available or not core_models_ready or not has_metrics:
        return {
            "key": "missing_evidence",
            "label": "Missing evidence",
            "detail": "This symbol is still missing some combination of staged data, trained models, or readiness validation evidence.",
        }
    if critical_stale or price_stale:
        return {
            "key": "stale_market_data",
            "label": "Stale market data",
            "detail": "Price inputs are stale enough to block trustworthy operator use right now.",
        }
    if headline in {"failed", "stale", "error"} or "news pipeline is degraded" in blocker_text:
        return {
            "key": "stale_news",
            "label": "Stale news context",
            "detail": "News inputs are stale or degraded enough that the current advisory context should not be trusted for execution.",
        }
    if calibration_drift or horizon_drift:
        return {
            "key": "calibration_drift",
            "label": "Calibration drift",
            "detail": "Recent calibration or horizon-accuracy drift is still active, so model evidence is not clean enough for higher-trust operation.",
        }
    if not fresh:
        return {
            "key": "stale_readiness",
            "label": "Stale readiness evidence",
            "detail": "The last readiness snapshot is stale, so deployment-stage conclusions should be treated as outdated.",
        }
    if any(
        token in blocker_text
        for token in (
            "pass streak / readiness gate",
            "horizon score is below the deployment floor",
            "window hit rate is too weak",
            "system check is failing",
        )
    ):
        return {
            "key": "weak_evidence",
            "label": "Weak / incomplete evidence",
            "detail": "Fresh artifacts exist, but the validation record is still too weak or inconsistent to support a stronger operating posture.",
        }
    if not input_fresh or calibration_evidence_stale or headline in {"missing", "thin", "degraded"} or monitor_flags:
        return {
            "key": "monitored_evidence",
            "label": "Usable, but monitored",
            "detail": "The evidence stack is usable for advisory work, but there are freshness or quality flags that still require operator caution.",
        }
    return {
        "key": "fresh_evidence",
        "label": "Fresh evidence",
        "detail": "Inputs, calibration evidence, and readiness state are currently aligned enough for the symbol's present stage.",
    }


def _operator_state(
    *,
    instrument_known: bool,
    onboarding_summary: dict,
    price_available: bool,
    core_models_ready: bool,
    has_metrics: bool,
    advisory_ready: bool,
    shadow_ready: bool,
    paper_ready: bool,
    live_ready_candidate: bool,
    rollout_actual_allowed: bool,
    evidence_state_key: str,
    symbol_maturity: dict,
) -> dict:
    onboarding_status = str((onboarding_summary or {}).get("status") or "not_started").strip().lower()
    if not instrument_known:
        return {
            "key": "unsupported",
            "label": "Unsupported",
            "detail": "This symbol is outside the registered runtime universe.",
        }
    if onboarding_status in {"running", "failed"} and not advisory_ready:
        return {
            "key": "blocked_incomplete_onboarding",
            "label": "Blocked: incomplete onboarding",
            "detail": str(symbol_maturity.get("detail") or "The symbol-specific onboarding pipeline has not reached a clean completion state yet."),
        }
    if not price_available:
        return {
            "key": "blocked_missing_maturity",
            "label": "Blocked: insufficient symbol maturity",
            "detail": "The platform supports this symbol in principle, but symbol-specific market data has not been staged yet.",
        }
    if not core_models_ready:
        return {
            "key": "blocked_missing_models",
            "label": "Blocked: missing trained models",
            "detail": "The workspace exists, but the core trained model set is still missing or incomplete for this symbol.",
        }
    if not has_metrics:
        return {
            "key": "blocked_missing_evidence",
            "label": "Blocked: missing readiness evidence",
            "detail": "Core models exist, but readiness metrics and stage evidence have not been generated cleanly enough yet.",
        }
    if evidence_state_key == "stale_market_data":
        return {
            "key": "blocked_stale_data",
            "label": "Blocked: stale data",
            "detail": "Signal direction may exist, but stale market inputs block trustworthy execution decisions right now.",
        }
    if evidence_state_key == "stale_news":
        return {
            "key": "blocked_stale_news",
            "label": "Blocked: stale news",
            "detail": "Signal direction may exist, but stale or degraded news context is blocking a higher-trust read.",
        }
    if evidence_state_key == "calibration_drift":
        return {
            "key": "blocked_calibration_drift",
            "label": "Blocked: calibration drift",
            "detail": "Recent drift flags are still active, so current advisory confidence should not be promoted into stronger operating modes.",
        }
    if evidence_state_key == "stale_readiness":
        return {
            "key": "blocked_stale_evidence",
            "label": "Blocked: stale readiness evidence",
            "detail": "The readiness state is too old to treat as current evidence, even if the symbol has trained models.",
        }
    if live_ready_candidate and rollout_actual_allowed:
        return {
            "key": "rollout_allowed",
            "label": "Rollout allowed",
            "detail": "The symbol is on the rollout list and currently clears the controlled-live candidate stage.",
        }
    if live_ready_candidate:
        return {
            "key": "live_candidate_rollout_gated",
            "label": "Live candidate / rollout gated",
            "detail": "The symbol is structurally at the controlled-live candidate stage, but actual rollout remains separately gated.",
        }
    if paper_ready and not rollout_actual_allowed:
        return {
            "key": "paper_candidate_rollout_gated",
            "label": "Paper candidate / rollout gated",
            "detail": "The symbol clears the paper-candidate bar, but the rollout list still keeps actual autonomy disabled for it.",
        }
    if paper_ready:
        return {
            "key": "paper_candidate",
            "label": "Paper candidate",
            "detail": "The symbol clears the current paper-autonomy candidate bar, but live deployment is still separately gated.",
        }
    if shadow_ready:
        return {
            "key": "shadow_candidate",
            "label": "Shadow candidate",
            "detail": "The symbol is ready for advisory use plus shadow observation, but not yet for paper autonomy.",
        }
    if advisory_ready and evidence_state_key in {"weak_evidence", "monitored_evidence"}:
        return {
            "key": "advisory_weak_evidence",
            "label": "Advisory usable, weak evidence",
            "detail": "The advisory stack is usable, but the current evidence is still weak or degraded enough that execution confidence should stay conservative.",
        }
    if advisory_ready:
        return {
            "key": "advisory_only",
            "label": "Advisory only",
            "detail": "The symbol is usable for operator-facing advisory work, but it has not earned shadow, paper, or live autonomy rights.",
        }
    return {
        "key": "trained_validation_pending",
        "label": "Trained, validation pending",
        "detail": str(symbol_maturity.get("detail") or "The symbol has not yet earned advisory-usable maturity."),
    }


def summarize_deployment_readiness(
    instrument_key: str,
    state: dict | None = None,
    system_report: dict | None = None,
    news_health: dict | None = None,
) -> dict:
    sym = str(instrument_key or "").upper()
    payload = dict(state or {})
    evaluation = dict(payload.get("evaluation") or {})
    metrics = dict(evaluation.get("metrics") or payload.get("metrics") or {})
    thresholds = dict(evaluation.get("thresholds") or payload.get("thresholds") or {})
    system = dict(system_report or payload.get("system_report") or payload.get("system_checks") or {})
    news = dict(news_health or payload.get("news_health") or {})
    artifact_summary = {
        "prices": _price_artifact_summary(sym, system_report=system),
        "models": _model_artifact_summary(sym),
        "news": _news_artifact_summary(sym, news_health=news),
    }
    history_summary = _history_summary(sym)
    observation_summary = _observation_summary(sym)
    stage_history_summary = _stage_history_summary(sym)
    calibration_summary = _calibration_summary(sym, system_report=system)
    onboarding_summary = _onboarding_summary(sym)
    price_available = bool((artifact_summary.get("prices") or {}).get("available"))
    core_models_ready = bool((artifact_summary.get("models") or {}).get("core_4min_ready"))
    any_models = bool((artifact_summary.get("models") or {}).get("has_any_models"))
    training_plan_guidance = _training_plan_guidance(onboarding_summary, core_models_ready=core_models_ready)
    model_runtime_guidance = _model_runtime_guidance(onboarding_summary, core_models_ready=core_models_ready)
    validation_evidence = _validation_evidence_from_sources(
        sym,
        state=payload,
        onboarding_summary=onboarding_summary,
        calibration_summary=calibration_summary,
    )

    blockers: list[str] = []
    monitor_flags: list[str] = []
    stage_key = "trained_not_validated"

    instrument_known = sym in INSTRUMENTS
    if not instrument_known:
        blockers.append("instrument is not registered in the runtime registry")

    readiness_error = str(payload.get("error") or evaluation.get("error") or "").strip()
    if readiness_error:
        blockers.append(readiness_error)

    fresh = bool(payload.get("fresh", False))
    live_ready = bool(payload.get("live_ready", False))
    pass_streak = int(payload.get("pass_streak_days") or 0)
    required_streak = max(1, int(payload.get("required_streak_days") or READINESS_MIN_PASS_STREAK_DAYS))

    hscore = _safe_float(metrics.get("horizon_weighted_score"))
    intr_hit = _safe_float(metrics.get("intraday_window_hit_rate"))
    day_hit = _safe_float(metrics.get("daily_window_hit_rate"))
    week_hit = _safe_float(metrics.get("weekend_window_hit_rate"))
    has_metrics = bool(metrics)

    system_status = str(system.get("status") or "").strip().lower()
    system_is_operational = system_status in {"", "ok"}
    models_missing = list(((system.get("models") or {}).get("missing") or []))
    if models_missing:
        blockers.append(f"missing models: {', '.join(str(x) for x in models_missing)}")
    if not price_available:
        blockers.append(str((artifact_summary.get("prices") or {}).get("error") or "no price data"))
    elif not core_models_ready:
        specific_training_blocker = str(
            training_plan_guidance.get("primary_blocker")
            or model_runtime_guidance.get("primary_blocker")
            or ""
        ).strip()
        if specific_training_blocker:
            blockers.append(specific_training_blocker)
        else:
            monitor_flags.append("core 4-minute model artifacts are incomplete")
    elif str(training_plan_guidance.get("monitor_flag") or "").strip():
        monitor_flags.append(str(training_plan_guidance.get("monitor_flag") or "").strip())
    if core_models_ready and str(model_runtime_guidance.get("monitor_flag") or "").strip():
        monitor_flags.append(str(model_runtime_guidance.get("monitor_flag") or "").strip())

    data_block = dict(system.get("data") or {})
    staleness = dict(data_block.get("staleness") or {})
    if bool(staleness.get("is_stale")):
        monitor_flags.append("cached market data is stale")

    pred_diag = dict((((system.get("predictions") or {}).get("4min") or {}).get("diagnostics") or {}))
    critical_stale = bool(pred_diag.get("critical_stale", False))
    readiness_locked = bool(pred_diag.get("readiness_locked", False))
    diversity = dict(pred_diag.get("diversity") or {})
    degenerate = bool(diversity.get("degenerate", False))
    if critical_stale:
        blockers.append("critical stale-data guard is active")
    if readiness_locked:
        blockers.append("readiness lock is still active")
    if degenerate:
        monitor_flags.append("4-minute model spread is degenerate")

    calibration = dict(system.get("calibration") or {})
    if bool(calibration.get("stale")):
        monitor_flags.append("calibration stats are stale")
    if bool(calibration.get("drift_flag")):
        blockers.append("calibration drift flag is active")
    if bool(calibration_summary.get("evidence_stale")):
        monitor_flags.append("calibration evidence is stale")

    horizon_accuracy = dict(system.get("horizon_accuracy") or {})
    if bool(horizon_accuracy.get("drift_flag")):
        blockers.append("daily/weekend drift flag is active")

    news_status = str(news.get("status") or "").strip().lower()
    headline_state = str(news.get("headline_state") or news_status).strip().lower()
    if headline_state in {"failed", "stale", "error"} or news_status in {"failed", "stale", "error"}:
        blockers.append("news pipeline is degraded")
    elif headline_state in {"missing", "thin"} or news_status in {"missing", "degraded"}:
        monitor_flags.append("news context is thin or degraded")

    for validation_message in _validation_blocker_messages(validation_evidence):
        monitor_flags.append(validation_message)

    if not fresh and payload:
        blockers.append("readiness state is stale")
    legacy_live_ready = bool(live_ready and fresh and not has_metrics and not models_missing and not readiness_error)
    if not has_metrics and core_models_ready and not models_missing and not readiness_error and not legacy_live_ready:
        blockers.append("readiness metrics are not available yet")

    rollout_symbols = {str(item or "").strip().upper() for item in AUTONOMY_SUPPORTED_SYMBOLS if str(item or "").strip()}
    rollout_actual_allowed = sym in rollout_symbols

    min_hscore = float(thresholds.get("min_horizon_score") or READINESS_MIN_HORIZON_SCORE)
    min_intr_hit = float(thresholds.get("min_intraday_window_hit_rate") or READINESS_MIN_INTRADAY_WINDOW_HIT_RATE)
    min_day_hit = float(thresholds.get("min_daily_window_hit_rate") or READINESS_MIN_DAILY_WINDOW_HIT_RATE)
    min_week_hit = float(thresholds.get("min_weekend_window_hit_rate") or READINESS_MIN_WEEKEND_WINDOW_HIT_RATE)

    stage_key = "supported_not_trained"
    if instrument_known and price_available and core_models_ready and not models_missing and not readiness_error:
        stage_key = "trained_not_validated"
    if has_metrics and instrument_known and price_available and core_models_ready and not models_missing and not readiness_error:
        stage_key = "advisory_only"
    if (
        has_metrics
        and instrument_known
        and price_available
        and core_models_ready
        and not models_missing
        and not readiness_error
        and fresh
        and not critical_stale
        and system_status != "fail"
    ):
        stage_key = "advisory_shadow_ready"

    paper_candidate = bool(
        (live_ready or legacy_live_ready)
        and fresh
        and instrument_known
        and price_available
        and core_models_ready
        and not models_missing
        and not critical_stale
        and system_is_operational
        and not bool(calibration.get("stale"))
        and not bool(calibration.get("drift_flag"))
        and not bool(horizon_accuracy.get("drift_flag"))
        and headline_state not in {"failed", "stale", "error"}
    )
    if has_metrics:
        paper_candidate = bool(
            paper_candidate
            and hscore is not None
            and hscore >= min_hscore
            and intr_hit is not None
            and intr_hit >= min_intr_hit
            and day_hit is not None
            and day_hit >= min_day_hit
            and week_hit is not None
            and week_hit >= min_week_hit
        )
    if paper_candidate:
        stage_key = "paper_autonomy_candidate"

    controlled_candidate = bool(
        paper_candidate
        and pass_streak >= (required_streak + 2)
    )
    if has_metrics:
        controlled_candidate = bool(
            controlled_candidate
            and hscore is not None
            and hscore >= max(min_hscore + 0.08, 0.70)
            and intr_hit is not None
            and intr_hit >= min(min_intr_hit + 0.05, 0.99)
            and day_hit is not None
            and day_hit >= min(min_day_hit + 0.05, 0.99)
            and week_hit is not None
            and week_hit >= min(min_week_hit + 0.05, 0.99)
        )
    if controlled_candidate:
        stage_key = "controlled_real_world_candidate"

    validation_stage_cap = _validation_stage_cap(validation_evidence)
    if _DEPLOYMENT_STAGE_ORDER.get(stage_key, 1) > _DEPLOYMENT_STAGE_ORDER.get(validation_stage_cap, 1):
        stage_key = validation_stage_cap

    stage_rank = int(_DEPLOYMENT_STAGE_ORDER.get(stage_key, 1))
    advisory_ready = bool(stage_rank >= _DEPLOYMENT_STAGE_ORDER["advisory_only"])
    shadow_ready = bool(stage_rank >= _DEPLOYMENT_STAGE_ORDER["advisory_shadow_ready"])
    paper_ready = bool(stage_rank >= _DEPLOYMENT_STAGE_ORDER["paper_autonomy_candidate"])
    live_ready_candidate = bool(stage_rank >= _DEPLOYMENT_STAGE_ORDER["controlled_real_world_candidate"])

    if not live_ready and has_metrics:
        blockers.append("pass streak / readiness gate is not yet satisfied")
    if has_metrics and hscore is not None and hscore < min_hscore:
        blockers.append(f"horizon score is below the deployment floor ({hscore:.2f} < {min_hscore:.2f})")
    if has_metrics and intr_hit is not None and intr_hit < min_intr_hit:
        blockers.append(f"4-minute window hit rate is too weak ({intr_hit:.0%})")
    if has_metrics and day_hit is not None and day_hit < min_day_hit:
        blockers.append(f"daily window hit rate is too weak ({day_hit:.0%})")
    if has_metrics and week_hit is not None and week_hit < min_week_hit:
        blockers.append(f"weekend window hit rate is too weak ({week_hit:.0%})")
    if system_status == "warn":
        monitor_flags.append("system check has warnings")
    elif system_status == "fail":
        blockers.append("system check is failing")

    clean_blockers = []
    seen = set()
    for item in blockers:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        clean_blockers.append(text)
    blockers = clean_blockers

    clean_flags = []
    seen_flags = set()
    for item in monitor_flags:
        text = str(item or "").strip()
        key = text.lower()
        if not text or key in seen_flags:
            continue
        seen_flags.add(key)
        clean_flags.append(text)
    monitor_flags = clean_flags

    primary_gap = blockers[0] if blockers else ""
    input_fresh = bool(
        not critical_stale
        and not bool((artifact_summary.get("prices") or {}).get("stale"))
        and headline_state not in {"failed", "stale", "error", "missing"}
        and not bool(calibration_summary.get("evidence_stale"))
    )
    symbol_maturity = _symbol_maturity_state(
        instrument_known=instrument_known,
        onboarding_summary=onboarding_summary,
        price_available=price_available,
        any_models=any_models,
        core_models_ready=core_models_ready,
        has_metrics=has_metrics,
        advisory_ready=advisory_ready,
        shadow_ready=shadow_ready,
        paper_ready=paper_ready,
        live_ready_candidate=live_ready_candidate,
    )
    evidence_state = _evidence_state(
        price_available=price_available,
        core_models_ready=core_models_ready,
        has_metrics=has_metrics,
        fresh=fresh,
        critical_stale=critical_stale,
        price_stale=bool((artifact_summary.get("prices") or {}).get("stale")),
        headline_state=headline_state,
        news_status=news_status,
        calibration_drift=bool(calibration.get("drift_flag")),
        horizon_drift=bool(horizon_accuracy.get("drift_flag")),
        calibration_evidence_stale=bool(calibration_summary.get("evidence_stale")),
        input_fresh=input_fresh,
        blockers=blockers,
        monitor_flags=monitor_flags,
    )
    operator_state = _operator_state(
        instrument_known=instrument_known,
        onboarding_summary=onboarding_summary,
        price_available=price_available,
        core_models_ready=core_models_ready,
        has_metrics=has_metrics,
        advisory_ready=advisory_ready,
        shadow_ready=shadow_ready,
        paper_ready=paper_ready,
        live_ready_candidate=live_ready_candidate,
        rollout_actual_allowed=rollout_actual_allowed,
        evidence_state_key=str(evidence_state.get("key") or ""),
        symbol_maturity=symbol_maturity,
    )

    rollout_label = "Shadow only rollout"
    rollout_detail = "Actual autonomy remains disabled for this symbol in the rollout list."
    if rollout_actual_allowed and live_ready_candidate:
        rollout_label = "Live rollout eligible"
        rollout_detail = "The symbol is on the rollout list and meets the current controlled-live candidate stage."
    elif rollout_actual_allowed and paper_ready:
        rollout_label = "Paper rollout eligible"
        rollout_detail = "The symbol is on the rollout list and meets the paper-autonomy candidate stage."
    elif rollout_actual_allowed:
        rollout_label = "Rollout listed / still gated"
        rollout_detail = "The symbol is on the rollout list, but the deployment stage is still below actual-autonomy readiness."

    base = {
        "instrument": sym,
        "stage_key": stage_key,
        "stage_label": _deployment_stage_label(stage_key),
        "stage_rank": stage_rank,
        "summary": "",
        "detail": "",
        "primary_blocker": primary_gap,
        "blockers": blockers,
        "monitor_flags": monitor_flags,
        "advisory_ready": advisory_ready,
        "shadow_ready": shadow_ready,
        "paper_ready": paper_ready,
        "live_ready_candidate": live_ready_candidate,
        "rollout_actual_allowed": rollout_actual_allowed,
        "actual_paper_allowed": bool(rollout_actual_allowed and paper_ready),
        "actual_live_allowed": bool(rollout_actual_allowed and live_ready_candidate),
        "rollout_label": rollout_label,
        "rollout_detail": rollout_detail,
        "system_status": system_status or "unknown",
        "news_status": headline_state or news_status or "unknown",
        "critical_stale": critical_stale,
        "fresh": fresh,
        "live_ready": live_ready,
        "input_fresh": input_fresh,
        "validation_evidence": validation_evidence,
        "validation_status": str(validation_evidence.get("validation_status") or "missing"),
        "validation_method": str(validation_evidence.get("validation_method") or "none"),
        "validation_summary": str(validation_evidence.get("summary") or ""),
        "validation_stage_cap": validation_stage_cap,
        "status_flags": _status_flags(
            instrument_known=instrument_known,
            price_available=price_available,
            core_models_ready=core_models_ready,
            has_metrics=has_metrics,
            advisory_ready=advisory_ready,
            shadow_ready=shadow_ready,
            paper_ready=paper_ready,
            live_ready_candidate=live_ready_candidate,
            rollout_actual_allowed=rollout_actual_allowed,
        ),
        "symbol_maturity_key": str(symbol_maturity.get("key") or ""),
        "symbol_maturity_label": str(symbol_maturity.get("label") or ""),
        "symbol_maturity_detail": str(symbol_maturity.get("detail") or ""),
        "evidence_state_key": str(evidence_state.get("key") or ""),
        "evidence_state_label": str(evidence_state.get("label") or ""),
        "evidence_state_detail": str(evidence_state.get("detail") or ""),
        "operator_state_key": str(operator_state.get("key") or ""),
        "operator_state_label": str(operator_state.get("label") or ""),
        "operator_state_detail": str(operator_state.get("detail") or ""),
    }
    evidence = build_deployment_evidence(sym, state={**payload, **base}, system_report=system, news_health=news, deployment_summary=base)
    prioritized_missing = _prioritize_requirements(list(evidence.get("missing_evidence") or []))
    prioritized_requirements = _prioritize_requirements(list(evidence.get("next_requirements") or []))
    primary_next_requirement = str(evidence.get("primary_next_requirement") or (prioritized_requirements[0] if prioritized_requirements else ""))
    next_requirement_category = _requirement_category(primary_next_requirement)
    summary_bits = [
        str(base.get("operator_state_label") or "Unknown state"),
        str(evidence.get("decision_state_label") or ""),
        f"Stage {str(base.get('stage_label') or '-')}",
        str(base.get("evidence_state_label") or "Evidence unknown"),
    ]
    detail_bits = [str(base.get("operator_state_detail") or "")]
    if str(evidence.get("decision_state_detail") or "").strip():
        detail_bits.append(str(evidence.get("decision_state_detail") or "").strip())
    if str(validation_evidence.get("summary") or "").strip() and str(validation_evidence.get("validation_status") or "") not in {
        "passed_paper_candidate",
        "passed_autonomy_candidate",
    }:
        detail_bits.append(f"Validation: {str(validation_evidence.get('summary') or '').strip()}")
    if primary_gap and primary_gap.lower() not in str(base.get("operator_state_detail") or "").lower():
        detail_bits.append(f"Blocker: {primary_gap}")
    if primary_next_requirement:
        detail_bits.append(f"Next: {primary_next_requirement}")
    base.update(
        {
            "artifact_summary": evidence.get("artifact_summary") or artifact_summary,
            "calibration_summary": evidence.get("calibration_summary") or calibration_summary,
            "validation_evidence": evidence.get("validation_evidence") or validation_evidence,
            "validation_status": str(evidence.get("validation_status") or validation_evidence.get("validation_status") or "missing"),
            "validation_status_label": str(evidence.get("validation_status_label") or validation_evidence.get("validation_status_label") or ""),
            "validation_method": str(evidence.get("validation_method") or validation_evidence.get("validation_method") or "none"),
            "validation_summary": str(evidence.get("validation_summary") or validation_evidence.get("summary") or ""),
            "validation_blockers": list(evidence.get("validation_blockers") or validation_evidence.get("blockers") or []),
            "validation_next_best_action": str(evidence.get("validation_next_best_action") or validation_evidence.get("next_best_action") or ""),
            "validation_stage_cap": validation_stage_cap,
            "freshness_summary": evidence.get("freshness_summary") or {},
            "history_summary": evidence.get("history_summary") or history_summary,
            "stage_history_summary": evidence.get("stage_history_summary") or stage_history_summary,
            "observation_summary": evidence.get("observation_summary") or observation_summary,
            "onboarding_summary": evidence.get("onboarding_summary") or onboarding_summary,
            "model_runtime_guidance": evidence.get("model_runtime_guidance") or model_runtime_guidance,
            "model_runtime_state_key": str(evidence.get("model_runtime_state_key") or ""),
            "model_runtime_state_label": str(evidence.get("model_runtime_state_label") or ""),
            "model_runtime_state_detail": str(evidence.get("model_runtime_state_detail") or ""),
            "regime_risk_summary": evidence.get("regime_risk_summary") or {},
            "portfolio_context": evidence.get("portfolio_context") or {},
            "risk_guidance": evidence.get("risk_guidance") or {},
            "decision_state_key": str(evidence.get("decision_state_key") or ""),
            "decision_state_label": str(evidence.get("decision_state_label") or ""),
            "decision_state_detail": str(evidence.get("decision_state_detail") or ""),
            "economics_state_key": str(evidence.get("economics_state_key") or ""),
            "economics_state_label": str(evidence.get("economics_state_label") or ""),
            "economics_state_detail": str(evidence.get("economics_state_detail") or ""),
            "blocker_groups": evidence.get("blocker_groups") or {},
            "missing_evidence": prioritized_missing,
            "next_requirements": prioritized_requirements,
            "primary_next_requirement": primary_next_requirement,
            "next_requirement_category": next_requirement_category,
            "checklist": evidence.get("checklist") or {},
            "next_stage_key": evidence.get("next_stage_key"),
            "next_stage_label": evidence.get("next_stage_label"),
            "summary": " | ".join(bit for bit in summary_bits if bit),
            "detail": " | ".join(bit for bit in detail_bits if bit),
        }
    )
    return base


def update_readiness_state(instrument_key: str, online: bool = False) -> dict:
    eval_row = evaluate_readiness_metrics(instrument_key, online=online)
    system_report = {}
    news_health = {}
    try:
        from system_checkup import _check_instrument
        system_report = _check_instrument(instrument_key, quick=True, online=online)
    except Exception:
        system_report = {}
    try:
        from news_pipeline import get_news_health
        news_health = get_news_health(instrument_key)
    except Exception:
        news_health = {}
    if eval_row.get("error"):
        state = {
            "instrument": instrument_key,
            "live_ready": False,
            "last_update": datetime.now(timezone.utc).isoformat(),
            "pass_streak_days": 0,
            "required_streak_days": int(READINESS_MIN_PASS_STREAK_DAYS),
            "error": eval_row.get("error"),
            "system_checks": system_report,
            "news_health": news_health,
        }
        state.update(summarize_deployment_readiness(instrument_key, state=state, system_report=system_report, news_health=news_health))
        _write_state(instrument_key, state)
        _persist_deployment_evidence(instrument_key, state)
        return state

    rows = _read_history(instrument_key)
    # Upsert today's entry.
    today = str(eval_row.get("date"))
    replaced = False
    for i, row in enumerate(rows):
        if str(row.get("date")) == today:
            rows[i] = eval_row
            replaced = True
            break
    if not replaced:
        rows.append(eval_row)
    rows = sorted(rows, key=lambda x: (str(x.get("date", "")), str(x.get("ts", ""))))
    # Keep recent history only.
    rows = rows[-120:]
    _write_history(instrument_key, rows)

    streak = _compute_pass_streak(rows)
    required = max(1, int(READINESS_MIN_PASS_STREAK_DAYS))
    live_ready = bool(streak >= required and bool(eval_row.get("passed", False)))
    state = {
        "instrument": instrument_key,
        "last_update": eval_row.get("ts"),
        "last_date": eval_row.get("date"),
        "last_passed": bool(eval_row.get("passed", False)),
        "pass_streak_days": int(streak),
        "required_streak_days": int(required),
        "live_ready": bool(live_ready),
        "fresh": True,
        "thresholds": eval_row.get("thresholds", {}),
        "checks": eval_row.get("checks", {}),
        "metrics": eval_row.get("metrics", {}),
        "sources": eval_row.get("sources", {}),
        "horizon_live": eval_row.get("horizon_live", {}),
        "horizon_backtest": eval_row.get("horizon_backtest", {}),
        "system_checks": system_report,
        "news_health": news_health,
        "evaluation": {
            "passed": bool(eval_row.get("passed", False)),
            "checks": eval_row.get("checks", {}),
            "metrics": eval_row.get("metrics", {}),
            "sources": eval_row.get("sources", {}),
            "horizon_live": eval_row.get("horizon_live", {}),
            "horizon_backtest": eval_row.get("horizon_backtest", {}),
            "thresholds": eval_row.get("thresholds", {}),
        },
    }
    state.update(summarize_deployment_readiness(instrument_key, state=state, system_report=system_report, news_health=news_health))
    _write_state(instrument_key, state)
    _persist_deployment_evidence(instrument_key, state)
    return state


def load_readiness_state(instrument_key: str) -> dict | None:
    path = _state_path(instrument_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            state = json.load(f)
        state["fresh"] = _is_state_fresh(state)
        if not bool(state["fresh"]):
            state["live_ready"] = False
        state.update(summarize_deployment_readiness(instrument_key, state=state))
        return state
    except Exception:
        return None
