from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from config import DATA_DIR
from data_hub import get_data_hub
from instrument_onboarding import register_instrument
from instrument_profile import build_instrument_meta, normalize_symbol
from model_zoo import build_model_runtime, summarize_model_stack, train_baseline_models
from symbol_pipeline import MODE_REQUIREMENTS, assess_symbol_history, build_training_plan, ensure_daily_long_history
from workspace_state import open_workspace


ONBOARDING_STAGE_ORDER = [
    ("validate_symbol", "Validate symbol"),
    ("resolve_profile", "Resolve profile"),
    ("stage_market_data", "Stage market data"),
    ("build_feature_readiness", "Build feature readiness"),
    ("train_baseline_models", "Train baseline models"),
    ("refresh_calibration", "Refresh calibration"),
    ("refresh_readiness", "Refresh readiness"),
    ("capture_deployment_evidence", "Capture deployment evidence"),
    ("finalize", "Finalize workspace readiness"),
]

_STAGE_STATUS_KEYS = ("completed", "running", "failed", "skipped", "partial", "blocked", "deferred", "pending")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"onboarding_state_{normalize_symbol(symbol)}.json")


def _history_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"onboarding_history_{normalize_symbol(symbol)}.jsonl")


def _stage_status_template(pending: int = 0) -> dict:
    payload = {key: 0 for key in _STAGE_STATUS_KEYS}
    payload["pending"] = int(max(0, pending))
    return payload


def _save_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def _append_jsonl(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as handle:
        handle.write(json.dumps(payload, sort_keys=True))
        handle.write("\n")


def load_onboarding_state(symbol: str) -> dict:
    path = _state_path(symbol)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def load_onboarding_history(symbol: str, limit: int | None = None) -> list[dict]:
    path = _history_path(symbol)
    if not os.path.exists(path):
        return []
    rows = []
    try:
        with open(path, "r") as handle:
            for raw in handle:
                raw = str(raw or "").strip()
                if not raw:
                    continue
                try:
                    item = json.loads(raw)
                except Exception:
                    continue
                if isinstance(item, dict):
                    rows.append(item)
    except Exception:
        return []
    if limit is not None and int(limit) > 0:
        return rows[-int(limit) :]
    return rows


def summarize_onboarding_state(symbol: str) -> dict:
    sym = normalize_symbol(symbol)
    state = load_onboarding_state(sym)
    if not state:
        return {
            "symbol": sym,
            "status": "not_started",
            "status_label": "Not started",
            "current_stage_key": "",
            "current_stage_label": "Not started",
            "progress_completed": 0,
            "progress_total": len(ONBOARDING_STAGE_ORDER),
            "summary": "No staged onboarding job has run for this symbol yet.",
            "last_error": "",
            "artifacts": {},
            "milestones": {},
            "completion_flags": {},
            "stage_status_counts": _stage_status_template(len(ONBOARDING_STAGE_ORDER)),
            "next_stage_key": ONBOARDING_STAGE_ORDER[0][0],
            "next_stage_label": ONBOARDING_STAGE_ORDER[0][1],
            "highest_completed_stage_key": "",
            "highest_completed_stage_label": "",
            "pipeline_outcome_key": "not_started",
            "pipeline_outcome_label": "Not started",
            "primary_next_requirement": "",
        }
    stages = dict(state.get("stages") or {})
    artifacts = dict(state.get("artifacts") or {})
    milestones = dict(state.get("milestones") or {})
    completion_flags = dict(state.get("completion_flags") or {})
    completed = sum(1 for key, _label in ONBOARDING_STAGE_ORDER if str((stages.get(key) or {}).get("status") or "") == "completed")
    stage_status_counts = _stage_status_template(0)
    highest_completed_stage_key = ""
    highest_completed_stage_label = ""
    next_stage_key = ""
    next_stage_label = ""
    for key, label in ONBOARDING_STAGE_ORDER:
        stage_status = str((stages.get(key) or {}).get("status") or "").strip().lower()
        if stage_status in stage_status_counts:
            stage_status_counts[stage_status] += 1
        else:
            stage_status_counts["pending"] += 1
        if stage_status == "completed":
            highest_completed_stage_key = key
            highest_completed_stage_label = label
        if not next_stage_key and stage_status not in {"completed", "skipped"}:
            next_stage_key = key
            next_stage_label = label
    if not next_stage_key:
        next_stage_key = highest_completed_stage_key
        next_stage_label = highest_completed_stage_label
    current_stage_key = str(state.get("current_stage_key") or "")
    current_stage_label = next((label for key, label in ONBOARDING_STAGE_ORDER if key == current_stage_key), current_stage_key.replace("_", " ").title() or "Unknown")
    status = str(state.get("status") or "unknown")
    status_label = {
        "running": "Running",
        "completed": "Completed",
        "failed": "Failed",
        "not_started": "Not started",
    }.get(status, status.replace("_", " ").title())
    detail = str(state.get("summary") or state.get("last_error") or current_stage_label or "-")
    pipeline_outcome_key = "complete_with_gaps" if any(stage_status_counts.get(key) for key in ("partial", "blocked", "deferred")) and status == "completed" else status
    pipeline_outcome_label = {
        "complete_with_gaps": "Completed with gaps",
        "completed": "Completed",
        "running": "Running",
        "failed": "Failed",
        "not_started": "Not started",
    }.get(pipeline_outcome_key, str(pipeline_outcome_key or "unknown").replace("_", " ").title())
    history_assessment = dict(artifacts.get("history_assessment") or {})
    readiness_artifact = dict(artifacts.get("readiness") or {})
    evidence_artifact = dict(artifacts.get("deployment_evidence") or {})
    primary_next_requirement = _best_primary_next_requirement(
        evidence_artifact.get("primary_next_requirement")
        or (list(evidence_artifact.get("next_requirements") or [])[:1] or [""])[0]
        or readiness_artifact.get("primary_next_requirement")
        or (list(readiness_artifact.get("next_requirements") or [])[:1] or [""])[0]
        or history_assessment.get("primary_next_requirement")
        or "",
        state.get("primary_next_requirement"),
    )
    return {
        "symbol": sym,
        "status": status,
        "status_label": status_label,
        "current_stage_key": current_stage_key,
        "current_stage_label": current_stage_label,
        "progress_completed": int(completed),
        "progress_total": int(len(ONBOARDING_STAGE_ORDER)),
        "summary": detail,
        "last_error": str(state.get("last_error") or ""),
        "train_requested": bool(state.get("train_requested")),
        "last_stage_detail": str(((stages.get(current_stage_key) or {}).get("detail") or "")),
        "stages": stages,
        "artifacts": artifacts,
        "milestones": milestones,
        "completion_flags": completion_flags,
        "readiness_stage_label": str(((artifacts.get("deployment_evidence") or {}).get("stage_label") or (artifacts.get("readiness") or {}).get("stage_label") or "")),
        "workspace_registered": bool(state.get("workspace_registered")),
        "stage_status_counts": stage_status_counts,
        "next_stage_key": next_stage_key,
        "next_stage_label": next_stage_label,
        "highest_completed_stage_key": highest_completed_stage_key,
        "highest_completed_stage_label": highest_completed_stage_label,
        "pipeline_outcome_key": pipeline_outcome_key,
        "pipeline_outcome_label": pipeline_outcome_label,
        "primary_next_requirement": primary_next_requirement,
        "resume_count": int(state.get("resume_count") or 0),
        "resumed_from_status": str(state.get("resumed_from_status") or ""),
    }


def _new_state(symbol: str, meta: dict | None = None, train_models: bool = True) -> dict:
    sym = normalize_symbol(symbol)
    return {
        "symbol": sym,
        "status": "running",
        "status_label": "Running",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "meta": dict(meta or {}),
        "train_requested": bool(train_models),
        "current_stage_key": "validate_symbol",
        "current_stage_label": "Validate symbol",
        "stages": {},
        "milestones": {},
        "artifacts": {},
        "completion_flags": {},
        "workspace_registered": False,
        "last_error": "",
        "summary": f"Starting onboarding for {sym}.",
        "primary_next_requirement": "",
    }


def _starting_state(symbol: str, meta: dict | None = None, train_models: bool = True) -> dict:
    sym = normalize_symbol(symbol)
    prior = load_onboarding_state(sym)
    if not prior:
        return _new_state(sym, meta=meta, train_models=train_models)
    state = dict(prior)
    state["symbol"] = sym
    state["status"] = "running"
    state["status_label"] = "Running"
    state["updated_at"] = _utc_now()
    state["train_requested"] = bool(train_models)
    state["current_stage_key"] = "validate_symbol"
    state["current_stage_label"] = "Validate symbol"
    state["last_error"] = ""
    state["summary"] = f"Resuming onboarding for {sym}."
    state["meta"] = {**dict(prior.get("meta") or {}), **dict(meta or {})}
    state["resume_count"] = int(prior.get("resume_count") or 0) + 1
    state["resumed_from_status"] = str(prior.get("status") or "")
    state.setdefault("artifacts", dict(prior.get("artifacts") or {}))
    state.setdefault("milestones", dict(prior.get("milestones") or {}))
    state.setdefault("stages", dict(prior.get("stages") or {}))
    state.setdefault("completion_flags", dict(prior.get("completion_flags") or {}))
    return state


def _persist_state(state: dict):
    sym = normalize_symbol(state.get("symbol"))
    _save_json(_state_path(sym), state)


def _merge_artifact(state: dict, artifact_key: str, payload):
    artifacts = dict(state.get("artifacts") or {})
    if isinstance(payload, dict):
        artifacts[str(artifact_key)] = dict(payload)
    else:
        artifacts[str(artifact_key)] = {"value": payload}
    state["artifacts"] = artifacts
    state["updated_at"] = _utc_now()


def _refresh_completion_flags(state: dict):
    stages = dict(state.get("stages") or {})
    artifacts = dict(state.get("artifacts") or {})

    def _stage_status(stage_key: str) -> str:
        return str((stages.get(stage_key) or {}).get("status") or "")

    flags = {
        "symbol_validated": _stage_status("validate_symbol") == "completed",
        "profile_resolved": _stage_status("resolve_profile") == "completed",
        "workspace_registered": bool(state.get("workspace_registered")),
        "market_data_staged": _stage_status("stage_market_data") == "completed" or bool((artifacts.get("market_data") or {}).get("rows")),
        "feature_readiness_built": _stage_status("build_feature_readiness") in {"completed", "partial"} or bool(
            (artifacts.get("feature_readiness") or {}).get("feature_ready_modes")
        ),
        "models_ready": bool((artifacts.get("model_runtime") or {}).get("summary", {}).get("core_ready")),
        "calibration_ready": _stage_status("refresh_calibration") in {"completed", "skipped"},
        "readiness_evaluated": _stage_status("refresh_readiness") == "completed",
        "deployment_evidence_captured": _stage_status("capture_deployment_evidence") == "completed" or bool(artifacts.get("deployment_evidence")),
    }
    required = [
        "symbol_validated",
        "profile_resolved",
        "workspace_registered",
        "market_data_staged",
        "feature_readiness_built",
        "readiness_evaluated",
        "deployment_evidence_captured",
    ]
    if bool(state.get("train_requested")):
        required.extend(["models_ready", "calibration_ready"])
    flags["honest_completion"] = all(bool(flags.get(key)) for key in required)
    state["completion_flags"] = flags


def _capture_milestone(state: dict, stage_key: str, status: str, detail: str):
    if status not in {"completed", "failed", "skipped", "partial", "blocked", "deferred"}:
        return
    milestones = dict(state.get("milestones") or {})
    milestones[str(stage_key)] = {
        "status": str(status),
        "detail": str(detail or ""),
        "updated_at": _utc_now(),
        "artifact_keys": sorted(list((state.get("artifacts") or {}).keys())),
    }
    state["milestones"] = milestones


def _join_modes(modes: list[str] | None) -> str:
    clean = [str(mode or "").strip() for mode in list(modes or []) if str(mode or "").strip()]
    return ", ".join(clean) if clean else "-"


def _normalized_modes(modes: list[str] | None) -> list[str]:
    clean = []
    seen = set()
    for mode in list(modes or []):
        text = str(mode or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        clean.append(text)
    return clean


def _primary_next_requirement_from_payload(payload) -> str:
    if not isinstance(payload, dict):
        return ""
    primary = str(payload.get("primary_next_requirement") or "").strip()
    if primary:
        return primary
    next_requirements = [str(item or "").strip() for item in list(payload.get("next_requirements") or []) if str(item or "").strip()]
    return next_requirements[0] if next_requirements else ""


def _is_generic_primary_next_requirement(text: str) -> bool:
    return str(text or "").strip().lower() in {
        "resolve the readiness refresh failure.",
    }


def _best_primary_next_requirement(current_text: str | None, prior_text: str | None = None) -> str:
    current = str(current_text or "").strip()
    prior = str(prior_text or "").strip()
    if current and not _is_generic_primary_next_requirement(current):
        return current
    if current and (not prior or current == prior):
        return current
    return prior or current


def _best_mode_list(current_modes: list[str] | None, prior_modes: list[str] | None = None) -> list[str]:
    current = _normalized_modes(current_modes)
    if current:
        return current
    return _normalized_modes(prior_modes)


def _feature_artifact_from_history(history_assessment: dict) -> dict:
    history = dict(history_assessment or {})
    paths = dict(history.get("paths") or {})
    intraday = dict(paths.get("4min") or {})
    daily = dict(paths.get("DailyHigh") or {})
    weekend = dict(paths.get("WeekEnd") or {})
    feature_ready_modes = list(history.get("feature_ready_modes") or [])
    eligible_modes = list(history.get("eligible_modes") or [])
    if not feature_ready_modes:
        status = "blocked"
        summary = "No supported feature path is producing usable rows yet."
    elif set(eligible_modes) >= {"4min", "DailyHigh", "WeekEnd"}:
        status = "completed"
        summary = "Intraday, daily, and weekend feature paths are currently training-eligible."
    else:
        status = "partial"
        summary = f"Feature coverage exists for {_join_modes(feature_ready_modes)}; training-eligible modes: {_join_modes(eligible_modes)}."
    return {
        "status": status,
        "summary": summary,
        "feature_ready_modes": feature_ready_modes,
        "eligible_modes": eligible_modes,
        "intraday_rows": int(intraday.get("feature_rows") or 0),
        "daily_rows": int(daily.get("feature_rows") or 0),
        "weekend_rows": int(weekend.get("feature_rows") or 0),
        "paths": paths,
        "primary_next_requirement": str(history.get("primary_next_requirement") or ""),
    }


def _training_plan_artifact(training_plan: dict | None, prior_artifact: dict | None = None) -> dict:
    plan = dict(training_plan or {})
    prior = dict(prior_artifact or {})
    eligible_modes = _best_mode_list(plan.get("eligible_modes"), prior.get("eligible_modes"))
    deferred_modes = _best_mode_list(plan.get("deferred_modes"), prior.get("deferred_modes"))
    blocked_modes = _best_mode_list(plan.get("blocked_modes"), prior.get("blocked_modes"))
    retained_fields = []
    for field, current, merged in (
        ("eligible_modes", plan.get("eligible_modes"), eligible_modes),
        ("deferred_modes", plan.get("deferred_modes"), deferred_modes),
        ("blocked_modes", plan.get("blocked_modes"), blocked_modes),
    ):
        if not _normalized_modes(current) and merged:
            retained_fields.append(field)
    return {
        "status": str(plan.get("status") or ""),
        "summary": str(plan.get("summary") or ""),
        "primary_next_requirement": str(plan.get("primary_next_requirement") or ""),
        "eligible_modes": eligible_modes,
        "deferred_modes": deferred_modes,
        "blocked_modes": blocked_modes,
        "retained_prior_fields": retained_fields,
        "retained_from_prior_run": bool(retained_fields),
    }


def _model_runtime_artifact(
    *,
    runtime: dict | None,
    training_plan: dict,
    available_modes: list[str] | None,
    prior_artifact: dict | None = None,
) -> dict:
    prior_model_artifact = dict(prior_artifact or {})
    requested_modes = list(training_plan.get("eligible_modes") or [])
    deferred_modes = list(training_plan.get("deferred_modes") or [])
    blocked_modes = list(training_plan.get("blocked_modes") or [])
    available = _normalized_modes(available_modes)
    retained_prior_modes = []
    trained_modes = list(available)
    failed_modes = [mode for mode in requested_modes if mode not in trained_modes]
    summary = dict((runtime or {}).get("summary") or {})
    if training_plan.get("status") == "skipped" and not trained_modes:
        retained_prior_modes = _normalized_modes(
            list(prior_model_artifact.get("trained_modes") or [])
            or list(prior_model_artifact.get("available_modes") or [])
        )
        if retained_prior_modes:
            trained_modes = list(retained_prior_modes)
            available = list(retained_prior_modes)
            prior_summary = dict(prior_model_artifact.get("summary") or {})
            if prior_summary and not summary:
                summary = prior_summary
    if trained_modes and not list(summary.get("available_modes") or []):
        summary["available_modes"] = list(trained_modes)
    mode_requests = dict(training_plan.get("mode_requests") or {})
    mode_outcomes = {}
    for mode in MODE_REQUIREMENTS:
        request = dict(mode_requests.get(mode) or {})
        outcome_status = str(request.get("status") or "blocked")
        outcome_reason = str(request.get("reason") or "")
        if mode in trained_modes:
            outcome_status = "trained"
            outcome_reason = f"{mode} artifacts are now available in the local model stack."
        elif mode in failed_modes:
            outcome_status = "failed"
            outcome_reason = f"{mode} training was attempted but no usable artifacts were available afterward."
        mode_outcomes[mode] = {
            "status": outcome_status,
            "label": str(request.get("label") or mode),
            "reason": outcome_reason,
            "feature_rows": int(request.get("feature_rows") or 0),
            "min_train_rows": int(request.get("min_train_rows") or 0),
        }
    if not trained_modes and requested_modes:
        status = "failed"
        detail = f"Training attempted for {_join_modes(requested_modes)}, but no usable model artifacts were detected."
    elif trained_modes and not requested_modes and training_plan.get("status") == "skipped":
        status = "skipped"
        if retained_prior_modes:
            detail = f"Training skipped for this pass; prior onboarding model coverage was retained: {_join_modes(trained_modes)}."
        else:
            detail = f"Training skipped for this pass; existing model coverage was retained: {_join_modes(trained_modes)}."
    elif trained_modes and not requested_modes and training_plan.get("status") in {"blocked", "deferred"}:
        status = "partial"
        detail = f"Existing model coverage {_join_modes(trained_modes)} was retained while additional paths remain deferred or blocked."
    elif trained_modes and not deferred_modes and not blocked_modes and not failed_modes:
        status = "completed"
        detail = f"Trained model coverage: {_join_modes(trained_modes)}."
    elif trained_modes:
        status = "partial"
        detail = f"Trained {_join_modes(trained_modes)} while {_join_modes(deferred_modes + blocked_modes + failed_modes)} remains deferred, blocked, or failed."
    elif training_plan.get("status") == "deferred":
        status = "deferred"
        detail = str(training_plan.get("summary") or "Training is deferred until more history is staged.")
    elif training_plan.get("status") == "skipped":
        status = "skipped"
        detail = str(training_plan.get("summary") or "Training was skipped for this pass.")
    else:
        status = "blocked"
        detail = str(training_plan.get("summary") or "Training is blocked by missing symbol-specific prerequisites.")
    return {
        "family": str((runtime or {}).get("family") or summary.get("family") or "tabular_baseline_v2"),
        "trained_at": (runtime or {}).get("trained_at"),
        "requested_modes": requested_modes,
        "trained_modes": trained_modes,
        "failed_modes": failed_modes,
        "deferred_modes": deferred_modes,
        "blocked_modes": blocked_modes,
        "attempted_modes": requested_modes,
        "available_modes": available,
        "retained_prior_modes": retained_prior_modes,
        "retained_from_prior_run": bool(retained_prior_modes),
        "mode_outcomes": mode_outcomes,
        "summary": summary,
        "status": status,
        "detail": detail,
        "primary_next_requirement": str(training_plan.get("primary_next_requirement") or ""),
    }


def _persist_artifact(state: dict, artifact_key: str, payload):
    state["primary_next_requirement"] = _best_primary_next_requirement(
        _primary_next_requirement_from_payload(payload),
        state.get("primary_next_requirement"),
    )
    _merge_artifact(state, artifact_key, payload)
    _refresh_completion_flags(state)
    _persist_state(state)


def _update_stage(state: dict, stage_key: str, status: str, detail: str, progress_cb=None):
    stage_label = next((label for key, label in ONBOARDING_STAGE_ORDER if key == stage_key), stage_key.replace("_", " ").title())
    stages = dict(state.get("stages") or {})
    entry = dict(stages.get(stage_key) or {})
    if status == "running":
        entry["started_at"] = _utc_now()
    entry["status"] = str(status)
    entry["detail"] = str(detail or "")
    entry["updated_at"] = _utc_now()
    if status in {"completed", "failed", "skipped", "partial", "blocked", "deferred"}:
        entry["finished_at"] = _utc_now()
    stages[stage_key] = entry
    state["stages"] = stages
    state["current_stage_key"] = stage_key
    state["current_stage_label"] = stage_label
    state["updated_at"] = _utc_now()
    state["summary"] = f"{stage_label}: {detail}".strip(": ")
    if status == "failed":
        state["status"] = "failed"
        state["status_label"] = "Failed"
        state["last_error"] = str(detail or stage_label)
    elif status == "running":
        state["status"] = "running"
        state["status_label"] = "Running"
    _capture_milestone(state, stage_key, status, detail)
    _refresh_completion_flags(state)
    _persist_state(state)
    _append_jsonl(
        _history_path(state.get("symbol")),
        {
            "ts": _utc_now(),
            "symbol": normalize_symbol(state.get("symbol")),
            "stage_key": stage_key,
            "stage_label": stage_label,
            "status": status,
            "detail": str(detail or ""),
        },
    )
    if progress_cb:
        try:
            progress_cb(stage_key, stage_label, dict(state))
        except Exception:
            pass


def _complete_job(state: dict, detail: str):
    state["status"] = "completed"
    state["status_label"] = "Completed"
    state["summary"] = str(detail or "Onboarding complete")
    state["updated_at"] = _utc_now()
    _refresh_completion_flags(state)
    _persist_state(state)
    _append_jsonl(
        _history_path(state.get("symbol")),
        {
            "ts": _utc_now(),
            "symbol": normalize_symbol(state.get("symbol")),
            "stage_key": "complete",
            "stage_label": "Complete",
            "status": "completed",
            "detail": str(detail or ""),
        },
    )


def run_onboarding_job(
    symbol: str,
    *,
    meta: dict | None = None,
    train_models: bool = True,
    progress_cb=None,
    data_hub=None,
) -> dict:
    sym = normalize_symbol(symbol)
    if not sym:
        raise ValueError("A symbol is required for onboarding.")
    hub = data_hub or get_data_hub()
    state = _starting_state(sym, meta=meta, train_models=train_models)
    _persist_state(state)

    _update_stage(state, "validate_symbol", "running", f"Validating {sym}", progress_cb=progress_cb)
    if len(sym) > 20:
        _update_stage(state, "validate_symbol", "failed", "symbol is too long for runtime onboarding", progress_cb=progress_cb)
        return state
    resolved_meta = build_instrument_meta(sym, overrides=meta or {})
    _update_stage(state, "validate_symbol", "completed", f"{sym} accepted for onboarding", progress_cb=progress_cb)

    _update_stage(state, "resolve_profile", "running", "Registering instrument profile", progress_cb=progress_cb)
    register_instrument(sym, meta=resolved_meta, persist=True)
    try:
        open_workspace(sym, meta=resolved_meta, activate=False)
        state["workspace_registered"] = True
    except Exception:
        state["workspace_registered"] = False
    state["meta"] = dict(resolved_meta)
    _persist_artifact(
        state,
        "profile",
        {
            "display_name": str(resolved_meta.get("display_name") or sym),
            "company_name": str(resolved_meta.get("company_name") or resolved_meta.get("display_name") or sym),
            "instrument_kind": str(resolved_meta.get("instrument_kind") or resolved_meta.get("type") or "equity"),
            "sector": str(resolved_meta.get("sector") or "Broad Market"),
            "industry": str(resolved_meta.get("industry") or resolved_meta.get("sector") or "-"),
            "news_aliases": list(resolved_meta.get("news_aliases") or []),
            "context_daily_symbols": dict(resolved_meta.get("context_daily_symbols") or {}),
        },
    )
    _update_stage(
        state,
        "resolve_profile",
        "completed",
        f"{resolved_meta.get('instrument_kind') or resolved_meta.get('type') or 'equity'} | {resolved_meta.get('sector') or 'Broad Market'}",
        progress_cb=progress_cb,
    )

    _update_stage(state, "stage_market_data", "running", "Refreshing canonical market data", progress_cb=progress_cb)
    bundle = hub.refresh_symbol(sym, force=True, reason="onboarding_stage_data", load_models=False)
    df = bundle.get("df")
    if df is None or getattr(df, "empty", True):
        _update_stage(state, "stage_market_data", "failed", "market data refresh did not produce usable price history", progress_cb=progress_cb)
        return state
    try:
        last_ts = pd.to_datetime(df.index[-1], utc=True, errors="coerce")
        last_ts_text = None if pd.isna(last_ts) else last_ts.isoformat()
    except Exception:
        last_ts_text = None
    daily_long_df, daily_long_summary = ensure_daily_long_history(sym, recent_df=df, persist=True)
    history_assessment = assess_symbol_history(df, instrument_key=sym, daily_long_history=daily_long_df)
    _persist_artifact(
        state,
        "market_data",
        {
            "rows": int(len(df)),
            "first_ts": history_assessment.get("first_ts"),
            "last_ts": last_ts_text,
            "data_source": str(bundle.get("data_source") or bundle.get("source") or "cache"),
            "freshness": dict(bundle.get("freshness") or {}),
            "support_status": dict((bundle.get("support_state") or {}).get("components") or {}),
            "span_days": int(history_assessment.get("span_days") or 0),
            "session_count": int(history_assessment.get("session_count") or 0),
            "daily_long_rows": int(history_assessment.get("daily_long_rows") or 0),
            "daily_long_summary": daily_long_summary,
        },
    )
    _persist_artifact(state, "history_assessment", {**history_assessment, "daily_long_summary": daily_long_summary})
    _update_stage(
        state,
        "stage_market_data",
        "completed",
        f"{int(len(df))} bars | daily-long {int(daily_long_summary.get('rows') or 0)} | source {bundle.get('data_source') or 'cache'} | {history_assessment.get('status_label') or 'history assessed'}",
        progress_cb=progress_cb,
    )

    _update_stage(state, "build_feature_readiness", "running", "Building baseline feature readiness", progress_cb=progress_cb)
    feature_artifact = _feature_artifact_from_history(history_assessment)
    _persist_artifact(state, "feature_readiness", feature_artifact)
    _update_stage(
        state,
        "build_feature_readiness",
        str(feature_artifact.get("status") or "completed"),
        str(feature_artifact.get("summary") or "Feature readiness assessed"),
        progress_cb=progress_cb,
    )

    training_plan = build_training_plan(history_assessment, train_requested=train_models)
    prior_training_plan_artifact = dict((state.get("artifacts") or {}).get("training_plan") or {})
    training_plan_artifact = _training_plan_artifact(training_plan, prior_artifact=prior_training_plan_artifact)
    _persist_artifact(state, "training_plan", training_plan_artifact)
    existing_runtime = build_model_runtime(sym, include_modes=True)
    if train_models:
        _update_stage(state, "train_baseline_models", "running", "Training default tabular baseline stack", progress_cb=progress_cb)
        runtime = dict(existing_runtime or {})
        model_artifact = {}
        try:
            if list(training_plan.get("eligible_raw_modes") or []):
                runtime = train_baseline_models(
                    df,
                    sym,
                    train_horizons="4min" in list(training_plan.get("eligible_raw_modes") or []),
                    modes_filter=list(training_plan.get("eligible_raw_modes") or []),
                    progress_cb=lambda msg: progress_cb("train_baseline_models", "Train baseline models", {**state, "summary": str(msg)}) if progress_cb else None,
                )
            runtime = dict(runtime or {})
            model_summary = summarize_model_stack(dict(runtime.get("models") or dict((existing_runtime or {}).get("models") or {})))
            runtime.setdefault("summary", model_summary)
            model_artifact = _model_runtime_artifact(
                runtime=runtime,
                training_plan=training_plan,
                available_modes=list(model_summary.get("available_modes") or []),
            )
        except Exception as exc:
            model_artifact = _model_runtime_artifact(
                runtime={"family": "tabular_baseline_v2", "summary": {"family": "tabular_baseline_v2", "core_ready": False, "available_modes": []}},
                training_plan=training_plan,
                available_modes=[],
            )
            model_artifact["status"] = "failed"
            model_artifact["detail"] = f"Model training failed unexpectedly: {exc}"
        _update_stage(
            state,
            "train_baseline_models",
            str(model_artifact.get("status") or "completed"),
            str(model_artifact.get("detail") or "Model runtime assessed"),
            progress_cb=progress_cb,
        )
        _persist_artifact(state, "model_runtime", model_artifact)
        if str(model_artifact.get("status")) == "failed":
            return state

        _update_stage(state, "refresh_calibration", "running", "Refreshing calibration and horizon evidence", progress_cb=progress_cb)
        if not bool((model_artifact.get("summary") or {}).get("core_ready")):
            prior_calibration = dict((state.get("artifacts") or {}).get("calibration") or {})
            calibration_artifact = {
                **prior_calibration,
                "status": "deferred",
                "reason": "Calibration is deferred until the core 4-minute model stack is trained for this symbol.",
                "eligible": False,
                "intraday": dict(prior_calibration.get("intraday") or {}),
                "horizons": dict(prior_calibration.get("horizons") or {}),
            }
            _persist_artifact(state, "calibration", calibration_artifact)
            _update_stage(
                state,
                "refresh_calibration",
                "deferred",
                str(calibration_artifact.get("reason") or "Calibration deferred"),
                progress_cb=progress_cb,
            )
        else:
            from calibration_tracker import update_live_calibration
            from horizon_accuracy_tracker import update_horizon_accuracy

            cal = update_live_calibration(sym, horizon_minutes=4)
            horiz = update_horizon_accuracy(sym)
            cal_status = str((cal or {}).get("status") or (cal or {}).get("last_update") or "updated")
            horiz_status = str((horiz or {}).get("status") or (horiz or {}).get("last_update") or "updated")
            _persist_artifact(
                state,
                "calibration",
                {
                    "intraday": dict(cal or {}),
                    "horizons": dict(horiz or {}),
                    "status": "updated",
                    "eligible": True,
                },
            )
            _update_stage(
                state,
                "refresh_calibration",
                "completed",
                f"calibration {cal_status} | horizon {horiz_status}",
                progress_cb=progress_cb,
            )
    else:
        prior_model_artifact = dict((state.get("artifacts") or {}).get("model_runtime") or {})
        model_artifact = _model_runtime_artifact(
            runtime=dict(existing_runtime or {}),
            training_plan=training_plan,
            available_modes=list(((existing_runtime or {}).get("summary") or {}).get("available_modes") or []),
            prior_artifact=prior_model_artifact,
        )
        _persist_artifact(state, "model_runtime", model_artifact)
        _update_stage(state, "train_baseline_models", "skipped", "Training skipped; workspace remains support-only until models are built.", progress_cb=progress_cb)
        prior_calibration = dict((state.get("artifacts") or {}).get("calibration") or {})
        _persist_artifact(
            state,
            "calibration",
            {
                **prior_calibration,
                "status": "skipped",
                "reason": "Calibration refresh was skipped because training did not run in this pass; prior calibration artifacts were preserved when available.",
                "eligible": False,
                "intraday": dict(prior_calibration.get("intraday") or {}),
                "horizons": dict(prior_calibration.get("horizons") or {}),
            },
        )
        _update_stage(state, "refresh_calibration", "skipped", "Calibration skipped because training did not run.", progress_cb=progress_cb)

    _update_stage(state, "refresh_readiness", "running", "Refreshing deployment evidence", progress_cb=progress_cb)
    from readiness_tracker import update_readiness_state

    readiness = {}
    try:
        readiness = update_readiness_state(sym, online=False)
        readiness_artifact = {
            "status": "updated",
            "stage_key": str(readiness.get("stage_key") or ""),
            "stage_label": str(readiness.get("stage_label") or ""),
            "summary": str(readiness.get("summary") or ""),
            "primary_blocker": str(readiness.get("primary_blocker") or ""),
            "primary_next_requirement": str(readiness.get("primary_next_requirement") or ""),
            "next_requirements": list(readiness.get("next_requirements") or []),
            "operator_state_label": str(readiness.get("operator_state_label") or ""),
            "evidence_state_label": str(readiness.get("evidence_state_label") or ""),
            "symbol_maturity_label": str(readiness.get("symbol_maturity_label") or ""),
            "rollout_label": str(readiness.get("rollout_label") or ""),
            "status_flags": dict(readiness.get("status_flags") or {}),
            "validation_evidence": dict(readiness.get("validation_evidence") or {}),
            "validation_status": str(readiness.get("validation_status") or ""),
            "validation_method": str(readiness.get("validation_method") or ""),
            "validation_summary": str(readiness.get("validation_summary") or ""),
            "validation_blockers": list(readiness.get("validation_blockers") or []),
        }
        _persist_artifact(state, "readiness", readiness_artifact)
        _update_stage(
            state,
            "refresh_readiness",
            "completed",
            f"{readiness.get('stage_label') or readiness.get('stage_key') or 'evidence refreshed'}",
            progress_cb=progress_cb,
        )
    except Exception as exc:
        readiness = {}
        _persist_artifact(
            state,
            "readiness",
            {
                "status": "blocked",
                "summary": "Readiness refresh failed before evidence could be updated.",
                "primary_blocker": str(exc),
                "next_requirements": [str(training_plan.get("primary_next_requirement") or "Resolve the readiness refresh failure.")],
            },
        )
        _update_stage(state, "refresh_readiness", "blocked", f"Readiness refresh failed: {exc}", progress_cb=progress_cb)

    _update_stage(state, "capture_deployment_evidence", "running", "Capturing deployment evidence summary", progress_cb=progress_cb)
    from readiness_tracker import load_deployment_evidence

    evidence = {}
    try:
        evidence = dict(load_deployment_evidence(sym) or {})
    except Exception:
        evidence = {}
    if not evidence:
        evidence = {
            "stage_key": str(readiness.get("stage_key") or ""),
            "stage_label": str(readiness.get("stage_label") or ""),
            "summary": str(readiness.get("summary") or ""),
            "next_stage_key": str(readiness.get("next_stage_key") or ""),
            "next_stage_label": str(readiness.get("next_stage_label") or ""),
            "missing_evidence": list(readiness.get("missing_evidence") or []),
            "next_requirements": list(readiness.get("next_requirements") or []),
            "primary_next_requirement": str(readiness.get("primary_next_requirement") or ""),
            "checklist": dict(readiness.get("checklist") or {}),
            "operator_state_label": str(readiness.get("operator_state_label") or ""),
            "evidence_state_label": str(readiness.get("evidence_state_label") or ""),
            "symbol_maturity_label": str(readiness.get("symbol_maturity_label") or ""),
            "status_flags": dict(readiness.get("status_flags") or {}),
            "validation_evidence": dict(readiness.get("validation_evidence") or {}),
            "validation_status": str(readiness.get("validation_status") or ""),
            "validation_method": str(readiness.get("validation_method") or ""),
            "validation_summary": str(readiness.get("validation_summary") or ""),
            "validation_blockers": list(readiness.get("validation_blockers") or []),
        }
    evidence_artifact = {
        "stage_key": str(evidence.get("stage_key") or readiness.get("stage_key") or ""),
        "stage_label": str(evidence.get("stage_label") or readiness.get("stage_label") or ""),
        "summary": str(evidence.get("summary") or readiness.get("summary") or ""),
        "next_stage_key": str(evidence.get("next_stage_key") or ""),
        "next_stage_label": str(evidence.get("next_stage_label") or ""),
        "missing_evidence": list(evidence.get("missing_evidence") or []),
        "next_requirements": list(evidence.get("next_requirements") or []),
        "primary_next_requirement": str(evidence.get("primary_next_requirement") or readiness.get("primary_next_requirement") or ""),
        "checklist_summary": dict((evidence.get("checklist") or {}).get("summary") or {}),
        "operator_state_label": str(evidence.get("operator_state_label") or readiness.get("operator_state_label") or ""),
        "evidence_state_label": str(evidence.get("evidence_state_label") or readiness.get("evidence_state_label") or ""),
        "symbol_maturity_label": str(evidence.get("symbol_maturity_label") or readiness.get("symbol_maturity_label") or ""),
        "status_flags": dict(evidence.get("status_flags") or readiness.get("status_flags") or {}),
        "validation_evidence": dict(evidence.get("validation_evidence") or readiness.get("validation_evidence") or {}),
        "validation_status": str(evidence.get("validation_status") or readiness.get("validation_status") or ""),
        "validation_method": str(evidence.get("validation_method") or readiness.get("validation_method") or ""),
        "validation_summary": str(evidence.get("validation_summary") or readiness.get("validation_summary") or ""),
        "validation_blockers": list(evidence.get("validation_blockers") or readiness.get("validation_blockers") or []),
        "generated_at": evidence.get("generated_at"),
    }
    _persist_artifact(state, "deployment_evidence", evidence_artifact)
    _update_stage(
        state,
        "capture_deployment_evidence",
        "completed" if evidence_artifact.get("stage_label") or evidence_artifact.get("summary") else "blocked",
        f"{evidence_artifact.get('stage_label') or readiness.get('stage_label') or 'evidence captured'}",
        progress_cb=progress_cb,
    )

    _update_stage(state, "finalize", "running", "Finalizing onboarding status", progress_cb=progress_cb)
    final_readiness = dict(readiness or {})
    final_stage = str(final_readiness.get("stage_label") or final_readiness.get("stage_key") or "Ready")
    stages = dict(state.get("stages") or {})
    outcome_statuses = [str((stages.get(key) or {}).get("status") or "") for key, _label in ONBOARDING_STAGE_ORDER]
    partial_outcome = any(status in {"partial", "blocked", "deferred"} for status in outcome_statuses)
    final_summary = f"Onboarding complete | {final_stage}"
    if partial_outcome:
        final_summary = f"Onboarding complete with partial maturity | {final_stage}"
    _persist_artifact(
        state,
        "workspace_status",
        {
            "workspace_registered": bool(state.get("workspace_registered")),
            "readiness_stage": final_stage,
            "train_requested": bool(state.get("train_requested")),
            "honest_completion": bool((state.get("completion_flags") or {}).get("honest_completion")),
            "pipeline_outcome": "complete_with_gaps" if partial_outcome else "completed",
            "primary_next_requirement": _best_primary_next_requirement(
                evidence_artifact.get("primary_next_requirement")
                or (list(evidence_artifact.get("next_requirements") or [])[:1] or [""])[0]
                or history_assessment.get("primary_next_requirement")
                or "",
                state.get("primary_next_requirement"),
            ),
            "eligible_modes": list(history_assessment.get("eligible_modes") or []),
            "trained_modes": list((model_artifact or {}).get("trained_modes") or []),
            "daily_long_rows": int(daily_long_summary.get("rows") or 0),
            "daily_long_sources": list(daily_long_summary.get("sources_used") or []),
            "resume_count": int(state.get("resume_count") or 0),
        },
    )
    _update_stage(state, "finalize", "completed", final_summary, progress_cb=progress_cb)
    _complete_job(state, final_summary)
    return state
