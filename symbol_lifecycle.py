from __future__ import annotations

from typing import Any

from config import INSTRUMENTS
from instrument_profile import build_instrument_meta
from model_zoo import summarize_model_stack


def _safe_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _clean_text(value: Any, fallback: str = "-") -> str:
    text = str(value or "").strip()
    return text or fallback


def _mode_coverage_label(available_modes: list[str] | None) -> str:
    modes = [str(item or "").strip() for item in list(available_modes or []) if str(item or "").strip()]
    if not modes:
        return "No model coverage"
    if set(modes) >= {"4min", "DailyHigh", "WeekEnd"}:
        return "4min + Daily + WeekEnd"
    return " + ".join(modes)


def _feature_state(onboarding_summary: dict | None, deployment: dict | None = None) -> dict:
    artifacts = dict((onboarding_summary or {}).get("artifacts") or {})
    history = dict(artifacts.get("history_assessment") or {})
    feature = dict(artifacts.get("feature_readiness") or {})
    paths = dict(feature.get("paths") or history.get("paths") or {})
    intraday_rows = _safe_int(feature.get("intraday_rows") or (paths.get("4min") or {}).get("feature_rows"))
    daily_rows = _safe_int(feature.get("daily_rows") or (paths.get("DailyHigh") or {}).get("feature_rows"))
    weekend_rows = _safe_int(feature.get("weekend_rows") or (paths.get("WeekEnd") or {}).get("feature_rows"))
    feature_ready_modes = list(feature.get("feature_ready_modes") or history.get("feature_ready_modes") or [])
    eligible_modes = list(feature.get("eligible_modes") or history.get("eligible_modes") or [])
    status_flags = dict((deployment or {}).get("status_flags") or {})
    if str(feature.get("status") or "").lower() == "blocked" and str(feature.get("summary") or "").strip():
        return {
            "key": "missing",
            "label": "Feature readiness missing",
            "detail": _clean_text(feature.get("summary"), "No feature-readiness artifacts have been captured for this symbol yet."),
        }
    if intraday_rows <= 0 and daily_rows <= 0 and weekend_rows <= 0:
        if bool(status_flags.get("trained")) or bool(status_flags.get("advisory_usable")):
            return {
                "key": "inferred_ready",
                "label": "Feature-ready",
                "detail": "Feature readiness artifacts were not captured in the onboarding state, but the symbol already has trained/readiness evidence.",
            }
        return {
            "key": "missing",
            "label": "Feature readiness missing",
                "detail": "No feature-readiness artifacts have been captured for this symbol yet.",
            }
    if set(eligible_modes) >= {"4min", "DailyHigh", "WeekEnd"}:
        return {
            "key": "multi_horizon",
            "label": "Intraday + daily + weekend",
            "detail": _clean_text(feature.get("summary"), f"Rows {intraday_rows}/{daily_rows}/{weekend_rows} for intraday, daily, and weekend paths."),
        }
    if feature_ready_modes and not eligible_modes:
        return {
            "key": "partial_horizons",
            "label": "Partial horizon coverage",
            "detail": _clean_text(feature.get("summary"), f"Rows {intraday_rows}/{daily_rows}/{weekend_rows}; some paths build usable features but none has reached its training floor yet."),
        }
    if intraday_rows > 0 and daily_rows > 0 and weekend_rows > 0:
        return {
            "key": "multi_horizon",
            "label": "Intraday + daily + weekend",
            "detail": f"Rows {intraday_rows}/{daily_rows}/{weekend_rows} for intraday, daily, and weekend paths.",
        }
    if intraday_rows > 0 and (daily_rows > 0 or weekend_rows > 0):
        return {
            "key": "partial_horizons",
            "label": "Partial horizon coverage",
            "detail": f"Rows {intraday_rows}/{daily_rows}/{weekend_rows}; some horizons are ready while others still need history.",
        }
    if intraday_rows > 0:
        return {
            "key": "intraday_only",
            "label": "Intraday only",
            "detail": f"Intraday feature rows {intraday_rows}; higher-horizon feature sets are still missing.",
        }
    return {
        "key": "insufficient",
        "label": "Insufficient history",
        "detail": f"Daily/weekend artifacts exist ({daily_rows}/{weekend_rows}), but intraday training rows are still missing.",
    }


def _data_state(runtime_bundle: dict | None, onboarding_summary: dict | None, deployment: dict | None) -> dict:
    artifacts = dict((onboarding_summary or {}).get("artifacts") or {})
    history = dict(artifacts.get("history_assessment") or {})
    if history:
        status_key = str(history.get("status_key") or "").strip().lower()
        status_label = _clean_text(history.get("status_label"), "")
        detail = _clean_text(history.get("detail"), "")
        artifact = dict((deployment or {}).get("artifact_summary", {}) or {})
        prices = dict(artifact.get("prices") or {})
        if status_key == "missing":
            onboarding_status = str((onboarding_summary or {}).get("status") or "").lower()
            if onboarding_status == "running":
                return {
                    "key": "staging",
                    "label": "Data staging started",
                    "detail": _clean_text((onboarding_summary or {}).get("summary"), "The onboarding pipeline has started but no canonical price history is staged yet."),
                }
            return {
                "key": "missing",
                "label": status_label or "No canonical price history",
                "detail": detail or "This symbol is supported in the registry, but canonical price history has not been staged yet.",
            }
        if status_key in {"insufficient_history"}:
            return {
                "key": "insufficient",
                "label": status_label or "Insufficient history",
                "detail": detail or "Price history exists, but there still is not enough usable history for supported feature paths.",
            }
        if status_key in {"feature_partial", "partial_horizon_eligible"}:
            return {
                "key": "partial",
                "label": "Partial history",
                "detail": detail or "Some history-dependent paths are available while others remain incomplete.",
            }
        if status_key in {"advisory_core_eligible", "multi_horizon_eligible"}:
            freshness = dict((runtime_bundle or {}).get("freshness") or {})
            price_freshness = dict(freshness.get("price") or {})
            if bool(price_freshness.get("is_stale")) or bool(prices.get("stale")):
                return {
                    "key": "stale",
                    "label": "Data staged, stale",
                    "detail": f"{_safe_int(history.get('price_rows'))} rows staged, but the current market-data snapshot is stale.",
                }
            return {
                "key": "ready",
                "label": "History staged",
                "detail": detail or f"{_safe_int(history.get('price_rows'))} rows staged with a usable local price history footprint.",
            }
    artifact = dict((deployment or {}).get("artifact_summary", {}) or {})
    prices = dict(artifact.get("prices") or {})
    market_data = dict(artifacts.get("market_data") or {})
    freshness = dict((runtime_bundle or {}).get("freshness") or {})
    price_freshness = dict(freshness.get("price") or {})
    rows = max(
        _safe_int(prices.get("rows")),
        _safe_int(market_data.get("rows")),
    )
    stale = bool(price_freshness.get("is_stale")) if price_freshness else bool(prices.get("stale"))
    if rows <= 0:
        onboarding_status = str((onboarding_summary or {}).get("status") or "").lower()
        if onboarding_status == "running":
            return {
                "key": "staging",
                "label": "Data staging started",
                "detail": _clean_text((onboarding_summary or {}).get("summary"), "The onboarding pipeline has started but no canonical price history is staged yet."),
            }
        return {
            "key": "missing",
            "label": "No canonical price history",
            "detail": "This symbol is supported in the registry, but canonical price history has not been staged yet.",
        }
    if stale:
        return {
            "key": "stale",
            "label": "Data staged, stale",
            "detail": f"{rows} rows staged, but the current market-data snapshot is stale.",
        }
    if rows < 240:
        return {
            "key": "insufficient",
            "label": "Insufficient history",
            "detail": f"{rows} rows staged; there is still not enough recent history for reliable feature and model coverage.",
        }
    if rows < 1200:
        return {
            "key": "partial",
            "label": "Partial history",
            "detail": f"{rows} rows staged; enough for partial coverage, but not yet deep multi-horizon maturity.",
        }
    return {
        "key": "ready",
        "label": "History staged",
        "detail": f"{rows} rows staged with a usable local price history footprint.",
    }


def _model_summary_from_deployment(deployment: dict | None) -> dict:
    artifact = dict((deployment or {}).get("artifact_summary", {}) or {})
    models = dict(artifact.get("models") or {})
    available_modes = []
    if bool(models.get("core_4min_ready")):
        available_modes.append("4min")
    horizon_ready = dict(models.get("horizon_ready") or {})
    if bool(horizon_ready.get("daily_high")) or bool(horizon_ready.get("daily_low")):
        available_modes.append("DailyHigh")
    if bool(horizon_ready.get("weekend")):
        available_modes.append("WeekEnd")
    return {
        "core_ready": bool(models.get("core_4min_ready")),
        "available_modes": available_modes,
    }


def _model_state(model_summary: dict | None, deployment: dict | None) -> dict:
    summary = dict(model_summary or {})
    if not summary or not list(summary.get("available_modes") or []):
        fallback = _model_summary_from_deployment(deployment)
        if not list(summary.get("available_modes") or []):
            summary["available_modes"] = list(fallback.get("available_modes") or [])
        if not bool(summary.get("core_ready")):
            summary["core_ready"] = bool(fallback.get("core_ready"))
    available_modes = list(summary.get("available_modes") or [])
    core_ready = bool(summary.get("core_ready"))
    if not available_modes:
        return {
            "key": "missing",
            "label": "No trained models",
            "detail": "No model artifacts are available for this symbol yet.",
        }
    coverage = _mode_coverage_label(available_modes)
    if core_ready and set(available_modes) >= {"4min", "DailyHigh", "WeekEnd"}:
        return {
            "key": "multi_horizon",
            "label": "Multi-horizon coverage",
            "detail": f"Core models are ready across {coverage}.",
        }
    if core_ready and "4min" in available_modes:
        return {
            "key": "core_only",
            "label": "Core advisory coverage",
            "detail": f"The core 4-minute stack is ready, with current coverage {coverage}.",
        }
    if available_modes:
        return {
            "key": "partial",
            "label": "Partial model coverage",
            "detail": f"Some model artifacts exist, but the full core stack is incomplete. Current coverage: {coverage}.",
        }
    evidence_state = str((deployment or {}).get("evidence_state_label") or "").lower()
    return {
        "key": "missing",
        "label": "No trained models",
        "detail": "No model artifacts are available yet." if evidence_state != "missing evidence" else "Symbol evidence is still incomplete and no usable model stack is present.",
    }


def _calibration_state(onboarding_summary: dict | None, deployment: dict | None) -> dict:
    calibration = dict((deployment or {}).get("calibration_summary") or {})
    onboarding_cal = dict(((onboarding_summary or {}).get("artifacts") or {}).get("calibration") or {})
    drift = bool(calibration.get("drift_flag"))
    stale = bool(calibration.get("evidence_stale"))
    available = bool(calibration.get("available")) or bool(onboarding_cal)
    if drift:
        return {
            "key": "drift",
            "label": "Calibration drift",
            "detail": "Recent calibration or horizon-accuracy drift is active for this symbol.",
        }
    if stale and available:
        return {
            "key": "stale",
            "label": "Calibration evidence stale",
            "detail": "Calibration artifacts exist, but their evidence window is stale.",
        }
    if available:
        return {
            "key": "ready",
            "label": "Calibration captured",
            "detail": "Calibration and horizon evidence have been captured for the current symbol path.",
        }
    if str(onboarding_cal.get("status") or "").lower() == "deferred":
        return {
            "key": "pending",
            "label": "Calibration deferred",
            "detail": _clean_text(onboarding_cal.get("reason"), "Calibration is deferred until the core 4-minute stack is trained."),
        }
    model_summary = dict(((onboarding_summary or {}).get("artifacts") or {}).get("model_runtime") or {}).get("summary") or {}
    if bool(model_summary.get("core_ready")):
        return {
            "key": "pending",
            "label": "Calibration pending",
            "detail": "Core models exist, but calibration evidence has not been captured cleanly yet.",
        }
    return {
        "key": "blocked",
        "label": "Calibration blocked",
        "detail": "Calibration is unavailable because the required symbol-specific models and evidence are not ready yet.",
    }


def _news_state(runtime_bundle: dict | None, deployment: dict | None, news_health: dict | None) -> dict:
    health = dict(news_health or (runtime_bundle or {}).get("news_health") or {})
    artifact = dict((deployment or {}).get("artifact_summary", {}) or {})
    news_artifact = dict(artifact.get("news") or {})
    headline_state = str(health.get("headline_state") or news_artifact.get("headline_state") or health.get("status") or "unknown").lower()
    history_state = str(health.get("history_state") or news_artifact.get("history_state") or "unknown").lower()
    rows = max(_safe_int(health.get("rows")), _safe_int(news_artifact.get("rows")))
    if headline_state in {"failed", "error", "stale"}:
        return {
            "key": "stale",
            "label": "News degraded",
            "detail": f"Headline state is {headline_state.replace('_', ' ')}.",
        }
    if rows <= 0 or headline_state in {"missing", "unknown"}:
        return {
            "key": "missing",
            "label": "No news context",
            "detail": "No local news history has been staged for this symbol yet.",
        }
    if history_state in {"missing", "insufficient_history", "thin", "degraded"}:
        return {
            "key": "partial",
            "label": "Thin news history",
            "detail": f"{rows} local items with history state {history_state.replace('_', ' ')}.",
        }
    return {
        "key": "ready",
        "label": "News context staged",
        "detail": f"{rows} local items with headline state {headline_state.replace('_', ' ')}.",
    }


def _lifecycle_stage(
    *,
    supported: bool,
    onboarding_summary: dict | None,
    data_state: dict,
    feature_state: dict,
    model_state: dict,
    calibration_state: dict,
    deployment: dict | None,
) -> dict:
    if not supported:
        return {
            "key": "unsupported",
            "label": "Unsupported",
            "detail": "This symbol is not registered in the runtime instrument set.",
        }
    onboarding_status = str((onboarding_summary or {}).get("status") or "not_started").lower()
    if onboarding_status == "running":
        return {
            "key": "onboarding_in_progress",
            "label": "Onboarding in progress",
            "detail": _clean_text((onboarding_summary or {}).get("summary"), "The symbol is progressing through the staged onboarding pipeline."),
        }
    if onboarding_status == "failed":
        return {
            "key": "onboarding_failed",
            "label": "Onboarding failed",
            "detail": _clean_text((onboarding_summary or {}).get("last_error") or (onboarding_summary or {}).get("summary"), "The latest onboarding pass failed before this symbol reached a stable workspace state."),
        }
    if data_state.get("key") in {"missing", "staging"}:
        return {
            "key": "profile_resolved",
            "label": "Profile resolved",
            "detail": "The symbol is registered, but price history staging is still the next gating step.",
        }
    if data_state.get("key") in {"insufficient", "partial"} and feature_state.get("key") == "missing":
        return {
            "key": "history_partial",
            "label": data_state.get("label") or "Partial history",
            "detail": data_state.get("detail") or "Data staging has started, but there still is not enough usable history for feature readiness.",
        }
    if feature_state.get("key") in {"intraday_only", "partial_horizons"} and model_state.get("key") == "missing":
        return {
            "key": "feature_ready_partial",
            "label": "Features partially ready",
            "detail": feature_state.get("detail") or "Feature readiness is partial and model training is the next gating step.",
        }
    if model_state.get("key") == "missing":
        return {
            "key": "data_ready_models_missing",
            "label": "Data ready, models pending",
            "detail": "Symbol data is staged, but no trained model coverage exists yet.",
        }
    if model_state.get("key") == "partial":
        return {
            "key": "models_partial",
            "label": "Models partially available",
            "detail": model_state.get("detail") or "Some model artifacts exist, but symbol coverage is still incomplete.",
        }
    if calibration_state.get("key") in {"blocked", "pending"}:
        return {
            "key": "calibration_pending",
            "label": "Calibration unavailable",
            "detail": calibration_state.get("detail") or "Calibration evidence is still missing for this symbol.",
        }
    maturity = str((deployment or {}).get("symbol_maturity_label") or "").strip()
    if maturity:
        return {
            "key": str((deployment or {}).get("symbol_maturity_key") or "maturity"),
            "label": maturity,
            "detail": _clean_text((deployment or {}).get("symbol_maturity_detail"), "The symbol-specific maturity state is available from the readiness system."),
        }
    return {
        "key": "supported_only",
        "label": "Supported only",
        "detail": "The platform supports this symbol, but symbol-specific maturity is still incomplete.",
    }


def build_symbol_lifecycle_snapshot(
    symbol: str,
    *,
    runtime_bundle: dict | None = None,
    workspace: dict | None = None,
    onboarding_summary: dict | None = None,
    deployment_summary: dict | None = None,
    news_health: dict | None = None,
    model_summary: dict | None = None,
) -> dict:
    sym = _safe_symbol(symbol)
    meta = build_instrument_meta(sym, overrides=INSTRUMENTS.get(sym, {}))
    if onboarding_summary is None:
        from onboarding_jobs import summarize_onboarding_state

        onboarding_summary = summarize_onboarding_state(sym)
    onboarding = dict(onboarding_summary or {})
    if deployment_summary is None:
        from readiness_tracker import load_readiness_state, summarize_deployment_readiness

        deployment_summary = summarize_deployment_readiness(
            sym,
            state=load_readiness_state(sym) or {},
            news_health=news_health,
        )
    deployment = dict(
        deployment_summary
    )
    bundle = dict(runtime_bundle or {})
    models_payload = dict(bundle.get("models") or {})
    model_meta = dict(model_summary or summarize_model_stack(models_payload))
    deployment_model_meta = _model_summary_from_deployment(deployment)
    data_state = _data_state(bundle, onboarding, deployment)
    feature = _feature_state(onboarding, deployment)
    models = _model_state(model_meta, deployment)
    calibration = _calibration_state(onboarding, deployment)
    news = _news_state(bundle, deployment, news_health)
    lifecycle = _lifecycle_stage(
        supported=bool(sym in INSTRUMENTS),
        onboarding_summary=onboarding,
        data_state=data_state,
        feature_state=feature,
        model_state=models,
        calibration_state=calibration,
        deployment=deployment,
    )
    status_flags = dict(deployment.get("status_flags") or {})
    modes = list(model_meta.get("available_modes") or []) or list(deployment_model_meta.get("available_modes") or [])
    history_assessment = dict((onboarding.get("artifacts") or {}).get("history_assessment") or {})
    next_requirement = _clean_text(
        deployment.get("primary_next_requirement")
        or onboarding.get("primary_next_requirement")
        or history_assessment.get("primary_next_requirement"),
        "",
    )
    operator = _clean_text(deployment.get("operator_state_label"), "Operator state unknown")
    evidence = _clean_text(deployment.get("evidence_state_label"), "Evidence unknown")
    summary_parts = [
        _clean_text(lifecycle.get("label"), sym),
        operator,
        evidence,
    ]
    if next_requirement:
        summary_parts.append(f"Next: {next_requirement}")
    onboarding_progress = f"{_safe_int(onboarding.get('progress_completed'))}/{_safe_int(onboarding.get('progress_total'))}"
    compact_coverage = _mode_coverage_label(modes)
    return {
        "symbol": sym,
        "display_name": _clean_text(meta.get("display_name"), sym),
        "instrument_kind": _clean_text(meta.get("instrument_kind") or meta.get("type"), "equity"),
        "sector": _clean_text(meta.get("sector"), "Broad Market"),
        "industry": _clean_text(meta.get("industry") or meta.get("sector"), "-"),
        "supported": bool(sym in INSTRUMENTS),
        "workspace_status": _clean_text((workspace or {}).get("last_status"), ""),
        "workspace_note": _clean_text((workspace or {}).get("last_note"), ""),
        "lifecycle_key": str(lifecycle.get("key") or ""),
        "lifecycle_label": _clean_text(lifecycle.get("label"), "Unknown"),
        "lifecycle_detail": _clean_text(lifecycle.get("detail"), "Lifecycle detail unavailable"),
        "stage_label": _clean_text(deployment.get("stage_label"), "-"),
        "rollout_label": _clean_text(deployment.get("rollout_label"), "-"),
        "operator_state_label": operator,
        "operator_state_detail": _clean_text(deployment.get("operator_state_detail"), "-"),
        "evidence_state_label": evidence,
        "evidence_state_detail": _clean_text(deployment.get("evidence_state_detail"), "-"),
        "symbol_maturity_label": _clean_text(deployment.get("symbol_maturity_label"), "-"),
        "symbol_maturity_detail": _clean_text(deployment.get("symbol_maturity_detail"), "-"),
        "primary_next_requirement": next_requirement,
        "next_requirement_category": _clean_text(deployment.get("next_requirement_category"), ""),
        "status_flags": status_flags,
        "onboarding_status": _clean_text(onboarding.get("status_label"), "Not started"),
        "onboarding_summary": _clean_text(onboarding.get("summary"), "No staged onboarding job has run for this symbol yet."),
        "onboarding_progress": onboarding_progress,
        "data_state_key": str(data_state.get("key") or ""),
        "data_state_label": _clean_text(data_state.get("label"), "-"),
        "data_state_detail": _clean_text(data_state.get("detail"), "-"),
        "feature_state_key": str(feature.get("key") or ""),
        "feature_state_label": _clean_text(feature.get("label"), "-"),
        "feature_state_detail": _clean_text(feature.get("detail"), "-"),
        "model_state_key": str(models.get("key") or ""),
        "model_state_label": _clean_text(models.get("label"), "-"),
        "model_state_detail": _clean_text(models.get("detail"), "-"),
        "model_coverage_label": compact_coverage,
        "available_modes": modes,
        "calibration_state_key": str(calibration.get("key") or ""),
        "calibration_state_label": _clean_text(calibration.get("label"), "-"),
        "calibration_state_detail": _clean_text(calibration.get("detail"), "-"),
        "news_state_key": str(news.get("key") or ""),
        "news_state_label": _clean_text(news.get("label"), "-"),
        "news_state_detail": _clean_text(news.get("detail"), "-"),
        "workspace_summary": " | ".join(bit for bit in summary_parts if bit),
        "overview_status": " | ".join(bit for bit in (operator, evidence, compact_coverage if compact_coverage != "No model coverage" else "", next_requirement[:72] if next_requirement else "") if bit),
        "empty_state_title": _clean_text(lifecycle.get("label"), "Symbol status"),
        "empty_state_detail": " | ".join(
            bit
            for bit in (
                _clean_text(lifecycle.get("detail"), ""),
                _clean_text(deployment.get("detail"), ""),
            )
            if bit
        ),
    }


def lifecycle_rows(snapshot: dict | None) -> list[tuple[str, str, str]]:
    payload = dict(snapshot or {})
    next_requirement = _clean_text(payload.get("primary_next_requirement"), "")
    rows = [
        (
            "Lifecycle",
            _clean_text(payload.get("lifecycle_label"), "-"),
            _clean_text(payload.get("lifecycle_detail"), "-"),
        ),
        (
            "Data / Features",
            f"{_clean_text(payload.get('data_state_label'), '-')} / {_clean_text(payload.get('feature_state_label'), '-')}",
            f"{_clean_text(payload.get('data_state_detail'), '-')} | {_clean_text(payload.get('feature_state_detail'), '-')}",
        ),
        (
            "Models",
            _clean_text(payload.get("model_state_label"), "-"),
            f"{_clean_text(payload.get('model_coverage_label'), '-')} | {_clean_text(payload.get('model_state_detail'), '-')}",
        ),
        (
            "Calibration / News",
            f"{_clean_text(payload.get('calibration_state_label'), '-')} / {_clean_text(payload.get('news_state_label'), '-')}",
            f"{_clean_text(payload.get('calibration_state_detail'), '-')} | {_clean_text(payload.get('news_state_detail'), '-')}",
        ),
        (
            "Operator Posture",
            _clean_text(payload.get("operator_state_label"), "-"),
            f"{_clean_text(payload.get('stage_label'), '-')} | {_clean_text(payload.get('evidence_state_label'), '-')}",
        ),
    ]
    if next_requirement:
        rows.append(
            (
                "Next Requirement",
                _clean_text(payload.get("next_requirement_category"), "other").replace("_", " ").title(),
                next_requirement,
            )
        )
    else:
        rows.append(
            (
                "Onboarding",
                _clean_text(payload.get("onboarding_status"), "-"),
                f"Progress {payload.get('onboarding_progress') or '-'} | {_clean_text(payload.get('onboarding_summary'), '-')}",
            )
        )
    return rows[:6]
