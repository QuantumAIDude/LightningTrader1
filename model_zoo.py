from __future__ import annotations

import time
from typing import Any

from model_trainers import train_all_models
from model_utils import load_models, predict_4min, predict_horizon


DEFAULT_MODEL_FAMILY = "tabular_baseline_v2"
_CORE_MODEL_KEYS = ("base", "forest", "boost", "extra_trees")
_MODE_ALIASES = {
    "4min": "4min",
    "4-minute": "4min",
    "intraday": "4min",
    "dailyhigh": "DailyHigh",
    "daily_high": "DailyHigh",
    "daily": "DailyHigh",
    "weekend": "WeekEnd",
    "week_end": "WeekEnd",
    "week": "WeekEnd",
}
_MODE_LABELS = {
    "4min": "4-minute intraday",
    "DailyHigh": "Daily range",
    "WeekEnd": "Weekend / weekly",
}
MODE_SPECS = {
    "4min": {
        "mode": "4min",
        "label": "4-minute intraday",
        "family": DEFAULT_MODEL_FAMILY,
        "horizon_group": "intraday",
        "prediction_key": "intraday",
        "requires_horizon_models": False,
        "default": True,
    },
    "DailyHigh": {
        "mode": "DailyHigh",
        "label": "Daily range",
        "family": DEFAULT_MODEL_FAMILY,
        "horizon_group": "session",
        "prediction_key": "daily",
        "requires_horizon_models": True,
        "default": False,
    },
    "WeekEnd": {
        "mode": "WeekEnd",
        "label": "Weekend / weekly",
        "family": DEFAULT_MODEL_FAMILY,
        "horizon_group": "swing",
        "prediction_key": "weekend",
        "requires_horizon_models": True,
        "default": False,
    },
}


def _safe_symbol(instrument_key: str | None) -> str:
    return str(instrument_key or "").strip().upper()


def normalize_mode(mode: str | None) -> str:
    raw = str(mode or "4min").strip()
    key = raw.lower().replace(" ", "").replace("-", "_")
    return _MODE_ALIASES.get(key, raw)


def load_model_stack(instrument_key: str | None, include_modes: bool = True) -> dict:
    return dict(load_models(include_modes=include_modes, instrument_key=_safe_symbol(instrument_key)) or {})


def get_mode_spec(mode: str | None = None) -> dict:
    canonical = normalize_mode(mode)
    return dict(MODE_SPECS.get(canonical) or MODE_SPECS["4min"])


def summarize_model_stack(models: dict | None) -> dict:
    payload = dict(models or {})
    core_present = [key for key in _CORE_MODEL_KEYS if isinstance(payload.get(key), dict)]
    modes = dict(payload.get("modes") or {})
    horizon_modes = {}
    available_modes = []
    if core_present:
        available_modes.append("4min")
    for mode_name, canonical in (("daily_high", "DailyHigh"), ("daily_low", "DailyHigh"), ("weekend", "WeekEnd")):
        group = dict(modes.get(mode_name) or {})
        ready = any(isinstance(group.get(key), dict) for key in _CORE_MODEL_KEYS)
        horizon_modes[canonical] = bool(horizon_modes.get(canonical) or ready)
    for canonical, ready in horizon_modes.items():
        if ready and canonical not in available_modes:
            available_modes.append(canonical)
    tradeability_head = bool(isinstance(payload.get("meta"), dict) or payload.get("meta") is not None)
    return {
        "family": DEFAULT_MODEL_FAMILY,
        "core_models": list(core_present),
        "missing_core_models": [key for key in _CORE_MODEL_KEYS if key not in core_present],
        "core_ready": len(core_present) == len(_CORE_MODEL_KEYS),
        "meta_present": bool(isinstance(payload.get("meta"), dict)),
        "regime_present": bool(payload.get("regime") is not None),
        "tradeability_head_present": tradeability_head,
        "horizon_modes": horizon_modes,
        "available_modes": available_modes,
    }


def list_mode_specs(bundle_or_models=None, *, available_only: bool = False) -> list[dict]:
    available = set(available_modes(bundle_or_models)) if available_only else set(MODE_SPECS.keys())
    rows = []
    for mode_name in ("4min", "DailyHigh", "WeekEnd"):
        spec = get_mode_spec(mode_name)
        spec["available"] = mode_name in available
        rows.append(spec)
    return rows


def build_model_runtime(instrument_key: str | None, include_modes: bool = True) -> dict:
    sym = _safe_symbol(instrument_key)
    models = load_model_stack(sym, include_modes=include_modes)
    return {
        "instrument": sym,
        "family": DEFAULT_MODEL_FAMILY,
        "loaded_at": time.time(),
        "models": models,
        "summary": summarize_model_stack(models),
        "mode_specs": list_mode_specs(models, available_only=False),
    }


def train_baseline_models(
    df,
    instrument_key: str | None,
    *,
    train_horizons: bool = True,
    modes_filter: list[str] | None = None,
    progress_cb=None,
    max_runtime_sec: float | None = None,
) -> dict:
    sym = _safe_symbol(instrument_key)
    trained = train_all_models(
        df,
        instrument_key=sym,
        train_horizons=bool(train_horizons),
        modes_filter=list(modes_filter or []) or None,
        progress_cb=progress_cb,
        max_runtime_sec=max_runtime_sec,
    )
    models = load_model_stack(sym, include_modes=True)
    return {
        "instrument": sym,
        "family": DEFAULT_MODEL_FAMILY,
        "trained_at": time.time(),
        "requested_modes": list(modes_filter or []),
        "models": models if models else dict(trained or {}),
        "summary": summarize_model_stack(models if models else trained),
    }


def _coerce_models(bundle_or_models: Any) -> dict:
    if isinstance(bundle_or_models, dict) and isinstance(bundle_or_models.get("models"), dict):
        return dict(bundle_or_models.get("models") or {})
    return dict(bundle_or_models or {})


def predict_intraday(bundle_or_models, df_latest, *, instrument_key: str | None = None, **kwargs) -> dict:
    sym = _safe_symbol(instrument_key)
    return dict(
        predict_4min(
            _coerce_models(bundle_or_models),
            df_latest,
            instrument_key=sym,
            **kwargs,
        )
        or {}
    )


def predict_mode(bundle_or_models, df_latest, *, instrument_key: str | None = None, mode: str | None = None, **kwargs) -> dict:
    sym = _safe_symbol(instrument_key)
    canonical = str(get_mode_spec(mode).get("mode") or "4min")
    if canonical == "4min":
        return predict_intraday(bundle_or_models, df_latest, instrument_key=sym, **kwargs)
    return dict(
        predict_horizon(
            _coerce_models(bundle_or_models),
            df_latest,
            instrument_key=sym,
            mode=canonical,
            **kwargs,
        )
        or {}
    )


def available_modes(bundle_or_models) -> list[str]:
    summary = summarize_model_stack(_coerce_models(bundle_or_models))
    return list(summary.get("available_modes") or [])


def build_inference_payload(bundle_or_models, df_latest, *, instrument_key: str | None = None, mode: str | None = None, **kwargs) -> dict:
    sym = _safe_symbol(instrument_key)
    spec = get_mode_spec(mode)
    models = _coerce_models(bundle_or_models)
    summary = summarize_model_stack(models)
    prediction = predict_mode(models, df_latest, instrument_key=sym, mode=spec.get("mode"), **kwargs)
    return {
        "instrument": sym,
        "family": DEFAULT_MODEL_FAMILY,
        "mode": str(spec.get("mode") or "4min"),
        "mode_label": str(spec.get("label") or mode_label(spec.get("mode"))),
        "prediction_key": str(spec.get("prediction_key") or ""),
        "mode_spec": spec,
        "summary": summary,
        "available_modes": list(summary.get("available_modes") or []),
        "prediction": prediction,
    }


def mode_label(mode: str | None) -> str:
    canonical = str(get_mode_spec(mode).get("mode") or "4min")
    return _MODE_LABELS.get(canonical, canonical)
