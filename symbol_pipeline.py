from __future__ import annotations

import os
from typing import Any

import pandas as pd

from config import DATA_DIR
from features import build_features_for_training, build_next_session_dataset, build_weekend_session_dataset


MODE_REQUIREMENTS = {
    "4min": {
        "label": "4-minute intraday",
        "raw_modes": ["4min"],
        "min_train_rows": 100,
        "source": "recent_intraday",
    },
    "DailyHigh": {
        "label": "Daily range",
        "raw_modes": ["daily_high", "daily_low"],
        "min_train_rows": 15,
        "source": "daily_long_history",
    },
    "WeekEnd": {
        "label": "Weekend / weekly",
        "raw_modes": ["weekend"],
        "min_train_rows": 6,
        "source": "daily_long_history",
    },
}


def _safe_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


def _safe_int(value: Any) -> int:
    try:
        return int(value or 0)
    except Exception:
        return 0


def _safe_text(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text or fallback


def _daily_long_history_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{_safe_symbol(symbol)}_daily_long.csv")


def _canonical_price_path(symbol: str) -> str:
    return os.path.join(DATA_DIR, f"prices_{_safe_symbol(symbol)}.csv")


def _provider_price_paths(symbol: str) -> list[str]:
    sym = _safe_symbol(symbol)
    return [
        os.path.join(DATA_DIR, f"prices_{sym}_alpaca.csv"),
        os.path.join(DATA_DIR, f"prices_{sym}_yfinance.csv"),
        os.path.join(DATA_DIR, f"prices_{sym}_auto.csv"),
        os.path.join(DATA_DIR, f"prices_{sym}_ibkr.csv"),
    ]


def load_daily_long_history(symbol: str) -> pd.DataFrame:
    path = _daily_long_history_path(symbol)
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        df.columns = [str(col).strip().lower() for col in df.columns]
        return df
    except Exception:
        return pd.DataFrame()


def _read_ohlcv_history(path: str) -> pd.DataFrame:
    if not path or not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df[~df.index.isna()].sort_index()
        df.columns = [str(col).strip().lower() for col in df.columns]
        keep = [col for col in ("open", "high", "low", "close", "volume") if col in df.columns]
        if "close" not in keep:
            return pd.DataFrame()
        out = df[keep].copy()
        for col in keep:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        out = out.dropna(subset=["close"])
        if "open" not in out.columns:
            out["open"] = out["close"]
        if "high" not in out.columns:
            out["high"] = out["close"]
        if "low" not in out.columns:
            out["low"] = out["close"]
        if "volume" not in out.columns:
            out["volume"] = 0.0
        out["open"] = out["open"].fillna(out["close"])
        out["high"] = out["high"].fillna(out["close"])
        out["low"] = out["low"].fillna(out["close"])
        out["volume"] = out["volume"].fillna(0.0)
        return out.sort_index()
    except Exception:
        return pd.DataFrame()


def _time_span_days(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    try:
        start = pd.to_datetime(df.index.min(), utc=True, errors="coerce")
        end = pd.to_datetime(df.index.max(), utc=True, errors="coerce")
        if pd.isna(start) or pd.isna(end):
            return 0
        return max(0, int((end - start).days))
    except Exception:
        return 0


def _session_count(df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0
    try:
        return int(pd.Series(pd.to_datetime(df.index, utc=True, errors="coerce")).dt.normalize().dropna().nunique())
    except Exception:
        return 0


def _frame_bounds(df: pd.DataFrame) -> tuple[str | None, str | None]:
    if df is None or df.empty:
        return None, None
    try:
        first = pd.to_datetime(df.index.min(), utc=True, errors="coerce")
        last = pd.to_datetime(df.index.max(), utc=True, errors="coerce")
        first_text = None if pd.isna(first) else first.isoformat()
        last_text = None if pd.isna(last) else last.isoformat()
        return first_text, last_text
    except Exception:
        return None, None


def _merge_ohlcv_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming.copy() if isinstance(incoming, pd.DataFrame) else pd.DataFrame()
    if incoming is None or incoming.empty:
        return existing.copy()
    merged = pd.concat([existing, incoming], axis=0, sort=False)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged


def _median_bar_minutes(df: pd.DataFrame) -> float | None:
    if df is None or df.empty or len(df.index) < 2:
        return None
    try:
        idx = pd.DatetimeIndex(pd.to_datetime(df.index, utc=True, errors="coerce")).dropna().sort_values().unique()
        if len(idx) < 2:
            return None
        diffs = (idx[1:] - idx[:-1]) / pd.Timedelta(minutes=1)
        if len(diffs) <= 0:
            return None
        return float(pd.Series(diffs).median())
    except Exception:
        return None


def _to_daily_history(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if df is None or df.empty:
        return pd.DataFrame(), {"rows_in": 0, "rows_out": 0, "cadence_minutes": None, "daily_like": False}
    frame = df.copy()
    try:
        frame.index = pd.to_datetime(frame.index, utc=True, errors="coerce")
        frame = frame[~frame.index.isna()].sort_index()
    except Exception:
        return pd.DataFrame(), {"rows_in": 0, "rows_out": 0, "cadence_minutes": None, "daily_like": False}
    keep = [col for col in ("open", "high", "low", "close", "volume") if col in frame.columns]
    if "close" not in keep:
        return pd.DataFrame(), {"rows_in": int(len(frame)), "rows_out": 0, "cadence_minutes": None, "daily_like": False}
    frame = frame[keep].copy()
    cadence_minutes = _median_bar_minutes(frame)
    daily_like = bool(cadence_minutes is not None and cadence_minutes >= 720.0)
    grouped = frame.copy()
    grouped["session_day"] = pd.DatetimeIndex(grouped.index).normalize()
    daily = (
        grouped.groupby("session_day", sort=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna(subset=["close"])
    )
    try:
        daily.index = pd.to_datetime(daily.index, utc=True, errors="coerce")
        daily = daily[~daily.index.isna()].sort_index()
    except Exception:
        pass
    return daily, {
        "rows_in": int(len(frame)),
        "rows_out": int(len(daily)),
        "cadence_minutes": cadence_minutes,
        "daily_like": daily_like,
    }


def ensure_daily_long_history(
    symbol: str,
    *,
    recent_df: pd.DataFrame | None = None,
    extra_frames: list[pd.DataFrame] | None = None,
    persist: bool = True,
) -> tuple[pd.DataFrame, dict]:
    sym = _safe_symbol(symbol)
    path = _daily_long_history_path(sym)
    existing = load_daily_long_history(sym)
    merged = existing.copy()
    source_rows = {}
    source_details = {}
    source_order: list[str] = []

    def _record_source(source_tag: str, frame: pd.DataFrame | None):
        nonlocal merged
        daily_frame, detail = _to_daily_history(frame if isinstance(frame, pd.DataFrame) else pd.DataFrame())
        if daily_frame.empty:
            if detail.get("rows_in"):
                source_details[source_tag] = detail
            return
        merged = _merge_ohlcv_frames(merged, daily_frame)
        source_rows[source_tag] = int(len(daily_frame))
        source_details[source_tag] = detail
        source_order.append(source_tag)

    if not existing.empty:
        source_rows["existing_daily_long"] = int(len(existing))
        source_details["existing_daily_long"] = {
            "rows_in": int(len(existing)),
            "rows_out": int(len(existing)),
            "cadence_minutes": 1440.0,
            "daily_like": True,
        }
        source_order.append("existing_daily_long")

    if isinstance(recent_df, pd.DataFrame) and not recent_df.empty:
        _record_source("recent_runtime", recent_df)

    for idx, frame in enumerate(list(extra_frames or [])):
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            _record_source(f"extra_frame_{idx + 1}", frame)

    _record_source("canonical_prices", _read_ohlcv_history(_canonical_price_path(sym)))
    for provider_path in _provider_price_paths(sym):
        provider_name = os.path.splitext(os.path.basename(provider_path))[0].rsplit("_", 1)[-1]
        _record_source(f"provider_{provider_name}", _read_ohlcv_history(provider_path))

    merged = merged[~merged.index.duplicated(keep="last")] if not merged.empty else merged
    merged = merged.sort_index() if not merged.empty else merged
    first_ts, last_ts = _frame_bounds(merged)
    changed = False
    if persist and not merged.empty:
        changed = (
            existing.empty
            or int(len(existing)) != int(len(merged))
            or existing.index.max() != merged.index.max()
            or existing.index.min() != merged.index.min()
        )
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            merged.to_csv(path, date_format="%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            changed = False
    summary = {
        "path": path,
        "rows": int(len(merged)) if isinstance(merged, pd.DataFrame) else 0,
        "first_ts": first_ts,
        "last_ts": last_ts,
        "span_days": int(_time_span_days(merged)),
        "session_count": int(_session_count(merged)),
        "source_rows": source_rows,
        "source_details": source_details,
        "sources_used": list(dict.fromkeys(source_order)),
        "updated": bool(changed),
        "available": bool(isinstance(merged, pd.DataFrame) and not merged.empty),
    }
    return merged, summary


def _path_status(label: str, feature_rows: int, min_train_rows: int, build_error: str = "", source_label: str = "") -> dict:
    rows = _safe_int(feature_rows)
    if build_error:
        return {
            "status": "blocked",
            "status_label": "Blocked",
            "detail": f"{label} feature build failed: {build_error}",
            "feature_rows": rows,
            "feature_ready": False,
            "train_eligible": False,
            "min_train_rows": int(min_train_rows),
            "source_label": source_label,
            "build_error": build_error,
        }
    if rows <= 0:
        return {
            "status": "blocked",
            "status_label": "Insufficient history",
            "detail": f"{label} feature build produced 0 usable rows from {source_label or 'available history'}.",
            "feature_rows": rows,
            "feature_ready": False,
            "train_eligible": False,
            "min_train_rows": int(min_train_rows),
            "source_label": source_label,
            "build_error": "",
        }
    if rows < int(min_train_rows):
        return {
            "status": "partial",
            "status_label": "Feature-ready only",
            "detail": f"{label} feature build produced {rows} usable rows; needs {int(min_train_rows)} for training eligibility.",
            "feature_rows": rows,
            "feature_ready": True,
            "train_eligible": False,
            "min_train_rows": int(min_train_rows),
            "source_label": source_label,
            "build_error": "",
        }
    return {
        "status": "eligible",
        "status_label": "Training-eligible",
        "detail": f"{label} feature build produced {rows} usable rows; meets the {int(min_train_rows)}-row training floor.",
        "feature_rows": rows,
        "feature_ready": True,
        "train_eligible": True,
        "min_train_rows": int(min_train_rows),
        "source_label": source_label,
        "build_error": "",
    }


def _primary_next_requirement(price_rows: int, paths: dict[str, dict]) -> str:
    if _safe_int(price_rows) <= 0:
        return "Stage canonical price history for this symbol."
    intraday = dict(paths.get("4min") or {})
    if not bool(intraday.get("feature_ready")):
        return "Advance intraday history until the 4-minute feature path produces usable rows."
    if not bool(intraday.get("train_eligible")):
        return "Advance intraday history until the 4-minute path reaches 100 usable feature rows."
    daily = dict(paths.get("DailyHigh") or {})
    if not bool(daily.get("feature_ready")):
        return "Extend session-level history until the daily range path produces usable rows."
    if not bool(daily.get("train_eligible")):
        return "Extend session-level history until the daily range path reaches 15 usable rows."
    weekend = dict(paths.get("WeekEnd") or {})
    if not bool(weekend.get("feature_ready")):
        return "Extend daily / weekly history until the weekend path produces usable rows."
    if not bool(weekend.get("train_eligible")):
        return "Extend daily / weekly history until the weekend path reaches 6 usable rows."
    return "Train the eligible model modes for this symbol."


def assess_symbol_history(
    df_recent: pd.DataFrame | None,
    *,
    instrument_key: str | None = None,
    daily_long_history: pd.DataFrame | None = None,
) -> dict:
    sym = _safe_symbol(instrument_key)
    df_recent = df_recent if isinstance(df_recent, pd.DataFrame) else pd.DataFrame()
    df_long = daily_long_history if isinstance(daily_long_history, pd.DataFrame) else load_daily_long_history(sym)
    price_rows = int(len(df_recent)) if isinstance(df_recent, pd.DataFrame) else 0
    first_ts, last_ts = _frame_bounds(df_recent)
    long_rows = int(len(df_long)) if isinstance(df_long, pd.DataFrame) else 0

    intraday_rows = 0
    daily_rows = 0
    weekend_rows = 0
    intraday_error = ""
    daily_error = ""
    weekend_error = ""

    if price_rows > 0:
        try:
            intraday_rows = int(len(build_features_for_training(df_recent, instrument_key=sym)))
        except Exception as exc:
            intraday_error = str(exc)
    daily_source = df_long if not df_long.empty else df_recent
    daily_source_label = "daily_long_history" if not df_long.empty else "recent_intraday"
    if not daily_source.empty:
        try:
            daily_rows = int(len(build_next_session_dataset(daily_source, instrument_key=sym)))
        except Exception as exc:
            daily_error = str(exc)
        try:
            weekend_rows = int(len(build_weekend_session_dataset(daily_source, instrument_key=sym)))
        except Exception as exc:
            weekend_error = str(exc)

    paths = {
        "4min": {
            "mode": "4min",
            "label": MODE_REQUIREMENTS["4min"]["label"],
            **_path_status(
                MODE_REQUIREMENTS["4min"]["label"],
                intraday_rows,
                MODE_REQUIREMENTS["4min"]["min_train_rows"],
                build_error=intraday_error,
                source_label="recent_intraday",
            ),
        },
        "DailyHigh": {
            "mode": "DailyHigh",
            "label": MODE_REQUIREMENTS["DailyHigh"]["label"],
            **_path_status(
                MODE_REQUIREMENTS["DailyHigh"]["label"],
                daily_rows,
                MODE_REQUIREMENTS["DailyHigh"]["min_train_rows"],
                build_error=daily_error,
                source_label=daily_source_label,
            ),
        },
        "WeekEnd": {
            "mode": "WeekEnd",
            "label": MODE_REQUIREMENTS["WeekEnd"]["label"],
            **_path_status(
                MODE_REQUIREMENTS["WeekEnd"]["label"],
                weekend_rows,
                MODE_REQUIREMENTS["WeekEnd"]["min_train_rows"],
                build_error=weekend_error,
                source_label=daily_source_label,
            ),
        },
    }
    feature_ready_modes = [mode for mode, payload in paths.items() if bool(payload.get("feature_ready"))]
    eligible_modes = [mode for mode, payload in paths.items() if bool(payload.get("train_eligible"))]
    if price_rows <= 0:
        status_key = "missing"
        status_label = "No canonical price history"
        detail = "Canonical price history is not staged for this symbol yet."
    elif not feature_ready_modes:
        status_key = "insufficient_history"
        status_label = "Insufficient history"
        detail = "Price history exists, but none of the supported feature paths can yet build usable rows."
    elif not eligible_modes:
        status_key = "feature_partial"
        status_label = "Features partially ready"
        detail = "Some feature paths can build usable rows, but none have reached their current training floors yet."
    elif set(eligible_modes) >= {"4min", "DailyHigh", "WeekEnd"}:
        status_key = "multi_horizon_eligible"
        status_label = "Multi-horizon eligible"
        detail = "The current staged history is sufficient for intraday, daily, and weekend training paths."
    elif "4min" in eligible_modes:
        status_key = "advisory_core_eligible"
        status_label = "Core advisory path eligible"
        detail = "The 4-minute path has enough history for training, while longer horizons remain partial or deferred."
    else:
        status_key = "partial_horizon_eligible"
        status_label = "Partial horizon eligibility"
        detail = "Some higher-horizon paths are eligible, but the core 4-minute advisory path is still below its training floor."
    return {
        "symbol": sym,
        "status_key": status_key,
        "status_label": status_label,
        "detail": detail,
        "price_rows": int(price_rows),
        "first_ts": first_ts,
        "last_ts": last_ts,
        "span_days": int(_time_span_days(df_recent)),
        "session_count": int(_session_count(df_recent)),
        "daily_long_rows": int(long_rows),
        "daily_long_first_ts": _frame_bounds(df_long)[0],
        "daily_long_last_ts": _frame_bounds(df_long)[1],
        "daily_long_span_days": int(_time_span_days(df_long)),
        "daily_source_label": daily_source_label,
        "feature_ready_modes": feature_ready_modes,
        "eligible_modes": eligible_modes,
        "blocked_modes": [mode for mode in MODE_REQUIREMENTS if mode not in eligible_modes],
        "primary_next_requirement": _primary_next_requirement(price_rows, paths),
        "paths": paths,
    }


def build_training_plan(history_assessment: dict | None, *, train_requested: bool = True) -> dict:
    history = dict(history_assessment or {})
    paths = dict(history.get("paths") or {})
    mode_requests = {}
    eligible_modes = []
    eligible_raw_modes = []
    deferred_modes = []
    blocked_modes = []
    for mode, spec in MODE_REQUIREMENTS.items():
        path = dict(paths.get(mode) or {})
        if not train_requested:
            status = "skipped"
            reason = "Training was not requested for this onboarding pass."
        elif bool(path.get("train_eligible")):
            status = "eligible"
            reason = path.get("detail") or f"{spec['label']} is eligible for training."
            eligible_modes.append(mode)
            eligible_raw_modes.extend(list(spec.get("raw_modes") or []))
        elif bool(path.get("feature_ready")):
            status = "deferred"
            reason = path.get("detail") or f"{spec['label']} can build features, but history is still below the training floor."
            deferred_modes.append(mode)
        else:
            status = "blocked"
            reason = path.get("detail") or f"{spec['label']} is blocked by missing or insufficient history."
            blocked_modes.append(mode)
        mode_requests[mode] = {
            "mode": mode,
            "label": str(spec.get("label") or mode),
            "status": status,
            "reason": _safe_text(reason),
            "raw_modes": list(spec.get("raw_modes") or []),
            "feature_rows": _safe_int(path.get("feature_rows")),
            "min_train_rows": _safe_int(path.get("min_train_rows")),
        }
    if not train_requested:
        status = "skipped"
        summary = "Training was skipped for this onboarding pass."
    elif eligible_modes and not deferred_modes and not blocked_modes:
        status = "eligible"
        summary = f"All supported paths are currently training-eligible: {', '.join(eligible_modes)}."
    elif eligible_modes:
        status = "partial"
        summary = f"Training can proceed for {', '.join(eligible_modes)} while other paths remain deferred or blocked."
    elif deferred_modes:
        status = "deferred"
        summary = "Feature rows exist for some paths, but no mode has yet reached its training floor."
    else:
        status = "blocked"
        summary = "No model path can train yet because staged history is still insufficient."
    return {
        "train_requested": bool(train_requested),
        "status": status,
        "summary": summary,
        "eligible_modes": eligible_modes,
        "eligible_raw_modes": eligible_raw_modes,
        "deferred_modes": deferred_modes,
        "blocked_modes": blocked_modes,
        "mode_requests": mode_requests,
        "primary_next_requirement": _safe_text(history.get("primary_next_requirement")),
    }
