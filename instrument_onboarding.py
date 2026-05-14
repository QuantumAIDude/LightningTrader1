#!/usr/bin/env python3
"""
instrument_onboarding.py

Runtime discovery and staged onboarding for new instruments.

Workflow:
1) Drop CSV files into data/new_instruments/<SYMBOL>.csv
2) The onboarding pass normalizes + saves data to data/prices_<SYMBOL>.csv
3) Instrument metadata is registered in data/instruments.json
4) Models are bootstrapped (and optionally fully retrained) for the symbol
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
from datetime import datetime, timezone

import pandas as pd

from config import (
    DATA_DIR,
    INSTRUMENTS_FILE,
    INSTRUMENTS,
    AUTO_ONBOARD_FROM_STAGING,
    AUTO_ONBOARD_BOOTSTRAP,
    AUTO_ONBOARD_FULL_TRAIN,
    AUTO_ONBOARD_COOLDOWN_SEC,
    NEW_INSTRUMENTS_DIR,
    NEW_INSTRUMENTS_PROCESSED_DIR,
    NEW_INSTRUMENTS_FAILED_DIR,
)
from instrument_profile import build_instrument_meta
from symbol_pipeline import assess_symbol_history, ensure_daily_long_history

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_LAST_RUN_TS = 0.0
_PROVIDER_SUFFIXES = {"alpaca", "yfinance", "auto", "ibkr"}
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9][A-Z0-9\.\-]{0,19}$")


def _default_context_daily_symbols(symbol: str) -> dict:
    return dict(build_instrument_meta(symbol).get("context_daily_symbols") or {})


def _default_equity_meta(symbol: str) -> dict:
    return build_instrument_meta(symbol)


def _canonical_price_path(symbol: str) -> str:
    safe = str(symbol or "").replace("/", "_").replace("\\", "_").upper()
    return os.path.join(DATA_DIR, f"prices_{safe}.csv")


def _load_registry_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            return payload
    except Exception:
        logger.exception("Failed reading instruments file")
    return {}


def _save_registry_file(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def register_instrument(symbol: str, meta: dict | None = None, persist: bool = True) -> bool:
    sym = str(symbol or "").strip().upper()
    if not sym:
        return False
    entry = build_instrument_meta(sym, overrides=meta or {})
    aliases = entry.get("news_aliases")
    if isinstance(aliases, (list, tuple, set)):
        clean_aliases = []
        for item in aliases:
            text = str(item or "").strip()
            if text:
                clean_aliases.append(text)
        entry["news_aliases"] = clean_aliases or [sym]
    else:
        entry["news_aliases"] = [sym]
    added = sym not in INSTRUMENTS
    INSTRUMENTS[sym] = entry
    if persist:
        payload = _load_registry_file(INSTRUMENTS_FILE)
        payload[sym] = entry
        _save_registry_file(INSTRUMENTS_FILE, payload)
    return added


def _symbol_from_price_filename(path: str) -> str | None:
    name = os.path.basename(path)
    m = re.fullmatch(r"prices_(.+)\.csv", name)
    if not m:
        return None
    raw = str(m.group(1) or "")
    if not raw:
        return None
    parts = raw.rsplit("_", 1)
    if len(parts) == 2 and parts[1].lower() in _PROVIDER_SUFFIXES:
        return None
    symbol = raw.upper().strip()
    if not _SYMBOL_PATTERN.match(symbol):
        return None
    return symbol


def refresh_registry_from_price_files() -> list[str]:
    """Discover symbols from canonical price files and register missing entries."""
    discovered = []
    try:
        names = os.listdir(DATA_DIR)
    except Exception:
        names = []
    for name in names:
        sym = _symbol_from_price_filename(name)
        if not sym:
            continue
        if sym not in INSTRUMENTS:
            register_instrument(sym, persist=True)
            discovered.append(sym)
    return sorted(set(discovered))


def list_runtime_instruments() -> list[str]:
    return sorted([str(k) for k in INSTRUMENTS.keys()])


def _coerce_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()
    try:
        if isinstance(out.columns, pd.MultiIndex):
            out.columns = out.columns.get_level_values(0)
    except Exception:
        pass
    out.columns = [str(c).strip().lower().replace(" ", "_") for c in out.columns]
    # detect timestamp column
    ts_col = None
    for c in ("timestamp", "datetime", "date", "time"):
        if c in out.columns:
            ts_col = c
            break
    if ts_col is None:
        # assume first column is timestamp
        ts_col = out.columns[0]
    out[ts_col] = pd.to_datetime(out[ts_col], utc=True, errors="coerce")
    out = out.dropna(subset=[ts_col]).set_index(ts_col)

    # normalize common column names
    if "adj_close" in out.columns and "close" not in out.columns:
        out = out.rename(columns={"adj_close": "close"})
    keep = [c for c in ("open", "high", "low", "close", "volume") if c in out.columns]
    if "close" not in keep:
        return pd.DataFrame()
    out = out[keep].copy()
    for c in ("open", "high", "low", "close", "volume"):
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
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
    out = out.sort_index()
    try:
        out = out[~out.index.duplicated(keep="last")]
    except Exception:
        pass
    return out


def _read_staged_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except Exception:
        logger.exception("Failed reading staged instrument file: %s", path)
        return pd.DataFrame()
    return _coerce_ohlcv(df)


def _safe_symbol_from_staged_filename(path: str) -> str | None:
    base = os.path.splitext(os.path.basename(path))[0]
    if base.lower().endswith("_news"):
        base = base[:-5]
    if base.lower().endswith(".meta"):
        base = base[:-5]
    symbol = str(base).upper().strip()
    if _SYMBOL_PATTERN.match(symbol):
        return symbol
    return None


def _load_staged_meta(symbol: str) -> dict:
    candidates = [
        os.path.join(NEW_INSTRUMENTS_DIR, f"{symbol}.meta.json"),
        os.path.join(NEW_INSTRUMENTS_DIR, f"{symbol}.json"),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r") as f:
                payload = json.load(f)
            if isinstance(payload, dict):
                return payload
        except Exception:
            logger.exception("Failed reading staged meta for %s", symbol)
    return {}


def _staged_news_files(symbol: str) -> list[str]:
    candidates = [
        os.path.join(NEW_INSTRUMENTS_DIR, f"{symbol}_news.csv"),
        os.path.join(NEW_INSTRUMENTS_DIR, f"{symbol}_news.jsonl"),
        os.path.join(NEW_INSTRUMENTS_DIR, f"news_{symbol}.csv"),
        os.path.join(NEW_INSTRUMENTS_DIR, f"news_{symbol}.jsonl"),
    ]
    return [path for path in candidates if os.path.exists(path)]


def _merge_price_frames(existing: pd.DataFrame, incoming: pd.DataFrame) -> pd.DataFrame:
    if existing is None or existing.empty:
        return incoming
    if incoming is None or incoming.empty:
        return existing
    merged = pd.concat([existing, incoming], axis=0, sort=False)
    merged = merged[~merged.index.duplicated(keep="last")]
    merged = merged.sort_index()
    return merged


def _read_existing_price_file(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df.columns = [str(c).lower() for c in df.columns]
        return _coerce_ohlcv(df.reset_index().rename(columns={"index": "timestamp"}))
    except Exception:
        logger.exception("Failed reading existing price file: %s", path)
        return pd.DataFrame()


def _archive_staged_file(path: str, symbol: str):
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    os.makedirs(NEW_INSTRUMENTS_PROCESSED_DIR, exist_ok=True)
    ext = os.path.splitext(path)[1] or ".dat"
    base = os.path.splitext(os.path.basename(path))[0]
    out = os.path.join(NEW_INSTRUMENTS_PROCESSED_DIR, f"{base}_{ts}{ext}")
    try:
        shutil.move(path, out)
    except Exception:
        # Fallback: keep original file if move fails.
        logger.exception("Failed moving staged file to processed folder: %s", path)


def _safe_slug(value: str, default: str = "unknown") -> str:
    raw = str(value or "").strip()
    if not raw:
        raw = default
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", raw)
    slug = slug.strip("_.-")
    return slug[:48] if slug else default


def _quarantine_staged_file(path: str, symbol: str | None, reason: str) -> str | None:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    sym = _safe_slug(symbol or os.path.splitext(os.path.basename(path))[0], default="UNKNOWN")
    reason_slug = _safe_slug(reason, default="failed")
    os.makedirs(NEW_INSTRUMENTS_FAILED_DIR, exist_ok=True)
    out = os.path.join(NEW_INSTRUMENTS_FAILED_DIR, f"{sym}_{ts}_{reason_slug}.csv")
    meta = {
        "path": path,
        "symbol": symbol,
        "reason": reason,
        "quarantined_at": datetime.now(timezone.utc).isoformat(),
        "quarantine_path": out,
    }
    try:
        shutil.move(path, out)
        with open(f"{out}.json", "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
        return out
    except Exception:
        logger.exception("Failed quarantining staged file: %s", path)
        return None


def _models_exist_for_symbol(symbol: str) -> bool:
    from model_utils import load_models

    try:
        models = load_models(instrument_key=symbol)
    except Exception:
        return False
    required = ["base", "forest", "boost", "extra_trees", "regime"]
    for key in required:
        if models.get(key) is None:
            return False
    modes = models.get("modes", {}) if isinstance(models, dict) else {}
    for mode in ("daily_high", "daily_low", "weekend"):
        block = modes.get(mode, {})
        if not isinstance(block, dict) or not block:
            return False
    return True


def _train_symbol_if_needed(symbol: str, full_train: bool):
    if _models_exist_for_symbol(symbol):
        return
    if AUTO_ONBOARD_BOOTSTRAP:
        from bootstrap_models_fast import bootstrap_fast_models

        bootstrap_fast_models(symbol)
    if full_train:
        from model_trainers import train_all_models

        path = _canonical_price_path(symbol)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
            df = df.sort_index()
            if not df.empty:
                train_all_models(df, instrument_key=symbol, train_horizons=True)


def process_staged_instrument_files(
    auto_train: bool = True,
    full_train: bool | None = None,
) -> dict:
    """
    Ingest staged CSV files from NEW_INSTRUMENTS_DIR.
    Returns a summary payload for diagnostics.
    """
    if full_train is None:
        full_train = bool(AUTO_ONBOARD_FULL_TRAIN)
    os.makedirs(NEW_INSTRUMENTS_DIR, exist_ok=True)
    os.makedirs(NEW_INSTRUMENTS_PROCESSED_DIR, exist_ok=True)
    os.makedirs(NEW_INSTRUMENTS_FAILED_DIR, exist_ok=True)

    staged = []
    try:
        staged = [
            os.path.join(NEW_INSTRUMENTS_DIR, n)
            for n in sorted(os.listdir(NEW_INSTRUMENTS_DIR))
            if n.lower().endswith(".csv")
            and not n.lower().endswith("_news.csv")
            and not n.lower().startswith("news_")
        ]
    except Exception:
        staged = []

    out = {"processed": [], "errors": [], "quarantined": []}
    for path in staged:
        symbol = _safe_symbol_from_staged_filename(path)
        if not symbol:
            quarantine_path = _quarantine_staged_file(path, symbol=None, reason="invalid_symbol_filename")
            out["errors"].append({"path": path, "error": "invalid_symbol_filename"})
            if quarantine_path:
                out["quarantined"].append({"path": quarantine_path, "reason": "invalid_symbol_filename"})
            continue
        df_new = _read_staged_csv(path)
        if df_new.empty:
            quarantine_path = _quarantine_staged_file(path, symbol=symbol, reason="invalid_or_empty_csv")
            out["errors"].append({"path": path, "symbol": symbol, "error": "invalid_or_empty_csv"})
            if quarantine_path:
                out["quarantined"].append({"path": quarantine_path, "symbol": symbol, "reason": "invalid_or_empty_csv"})
            continue
        data_path = _canonical_price_path(symbol)
        df_old = _read_existing_price_file(data_path)
        df_all = _merge_price_frames(df_old, df_new)
        if df_all.empty:
            quarantine_path = _quarantine_staged_file(path, symbol=symbol, reason="no_rows_after_merge")
            out["errors"].append({"path": path, "symbol": symbol, "error": "no_rows_after_merge"})
            if quarantine_path:
                out["quarantined"].append({"path": quarantine_path, "symbol": symbol, "reason": "no_rows_after_merge"})
            continue
        df_all.to_csv(data_path, date_format="%Y-%m-%dT%H:%M:%SZ")
        daily_long_df, daily_long_summary = ensure_daily_long_history(
            symbol,
            recent_df=df_all,
            extra_frames=[df_new],
            persist=True,
        )
        meta = _load_staged_meta(symbol)
        added = register_instrument(symbol, meta=meta, persist=True)
        imported_news = None
        extra_news_count = 0
        try:
            from news_pipeline import import_local_historical_news

            extra_news = _staged_news_files(symbol)
            extra_news_count = int(len(extra_news))
            if extra_news:
                imported_news = import_local_historical_news(symbol, extra_paths=extra_news)
                for news_path in extra_news:
                    _archive_staged_file(news_path, symbol)
        except Exception:
            logger.exception("Staged news import failed for %s", symbol)
        try:
            meta_path = next(
                (
                    p
                    for p in [
                        os.path.join(NEW_INSTRUMENTS_DIR, f"{symbol}.meta.json"),
                        os.path.join(NEW_INSTRUMENTS_DIR, f"{symbol}.json"),
                    ]
                    if os.path.exists(p)
                ),
                None,
            )
            if meta_path:
                _archive_staged_file(meta_path, symbol)
        except Exception:
            logger.exception("Failed archiving staged meta for %s", symbol)
        trained = False
        if auto_train:
            try:
                _train_symbol_if_needed(symbol, full_train=bool(full_train))
                trained = True
            except Exception:
                logger.exception("Auto-train failed for %s", symbol)
        history = assess_symbol_history(df_all, instrument_key=symbol, daily_long_history=daily_long_df)
        _archive_staged_file(path, symbol)
        out["processed"].append(
            {
                "symbol": symbol,
                "rows": int(len(df_all)),
                "new_rows": int(len(df_new)),
                "registered": bool(added),
                "profile_resolved": True,
                "trained": bool(trained),
                "path": data_path,
                "daily_long_rows": int(daily_long_summary.get("rows") or 0),
                "daily_long_updated": bool(daily_long_summary.get("updated")),
                "daily_long_sources": list(daily_long_summary.get("sources_used") or []),
                "news_import": imported_news,
                "news_files_staged": int(extra_news_count),
                "history_status": str(history.get("status_label") or ""),
                "feature_ready_modes": list(history.get("feature_ready_modes") or []),
                "eligible_modes": list(history.get("eligible_modes") or []),
                "primary_next_requirement": str(history.get("primary_next_requirement") or ""),
                "status": "processed",
            }
        )
    return out


def ensure_runtime_instruments(force: bool = False, auto_onboard: bool = True) -> dict:
    """
    Periodically refresh runtime instruments from disk and optionally process staged files.
    Cooldown avoids expensive work on every UI refresh.
    """
    global _LAST_RUN_TS
    now = time.time()
    with _LOCK:
        if not force and (now - _LAST_RUN_TS) < max(5, int(AUTO_ONBOARD_COOLDOWN_SEC)):
            return {"discovered": [], "onboarded": {"processed": [], "errors": []}, "skipped": True}
        discovered = refresh_registry_from_price_files()
        onboarded = {"processed": [], "errors": []}
        if auto_onboard and AUTO_ONBOARD_FROM_STAGING:
            onboarded = process_staged_instrument_files(auto_train=True, full_train=AUTO_ONBOARD_FULL_TRAIN)
        _LAST_RUN_TS = now
    return {"discovered": discovered, "onboarded": onboarded, "skipped": False}
