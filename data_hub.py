from __future__ import annotations

import threading
import time
import uuid
from collections import defaultdict
from typing import Callable, Iterable

import pandas as pd

from config import AUTO_PREDICT_SECONDS
from data_fetch import refresh_support_data, update_price_data
from model_utils import _staleness_info
from model_zoo import build_model_runtime, summarize_model_stack
from news_pipeline import build_news_runtime_summary, get_news_health
from readiness_tracker import load_readiness_state, summarize_deployment_readiness
from symbol_lifecycle import build_symbol_lifecycle_snapshot


def _safe_symbol(symbol: str | None) -> str:
    return str(symbol or "").strip().upper()


class DataHub:
    """
    Lightweight shared runtime owner for symbol refresh state, freshness, cached runtime bundles,
    and refresh fanout. This deliberately wraps the existing file-backed pipelines instead of
    replacing them.
    """

    def __init__(
        self,
        *,
        refresh_ttl_sec: float | None = None,
        bundle_ttl_sec: float | None = None,
    ):
        self._lock = threading.RLock()
        self._symbol_locks: dict[str, threading.Lock] = {}
        self._bundles: dict[str, dict] = {}
        self._states: dict[str, dict] = {}
        self._subscribers: dict[str, dict[str, Callable[[dict], None]]] = defaultdict(dict)
        self._refresh_ttl_sec = float(refresh_ttl_sec if refresh_ttl_sec is not None else max(30, int(AUTO_PREDICT_SECONDS * 0.5)))
        self._bundle_ttl_sec = float(bundle_ttl_sec if bundle_ttl_sec is not None else max(300, int(AUTO_PREDICT_SECONDS * 3)))

    @staticmethod
    def _bundle_age_sec(bundle: dict | None) -> float | None:
        if not isinstance(bundle, dict):
            return None
        loaded_at = bundle.get("loaded_at")
        try:
            return max(0.0, time.time() - float(loaded_at or 0.0))
        except Exception:
            return None

    def _symbol_lock(self, symbol: str) -> threading.Lock:
        sym = _safe_symbol(symbol)
        with self._lock:
            lock = self._symbol_locks.get(sym)
            if lock is None:
                lock = threading.Lock()
                self._symbol_locks[sym] = lock
            return lock

    def subscribe(self, symbol: str, callback: Callable[[dict], None]) -> str:
        sym = _safe_symbol(symbol)
        token = str(uuid.uuid4())
        with self._lock:
            self._subscribers[sym][token] = callback
        return token

    def unsubscribe(self, symbol: str, token: str):
        sym = _safe_symbol(symbol)
        with self._lock:
            callbacks = self._subscribers.get(sym)
            if isinstance(callbacks, dict):
                callbacks.pop(str(token), None)

    def _notify(self, symbol: str, payload: dict):
        sym = _safe_symbol(symbol)
        with self._lock:
            callbacks = list((self._subscribers.get(sym) or {}).values())
        for callback in callbacks:
            try:
                callback(dict(payload or {}))
            except Exception:
                continue

    def _freshness_snapshot(self, symbol: str, df: pd.DataFrame | None, news_health: dict | None = None) -> dict:
        news = dict(news_health or {})
        staleness = {}
        if isinstance(df, pd.DataFrame) and not df.empty:
            try:
                staleness = dict(_staleness_info(df, instrument_key=symbol) or {})
            except Exception:
                staleness = {}
        return {
            "price": staleness,
            "news": {
                "status": str(news.get("status") or ""),
                "headline_state": str(news.get("headline_state") or ""),
                "history_state": str(news.get("history_state") or ""),
                "age_hours": news.get("age_hours"),
            },
            "inputs_fresh": bool(
                not bool(staleness.get("is_stale"))
                and str(news.get("headline_state") or "").lower() not in {"failed", "stale", "error", "missing"}
            ),
        }

    @staticmethod
    def _aggregate_support_status(support_refresh: dict | None, news_health: dict | None = None, loaded_at: float | None = None) -> dict:
        support = dict(support_refresh or {})
        component_names = [
            "news",
            "context",
            "macro",
            "event",
            "options",
            "breadth",
            "catalyst",
        ]
        components: dict[str, dict] = {}
        statuses = []
        for name in component_names:
            payload = dict(support.get(name) or {})
            if name == "news":
                health = dict(news_health or {})
                if not payload:
                    payload = {
                        "status": str(health.get("status") or health.get("headline_state") or "unknown"),
                        "detail": str(health.get("headline_state") or health.get("history_state") or health.get("status") or "-"),
                    }
                if health:
                    payload.setdefault("headline_state", str(health.get("headline_state") or ""))
                    payload.setdefault("history_state", str(health.get("history_state") or ""))
                    payload.setdefault("age_hours", health.get("age_hours"))
            if loaded_at is not None and "updated_at" not in payload:
                payload["updated_at"] = loaded_at
            components[name] = payload
            status = str(payload.get("status") or "").strip().lower()
            if status:
                statuses.append(status)

        aggregate = "unknown"
        if statuses:
            if any(status in {"error", "failed"} for status in statuses):
                aggregate = "error"
            elif any(status in {"degraded", "stale", "missing", "thin", "warn"} for status in statuses):
                aggregate = "degraded"
            elif all(status in {"ok", "cached", "disabled", "skipped"} for status in statuses):
                aggregate = "ok"
            else:
                aggregate = "mixed"

        return {
            "status": aggregate,
            "updated_at": loaded_at,
            "components": components,
        }

    def get_runtime_bundle(self, symbol: str, *, allow_stale: bool = True) -> dict | None:
        sym = _safe_symbol(symbol)
        with self._lock:
            bundle = dict(self._bundles.get(sym) or {})
        if not bundle:
            return None
        age = self._bundle_age_sec(bundle)
        if age is None:
            age = float("inf")
        if not allow_stale and age > float(self._bundle_ttl_sec):
            return None
        bundle["age_sec"] = age
        return bundle

    def get_runtime_state(self, symbol: str) -> dict:
        sym = _safe_symbol(symbol)
        with self._lock:
            state = dict(self._states.get(sym) or {"symbol": sym, "status": "idle"})
        bundle = self.get_runtime_bundle(sym, allow_stale=True)
        if bundle:
            state.setdefault("loaded_at", bundle.get("loaded_at"))
            state.setdefault("bundle_age_sec", bundle.get("age_sec"))
            state.setdefault("freshness", dict(bundle.get("freshness") or {}))
        return state

    def invalidate_symbol(self, symbol: str, *, clear_state: bool = False, reason: str | None = None) -> dict:
        sym = _safe_symbol(symbol)
        if not sym:
            return {"symbol": "", "status": "invalid"}
        with self._lock:
            self._bundles.pop(sym, None)
            state = dict(self._states.get(sym) or {"symbol": sym})
            if clear_state:
                state = {"symbol": sym}
            state.update(
                {
                    "status": "invalidated",
                    "reason": str(reason or "manual"),
                    "updated_at": time.time(),
                }
            )
            self._states[sym] = state
        return dict(state)

    def snapshot_symbols(
        self,
        symbols: Iterable[str],
        *,
        allow_stale: bool = True,
        prefetch_missing: bool = False,
        force: bool = False,
        reason: str | None = None,
        load_models: bool = True,
    ) -> dict[str, dict]:
        out: dict[str, dict] = {}
        seen = set()
        for raw_symbol in list(symbols or []):
            sym = _safe_symbol(raw_symbol)
            if not sym or sym in seen:
                continue
            seen.add(sym)
            bundle = self.get_runtime_bundle(sym, allow_stale=allow_stale)
            if bundle is None and prefetch_missing:
                try:
                    bundle = self.refresh_symbol(
                        sym,
                        force=force,
                        reason=str(reason or "snapshot_prefetch"),
                        load_models=load_models,
                    )
                except Exception as exc:
                    out[sym] = {
                        "symbol": sym,
                        "error": str(exc),
                        "state": self.get_runtime_state(sym),
                    }
                    continue
            if bundle is None:
                out[sym] = {"symbol": sym, "state": self.get_runtime_state(sym)}
                continue
            payload = dict(bundle)
            payload["state"] = self.get_runtime_state(sym)
            out[sym] = payload
        return out

    def store_runtime_bundle(
        self,
        symbol: str,
        *,
        df: pd.DataFrame | None,
        models: dict | None = None,
        news_health: dict | None = None,
        news_runtime: dict | None = None,
        support_refresh: dict | None = None,
        deployment_meta: dict | None = None,
        source: str | None = None,
    ) -> dict:
        sym = _safe_symbol(symbol)
        models = dict(models or {})
        news_health = dict(news_health or get_news_health(sym))
        deployment_meta = dict(deployment_meta or summarize_deployment_readiness(sym, state=load_readiness_state(sym) or {}, news_health=news_health))
        freshness = self._freshness_snapshot(sym, df, news_health=news_health)
        loaded_at = time.time()
        support_state = self._aggregate_support_status(support_refresh, news_health=news_health, loaded_at=loaded_at)
        model_summary = summarize_model_stack(models)
        lifecycle_meta = build_symbol_lifecycle_snapshot(
            sym,
            runtime_bundle={
                "symbol": sym,
                "df": df,
                "models": models,
                "freshness": freshness,
                "news_health": news_health,
            },
            deployment_summary=deployment_meta,
            news_health=news_health,
            model_summary=model_summary,
        )
        bundle = {
            "symbol": sym,
            "loaded_at": loaded_at,
            "df": df.copy() if isinstance(df, pd.DataFrame) else df,
            "models": models,
            "model_summary": model_summary,
            "news_health": news_health,
            "news_runtime": dict(news_runtime or build_news_runtime_summary(sym, news_health=news_health)),
            "support_refresh": dict(support_refresh or {}),
            "support_state": support_state,
            "deployment_meta": deployment_meta,
            "lifecycle_meta": lifecycle_meta,
            "freshness": freshness,
            "data_source": getattr(df, "attrs", {}).get("data_source") if isinstance(df, pd.DataFrame) else None,
            "data_error": getattr(df, "attrs", {}).get("data_error") if isinstance(df, pd.DataFrame) else None,
            "source": str(source or "manual"),
        }
        state = {
            "symbol": sym,
            "status": "ready",
            "loaded_at": bundle.get("loaded_at"),
            "bundle_age_sec": 0.0,
            "freshness": freshness,
            "data_source": bundle.get("data_source"),
            "last_error": None,
            "support_status": dict(support_state),
            "deployment_stage": str((deployment_meta or {}).get("stage_key") or ""),
            "validation_status": str((deployment_meta or {}).get("validation_status") or ""),
            "validation_method": str((deployment_meta or {}).get("validation_method") or ""),
            "lifecycle": {
                "label": str((lifecycle_meta or {}).get("lifecycle_label") or ""),
                "operator_state_label": str((lifecycle_meta or {}).get("operator_state_label") or ""),
                "evidence_state_label": str((lifecycle_meta or {}).get("evidence_state_label") or ""),
                "next_requirement": str((lifecycle_meta or {}).get("primary_next_requirement") or ""),
            },
        }
        with self._lock:
            self._bundles[sym] = bundle
            self._states[sym] = state
        self._notify(sym, bundle)
        return bundle

    def refresh_symbol(
        self,
        symbol: str,
        *,
        force: bool = False,
        reason: str | None = None,
        load_models: bool = True,
    ) -> dict:
        sym = _safe_symbol(symbol)
        cached = self.get_runtime_bundle(sym, allow_stale=True)
        if not force and cached:
            age = self._bundle_age_sec(cached)
            if age is None:
                age = float("inf")
            if age <= float(self._refresh_ttl_sec):
                cached = dict(cached)
                cached["cache_hit"] = True
                return cached

        lock = self._symbol_lock(sym)
        with lock:
            cached = self.get_runtime_bundle(sym, allow_stale=True)
            if not force and cached:
                age = self._bundle_age_sec(cached)
                if age is None:
                    age = float("inf")
                if age <= float(self._refresh_ttl_sec):
                    cached = dict(cached)
                    cached["cache_hit"] = True
                    return cached
            with self._lock:
                self._states[sym] = {
                    "symbol": sym,
                    "status": "refreshing",
                    "reason": str(reason or "manual"),
                    "started_at": time.time(),
                    "last_error": None,
                }
            try:
                df = update_price_data(sym)
                if load_models:
                    runtime = build_model_runtime(sym, include_modes=True)
                    models = dict(runtime.get("models") or {})
                else:
                    models = dict((cached or {}).get("models") or {})
                support_refresh = {}
                if isinstance(df, pd.DataFrame):
                    support_refresh = dict(getattr(df, "attrs", {}).get("support_refresh") or {})
                if not support_refresh:
                    current_price = None
                    try:
                        if isinstance(df, pd.DataFrame) and not df.empty and "close" in df.columns:
                            current_price = float(pd.to_numeric(df["close"], errors="coerce").iloc[-1])
                    except Exception:
                        current_price = None
                    support_refresh = dict(refresh_support_data(sym, current_price=current_price, force=False) or {})
                news_health = get_news_health(sym)
                news_runtime = build_news_runtime_summary(sym, news_health=news_health)
                deployment_meta = summarize_deployment_readiness(sym, state=load_readiness_state(sym) or {}, news_health=news_health)
                bundle = self.store_runtime_bundle(
                    sym,
                    df=df,
                    models=models,
                    news_health=news_health,
                    news_runtime=news_runtime,
                    support_refresh=support_refresh,
                    deployment_meta=deployment_meta,
                    source=str(reason or "manual"),
                )
                bundle["cache_hit"] = False
                return bundle
            except Exception as exc:
                with self._lock:
                    self._states[sym] = {
                        "symbol": sym,
                        "status": "error",
                        "reason": str(reason or "manual"),
                        "last_error": str(exc),
                        "updated_at": time.time(),
                    }
                raise


_DATA_HUB: DataHub | None = None
_DATA_HUB_LOCK = threading.Lock()


def get_data_hub() -> DataHub:
    global _DATA_HUB
    if _DATA_HUB is not None:
        return _DATA_HUB
    with _DATA_HUB_LOCK:
        if _DATA_HUB is None:
            _DATA_HUB = DataHub()
    return _DATA_HUB
