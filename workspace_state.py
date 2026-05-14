from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone

from config import DATA_DIR, DEFAULT_INSTRUMENT, INSTRUMENTS, TRADER_PROFILE_DEFAULT
from instrument_profile import build_instrument_meta, normalize_symbol


_LOCK = threading.RLock()
_VERSION = 1


def _state_path() -> str:
    return os.path.join(DATA_DIR, "workspace_state.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_workspace(symbol: str, meta: dict | None = None) -> dict:
    sym = normalize_symbol(symbol)
    payload = build_instrument_meta(sym, overrides=meta or INSTRUMENTS.get(sym))
    now_iso = _utc_now_iso()
    title = str(payload.get("display_name") or sym).strip() or sym
    return {
        "symbol": sym,
        "title": title,
        "display_name": title,
        "instrument_kind": str(payload.get("instrument_kind") or payload.get("type") or "equity"),
        "sector": str(payload.get("sector") or "Broad Market"),
        "industry": str(payload.get("industry") or payload.get("sector") or "-"),
        "opened": True,
        "pinned": bool(sym == normalize_symbol(DEFAULT_INSTRUMENT)),
        "created_at": now_iso,
        "updated_at": now_iso,
        "last_opened_at": now_iso,
        "view_mode": "4min",
        "profile_name": str(TRADER_PROFILE_DEFAULT or "balanced"),
        "last_status": "new",
        "last_note": "",
    }


def _normalize_state(payload: dict | None) -> dict:
    state = payload if isinstance(payload, dict) else {}
    workspaces = state.get("workspaces")
    if not isinstance(workspaces, dict):
        workspaces = {}
    order = [normalize_symbol(item) for item in list(state.get("order") or []) if normalize_symbol(item)]
    normalized: dict[str, dict] = {}
    for symbol, entry in workspaces.items():
        sym = normalize_symbol(symbol)
        if not sym:
            continue
        base = _default_workspace(sym, meta=entry if isinstance(entry, dict) else None)
        if isinstance(entry, dict):
            base.update(entry)
        base["symbol"] = sym
        base["updated_at"] = str(base.get("updated_at") or _utc_now_iso())
        base["display_name"] = str(base.get("display_name") or base.get("title") or sym)
        base["title"] = str(base.get("title") or base.get("display_name") or sym)
        base["view_mode"] = str(base.get("view_mode") or "4min")
        base["profile_name"] = str(base.get("profile_name") or TRADER_PROFILE_DEFAULT or "balanced")
        base["opened"] = bool(base.get("opened", True))
        base["pinned"] = bool(base.get("pinned", False))
        normalized[sym] = base

    default_symbol = normalize_symbol(DEFAULT_INSTRUMENT) or "QQQ"
    if default_symbol not in normalized:
        normalized[default_symbol] = _default_workspace(default_symbol)

    seen = set()
    clean_order = []
    for sym in order + list(normalized.keys()):
        if sym in seen or sym not in normalized:
            continue
        seen.add(sym)
        clean_order.append(sym)

    active_symbol = normalize_symbol(state.get("active_symbol"))
    if active_symbol not in normalized:
        active_symbol = default_symbol
    normalized[active_symbol]["opened"] = True

    return {
        "version": _VERSION,
        "active_symbol": active_symbol,
        "order": clean_order,
        "workspaces": normalized,
    }


def load_workspace_state() -> dict:
    path = _state_path()
    with _LOCK:
        if not os.path.exists(path):
            return _normalize_state({})
        try:
            with open(path, "r") as handle:
                payload = json.load(handle)
        except Exception:
            payload = {}
        return _normalize_state(payload)


def save_workspace_state(state: dict) -> dict:
    normalized = _normalize_state(state)
    path = _state_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with _LOCK:
        with open(tmp, "w") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
        os.replace(tmp, path)
    return normalized


def get_workspace(symbol: str) -> dict | None:
    sym = normalize_symbol(symbol)
    if not sym:
        return None
    state = load_workspace_state()
    entry = state.get("workspaces", {}).get(sym)
    return dict(entry) if isinstance(entry, dict) else None


def open_workspace(
    symbol: str,
    meta: dict | None = None,
    *,
    activate: bool = True,
    pinned: bool | None = None,
    title: str | None = None,
) -> dict:
    sym = normalize_symbol(symbol)
    if not sym:
        return load_workspace_state()
    state = load_workspace_state()
    workspaces = dict(state.get("workspaces") or {})
    entry = dict(workspaces.get(sym) or _default_workspace(sym, meta=meta))
    if isinstance(meta, dict):
        payload = build_instrument_meta(sym, overrides=meta)
        entry["display_name"] = str(payload.get("display_name") or entry.get("display_name") or sym)
        entry["title"] = str(title or payload.get("display_name") or entry.get("title") or sym)
        entry["instrument_kind"] = str(payload.get("instrument_kind") or entry.get("instrument_kind") or "equity")
        entry["sector"] = str(payload.get("sector") or entry.get("sector") or "Broad Market")
        entry["industry"] = str(payload.get("industry") or entry.get("industry") or "-")
    elif title:
        entry["title"] = str(title)
    if pinned is not None:
        entry["pinned"] = bool(pinned)
    entry["opened"] = True
    entry["last_opened_at"] = _utc_now_iso()
    entry["updated_at"] = entry["last_opened_at"]
    workspaces[sym] = entry

    order = [item for item in list(state.get("order") or []) if normalize_symbol(item) != sym]
    order.insert(0, sym)
    state["order"] = order
    state["workspaces"] = workspaces
    if activate:
        state["active_symbol"] = sym
    return save_workspace_state(state)


def activate_workspace(symbol: str) -> dict:
    return open_workspace(symbol, activate=True)


def close_workspace(symbol: str) -> dict:
    sym = normalize_symbol(symbol)
    state = load_workspace_state()
    workspaces = dict(state.get("workspaces") or {})
    if sym not in workspaces:
        return state
    workspaces[sym]["opened"] = False
    workspaces[sym]["updated_at"] = _utc_now_iso()
    opened_symbols = [item for item in list(state.get("order") or []) if item != sym and bool(workspaces.get(item, {}).get("opened"))]
    active_symbol = normalize_symbol(state.get("active_symbol"))
    if active_symbol == sym:
        fallback = opened_symbols[0] if opened_symbols else normalize_symbol(DEFAULT_INSTRUMENT) or sym
        state["active_symbol"] = fallback
        if fallback in workspaces:
            workspaces[fallback]["opened"] = True
    state["workspaces"] = workspaces
    return save_workspace_state(state)


def remember_workspace_view(symbol: str, *, mode: str | None = None, profile_name: str | None = None, note: str | None = None, status: str | None = None) -> dict:
    sym = normalize_symbol(symbol)
    state = load_workspace_state()
    workspaces = dict(state.get("workspaces") or {})
    entry = dict(workspaces.get(sym) or _default_workspace(sym))
    if mode is not None:
        entry["view_mode"] = str(mode)
    if profile_name is not None:
        entry["profile_name"] = str(profile_name)
    if note is not None:
        entry["last_note"] = str(note)
    if status is not None:
        entry["last_status"] = str(status)
    entry["updated_at"] = _utc_now_iso()
    workspaces[sym] = entry
    state["workspaces"] = workspaces
    return save_workspace_state(state)


def list_workspaces(open_only: bool = True) -> list[dict]:
    def _sort_ts(item: dict) -> float:
        raw = str(item.get("last_opened_at") or item.get("updated_at") or "1970-01-01T00:00:00+00:00")
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except Exception:
            return 0.0

    state = load_workspace_state()
    workspaces = dict(state.get("workspaces") or {})
    active = normalize_symbol(state.get("active_symbol"))
    ordered = []
    for sym in list(state.get("order") or []):
        entry = workspaces.get(sym)
        if not isinstance(entry, dict):
            continue
        if open_only and not bool(entry.get("opened")):
            continue
        ordered.append(dict(entry))
    for sym, entry in workspaces.items():
        if any(str(item.get("symbol")) == sym for item in ordered):
            continue
        if open_only and not bool(entry.get("opened")):
            continue
        ordered.append(dict(entry))
    ordered.sort(
        key=lambda item: (
            0 if normalize_symbol(item.get("symbol")) == active else 1,
            0 if bool(item.get("pinned")) else 1,
            -_sort_ts(item),
        )
    )
    return ordered


def list_workspace_symbols(open_only: bool = True, limit: int | None = None) -> list[str]:
    rows = [normalize_symbol(item.get("symbol")) for item in list_workspaces(open_only=open_only)]
    rows = [item for item in rows if item]
    if limit is None or limit <= 0:
        return rows
    return rows[: int(limit)]
