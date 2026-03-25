from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

from app_meta import APP_VERSION, GITHUB_RELEASES_API_URL, GITHUB_RELEASES_REPO, GITHUB_RELEASES_URL
from credential_store import get_local_settings_dir


UPDATE_CHECK_TIMEOUT_SEC = 6.0
UPDATE_CACHE_TTL_HOURS = 8.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_path():
    return get_local_settings_dir() / "update_status.json"


def normalize_version(value: str | None) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("v"):
        text = text[1:]
    return text


def _version_key(value: str | None) -> tuple:
    text = normalize_version(value)
    if not text:
        return tuple()
    parts = []
    for chunk in text.replace("-", ".").split("."):
        if not chunk:
            continue
        num = "".join(ch for ch in chunk if ch.isdigit())
        if num:
            parts.append(int(num))
        else:
            parts.append(chunk.lower())
    return tuple(parts)


def is_newer_version(latest: str | None, current: str | None = None) -> bool:
    current = APP_VERSION if current is None else current
    return _version_key(latest) > _version_key(current)


def _read_cache() -> dict:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_cache(payload: dict):
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass


def _fetch_latest_release() -> dict:
    if not GITHUB_RELEASES_API_URL:
        raise RuntimeError("GitHub release repository is not configured")
    req = urllib.request.Request(
        GITHUB_RELEASES_API_URL,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "LightningTrader4000 Update Checker",
        },
    )
    with urllib.request.urlopen(req, timeout=UPDATE_CHECK_TIMEOUT_SEC) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return {
        "latest_version": normalize_version(payload.get("tag_name") or payload.get("name")),
        "release_name": str(payload.get("name") or payload.get("tag_name") or "").strip(),
        "release_url": str(payload.get("html_url") or GITHUB_RELEASES_URL),
        "published_at": payload.get("published_at"),
        "source_kind": "live",
        "repo": GITHUB_RELEASES_REPO,
    }


def get_update_status(force: bool = False) -> dict:
    cache = _read_cache()
    now = datetime.now(timezone.utc)
    if not force and cache:
        try:
            checked_at = datetime.fromisoformat(str(cache.get("checked_at")))
            if checked_at.tzinfo is None:
                checked_at = checked_at.replace(tzinfo=timezone.utc)
            if now - checked_at <= timedelta(hours=UPDATE_CACHE_TTL_HOURS):
                return dict(cache)
        except Exception:
            pass
    try:
        live = _fetch_latest_release()
        latest_version = normalize_version(live.get("latest_version"))
        status = "update_available" if is_newer_version(latest_version, APP_VERSION) else "up_to_date"
        message = (
            f"Version {latest_version} is available."
            if status == "update_available"
            else f"You are up to date on {normalize_version(APP_VERSION)}."
        )
        payload = {
            "status": status,
            "current_version": normalize_version(APP_VERSION),
            "latest_version": latest_version,
            "release_name": live.get("release_name"),
            "release_url": live.get("release_url") or GITHUB_RELEASES_URL,
            "published_at": live.get("published_at"),
            "checked_at": _utc_now(),
            "source_kind": "live",
            "message": message,
            "repo": GITHUB_RELEASES_REPO,
        }
        _write_cache(payload)
        return payload
    except urllib.error.URLError:
        cache = _read_cache()
        if cache:
            cache = dict(cache)
            cache["status"] = "cached"
            cache["source_kind"] = "cache"
            cache["message"] = str(cache.get("message") or "Showing the last cached update check.")
            return cache
        return {
            "status": "unavailable",
            "current_version": normalize_version(APP_VERSION),
            "latest_version": None,
            "release_url": GITHUB_RELEASES_URL,
            "checked_at": _utc_now(),
            "source_kind": "unavailable",
            "message": "Update check could not reach GitHub right now.",
            "repo": GITHUB_RELEASES_REPO,
        }
    except Exception as exc:
        return {
            "status": "error",
            "current_version": normalize_version(APP_VERSION),
            "latest_version": None,
            "release_url": GITHUB_RELEASES_URL,
            "checked_at": _utc_now(),
            "source_kind": "error",
            "message": str(exc) or "Update check failed.",
            "repo": GITHUB_RELEASES_REPO,
        }
