from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app_meta import APP_VERSION, GITHUB_RELEASES_API_URL, GITHUB_RELEASES_REPO, GITHUB_RELEASES_URL
from credential_store import get_local_settings_dir


UPDATE_CHECK_TIMEOUT_SEC = 6.0
UPDATE_DOWNLOAD_TIMEOUT_SEC = 60.0
UPDATE_CACHE_TTL_HOURS = 8.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cache_path():
    return get_local_settings_dir() / "update_status.json"


def _download_dir() -> Path:
    path = get_local_settings_dir() / "updates"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    assets = []
    for item in list(payload.get("assets") or []):
        if not isinstance(item, dict):
            continue
        assets.append(
            {
                "name": str(item.get("name") or "").strip(),
                "browser_download_url": str(item.get("browser_download_url") or "").strip(),
                "content_type": str(item.get("content_type") or "").strip(),
                "size": item.get("size"),
            }
        )
    return {
        "latest_version": normalize_version(payload.get("tag_name") or payload.get("name")),
        "release_name": str(payload.get("name") or payload.get("tag_name") or "").strip(),
        "release_url": str(payload.get("html_url") or GITHUB_RELEASES_URL),
        "published_at": payload.get("published_at"),
        "assets": assets,
        "zipball_url": str(payload.get("zipball_url") or "").strip(),
        "tarball_url": str(payload.get("tarball_url") or "").strip(),
        "source_kind": "live",
        "repo": GITHUB_RELEASES_REPO,
    }


def _platform_tokens(platform_name: str | None = None) -> list[str]:
    name = str(platform_name or sys.platform or "").lower()
    if name.startswith("darwin"):
        return ["macos", "mac", "darwin", "osx"]
    if name.startswith("win"):
        return ["windows", "win", "exe", "msi"]
    if name.startswith("linux"):
        return ["linux", "appimage"]
    return [name] if name else []


def _safe_filename(value: str, fallback: str = "LightningTrader4000-update.zip") -> str:
    text = str(value or "").strip() or fallback
    text = re.sub(r"[^A-Za-z0-9 ._()-]+", "_", text)
    text = text.strip(" ._")
    if not text:
        text = fallback
    return text[:160]


def select_release_asset(release: dict | None, platform_name: str | None = None) -> dict:
    payload = dict(release or {})
    assets = [dict(item) for item in list(payload.get("assets") or []) if isinstance(item, dict)]
    tokens = _platform_tokens(platform_name)
    preferred_ext = (".zip", ".dmg", ".exe", ".msi", ".pkg")

    def _score(asset: dict) -> tuple[int, int, int]:
        name = str(asset.get("name") or "").lower()
        token_score = 4 if any(token and token in name for token in tokens) else 0
        ext_score = 2 if name.endswith(preferred_ext) else 0
        app_score = 1 if "lightning" in name or "trader" in name else 0
        return token_score + ext_score + app_score, ext_score, app_score

    downloadable = [item for item in assets if str(item.get("browser_download_url") or "").strip()]
    if downloadable:
        ranked = sorted(downloadable, key=_score, reverse=True)
        best = ranked[0]
        return {
            "kind": "asset",
            "name": _safe_filename(best.get("name")),
            "url": str(best.get("browser_download_url") or "").strip(),
            "size": best.get("size"),
            "content_type": best.get("content_type"),
        }
    zipball_url = str(payload.get("zipball_url") or "").strip()
    latest = normalize_version(payload.get("latest_version")) or "latest"
    repo_name = str(payload.get("repo") or GITHUB_RELEASES_REPO or "LightningTrader4000").split("/")[-1]
    if zipball_url:
        return {
            "kind": "source_zip",
            "name": _safe_filename(f"{repo_name}-v{latest}-source.zip"),
            "url": zipball_url,
            "size": None,
            "content_type": "application/zip",
        }
    return {"kind": "none", "name": "", "url": "", "size": None, "content_type": ""}


def _download_url_to_file(url: str, destination: Path) -> int:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream, application/vnd.github+json;q=0.8, */*;q=0.5",
            "User-Agent": "LightningTrader4000 Update Downloader",
        },
    )
    tmp = destination.with_suffix(destination.suffix + ".part")
    total = 0
    with urllib.request.urlopen(req, timeout=UPDATE_DOWNLOAD_TIMEOUT_SEC) as response:
        with open(tmp, "wb") as handle:
            while True:
                chunk = response.read(1024 * 256)
                if not chunk:
                    break
                handle.write(chunk)
                total += len(chunk)
    os.replace(tmp, destination)
    return int(total)


def download_latest_release(force: bool = False, platform_name: str | None = None) -> dict:
    release = _fetch_latest_release()
    latest_version = normalize_version(release.get("latest_version"))
    status = "update_available" if is_newer_version(latest_version, APP_VERSION) else "up_to_date"
    asset = select_release_asset(release, platform_name=platform_name)
    if not asset.get("url"):
        return {
            "status": "unavailable",
            "current_version": normalize_version(APP_VERSION),
            "latest_version": latest_version,
            "release_url": release.get("release_url") or GITHUB_RELEASES_URL,
            "message": "No downloadable GitHub release asset or source archive is available.",
            "asset": asset,
            "repo": GITHUB_RELEASES_REPO,
        }
    destination = _download_dir() / _safe_filename(asset.get("name"))
    expected_size = _safe_float(asset.get("size"))
    if destination.exists() and not force:
        actual_size = destination.stat().st_size
        if expected_size is None or int(actual_size) == int(expected_size):
            return {
                "status": "downloaded",
                "download_status": "cached",
                "current_version": normalize_version(APP_VERSION),
                "latest_version": latest_version,
                "release_url": release.get("release_url") or GITHUB_RELEASES_URL,
                "path": str(destination),
                "bytes": int(actual_size),
                "asset": asset,
                "message": f"Update v{latest_version} is already downloaded.",
                "repo": GITHUB_RELEASES_REPO,
            }
    bytes_written = _download_url_to_file(str(asset.get("url")), destination)
    return {
        "status": "downloaded",
        "download_status": status,
        "current_version": normalize_version(APP_VERSION),
        "latest_version": latest_version,
        "release_url": release.get("release_url") or GITHUB_RELEASES_URL,
        "path": str(destination),
        "bytes": int(bytes_written),
        "asset": asset,
        "message": f"Downloaded v{latest_version} to {destination}.",
        "repo": GITHUB_RELEASES_REPO,
    }


def _safe_float(value):
    try:
        out = float(value)
    except Exception:
        return None
    return out


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
            "message": "Update check could not reach GitHub right now. You can still open the Releases page manually.",
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
