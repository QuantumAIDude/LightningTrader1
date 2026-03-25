from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

from app_meta import APP_NAME
LOCAL_STORAGE_KIND = "local_file"
LOCAL_STORAGE_NOTE = (
    "Stored locally on this computer in a private app settings file. "
    "This is not OS keychain storage."
)


def _parse_bool(value, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _normalize_broker_mode(value: str | None, default: str = "local_paper") -> str:
    mode = str(value or "").strip().lower()
    if mode in {"paper", "local_paper", "paper_local"}:
        return "local_paper"
    if mode in {"alpaca_paper", "broker_paper", "paper_broker"}:
        return "alpaca_paper"
    if mode in {"alpaca_live", "live", "broker_live"}:
        return "alpaca_live"
    if mode in {"disabled", "off", "none"}:
        return "disabled"
    return str(default or "local_paper")


def get_local_settings_dir() -> Path:
    override = os.environ.get("LIGHTNINGTRADER_SETTINGS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "").strip()
        return Path(appdata).expanduser() / APP_NAME if appdata else home / APP_NAME
    return home / ".config" / APP_NAME


def get_alpaca_credentials_path() -> Path:
    return get_local_settings_dir() / "alpaca_credentials.json"


def _write_private_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    fd = os.open(str(path), flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
    finally:
        try:
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
        except Exception:
            pass


def load_saved_alpaca_credentials() -> dict:
    path = get_alpaca_credentials_path()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return {
        "provider": str(payload.get("provider") or "alpaca").strip().lower(),
        "api_key": str(payload.get("api_key") or "").strip(),
        "secret_key": str(payload.get("secret_key") or "").strip(),
        "paper": _parse_bool(payload.get("paper"), default=True),
        "broker_mode": _normalize_broker_mode(payload.get("broker_mode"), default="alpaca_paper"),
        "saved_at": payload.get("saved_at"),
        "storage_kind": LOCAL_STORAGE_KIND,
        "storage_path": str(path),
    }


def save_alpaca_credentials(
    api_key: str,
    secret_key: str,
    *,
    paper: bool = True,
    broker_mode: str = "alpaca_paper",
    provider: str = "alpaca",
    saved_at: str | None = None,
) -> dict:
    payload = {
        "provider": str(provider or "alpaca").strip().lower(),
        "api_key": str(api_key or "").strip(),
        "secret_key": str(secret_key or "").strip(),
        "paper": bool(paper),
        "broker_mode": _normalize_broker_mode(broker_mode, default="alpaca_paper"),
        "saved_at": saved_at,
    }
    _write_private_json(get_alpaca_credentials_path(), payload)
    return load_saved_alpaca_credentials()


def clear_saved_alpaca_credentials() -> bool:
    path = get_alpaca_credentials_path()
    if not path.exists():
        return False
    try:
        path.unlink()
        return True
    except Exception:
        return False


def get_runtime_broker_settings(override: dict | None = None) -> dict:
    override = dict(override or {})
    saved = load_saved_alpaca_credentials()

    env_provider = os.environ.get("LIVE_BROKER_PROVIDER")
    env_mode = os.environ.get("BROKER_MODE_DEFAULT")
    env_paper = os.environ.get("ALPACA_PAPER")
    env_api = os.environ.get("ALPACA_API_KEY")
    env_secret = os.environ.get("ALPACA_SECRET_KEY")

    provider = str(
        override.get("provider")
        or env_provider
        or saved.get("provider")
        or "alpaca"
    ).strip().lower()
    provider_source = (
        "override" if override.get("provider") is not None else
        "env" if env_provider is not None else
        "local_store" if saved.get("provider") else
        "default"
    )

    broker_mode = _normalize_broker_mode(
        override.get("broker_mode")
        if override.get("broker_mode") is not None
        else env_mode
        if env_mode is not None
        else saved.get("broker_mode")
        if saved.get("broker_mode")
        else ("alpaca_paper" if saved.get("api_key") and saved.get("secret_key") else "local_paper"),
        default="local_paper",
    )
    broker_mode_source = (
        "override" if override.get("broker_mode") is not None else
        "env" if env_mode is not None else
        "local_store" if saved.get("broker_mode") else
        "default"
    )

    paper = (
        bool(override.get("paper"))
        if override.get("paper") is not None
        else _parse_bool(env_paper)
        if env_paper is not None
        else bool(saved.get("paper"))
        if "paper" in saved
        else True
    )
    paper_source = (
        "override" if override.get("paper") is not None else
        "env" if env_paper is not None else
        "local_store" if "paper" in saved else
        "default"
    )

    api_key = (
        str(override.get("api_key") or "").strip()
        if override.get("api_key") is not None
        else str(env_api or "").strip()
        if env_api is not None
        else str(saved.get("api_key") or "").strip()
    )
    secret_key = (
        str(override.get("secret_key") or "").strip()
        if override.get("secret_key") is not None
        else str(env_secret or "").strip()
        if env_secret is not None
        else str(saved.get("secret_key") or "").strip()
    )

    if override.get("api_key") is not None or override.get("secret_key") is not None:
        credentials_source = "override"
    elif env_api and env_secret:
        credentials_source = "env"
    elif saved.get("api_key") and saved.get("secret_key"):
        credentials_source = "local_store"
    elif env_api or env_secret:
        credentials_source = "env_partial"
    else:
        credentials_source = "missing"

    return {
        "provider": provider,
        "provider_source": provider_source,
        "broker_mode": broker_mode,
        "broker_mode_source": broker_mode_source,
        "paper": bool(paper),
        "paper_source": paper_source,
        "api_key": api_key,
        "secret_key": secret_key,
        "credentials_present": bool(api_key and secret_key),
        "credentials_source": credentials_source,
        "has_saved_credentials": bool(saved.get("api_key") and saved.get("secret_key")),
        "saved_credentials": saved,
        "storage_kind": LOCAL_STORAGE_KIND,
        "storage_path": str(get_alpaca_credentials_path()),
        "storage_note": LOCAL_STORAGE_NOTE,
    }
