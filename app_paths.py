from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

from app_meta import APP_NAME
from credential_store import get_local_settings_dir
from icon_build import CANONICAL_ICON_NAME


PROJECT_ROOT = Path(__file__).resolve().parent


def is_frozen_app() -> bool:
    return bool(getattr(sys, "frozen", False))


def bundle_root() -> Path:
    if is_frozen_app():
        return Path(getattr(sys, "_MEIPASS", PROJECT_ROOT))
    return PROJECT_ROOT


def resource_path(relative_path: str) -> Path:
    return bundle_root() / relative_path


def get_runtime_root() -> Path:
    return get_local_settings_dir()


def get_runtime_data_dir() -> Path:
    if os.environ.get("DATA_DIR", "").strip():
        return Path(os.environ["DATA_DIR"]).expanduser()
    if is_frozen_app():
        return get_runtime_root() / "data"
    return PROJECT_ROOT / "data"


def get_runtime_model_dir() -> Path:
    if os.environ.get("MODEL_DIR", "").strip():
        return Path(os.environ["MODEL_DIR"]).expanduser()
    if is_frozen_app():
        return get_runtime_root() / "models"
    return PROJECT_ROOT / "models"


def get_runtime_logs_dir() -> Path:
    if is_frozen_app():
        return get_runtime_root() / "logs"
    return PROJECT_ROOT


def get_branding_dir() -> Path:
    return resource_path("assets/branding")


def get_window_icon_path() -> Path | None:
    branding_dir = get_branding_dir()
    for name in (CANONICAL_ICON_NAME, "lightning_trader.png", "lightning_trader_icon.ppm"):
        candidate = branding_dir / name
        if candidate.exists():
            return candidate
    return None


def get_packaging_icon_path(platform_name: str | None = None) -> Path | None:
    platform_name = str(platform_name or sys.platform).lower()
    branding_dir = PROJECT_ROOT / "assets" / "branding"
    if platform_name.startswith("darwin"):
        candidate = branding_dir / "lightning_trader.icns"
        return candidate if candidate.exists() else None
    if platform_name.startswith("win"):
        candidate = branding_dir / "lightning_trader.ico"
        return candidate if candidate.exists() else None
    candidate = branding_dir / CANONICAL_ICON_NAME
    if candidate.exists():
        return candidate
    candidate = branding_dir / "lightning_trader.png"
    return candidate if candidate.exists() else None


def _copy_seed_tree(src: Path, dst: Path):
    if not src.exists():
        return
    for path in src.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        out = dst / rel
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, out)


def seed_packaged_runtime_assets():
    if not is_frozen_app():
        return
    data_seed = resource_path("bundle_assets/data_seed")
    model_seed = resource_path("bundle_assets/models_seed")
    runtime_data = get_runtime_data_dir()
    runtime_models = get_runtime_model_dir()
    runtime_data.mkdir(parents=True, exist_ok=True)
    runtime_models.mkdir(parents=True, exist_ok=True)
    _copy_seed_tree(data_seed, runtime_data)
    _copy_seed_tree(model_seed, runtime_models)


def bootstrap_runtime_environment() -> dict:
    runtime_data = get_runtime_data_dir()
    runtime_models = get_runtime_model_dir()
    runtime_data.mkdir(parents=True, exist_ok=True)
    runtime_models.mkdir(parents=True, exist_ok=True)
    if is_frozen_app():
        os.environ.setdefault("DATA_DIR", str(runtime_data))
        os.environ.setdefault("MODEL_DIR", str(runtime_models))
        os.environ.setdefault("INSTRUMENTS_FILE", str(runtime_data / "instruments.json"))
        seed_packaged_runtime_assets()
    return {
        "app_name": APP_NAME,
        "frozen": is_frozen_app(),
        "bundle_root": str(bundle_root()),
        "runtime_root": str(get_runtime_root()),
        "data_dir": str(runtime_data),
        "model_dir": str(runtime_models),
        "logs_dir": str(get_runtime_logs_dir()),
        "window_icon": str(get_window_icon_path()) if get_window_icon_path() else None,
    }
