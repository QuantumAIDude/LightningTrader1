from __future__ import annotations

import os
from pathlib import Path

from app_meta import APP_DISPLAY_NAME
from app_paths import get_packaging_icon_path
from icon_build import canonical_icon_source, generated_icon_paths


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def get_runtime_instrument() -> str:
    return str(os.environ.get("DEFAULT_INSTRUMENT", "QQQ")).strip().upper() or "QQQ"


def collect_model_seed_files(root: Path | None = None, instrument: str | None = None) -> list[Path]:
    root = Path(root or _project_root())
    instrument = str(instrument or get_runtime_instrument()).upper()
    model_dir = root / "models"
    if not model_dir.exists():
        return []
    files = sorted(p for p in model_dir.glob(f"*_{instrument}.pkl") if p.is_file())
    return files


def collect_data_seed_files(root: Path | None = None) -> list[Path]:
    root = Path(root or _project_root())
    data_dir = root / "data"
    if not data_dir.exists():
        return []
    prefixes = (
        "prices_",
        "context_",
        "news_",
        "event_snapshot_",
        "options_snapshot_",
        "readiness_state_",
        "walk_forward_cache_",
        "horizon_overlay_",
        "horizon_accuracy_",
        "calibration_live_",
    )
    allowed = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name == "instruments.json" or path.name.startswith(prefixes):
            allowed.append(path)
    return allowed


def collect_branding_assets(root: Path | None = None) -> list[Path]:
    root = Path(root or _project_root())
    branding_dir = root / "assets" / "branding"
    if not branding_dir.exists():
        return []
    allowed_exts = {".png", ".ppm", ".ico", ".icns"}
    return sorted(p for p in branding_dir.rglob("*") if p.is_file() and p.suffix.lower() in allowed_exts)


def build_pyinstaller_datas(root: Path | None = None) -> list[tuple[str, str]]:
    root = Path(root or _project_root())
    datas: list[tuple[str, str]] = []
    for path in collect_data_seed_files(root):
        rel_parent = path.relative_to(root / "data").parent
        dest = str(Path("bundle_assets") / "data_seed" / rel_parent)
        datas.append((str(path), dest))
    for path in collect_model_seed_files(root):
        dest = str(Path("bundle_assets") / "models_seed")
        datas.append((str(path), dest))
    for path in collect_branding_assets(root):
        rel_parent = path.relative_to(root / "assets").parent
        dest = str(Path("assets") / rel_parent)
        datas.append((str(path), dest))
    return datas


def packaging_manifest(root: Path | None = None) -> dict:
    root = Path(root or _project_root())
    model_files = collect_model_seed_files(root)
    data_files = collect_data_seed_files(root)
    branding_files = collect_branding_assets(root)
    generated_icons = generated_icon_paths(root)
    return {
        "app_name": APP_DISPLAY_NAME,
        "instrument": get_runtime_instrument(),
        "model_files": [str(p.relative_to(root)) for p in model_files],
        "data_files": [str(p.relative_to(root)) for p in data_files],
        "branding_files": [str(p.relative_to(root)) for p in branding_files],
        "canonical_icon": str(canonical_icon_source(root) or ""),
        "generated_icons": {key: str(path.relative_to(root)) for key, path in generated_icons.items() if path.exists()},
        "packaging_icon": str(get_packaging_icon_path() or ""),
    }
