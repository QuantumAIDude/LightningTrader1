from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image


CANONICAL_ICON_NAME = "icon.png"
MAC_ICON_NAME = "lightning_trader.icns"
WIN_ICON_NAME = "lightning_trader.ico"


def _project_root(root: Path | None = None) -> Path:
    return Path(root or Path(__file__).resolve().parent)


def branding_dir(root: Path | None = None) -> Path:
    return _project_root(root) / "assets" / "branding"


def canonical_icon_source(root: Path | None = None) -> Path | None:
    candidate = branding_dir(root) / CANONICAL_ICON_NAME
    return candidate if candidate.exists() else None


def generated_icon_paths(root: Path | None = None) -> dict[str, Path]:
    base = branding_dir(root)
    return {
        "png": base / CANONICAL_ICON_NAME,
        "ico": base / WIN_ICON_NAME,
        "icns": base / MAC_ICON_NAME,
    }


def _needs_regeneration(source: Path, target: Path, force: bool = False) -> bool:
    if force or not target.exists():
        return True
    try:
        return source.stat().st_mtime > target.stat().st_mtime
    except Exception:
        return True


def _generate_ico(source: Path, target: Path):
    image = Image.open(source).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="ICO", sizes=sizes)


def _write_iconset_pngs(source: Path, iconset_dir: Path):
    image = Image.open(source).convert("RGBA")
    iconset_dir.mkdir(parents=True, exist_ok=True)
    size_map = {
        "icon_16x16.png": 16,
        "icon_16x16@2x.png": 32,
        "icon_32x32.png": 32,
        "icon_32x32@2x.png": 64,
        "icon_128x128.png": 128,
        "icon_128x128@2x.png": 256,
        "icon_256x256.png": 256,
        "icon_256x256@2x.png": 512,
        "icon_512x512.png": 512,
        "icon_512x512@2x.png": 1024,
    }
    for filename, size in size_map.items():
        resized = image.resize((size, size), Image.Resampling.LANCZOS)
        resized.save(iconset_dir / filename, format="PNG")


def _generate_icns(source: Path, target: Path):
    iconutil = shutil.which("iconutil")
    target.parent.mkdir(parents=True, exist_ok=True)
    if iconutil:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                iconset_dir = Path(tmp) / "icon.iconset"
                _write_iconset_pngs(source, iconset_dir)
                subprocess.run(
                    [iconutil, "-c", "icns", str(iconset_dir), "-o", str(target)],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            return
        except Exception:
            pass
    image = Image.open(source).convert("RGBA")
    image.save(target, format="ICNS")


def ensure_generated_icons(root: Path | None = None, force: bool = False) -> dict:
    source = canonical_icon_source(root)
    targets = generated_icon_paths(root)
    if not source:
        return {
            "status": "missing_source",
            "source": None,
            "targets": {key: str(path) for key, path in targets.items()},
            "generated": [],
            "errors": {"source": "Canonical icon source assets/branding/icon.png is missing."},
        }

    generated = []
    errors: dict[str, str] = {}
    for kind in ("ico", "icns"):
        target = targets[kind]
        if not _needs_regeneration(source, target, force=force):
            continue
        try:
            if kind == "ico":
                _generate_ico(source, target)
            else:
                _generate_icns(source, target)
            generated.append(kind)
        except Exception as exc:
            errors[kind] = str(exc)

    return {
        "status": "ok" if not errors else ("partial" if generated else "error"),
        "source": str(source),
        "targets": {key: str(path) for key, path in targets.items()},
        "generated": generated,
        "errors": errors,
    }


if __name__ == "__main__":
    print(ensure_generated_icons())
