from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from app_meta import APP_DISPLAY_NAME, APP_VERSION
from icon_build import ensure_generated_icons
from packaging_support import packaging_manifest


PROJECT_ROOT = Path(__file__).resolve().parent


def _pyinstaller_command() -> list[str] | None:
    pyinstaller = shutil.which("pyinstaller") or shutil.which("pyinstaller.exe")
    if pyinstaller:
        return [pyinstaller]
    try:
        import PyInstaller  # noqa: F401
    except Exception:
        return None
    return [sys.executable, "-m", "PyInstaller"]


def _build_env() -> dict:
    env = dict(os.environ)
    cache_root = PROJECT_ROOT / "build" / ".pyinstaller-cache"
    mpl_root = PROJECT_ROOT / "build" / ".mplconfig"
    cache_root.mkdir(parents=True, exist_ok=True)
    mpl_root.mkdir(parents=True, exist_ok=True)
    env.setdefault("PYINSTALLER_CONFIG_DIR", str(cache_root))
    env.setdefault("MPLCONFIGDIR", str(mpl_root))
    return env


def _zip_macos_app(dist_dir: Path) -> str | None:
    app_dir = dist_dir / f"{APP_DISPLAY_NAME}.app"
    if not app_dir.exists():
        return None
    archive_base = dist_dir / f"{APP_DISPLAY_NAME}-macOS-v{APP_VERSION}"
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=dist_dir, base_dir=app_dir.name))
    return str(zip_path)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare or build the Lightning Trader desktop app.")
    parser.add_argument("--prepare-only", action="store_true", help="Print the packaging manifest without running PyInstaller.")
    parser.add_argument("--clean", action="store_true", help="Remove previous build artifacts before building.")
    parser.add_argument("--skip-icon-build", action="store_true", help="Skip generating .ico/.icns assets from assets/branding/icon.png.")
    args = parser.parse_args(argv)

    icon_result = {"status": "skipped"}
    if not args.skip_icon_build:
        icon_result = ensure_generated_icons(PROJECT_ROOT)

    manifest = packaging_manifest(PROJECT_ROOT)
    manifest["icon_generation"] = icon_result
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if args.prepare_only:
        return 0

    pyinstaller_cmd = _pyinstaller_command()
    if not pyinstaller_cmd:
        print("PyInstaller is not installed. Install requirements-build.txt before running a build.")
        return 1

    if args.clean:
        for name in ("build", "dist"):
            shutil.rmtree(PROJECT_ROOT / name, ignore_errors=True)

    cmd = [*pyinstaller_cmd, "--clean", "lightning_trader.spec"]
    print(f"Building {APP_DISPLAY_NAME} v{APP_VERSION}...")
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, env=_build_env())
    if completed.returncode != 0:
        return int(completed.returncode)
    zip_path = _zip_macos_app(PROJECT_ROOT / "dist")
    if zip_path:
        print(f"Created release archive: {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
