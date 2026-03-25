# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys

from PyInstaller.utils.hooks import collect_submodules

ROOT = Path.cwd().resolve()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app_meta import APP_DISPLAY_NAME, APP_VERSION
from packaging_support import build_pyinstaller_datas
from app_paths import get_packaging_icon_path

datas = build_pyinstaller_datas(ROOT)
icon_path = get_packaging_icon_path()

hiddenimports = [
    "alpaca.trading.client",
    "alpaca.data.historical",
    "alpaca.data.requests",
    "alpaca.data.timeframe",
]
hiddenimports += [m for m in collect_submodules("pandas") if "tests" not in m]

a = Analysis(
    ["desktop_app.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_DISPLAY_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=str(icon_path) if icon_path else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_DISPLAY_NAME,
)

app = BUNDLE(
    coll,
    name=f"{APP_DISPLAY_NAME}.app",
    icon=str(icon_path) if icon_path and str(icon_path).endswith(".icns") else None,
    bundle_identifier="com.lightningtrader.terminal",
    version=APP_VERSION,
)
