# Branding Assets

This folder is ready for production branding assets.

Canonical source:

- `icon.png`
  Master app icon source used by the GUI and by the build pipeline.

Generated release assets:

- `lightning_trader.icns`
  macOS app-bundle icon generated from `icon.png`.
- `lightning_trader.ico`
  Windows executable icon generated from `icon.png`.

Fallback support:

- `lightning_trader_icon.ppm`
  Last-resort in-app fallback if `icon.png` is missing.
- `lightning_trader.png`
  Optional generic branding export for future splash/about surfaces.

Build behavior:

- `python3 build_release.py --prepare-only`
- `python3 build_release.py --clean`

Both commands now try to generate `.icns` and `.ico` from `icon.png` before printing the packaging manifest or invoking PyInstaller.
