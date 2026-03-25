# Deployment Guide

This project is ready for internal desktop packaging, but a few release steps still require a human operator.

## Versioning

The authoritative app version lives in [app_meta.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/app_meta.py).

Before a release:

1. bump `APP_VERSION`
2. confirm `GITHUB_RELEASES_REPO` points at the correct repository
3. commit the version bump before creating the GitHub Release tag

## Build Requirements

Install build tooling:

```bash
python3 -m pip install -r requirements-build.txt
```

Validate the packaging manifest and generate icon assets without building:

```bash
python3 build_release.py --prepare-only
```

Run the macOS-oriented PyInstaller build:

```bash
python3 build_release.py --clean
```

Helper wrapper:

```bash
./scripts/build_macos.sh
```

Manual Windows build on a Windows machine:

```powershell
./scripts/build_windows.ps1
```

## Packaging Notes

- PyInstaller entrypoint: [desktop_app.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/desktop_app.py)
- Spec file: [lightning_trader.spec](/Users/quinnesser/IdeaProjects/LightningTrader4000/lightning_trader.spec)
- Packaging asset manifest logic: [packaging_support.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/packaging_support.py)
- User runtime directories and bundled asset seeding: [app_paths.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/app_paths.py)

The packaging manifest deliberately bundles:

- instrument-scoped runtime models such as `*_QQQ.pkl`
- curated data seeds needed for the first launch
- branding assets from `assets/branding/`

It intentionally does not try to bundle every legacy model artifact in `models/`.

Successful build outputs land in `dist/`:

- `Lightning Trader Terminal.app`
- `Lightning Trader Terminal-macOS-v<APP_VERSION>.zip`
- `Lightning Trader Terminal-Windows-v<APP_VERSION>.zip`

The zip files are the intended GitHub Releases upload artifacts.

## Branding Assets

Canonical source:

- [assets/branding/icon.png](/Users/quinnesser/IdeaProjects/LightningTrader4000/assets/branding/icon.png)

Generated packaging assets:

- `lightning_trader.icns` for macOS app bundles
- `lightning_trader.ico` for Windows packaging
- `lightning_trader.png` for cross-platform packaging fallback if you decide to export one

`build_release.py` now attempts to generate `.icns` and `.ico` from `icon.png` before printing the manifest or running PyInstaller.

If generation fails on a release machine, check:

1. `Pillow` is installed from [requirements-build.txt](/Users/quinnesser/IdeaProjects/LightningTrader4000/requirements-build.txt)
2. on macOS, `iconutil` is available
3. `icon.png` still exists and is readable

The app window uses `icon.png` directly in source mode and packaged runtime mode, with a graceful fallback to the placeholder PPM only if `icon.png` is missing.

## Automated Windows Release Build

GitHub Actions now builds the Windows artifact on a Windows runner and uploads it to the matching GitHub Release:

- workflow: `.github/workflows/release-windows.yml`
- trigger: `release.published`
- optional manual rerun: `workflow_dispatch`
- upload target: the same GitHub Release tag that matches `APP_VERSION`

The workflow does the following:

1. checks out the repository on `windows-latest`
2. installs runtime and PyInstaller build dependencies
3. confirms the release tag matches `APP_VERSION`
4. runs `python build_release.py --clean`
5. uploads `dist/Lightning Trader Terminal-Windows-v<APP_VERSION>.zip` to the GitHub Release

This is the supported way to produce the Windows asset. PyInstaller does not cross-compile Windows executables from macOS.

## GitHub Releases Update Checks

Update checks use the GitHub Releases API via [update_checker.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/update_checker.py).

Expected release flow:

1. bump `APP_VERSION`
2. build and test the app
3. publish a GitHub Release whose tag matches the version
4. users choose `Check for Updates` inside the app
5. if a newer release exists, the app opens the Releases page for manual download

Current implementation is intentionally manual-update only. It does not self-patch installed binaries.

## Human Release Checklist

1. Confirm Alpaca and saved-credential flows on a clean machine.
2. Build the app with PyInstaller on the target OS.
3. Test first launch, update checks, broker setup, replay, and local paper routing on that clean machine.
4. Provide final branding assets if you want the packaged app and dock/taskbar icon to look official.
5. Sign and notarize the macOS app if employees will install it outside developer-controlled environments.
6. Publish the GitHub Release after upload artifacts are ready.

## Still Human-Owned

- Apple signing and notarization
- Windows signing, if Windows distribution is added
- final icon/brand assets
- clean-machine install testing
- GitHub Release publishing
- release operations and employee distribution policy
