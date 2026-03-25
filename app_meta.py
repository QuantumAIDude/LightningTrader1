from __future__ import annotations

import os


APP_NAME = "LightningTrader4000"
APP_DISPLAY_NAME = "Lightning Trader Terminal"
APP_VERSION = "0.10.0"
APP_COPYRIGHT = "Internal use only"
APP_BUNDLE_ID = "com.lightningtrader.terminal"
APP_VENDOR = "LightningTrader"

GITHUB_RELEASES_REPO = os.environ.get("LT4000_GITHUB_REPO", "QuantumAIDude/StockBotV2").strip()
GITHUB_RELEASES_URL = f"https://github.com/{GITHUB_RELEASES_REPO}/releases" if GITHUB_RELEASES_REPO else ""
GITHUB_RELEASES_API_URL = (
    f"https://api.github.com/repos/{GITHUB_RELEASES_REPO}/releases/latest" if GITHUB_RELEASES_REPO else ""
)


def app_title() -> str:
    return f"{APP_DISPLAY_NAME} v{APP_VERSION}"
