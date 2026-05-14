# Lightning Trader Terminal v2.00.0

Lightning Trader Terminal is a desktop trading workstation with paper-safe broker defaults, replay, advisory context, and a beginner-friendly Alpaca setup flow.

The app is designed to be usable without terminal setup on normal launches:

- paper trading is the safe default
- Alpaca keys can be entered and saved from inside the GUI
- replay is clearly labeled as simulation
- update checks and downloads use GitHub Releases and never self-update silently

## What Normal Users Do

1. Launch the app.
2. If Alpaca paper keys are missing, use the in-app `Alpaca Setup` flow.
3. Keep paper mode on unless you intentionally want live routing.
4. Watch the `Broker` card to confirm whether orders route to `Local Paper` or `Alpaca Paper`.
5. Use `Replay` to practice safely without touching live markets.
6. Use `Check for Updates` or `Download Update` from the app menu or Settings screen.
7. Add new stocks from `Add Stock`; the app registers the profile, news aliases, data context, and onboarding pipeline for that symbol.

## Autonomy Modes

The auto-trading controller separates the route from the decision style:

- `Full Autonomy`: the app uses the model stack, regime model, news model, risk gates, and readiness evidence to decide whether to trade.
- `Strategy Autonomy`: the operator selects a known strategy for a stock while the same regime, news, broker, readiness, and risk gates remain active.

The built-in strategy list covers common algorithmic families: trend following, momentum, mean reversion, breakout, VWAP reversion, opening range breakout, volatility breakout, and pairs/statistical arbitrage. Pairs/stat arb stays blocked until a pair feed is configured.

Use `Auto Risk` to set the controller stop loss. If realized plus tracked open autonomous PnL breaches that dollar or percentage limit, the controller turns on the kill switch and attempts to flatten any tracked autonomous position.

## Broker Modes

- `local_paper`: orders stay inside the built-in paper engine
- `alpaca_paper`: orders route to Alpaca paper only when the SDK, credentials, and connection all pass verification
- `alpaca_live`: live routing path exists, but should only be enabled intentionally
- `disabled`: broker routing stays off

If broker routing is unavailable, the workstation falls back to Local Paper and labels that fallback explicitly in the dashboard and GUI.

## Alpaca Setup

The easiest path is now inside the app:

- open `Broker -> Alpaca Setup`
- enter your Alpaca API key and secret
- leave paper mode enabled
- click `Test Connection` or `Save and Verify`

Saved credentials are stored locally on that computer so the app can reuse them on future launches. The current implementation uses a private local settings file, not the OS keychain.

For advanced/operator workflows, you can still use environment variables:

```bash
export BROKER_MODE_DEFAULT=alpaca_paper
export LIVE_BROKER_PROVIDER=alpaca
export ALPACA_PAPER=true
export ALPACA_API_KEY=your_key_here
export ALPACA_SECRET_KEY=your_secret_here
```

## Replay Quickstart

Replay is simulation only.

Use the GUI controls in this order:

1. `Replay Start`
2. `Replay Step` for manual stepping, or `Replay Resume` / `Replay Pause`
3. `Replay Buy` / `Replay Sell` / `Replay Flat` for replay-paper trades
4. `Replay Reset` to rewind to the starting bar and clear replay-paper trades
5. `Replay Stop` to leave simulation mode

The replay panels show current bar, replay timestamp, replay-paper route, PnL, and trade count.

## Updates

The app checks GitHub Releases and compares the installed version to the latest published release. It can also download the latest release asset or source archive into the local app updates folder; installation remains an explicit human step.

- current version comes from [app_meta.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/app_meta.py)
- update checks are handled by [update_checker.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/update_checker.py)
- downloads are handled by [update_checker.py](/Users/quinnesser/IdeaProjects/LightningTrader4000/update_checker.py) and surfaced in the GUI as `Download Update`

## Packaging And Release Notes

Maintainer-focused packaging instructions live in [docs/DEPLOYMENT.md](/Users/quinnesser/IdeaProjects/LightningTrader4000/docs/DEPLOYMENT.md).

The current PyInstaller build path is:

```bash
python3 build_release.py --prepare-only
python3 build_release.py --clean
```

Artifacts are written to `dist/` when PyInstaller is installed and the build completes successfully.

Current macOS release outputs:

- `dist/Lightning Trader Terminal.app`
- `dist/Lightning Trader Terminal-macOS-v2.00.0.zip`

Windows release output:

- `dist/Lightning Trader Terminal-Windows-v2.00.0.zip`

The Windows zip is built automatically on GitHub Actions when a GitHub Release is published, using the same PyInstaller entrypoint and runtime asset manifest as the macOS build.

## Honesty Rules

The workstation distinguishes:

- model outputs
- heuristic estimates
- live feed context
- cached data
- local paper execution
- broker-routed execution
- replay simulation
- unavailable or placeholder paths

If something is not connected, the UI should say so directly instead of pretending it worked.
