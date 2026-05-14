#!/usr/bin/env python3
"""
manage_instruments.py
Utility for onboarding equity instruments without editing config.py.
"""

import argparse
import json
import os

from config import INSTRUMENTS_FILE, INSTRUMENTS
from instrument_onboarding import ensure_runtime_instruments
from instrument_profile import build_instrument_meta


def _default_equity_meta(symbol: str) -> dict:
    return build_instrument_meta(symbol)


def _load_file(path: str) -> dict:
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _save_file(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def _print_list():
    for key in sorted(INSTRUMENTS.keys()):
        meta = INSTRUMENTS.get(key, {})
        print(f"{key}: type={meta.get('type')} yahoo={meta.get('yahoo')} tz={meta.get('tz')}")


def main():
    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass
    parser = argparse.ArgumentParser(description="Manage LightningTrader instruments")
    parser.add_argument("--list", action="store_true", help="List currently loaded instruments")
    parser.add_argument("--symbol", help="Ticker symbol to add/update (e.g., AAPL)")
    parser.add_argument("--yahoo", default=None, help="Yahoo ticker override (default: symbol)")
    parser.add_argument("--alpaca", default=None, help="Alpaca ticker override (default: symbol)")
    parser.add_argument("--tz", default="America/New_York", help="Timezone")
    parser.add_argument("--tick-size", type=float, default=0.01, help="Tick size")
    parser.add_argument("--target-vol", type=float, default=0.004, help="Target volatility for sizing")
    parser.add_argument("--vol-floor", type=float, default=0.00025, help="Volatility floor")
    parser.add_argument("--open-minute", type=int, default=9 * 60 + 30, help="Session open minute (local tz)")
    parser.add_argument("--close-minute", type=int, default=16 * 60, help="Session close minute (local tz)")
    parser.add_argument("--fetch", action="store_true", help="Fetch data immediately after add/update")
    parser.add_argument("--train", action="store_true", help="Train fast model bundle after add/update")
    args = parser.parse_args()

    if args.list and not args.symbol:
        _print_list()
        return

    if not args.symbol:
        parser.error("--symbol is required unless using --list only")

    symbol = str(args.symbol).strip().upper()
    if not symbol:
        parser.error("symbol cannot be empty")

    file_payload = _load_file(INSTRUMENTS_FILE)
    entry = _default_equity_meta(symbol)
    entry.update({
        "type": "equity",
        "yahoo": args.yahoo or symbol,
        "alpaca": args.alpaca or symbol,
        "tz": args.tz,
        "tick_size": float(args.tick_size),
        "target_vol": float(args.target_vol),
        "vol_floor": float(args.vol_floor),
        "session_open_minute": int(args.open_minute),
        "session_close_minute": int(args.close_minute),
    })
    file_payload[symbol] = entry
    _save_file(INSTRUMENTS_FILE, file_payload)

    # Update in-memory registry for immediate fetch/train calls in this process.
    INSTRUMENTS[symbol] = entry

    print(f"Saved {symbol} to {INSTRUMENTS_FILE}")

    if args.fetch:
        from data_fetch import update_price_data
        df = update_price_data(symbol)
        rows = 0 if df is None else len(df)
        print(f"Fetched data for {symbol}: rows={rows}")

    if args.train:
        from bootstrap_models_fast import bootstrap_fast_models
        bootstrap_fast_models(symbol)
        print(f"Trained fast models for {symbol}")

    if args.list:
        _print_list()


if __name__ == "__main__":
    main()
