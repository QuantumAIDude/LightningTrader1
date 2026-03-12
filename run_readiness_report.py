#!/usr/bin/env python3
"""
run_readiness_report.py

Daily readiness report updater.
"""

import argparse
import json

from config import INSTRUMENTS
from instrument_onboarding import ensure_runtime_instruments, list_runtime_instruments
from readiness_tracker import update_readiness_state


def _parse_instruments(raw: str) -> list[str]:
    v = str(raw or "").strip()
    if not v or v.lower() == "all":
        symbols = list_runtime_instruments() or list(INSTRUMENTS.keys())
        return sorted(set(symbols))
    return [s.strip().upper() for s in v.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Update daily deployment readiness state")
    parser.add_argument("--instrument", default="all", help="Comma list or all")
    parser.add_argument("--online", action="store_true", help="Fetch latest prices before backtest eval")
    args = parser.parse_args()

    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass

    out = {"instruments": {}}
    for sym in _parse_instruments(args.instrument):
        out["instruments"][sym] = update_readiness_state(sym, online=args.online)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

