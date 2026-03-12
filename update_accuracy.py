#!/usr/bin/env python3
"""
update_accuracy.py

Runs accuracy resolution for all horizons:
- 4min live calibration stats
- daily high/low + weekend resolved error metrics
"""

import argparse
import json

from calibration_tracker import update_live_calibration
from config import INSTRUMENTS
from horizon_accuracy_tracker import update_horizon_accuracy
from instrument_onboarding import ensure_runtime_instruments, list_runtime_instruments


def _parse_instruments(raw: str) -> list[str]:
    v = str(raw or "").strip()
    if not v or v.lower() == "all":
        symbols = list_runtime_instruments() or list(INSTRUMENTS.keys())
        return sorted(set(symbols))
    return [s.strip().upper() for s in v.split(",") if s.strip()]


def main():
    parser = argparse.ArgumentParser(description="Update 4min + horizon accuracy metrics")
    parser.add_argument("--instrument", default="all", help="Comma list or 'all'")
    args = parser.parse_args()

    try:
        ensure_runtime_instruments(force=False, auto_onboard=True)
    except Exception:
        pass

    out = {"instruments": {}}
    for sym in _parse_instruments(args.instrument):
        out["instruments"][sym] = {
            "calibration_4min": update_live_calibration(sym, horizon_minutes=4),
            "horizon_accuracy": update_horizon_accuracy(sym),
        }
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

