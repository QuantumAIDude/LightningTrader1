#!/usr/bin/env python3
"""
auto_onboard.py

Manual/cron entrypoint for runtime instrument onboarding.
"""

import argparse
import json

from instrument_onboarding import ensure_runtime_instruments


def main():
    parser = argparse.ArgumentParser(description="Process staged instrument files and refresh registry")
    parser.add_argument("--force", action="store_true", help="Ignore cooldown and run immediately")
    parser.add_argument(
        "--no-staging",
        action="store_true",
        help="Skip staged file ingestion and only refresh from existing price files",
    )
    args = parser.parse_args()

    report = ensure_runtime_instruments(force=args.force, auto_onboard=(not args.no_staging))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

