#!/usr/bin/env python3
"""
run_news_pipeline.py

Manual/cron helper for refreshing the configured news pipeline symbol.
"""

import argparse
import json

from config import NEWS_PIPELINE_SYMBOL
from news_pipeline import maybe_refresh_news, update_news_store


def main():
    parser = argparse.ArgumentParser(description="Refresh sentiment news store")
    parser.add_argument("--symbol", default=NEWS_PIPELINE_SYMBOL, help="Ticker symbol (default from config)")
    parser.add_argument("--force", action="store_true", help="Ignore cooldown and fetch immediately")
    args = parser.parse_args()

    symbol = str(args.symbol or NEWS_PIPELINE_SYMBOL).upper()
    if args.force:
        out = update_news_store(symbol=symbol, max_items=300)
    else:
        out = maybe_refresh_news(symbol=symbol)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

