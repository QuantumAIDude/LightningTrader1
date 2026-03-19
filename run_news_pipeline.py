#!/usr/bin/env python3
"""
run_news_pipeline.py

Manual/cron helper for refreshing the configured news pipeline symbol.
"""

import argparse
import json

from config import NEWS_ARCHIVE_CHUNK_DAYS, NEWS_ARCHIVE_LOOKBACK_DAYS, NEWS_PIPELINE_SYMBOL
from news_pipeline import backfill_news_archive, maybe_refresh_news, update_news_store


def main():
    parser = argparse.ArgumentParser(description="Refresh sentiment news store")
    parser.add_argument("--symbol", default=NEWS_PIPELINE_SYMBOL, help="Ticker symbol (default from config)")
    parser.add_argument("--force", action="store_true", help="Ignore cooldown and fetch immediately")
    parser.add_argument("--backfill", action="store_true", help="Expand the historical archive before refreshing")
    parser.add_argument("--lookback-days", type=int, default=NEWS_ARCHIVE_LOOKBACK_DAYS, help="Archive lookback window in days")
    parser.add_argument("--chunk-days", type=int, default=NEWS_ARCHIVE_CHUNK_DAYS, help="Archive request chunk size in days")
    args = parser.parse_args()

    symbol = str(args.symbol or NEWS_PIPELINE_SYMBOL).upper()
    if args.backfill:
        out = backfill_news_archive(symbol=symbol, lookback_days=args.lookback_days, chunk_days=args.chunk_days)
    elif args.force:
        out = update_news_store(symbol=symbol, max_items=300)
    else:
        out = maybe_refresh_news(symbol=symbol)
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
