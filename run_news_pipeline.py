#!/usr/bin/env python3
"""
run_news_pipeline.py

Manual/cron helper for refreshing live news quickly and expanding history on a slower cadence.
"""

import argparse
import json

from config import INSTRUMENTS, NEWS_ARCHIVE_CHUNK_DAYS, NEWS_ARCHIVE_LOOKBACK_DAYS, NEWS_PIPELINE_SYMBOL
from news_pipeline import backfill_news_archive, maybe_refresh_news, update_news_store


def main():
    parser = argparse.ArgumentParser(description="Refresh sentiment news store")
    parser.add_argument("--symbol", default=NEWS_PIPELINE_SYMBOL, help="Ticker symbol (default from config)")
    parser.add_argument("--all-symbols", action="store_true", help="Refresh every news-enabled runtime symbol")
    parser.add_argument("--force", action="store_true", help="Ignore cooldown and fetch immediately")
    parser.add_argument("--backfill", action="store_true", help="Expand the historical archive before refreshing")
    parser.add_argument("--latest-first", action="store_true", help="Fetch current news first, then backfill history")
    parser.add_argument("--lookback-days", type=int, default=NEWS_ARCHIVE_LOOKBACK_DAYS, help="Archive lookback window in days")
    parser.add_argument("--chunk-days", type=int, default=NEWS_ARCHIVE_CHUNK_DAYS, help="Archive request chunk size in days")
    args = parser.parse_args()

    if args.all_symbols:
        symbols = sorted(
            set(
                [
                    str(sym).upper()
                    for sym, meta in INSTRUMENTS.items()
                    if isinstance(meta, dict) and bool(meta.get("news_enabled", True))
                ]
            )
        )
    else:
        symbols = [str(args.symbol or NEWS_PIPELINE_SYMBOL).upper()]

    out = {}
    for symbol in symbols:
        if args.latest_first:
            latest = update_news_store(symbol=symbol, max_items=300) if args.force else maybe_refresh_news(symbol=symbol)
            archive = backfill_news_archive(symbol=symbol, lookback_days=args.lookback_days, chunk_days=args.chunk_days) if args.backfill else None
            out[symbol] = {"latest": latest, "archive": archive}
        elif args.backfill:
            out[symbol] = backfill_news_archive(symbol=symbol, lookback_days=args.lookback_days, chunk_days=args.chunk_days)
        elif args.force:
            out[symbol] = update_news_store(symbol=symbol, max_items=300)
        else:
            out[symbol] = maybe_refresh_news(symbol=symbol)
    print(json.dumps(out if len(symbols) > 1 or args.latest_first else out[symbols[0]], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
