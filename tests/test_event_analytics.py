import os
import tempfile
import unittest

import event_analytics as ea


class EventAnalyticsTests(unittest.TestCase):
    def test_get_event_snapshot_uses_fresh_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = ea.DATA_DIR
            try:
                ea.DATA_DIR = tmp
                ea._save_cached_snapshot(
                    "QQQ",
                    {
                        "status": "ok",
                        "symbol": "QQQ",
                        "next_earnings": "2026-03-28T00:00:00+00:00",
                        "days_to_earnings": 5.0,
                        "fetched_at": "2026-03-24T00:00:00+00:00",
                    },
                )
                out = ea.get_event_snapshot("QQQ", max_age_sec=10**9)
                self.assertEqual(out.get("status"), "ok")
                self.assertEqual(out.get("source"), "cache")
            finally:
                ea.DATA_DIR = old_dir

    def test_get_event_snapshot_caches_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = ea.DATA_DIR
            old_fetch = ea._fetch_remote_event_snapshot
            try:
                ea.DATA_DIR = tmp
                ea._fetch_remote_event_snapshot = lambda symbol: {"status": "unavailable", "reason": "no_earnings_date"}
                out = ea.get_event_snapshot("QQQ", max_age_sec=0)
                self.assertEqual(out.get("status"), "unavailable")
                cached = ea.get_event_snapshot("QQQ", max_age_sec=10**9)
                self.assertEqual(cached.get("source"), "cache_failure")
            finally:
                ea.DATA_DIR = old_dir
                ea._fetch_remote_event_snapshot = old_fetch


if __name__ == "__main__":
    unittest.main()
