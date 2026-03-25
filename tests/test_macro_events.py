import json
import os
import tempfile
import unittest

import macro_events as me


class MacroEventsTests(unittest.TestCase):
    def test_macro_snapshot_loads_and_scores_upcoming_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "macro_calendar.json")
            with open(path, "w") as f:
                json.dump(
                    [
                        {"title": "US CPI", "scheduled_at": "2030-01-15T13:30:00Z", "severity": "high"},
                        {"title": "Treasury Auction", "scheduled_at": "2030-01-16T15:00:00Z", "severity": "medium"},
                    ],
                    f,
                )
            old_path = me._calendar_path
            try:
                me._calendar_path = lambda: path
                snap = me.get_macro_event_snapshot(lookahead_hours=24 * 3650)
                self.assertEqual(snap["status"], "ok")
                self.assertTrue(snap["events"])
                self.assertGreater(snap["risk_score"], 0.0)
            finally:
                me._calendar_path = old_path

    def test_macro_snapshot_uses_live_provider_when_available(self):
        old_calendar = me._calendar_path
        old_cache = me._cache_path
        old_fetch = me._fetch_remote_macro_events
        with tempfile.TemporaryDirectory() as tmp:
            try:
                me._calendar_path = lambda: os.path.join(tmp, "missing.json")
                me._cache_path = lambda: os.path.join(tmp, "macro_cache.json")
                me._fetch_remote_macro_events = lambda lookahead_hours=96: {
                    "status": "ok",
                    "events": [
                        {
                            "id": "cpi",
                            "event_type": "macro",
                            "category": "cpi",
                            "title": "US CPI",
                            "scheduled_at": "2030-01-15T13:30:00Z",
                            "severity": "high",
                            "scope": "market",
                            "source": "tradingeconomics_live",
                            "status": "scheduled",
                            "heuristic": False,
                        }
                    ],
                    "provider_note": "TradingEconomics live calendar",
                    "source_kind": "live",
                    "fetched_at": "2030-01-10T00:00:00Z",
                }
                snap = me.get_macro_event_snapshot(lookahead_hours=24 * 3650)
                self.assertEqual(snap["status"], "ok")
                self.assertEqual(snap["source_kind"], "live")
                self.assertEqual(snap["next_event"]["title"], "US CPI")
                self.assertEqual(snap["next_event"]["time_to_event_text"][-1], "d")
            finally:
                me._calendar_path = old_calendar
                me._cache_path = old_cache
                me._fetch_remote_macro_events = old_fetch

    def test_macro_snapshot_uses_cached_events_when_live_refresh_fails(self):
        old_calendar = me._calendar_path
        old_cache = me._cache_path
        old_fetch = me._fetch_remote_macro_events
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "macro_cache.json")
            with open(cache_path, "w") as f:
                json.dump(
                    {
                        "fetched_at": "2020-01-10T00:00:00Z",
                        "provider_note": "TradingEconomics live calendar",
                        "events": [
                            {
                                "id": "fomc",
                                "event_type": "macro",
                                "category": "fomc",
                                "title": "FOMC Rate Decision",
                                "scheduled_at": "2030-01-20T19:00:00Z",
                                "severity": "high",
                                "scope": "market",
                                "source": "tradingeconomics_live",
                                "status": "scheduled",
                                "heuristic": False,
                            }
                        ],
                    },
                    f,
                )
            try:
                me._calendar_path = lambda: os.path.join(tmp, "missing.json")
                me._cache_path = lambda: cache_path
                me._fetch_remote_macro_events = lambda lookahead_hours=96: {"status": "error", "reason": "dns_failed"}
                snap = me.get_macro_event_snapshot(lookahead_hours=24 * 3650)
                self.assertEqual(snap["status"], "ok")
                self.assertEqual(snap["source_kind"], "cache")
                self.assertIn("live refresh failed", snap["provider_note"])
            finally:
                me._calendar_path = old_calendar
                me._cache_path = old_cache
                me._fetch_remote_macro_events = old_fetch

    def test_maybe_refresh_macro_events_writes_cache(self):
        old_calendar = me._calendar_path
        old_cache = me._cache_path
        old_fetch = me._fetch_remote_macro_events
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = os.path.join(tmp, "macro_cache.json")
            try:
                me._calendar_path = lambda: os.path.join(tmp, "missing.json")
                me._cache_path = lambda: cache_path
                me._fetch_remote_macro_events = lambda lookahead_hours=96: {
                    "status": "ok",
                    "events": [
                        {
                            "id": "cpi",
                            "event_type": "macro",
                            "category": "cpi",
                            "title": "US CPI",
                            "scheduled_at": "2030-01-15T13:30:00Z",
                            "severity": "high",
                            "scope": "market",
                            "source": "tradingeconomics_live",
                            "status": "scheduled",
                            "heuristic": False,
                        }
                    ],
                    "provider_note": "TradingEconomics live calendar",
                    "source_kind": "live",
                    "fetched_at": "2030-01-10T00:00:00Z",
                }
                snap = me.maybe_refresh_macro_events(lookahead_hours=24 * 3650, force=True)
                self.assertEqual(snap["source_kind"], "live")
                self.assertTrue(os.path.exists(cache_path))
            finally:
                me._calendar_path = old_calendar
                me._cache_path = old_cache
                me._fetch_remote_macro_events = old_fetch


if __name__ == "__main__":
    unittest.main()
