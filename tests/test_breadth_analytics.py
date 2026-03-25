import json
import os
import tempfile
import unittest

import pandas as pd

import breadth_analytics as ba


class BreadthAnalyticsTests(unittest.TestCase):
    def test_proxy_snapshot_uses_context_histories(self):
        old_loader = ba.load_context_daily_history
        old_instruments = dict(ba.INSTRUMENTS)
        old_fetch = ba._fetch_live_partial_breadth
        try:
            hist = pd.DataFrame({"close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]})
            vix = pd.DataFrame({"close": [20.0, 20.5, 20.2, 19.8, 19.6, 19.4]})
            ba.load_context_daily_history = lambda instrument_key, role, allow_remote=False: vix.copy() if role == "vol" else hist.copy()
            ba._fetch_live_partial_breadth = lambda: {"status": "unavailable", "reason": "test"}
            ba.INSTRUMENTS.clear()
            ba.INSTRUMENTS.update({"QQQ": {"context_daily_symbols": {"market": "SPY", "sector": "XLK", "vol": "^VXN", "lead_fut": "NQ=F", "rates": "TLT"}}})
            snap = ba.get_breadth_snapshot("QQQ")
            self.assertEqual(snap["status"], "proxy_only")
            self.assertTrue(snap["rows"])
        finally:
            ba.load_context_daily_history = old_loader
            ba._fetch_live_partial_breadth = old_fetch
            ba.INSTRUMENTS.clear()
            ba.INSTRUMENTS.update(old_instruments)

    def test_direct_snapshot_reads_local_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "market_breadth_snapshot.json")
            with open(path, "w") as f:
                json.dump({"advancers": 3200, "decliners": 1800, "up_volume": 8_000_000, "down_volume": 4_000_000, "breadth_thrust": 0.64}, f)
            old_path = ba._breadth_path
            try:
                ba._breadth_path = lambda: path
                snap = ba.get_breadth_snapshot("QQQ")
                self.assertEqual(snap["status"], "direct")
                self.assertTrue(snap["direct_feed"])
                self.assertEqual(snap["rows"][0][1], "Direct")
                self.assertEqual(snap["rows"][3][0], "Breadth Thrust")
            finally:
                ba._breadth_path = old_path

    def test_live_partial_snapshot_uses_remote_provider_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = ba._breadth_path
            old_cache = ba._cache_path
            old_fetch = ba._fetch_live_partial_breadth
            try:
                ba._breadth_path = lambda: os.path.join(tmp, "missing.json")
                ba._cache_path = lambda: os.path.join(tmp, "market_breadth_live.json")
                ba._fetch_live_partial_breadth = lambda: {
                    "status": "ok",
                    "fetched_at": "2030-01-10T00:00:00Z",
                    "risk_on_score": 0.68,
                    "sector_up": 8,
                    "sector_down": 3,
                    "sector_participation": 8 / 11,
                    "index_participation": 0.75,
                    "vol_ret_1": -0.03,
                    "leader_text": "QQQ +1.20%, SPY +0.90%",
                    "index_returns": {"QQQ": 0.012, "SPY": 0.009},
                    "sector_returns": {"XLK": 0.014, "XLF": -0.002},
                    "provider_note": "Live market breadth proxy from major indices and sector ETFs",
                }
                snap = ba.get_breadth_snapshot("QQQ")
                self.assertEqual(snap["status"], "live_partial")
                self.assertEqual(snap["source_kind"], "live")
                self.assertIn("Live partial", snap["rows"][0][1])
            finally:
                ba._breadth_path = old_path
                ba._cache_path = old_cache
                ba._fetch_live_partial_breadth = old_fetch

    def test_direct_cache_snapshot_can_be_injected_cleanly(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_path = ba._breadth_path
            old_direct_cache = ba._direct_cache_path
            try:
                ba._breadth_path = lambda: os.path.join(tmp, "missing.json")
                ba._direct_cache_path = lambda: os.path.join(tmp, "market_breadth_direct_cache.json")
                ok = ba.save_direct_breadth_snapshot(
                    {
                        "advancers": 2800,
                        "decliners": 2200,
                        "up_volume": 7_000_000,
                        "down_volume": 5_000_000,
                        "source": "collector",
                    }
                )
                self.assertTrue(ok)
                snap = ba.get_breadth_snapshot("QQQ")
                self.assertEqual(snap["status"], "direct")
                self.assertEqual(snap["source_kind"], "cache")
                self.assertIn("Direct cache", snap["rows"][0][1])
            finally:
                ba._breadth_path = old_path
                ba._direct_cache_path = old_direct_cache


if __name__ == "__main__":
    unittest.main()
