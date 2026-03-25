import json
import os
import tempfile
import unittest

import catalyst_providers as cp
import pandas as pd


class CatalystProvidersTests(unittest.TestCase):
    def test_provider_snapshot_reads_local_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_dir = cp.DATA_DIR
            old_news = cp.load_recent_headlines
            cp.DATA_DIR = tmp
            try:
                cp.load_recent_headlines = lambda symbol, limit=6: []
                with open(os.path.join(tmp, "sec_events_QQQ.json"), "w") as f:
                    json.dump([{"form": "8-K", "filed_at": "2030-01-10T14:00:00Z", "detail": "Material event"}], f)
                with open(os.path.join(tmp, "analyst_actions_QQQ.json"), "w") as f:
                    json.dump([{"action": "Upgrade", "firm": "Test Bank", "published_at": "2030-01-09T12:00:00Z"}], f)
                with open(os.path.join(tmp, "social_sentiment_QQQ.json"), "w") as f:
                    json.dump([{"as_of": "2030-01-09T15:00:00Z", "label": "positive", "score": 0.7}], f)
                snap = cp.get_catalyst_provider_snapshot("QQQ")
                self.assertEqual(snap["status"], "ok")
                self.assertEqual(len(snap["recent"]), 3)
                self.assertTrue(snap["rows"])
                self.assertEqual(snap["rows"][0][0], "Active Catalyst")
                self.assertEqual(snap["rows"][1][0], "Catalyst Risk")
            finally:
                cp.DATA_DIR = old_dir
                cp.load_recent_headlines = old_news

    def test_provider_snapshot_uses_live_sec_and_analyst_paths_when_available(self):
        old_dir = cp.DATA_DIR
        old_sec = cp._fetch_sec_records_remote
        old_analyst = cp._fetch_analyst_records_remote
        old_news = cp.load_recent_headlines
        now = pd.Timestamp.now(tz="UTC")
        try:
            with tempfile.TemporaryDirectory() as tmp:
                cp.DATA_DIR = tmp
                cp._fetch_sec_records_remote = lambda symbol: [
                    {"type": "sec", "title": "8-K", "scheduled_at": (now - pd.Timedelta(hours=2)).isoformat(), "sentiment": None, "detail": "material event", "source": "sec_live"}
                ]
                cp._fetch_analyst_records_remote = lambda symbol: [
                    {"type": "analyst", "title": "Upgrade Hold->Buy", "scheduled_at": (now - pd.Timedelta(hours=3)).isoformat(), "sentiment": "positive", "detail": "Test Bank", "source": "yfinance_live"}
                ]
                cp.load_recent_headlines = lambda symbol, limit=6: [
                    {"published_at": (now - pd.Timedelta(hours=1)).isoformat(), "headline": "Policy risk eases", "label": "positive", "score": 0.2, "source": "news_store", "political_relevance": 0.3}
                ]
                snap = cp.get_catalyst_provider_snapshot("QQQ")
                self.assertEqual(snap["status"], "ok")
                row_map = {row[0]: row for row in snap["rows"]}
                self.assertEqual(row_map["SEC Feed"][1], "Live")
                self.assertEqual(row_map["Analyst Feed"][1], "Live")
                self.assertGreater(snap["catalyst_risk_score"], 0.0)
                self.assertGreaterEqual(len(snap["recent"]), 3)
                self.assertIn("ambiguity", snap["rows"][0][2])
                self.assertIn("active_catalyst", snap)
        finally:
            cp.DATA_DIR = old_dir
            cp._fetch_sec_records_remote = old_sec
            cp._fetch_analyst_records_remote = old_analyst
            cp.load_recent_headlines = old_news


if __name__ == "__main__":
    unittest.main()
