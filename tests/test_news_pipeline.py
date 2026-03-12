import os
import tempfile
import unittest

import pandas as pd

import news_pipeline as npipe


class NewsPipelineTests(unittest.TestCase):
    def test_headline_sentiment_scoring(self):
        pos_score, pos_label = npipe.score_headline_sentiment("Tech stocks surge on strong growth upgrade")
        neg_score, neg_label = npipe.score_headline_sentiment("Semiconductor selloff as guidance cut warns of slowdown")
        self.assertGreater(pos_score, 0.0)
        self.assertEqual(pos_label, "positive")
        self.assertLess(neg_score, 0.0)
        self.assertEqual(neg_label, "negative")

    def test_news_features_are_lagged_by_one_day(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = npipe.DATA_DIR
            try:
                npipe.DATA_DIR = tmp
                os.makedirs(tmp, exist_ok=True)
                news_path = os.path.join(tmp, "news_QQQ.csv")
                pd.DataFrame(
                    [
                        {
                            "symbol": "QQQ",
                            "headline_id": "a1",
                            "published_at": "2026-03-01T15:00:00Z",
                            "headline": "Tech gains on strong growth",
                            "source": "test",
                            "url": "http://example.com/a1",
                            "sentiment_score": 1.0,
                            "sentiment_label": "positive",
                            "ingested_at": "2026-03-01T15:10:00Z",
                        },
                        {
                            "symbol": "QQQ",
                            "headline_id": "a2",
                            "published_at": "2026-03-02T15:00:00Z",
                            "headline": "Tech drops after warning",
                            "source": "test",
                            "url": "http://example.com/a2",
                            "sentiment_score": -1.0,
                            "sentiment_label": "negative",
                            "ingested_at": "2026-03-02T15:10:00Z",
                        },
                    ]
                ).to_csv(news_path, index=False)

                idx = pd.DatetimeIndex(
                    [
                        pd.Timestamp("2026-03-02T16:00:00Z"),
                        pd.Timestamp("2026-03-03T16:00:00Z"),
                    ]
                )
                feats = npipe.load_news_features_for_index("QQQ", idx, tz="America/New_York")
                self.assertGreater(float(feats.iloc[0]["news_sent_mean_d1"]), 0.0)
                self.assertLess(float(feats.iloc[1]["news_sent_mean_d1"]), 0.0)
                self.assertGreaterEqual(float(feats.iloc[0]["news_headline_count_d1"]), 1.0)
            finally:
                npipe.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()
