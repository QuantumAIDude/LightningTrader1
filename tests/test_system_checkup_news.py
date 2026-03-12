import os
import tempfile
import unittest

import pandas as pd

import system_checkup as sc


class SystemCheckupNewsTests(unittest.TestCase):
    def test_news_health_degraded_and_ok_states(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = sc.DATA_DIR
            old_enabled = sc.NEWS_PIPELINE_ENABLED
            old_symbol = sc.NEWS_PIPELINE_SYMBOL
            try:
                sc.DATA_DIR = tmp
                sc.NEWS_PIPELINE_ENABLED = True
                sc.NEWS_PIPELINE_SYMBOL = "QQQ"
                path = os.path.join(tmp, "news_QQQ.csv")

                # Too few unique days -> degraded.
                df_bad = pd.DataFrame(
                    [
                        {
                            "published_at": "2026-03-10T12:00:00Z",
                            "headline": "Apple beats estimates",
                            "sentiment_score": 0.4,
                        },
                        {
                            "published_at": "2026-03-10T15:00:00Z",
                            "headline": "Nvidia raises guidance",
                            "sentiment_score": 0.5,
                        },
                    ]
                )
                df_bad.to_csv(path, index=False)
                bad = sc._load_news_health("QQQ")
                self.assertEqual(bad.get("status"), "degraded")
                self.assertFalse(bool(bad.get("signal_ready")))

                # Fresh and varied across >=5 days -> ok.
                rows = []
                base = pd.Timestamp.now(tz="UTC").floor("D")
                for i in range(6):
                    rows.append(
                        {
                            "published_at": (base - pd.Timedelta(days=i)).isoformat(),
                            "headline": f"Tech headline {i}",
                            "sentiment_score": (0.3 if i % 2 == 0 else -0.2),
                        }
                    )
                pd.DataFrame(rows).to_csv(path, index=False)
                good = sc._load_news_health("QQQ")
                self.assertEqual(good.get("status"), "ok")
                self.assertTrue(bool(good.get("signal_ready")))
            finally:
                sc.DATA_DIR = old_data_dir
                sc.NEWS_PIPELINE_ENABLED = old_enabled
                sc.NEWS_PIPELINE_SYMBOL = old_symbol


if __name__ == "__main__":
    unittest.main()
