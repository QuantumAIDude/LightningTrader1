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
        legacy_detail = npipe._headline_sentiment_detail(
            "Exchange-Traded Funds Rise as US Equities Move Higher After Midday",
            instrument_symbol="QQQ",
        )
        self.assertGreater(float(legacy_detail["relevance"]), 0.25)
        self.assertGreater(float(legacy_detail["score"]), 0.0)
        political_detail = npipe._headline_sentiment_detail(
            "White House weighs new chip export controls on China",
            instrument_symbol="QQQ",
        )
        self.assertTrue(bool(political_detail["political_relevant"]))
        self.assertGreater(float(political_detail["political_relevance"]), 0.25)
        self.assertLess(float(political_detail["score"]), 0.0)

    def test_news_relevance_is_profile_aware_for_multi_stock(self):
        old_instruments = dict(npipe.INSTRUMENTS)
        try:
            npipe.INSTRUMENTS.clear()
            npipe.INSTRUMENTS.update(
                {
                    "AAPL": {
                        "display_name": "Apple",
                        "company_name": "Apple Inc",
                        "news_aliases": ["AAPL", "Apple", "Apple Inc"],
                        "sector": "Technology",
                        "type": "equity",
                    },
                    "COST": {
                        "display_name": "Costco",
                        "company_name": "Costco Wholesale",
                        "news_aliases": ["COST", "Costco", "Costco Wholesale"],
                        "sector": "Consumer Staples",
                        "type": "equity",
                    },
                }
            )
            tech_detail = npipe._headline_sentiment_detail(
                "Semiconductor and AI stocks rally after cloud demand surge",
                instrument_symbol="AAPL",
            )
            defensive_detail = npipe._headline_sentiment_detail(
                "Semiconductor and AI stocks rally after cloud demand surge",
                instrument_symbol="COST",
            )
            company_detail = npipe._headline_sentiment_detail(
                "Apple launches a stronger iPhone cycle with upbeat demand",
                instrument_symbol="AAPL",
            )
            self.assertGreater(float(tech_detail["relevance"]), float(defensive_detail["relevance"]))
            self.assertEqual(company_detail["relevance_scope"], "symbol_specific")
            self.assertTrue(bool(company_detail["symbol_specific"]))
        finally:
            npipe.INSTRUMENTS.clear()
            npipe.INSTRUMENTS.update(old_instruments)

    def test_news_runtime_summary_supports_layered_optional_scorers(self):
        old_scorers = dict(npipe._OPTIONAL_NEWS_SCORERS)
        try:
            npipe._OPTIONAL_NEWS_SCORERS.clear()
            npipe.register_optional_news_scorer(
                "unit",
                lambda symbol, headlines, meta: [{"headline": headlines[0]["headline"], "score": 0.9, "symbol": symbol}],
            )
            summary = npipe.build_news_runtime_summary(
                "AAPL",
                headlines=[
                    {
                        "headline": "Apple launches new device lineup",
                        "relevance_scope": "symbol_specific",
                        "theme_bucket": "technology",
                        "relevance": 0.92,
                        "impact_score": 0.63,
                    }
                ],
                news_health={"status": "ok", "headline_state": "ok", "history_state": "ok"},
            )

            self.assertEqual(summary["layer_a"]["top_scope"], "symbol_specific")
            self.assertEqual(summary["layer_b"]["status"], "ready")
            self.assertEqual(summary["layer_b"]["candidate_count"], 1)
            self.assertEqual(summary["layer_b"]["scored_count"], 1)
            self.assertEqual(summary["layer_b"]["results"][0]["status"], "ok")
            self.assertTrue(summary["profile"]["profile_exposures"]["dominant_themes"])
        finally:
            npipe._OPTIONAL_NEWS_SCORERS.clear()
            npipe._OPTIONAL_NEWS_SCORERS.update(old_scorers)

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
                            "relevance_score": 1.0,
                            "tech_relevant": 1,
                            "ingested_at": "2026-03-01T15:10:00Z",
                        },
                        {
                            "symbol": "QQQ",
                            "headline_id": "a1b",
                            "published_at": "2026-03-01T17:00:00Z",
                            "headline": "Tech firms face antitrust probe",
                            "source": "test",
                            "url": "http://example.com/a1b",
                            "sentiment_score": -0.5,
                            "sentiment_label": "negative",
                            "relevance_score": 0.9,
                            "tech_relevant": 1,
                            "ingested_at": "2026-03-01T17:10:00Z",
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
                            "relevance_score": 1.0,
                            "tech_relevant": 1,
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
                self.assertNotEqual(float(feats.iloc[0]["news_sent_mean_d1"]), 0.0)
                self.assertLess(float(feats.iloc[1]["news_sent_mean_d1"]), 0.0)
                self.assertGreaterEqual(float(feats.iloc[0]["news_headline_count_d1"]), 1.0)
                self.assertIn("news_sent_ewm_d7", feats.columns)
                self.assertIn("news_intensity_ewm_d7", feats.columns)
                self.assertIn("news_entropy_d3", feats.columns)
                self.assertIn("news_relevance_mean_d3", feats.columns)
                self.assertIn("news_sent_surprise_d1", feats.columns)
                self.assertGreater(float(feats.iloc[0]["news_intensity_ewm_d7"]), 0.0)
                self.assertGreater(float(feats.iloc[0]["news_entropy_d3"]), 0.0)
                self.assertGreater(float(feats.iloc[0]["news_relevance_mean_d3"]), 0.0)
            finally:
                npipe.DATA_DIR = old_data_dir

    def test_legacy_news_rows_are_rescored_in_context(self):
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
                            "headline_id": "b1",
                            "published_at": "2026-03-10T17:17:51Z",
                            "headline": "Exchange-Traded Funds Rise as US Equities Move Higher After Midday",
                            "source": "test",
                            "url": "http://example.com/b1",
                            "sentiment_score": 0.0,
                            "sentiment_label": "neutral",
                            "ingested_at": "2026-03-10T17:20:00Z",
                        },
                        {
                            "symbol": "QQQ",
                            "headline_id": "b2",
                            "published_at": "2026-03-11T12:52:34Z",
                            "headline": "Exchange-Traded Funds, Equity Futures Lower Pre-Bell Wednesday Amid CPI Inflation Data",
                            "source": "test",
                            "url": "http://example.com/b2",
                            "sentiment_score": 0.0,
                            "sentiment_label": "neutral",
                            "ingested_at": "2026-03-11T13:00:00Z",
                        },
                    ]
                ).to_csv(news_path, index=False)
                idx = pd.DatetimeIndex([pd.Timestamp("2026-03-12T16:00:00Z")])
                feats = npipe.load_news_features_for_index("QQQ", idx, tz="America/New_York")
                self.assertGreater(float(feats.iloc[0]["news_headline_count_d1"]), 0.0)
                self.assertGreater(float(feats.iloc[0]["news_relevance_mean_d3"]), 0.0)
                self.assertLess(float(feats.iloc[0]["news_sent_mean_d1"]), 0.0)
                self.assertGreater(float(feats.iloc[0]["news_intensity_ewm_d7"]), 0.0)
            finally:
                npipe.DATA_DIR = old_data_dir

    def test_news_archive_coverage_requires_depth(self):
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
                            "headline_id": "c1",
                            "published_at": "2026-03-10T17:17:51Z",
                            "headline": "Nasdaq 100 rises on chip strength",
                            "source": "test",
                            "url": "http://example.com/c1",
                            "sentiment_score": 0.3,
                            "sentiment_label": "positive",
                            "relevance_score": 0.9,
                            "tech_relevant": 1,
                            "ingested_at": "2026-03-10T17:20:00Z",
                        },
                        {
                            "symbol": "QQQ",
                            "headline_id": "c2",
                            "published_at": "2026-03-11T17:17:51Z",
                            "headline": "Big tech slips as yields climb",
                            "source": "test",
                            "url": "http://example.com/c2",
                            "sentiment_score": -0.2,
                            "sentiment_label": "negative",
                            "relevance_score": 0.8,
                            "tech_relevant": 1,
                            "ingested_at": "2026-03-11T17:20:00Z",
                        },
                    ]
                ).to_csv(news_path, index=False)
                coverage = npipe.news_archive_coverage(
                    "QQQ",
                    trading_dates=pd.DatetimeIndex(
                        [
                            pd.Timestamp("2026-03-10T16:00:00Z"),
                            pd.Timestamp("2026-03-11T16:00:00Z"),
                        ]
                    ),
                )
                self.assertEqual(coverage["active_days"], 2)
                self.assertEqual(coverage["trading_days"], 2)
                self.assertFalse(coverage["sufficient_long_horizon"])
            finally:
                npipe.DATA_DIR = old_data_dir

    def test_import_local_historical_news_reads_local_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = npipe.DATA_DIR
            try:
                npipe.DATA_DIR = tmp
                os.makedirs(tmp, exist_ok=True)
                archive_path = os.path.join(tmp, "news_history_QQQ.csv")
                pd.DataFrame(
                    [
                        {
                            "published": "2026-02-10T15:00:00Z",
                            "title": "Nasdaq 100 jumps on chip demand",
                            "publisher": "local-feed",
                            "link": "http://example.com/local1",
                        },
                        {
                            "published": "2026-02-11T16:00:00Z",
                            "title": "Big tech slips as yields rise",
                            "publisher": "local-feed",
                            "link": "http://example.com/local2",
                        },
                    ]
                ).to_csv(archive_path, index=False)
                result = npipe.import_local_historical_news("QQQ")
                self.assertEqual(result["imported_rows"], 2)
                self.assertEqual(len(result["imported_files"]), 1)
                stored = pd.read_csv(os.path.join(tmp, "news_QQQ.csv"))
                self.assertEqual(len(stored), 2)
                self.assertIn("relevance_score", stored.columns)
                self.assertIn("sentiment_score", stored.columns)
                self.assertIn("political_relevance_score", stored.columns)
            finally:
                npipe.DATA_DIR = old_data_dir

    def test_recent_headlines_and_health_respect_as_of_cutoff(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = npipe.DATA_DIR
            try:
                npipe.DATA_DIR = tmp
                os.makedirs(tmp, exist_ok=True)
                pd.DataFrame(
                    [
                        {
                            "symbol": "QQQ",
                            "headline_id": "h1",
                            "published_at": "2026-03-10T15:00:00Z",
                            "headline": "QQQ rises on semiconductor strength",
                            "source": "test",
                            "url": "http://example.com/h1",
                            "sentiment_score": 0.3,
                            "sentiment_label": "positive",
                            "relevance_score": 0.8,
                            "importance_score": 0.5,
                            "impact_score": 0.4,
                            "direction_score": 0.2,
                            "tech_relevant": 1,
                            "ingested_at": "2026-03-10T15:05:00Z",
                        },
                        {
                            "symbol": "QQQ",
                            "headline_id": "h2",
                            "published_at": "2026-03-12T15:00:00Z",
                            "headline": "QQQ slips as chip guidance disappoints",
                            "source": "test",
                            "url": "http://example.com/h2",
                            "sentiment_score": -0.4,
                            "sentiment_label": "negative",
                            "relevance_score": 0.9,
                            "importance_score": 0.6,
                            "impact_score": 0.5,
                            "direction_score": -0.3,
                            "tech_relevant": 1,
                            "ingested_at": "2026-03-12T15:05:00Z",
                        },
                    ]
                ).to_csv(os.path.join(tmp, "news_QQQ.csv"), index=False)
                rows = npipe.load_recent_headlines("QQQ", limit=8, as_of="2026-03-11T12:00:00Z")
                self.assertEqual(len(rows), 1)
                self.assertIn("semiconductor", rows[0]["headline"].lower())
                health = npipe.get_news_health(
                    "QQQ",
                    trading_dates=pd.DatetimeIndex([pd.Timestamp("2026-03-10T16:00:00Z"), pd.Timestamp("2026-03-11T16:00:00Z")]),
                    as_of="2026-03-11T12:00:00Z",
                )
                self.assertEqual(health["rows"], 1)
                self.assertLessEqual(float(health["age_hours"]), 24.0)
            finally:
                npipe.DATA_DIR = old_data_dir

    def test_adaptive_news_weight_uses_past_price_reaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = npipe.DATA_DIR
            try:
                npipe.DATA_DIR = tmp
                os.makedirs(tmp, exist_ok=True)
                news_rows = []
                price_rows = []
                base_close = 100.0
                trade_days = pd.bdate_range("2026-01-05", periods=36)
                for i, day in enumerate(trade_days):
                    positive = (i % 2) == 0
                    headline = (
                        "White House backs semiconductor subsidy support"
                        if positive
                        else "Senate signals new chip export controls on China"
                    )
                    sent = 0.8 if positive else -0.8
                    news_rows.append(
                        {
                            "symbol": "QQQ",
                            "headline_id": f"p{i}",
                            "published_at": (pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=14)).isoformat(),
                            "headline": headline,
                            "source": "test",
                            "url": f"http://example.com/{i}",
                            "sentiment_score": sent,
                            "sentiment_label": "positive" if positive else "negative",
                            "relevance_score": 0.9,
                            "tech_relevant": 1,
                            "political_relevance_score": 0.9,
                            "political_relevant": 1,
                            "ingested_at": (pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=14, minutes=5)).isoformat(),
                        }
                    )
                    close_move = 1.5 if positive else -1.5
                    high = base_close + max(close_move, 0.0) + 0.7
                    low = base_close + min(close_move, 0.0) - 0.7
                    close = base_close + close_move
                    price_rows.append(
                        {
                            "Date": day.strftime("%Y-%m-%d"),
                            "Open": base_close,
                            "High": high,
                            "Low": low,
                            "Close": close,
                            "Volume": 1_000_000 + (i * 1000),
                        }
                    )
                    base_close = close
                pd.DataFrame(news_rows).to_csv(os.path.join(tmp, "news_QQQ.csv"), index=False)
                pd.DataFrame(price_rows).to_csv(os.path.join(tmp, "prices_QQQ_daily_long.csv"), index=False)
                idx = pd.DatetimeIndex(
                    [pd.Timestamp(day).tz_localize("UTC") + pd.Timedelta(hours=20) for day in trade_days[-6:]]
                )
                feats = npipe.load_news_features_for_index("QQQ", idx, tz="America/New_York")
                self.assertIn("news_adaptive_weight_d7", feats.columns)
                self.assertIn("news_weighted_political_d7", feats.columns)
                self.assertGreater(float(feats["news_adaptive_weight_d7"].iloc[-1]), 1.0)
                self.assertGreater(abs(float(feats["news_weighted_political_d7"].iloc[-1])), 0.0)
            finally:
                npipe.DATA_DIR = old_data_dir

    def test_structured_event_fields_capture_novelty_and_noise(self):
        frame = pd.DataFrame(
            [
                {
                    "symbol": "QQQ",
                    "headline_id": "n1",
                    "published_at": "2026-03-10T14:00:00Z",
                    "headline": "Chipmaker raises guidance after strong demand",
                    "source": "MT Newswires",
                },
                {
                    "symbol": "QQQ",
                    "headline_id": "n2",
                    "published_at": "2026-03-10T15:00:00Z",
                    "headline": "Chipmaker raises guidance after strong demand",
                    "source": "MT Newswires",
                },
                {
                    "symbol": "QQQ",
                    "headline_id": "n3",
                    "published_at": "2026-03-10T16:00:00Z",
                    "headline": "Should you buy QQQ after the sell-off? History offers a clear answer",
                    "source": "Motley Fool",
                },
            ]
        )
        norm = npipe._normalize_news_frame(frame, default_symbol="QQQ")
        self.assertEqual(str(norm.iloc[0]["event_type"]), "guidance")
        self.assertGreater(float(norm.iloc[0]["importance_score"]), float(norm.iloc[2]["importance_score"]))
        self.assertLess(float(norm.iloc[1]["novelty_score"]), float(norm.iloc[0]["novelty_score"]))
        self.assertGreater(float(norm.iloc[2]["noise_score"]), float(norm.iloc[0]["noise_score"]))

    def test_get_news_health_marks_stale_news_and_thin_history(self):
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
                            "headline_id": "h1",
                            "published_at": "2026-03-01T15:00:00Z",
                            "headline": "Chip export controls pressure semiconductor stocks",
                            "source": "Reuters",
                        }
                    ]
                ).to_csv(news_path, index=False)
                health = npipe.get_news_health("QQQ")
                self.assertEqual(health["headline_state"], "stale")
                self.assertEqual(health["history_state"], "insufficient_history")
                self.assertFalse(bool(health["signal_ready"]))
            finally:
                npipe.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()
