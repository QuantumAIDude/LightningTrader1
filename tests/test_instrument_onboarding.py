import os
import tempfile
import unittest

import pandas as pd

import instrument_onboarding as io
import symbol_pipeline as sp


class InstrumentOnboardingTests(unittest.TestCase):
    def test_process_staged_file_registers_symbol_and_writes_canonical_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = io.DATA_DIR
            old_stage_dir = io.NEW_INSTRUMENTS_DIR
            old_processed_dir = io.NEW_INSTRUMENTS_PROCESSED_DIR
            old_failed_dir = io.NEW_INSTRUMENTS_FAILED_DIR
            old_instruments_file = io.INSTRUMENTS_FILE
            old_instruments = dict(io.INSTRUMENTS)
            old_symbol_pipeline_data_dir = sp.DATA_DIR
            try:
                io.DATA_DIR = tmp
                sp.DATA_DIR = tmp
                io.NEW_INSTRUMENTS_DIR = os.path.join(tmp, "new_instruments")
                io.NEW_INSTRUMENTS_PROCESSED_DIR = os.path.join(io.NEW_INSTRUMENTS_DIR, "processed")
                io.NEW_INSTRUMENTS_FAILED_DIR = os.path.join(io.NEW_INSTRUMENTS_DIR, "failed")
                io.INSTRUMENTS_FILE = os.path.join(tmp, "instruments.json")
                io.INSTRUMENTS.clear()
                os.makedirs(io.NEW_INSTRUMENTS_DIR, exist_ok=True)
                import news_pipeline as npipe
                old_news_data_dir = npipe.DATA_DIR
                old_news_stage_dir = getattr(npipe, "NEW_INSTRUMENTS_DIR", io.NEW_INSTRUMENTS_DIR)
                npipe.DATA_DIR = tmp
                npipe.NEW_INSTRUMENTS_DIR = io.NEW_INSTRUMENTS_DIR

                staged = os.path.join(io.NEW_INSTRUMENTS_DIR, "ABC.csv")
                staged_meta = os.path.join(io.NEW_INSTRUMENTS_DIR, "ABC.meta.json")
                staged_news = os.path.join(io.NEW_INSTRUMENTS_DIR, "ABC_news.csv")
                ts = pd.date_range("2026-01-05 14:30:00+00:00", periods=4, freq="5min")
                pd.DataFrame(
                    {
                        "timestamp": ts,
                        "open": [100, 101, 102, 103],
                        "high": [101, 102, 103, 104],
                        "low": [99, 100, 101, 102],
                        "close": [100.5, 101.2, 102.3, 103.8],
                        "volume": [1000, 1200, 1400, 1600],
                    }
                ).to_csv(staged, index=False)
                pd.DataFrame(
                    [
                        {"published": "2026-01-04T15:00:00Z", "title": "ABC wins tariff relief", "publisher": "local", "link": "http://example.com/1"},
                    ]
                ).to_csv(staged_news, index=False)
                with open(staged_meta, "w") as f:
                    f.write('{"display_name":"ABC Corp","news_aliases":["ABC","ABC Corp"]}')

                rep = io.process_staged_instrument_files(auto_train=False, full_train=False)
                self.assertEqual(len(rep.get("processed", [])), 1)
                processed = rep["processed"][0]
                self.assertIn("ABC", io.INSTRUMENTS)
                self.assertEqual(io.INSTRUMENTS["ABC"].get("display_name"), "ABC Corp")
                self.assertEqual(processed["new_rows"], 4)
                self.assertTrue(processed["profile_resolved"])
                self.assertIn(processed["history_status"], {"Insufficient history", "Features partially ready"})
                self.assertEqual(processed["eligible_modes"], [])
                self.assertIn("Advance intraday history", processed["primary_next_requirement"])
                self.assertEqual(processed["status"], "processed")
                self.assertTrue(os.path.exists(os.path.join(tmp, "prices_ABC.csv")))
                self.assertTrue(os.path.exists(os.path.join(tmp, "prices_ABC_daily_long.csv")))
                self.assertGreaterEqual(processed["daily_long_rows"], 1)
                self.assertTrue(processed["daily_long_sources"])
                self.assertTrue(os.path.exists(os.path.join(tmp, "news_ABC.csv")))
                self.assertTrue(os.path.exists(io.INSTRUMENTS_FILE))
            finally:
                io.DATA_DIR = old_data_dir
                io.NEW_INSTRUMENTS_DIR = old_stage_dir
                io.NEW_INSTRUMENTS_PROCESSED_DIR = old_processed_dir
                io.NEW_INSTRUMENTS_FAILED_DIR = old_failed_dir
                io.INSTRUMENTS_FILE = old_instruments_file
                io.INSTRUMENTS.clear()
                io.INSTRUMENTS.update(old_instruments)
                sp.DATA_DIR = old_symbol_pipeline_data_dir
                npipe.DATA_DIR = old_news_data_dir
                npipe.NEW_INSTRUMENTS_DIR = old_news_stage_dir

    def test_invalid_staged_file_is_quarantined(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = io.DATA_DIR
            old_stage_dir = io.NEW_INSTRUMENTS_DIR
            old_processed_dir = io.NEW_INSTRUMENTS_PROCESSED_DIR
            old_failed_dir = io.NEW_INSTRUMENTS_FAILED_DIR
            old_instruments_file = io.INSTRUMENTS_FILE
            old_instruments = dict(io.INSTRUMENTS)
            old_symbol_pipeline_data_dir = sp.DATA_DIR
            try:
                io.DATA_DIR = tmp
                sp.DATA_DIR = tmp
                io.NEW_INSTRUMENTS_DIR = os.path.join(tmp, "new_instruments")
                io.NEW_INSTRUMENTS_PROCESSED_DIR = os.path.join(io.NEW_INSTRUMENTS_DIR, "processed")
                io.NEW_INSTRUMENTS_FAILED_DIR = os.path.join(io.NEW_INSTRUMENTS_DIR, "failed")
                io.INSTRUMENTS_FILE = os.path.join(tmp, "instruments.json")
                io.INSTRUMENTS.clear()
                os.makedirs(io.NEW_INSTRUMENTS_DIR, exist_ok=True)

                staged = os.path.join(io.NEW_INSTRUMENTS_DIR, "BAD.csv")
                pd.DataFrame({"timestamp": ["bad-ts"], "close": [None]}).to_csv(staged, index=False)

                rep = io.process_staged_instrument_files(auto_train=False, full_train=False)
                self.assertEqual(len(rep.get("processed", [])), 0)
                self.assertEqual(len(rep.get("errors", [])), 1)
                self.assertEqual(len(rep.get("quarantined", [])), 1)
                self.assertFalse(os.path.exists(staged))
                failed_dir = io.NEW_INSTRUMENTS_FAILED_DIR
                failed_files = [x for x in os.listdir(failed_dir) if x.endswith(".csv")]
                self.assertEqual(len(failed_files), 1)
                self.assertTrue(os.path.exists(os.path.join(failed_dir, failed_files[0] + ".json")))
            finally:
                io.DATA_DIR = old_data_dir
                io.NEW_INSTRUMENTS_DIR = old_stage_dir
                io.NEW_INSTRUMENTS_PROCESSED_DIR = old_processed_dir
                io.NEW_INSTRUMENTS_FAILED_DIR = old_failed_dir
                io.INSTRUMENTS_FILE = old_instruments_file
                io.INSTRUMENTS.clear()
                io.INSTRUMENTS.update(old_instruments)
                sp.DATA_DIR = old_symbol_pipeline_data_dir


if __name__ == "__main__":
    unittest.main()
