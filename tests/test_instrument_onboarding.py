import os
import tempfile
import unittest

import pandas as pd

import instrument_onboarding as io


class InstrumentOnboardingTests(unittest.TestCase):
    def test_process_staged_file_registers_symbol_and_writes_canonical_prices(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = io.DATA_DIR
            old_stage_dir = io.NEW_INSTRUMENTS_DIR
            old_processed_dir = io.NEW_INSTRUMENTS_PROCESSED_DIR
            old_instruments_file = io.INSTRUMENTS_FILE
            old_instruments = dict(io.INSTRUMENTS)
            try:
                io.DATA_DIR = tmp
                io.NEW_INSTRUMENTS_DIR = os.path.join(tmp, "new_instruments")
                io.NEW_INSTRUMENTS_PROCESSED_DIR = os.path.join(io.NEW_INSTRUMENTS_DIR, "processed")
                io.INSTRUMENTS_FILE = os.path.join(tmp, "instruments.json")
                io.INSTRUMENTS.clear()
                os.makedirs(io.NEW_INSTRUMENTS_DIR, exist_ok=True)

                staged = os.path.join(io.NEW_INSTRUMENTS_DIR, "ABC.csv")
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

                rep = io.process_staged_instrument_files(auto_train=False, full_train=False)
                self.assertEqual(len(rep.get("processed", [])), 1)
                self.assertIn("ABC", io.INSTRUMENTS)
                self.assertTrue(os.path.exists(os.path.join(tmp, "prices_ABC.csv")))
                self.assertTrue(os.path.exists(io.INSTRUMENTS_FILE))
            finally:
                io.DATA_DIR = old_data_dir
                io.NEW_INSTRUMENTS_DIR = old_stage_dir
                io.NEW_INSTRUMENTS_PROCESSED_DIR = old_processed_dir
                io.INSTRUMENTS_FILE = old_instruments_file
                io.INSTRUMENTS.clear()
                io.INSTRUMENTS.update(old_instruments)


if __name__ == "__main__":
    unittest.main()

