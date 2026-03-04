import csv
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

import pandas as pd

import calibration_tracker
import model_utils


class CalibrationIntegrityTests(unittest.TestCase):
    def test_last_processed_does_not_advance_when_targets_unresolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = calibration_tracker.DATA_DIR
            calibration_tracker.DATA_DIR = tmp
            try:
                instr = "TEST"
                ts = datetime.now(timezone.utc) - timedelta(minutes=1)

                preds_path = os.path.join(tmp, f"predictions_{instr}.csv")
                with open(preds_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=["timestamp", "instrument", "last_price", "base_pred_ret", "base_sig_bucket"],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": ts.isoformat(),
                            "instrument": instr,
                            "last_price": 100.0,
                            "base_pred_ret": 0.001,
                            "base_sig_bucket": "Medium",
                        }
                    )

                prices_path = os.path.join(tmp, f"prices_{instr}.csv")
                pd.DataFrame({"close": [100.0]}, index=[pd.Timestamp(ts)]).to_csv(prices_path)

                out = calibration_tracker.update_live_calibration(instr, horizon_minutes=4)
                self.assertEqual(out.get("updated"), 0)
                self.assertGreaterEqual(out.get("skipped", 0), 1)

                state = calibration_tracker._load_state(instr)
                self.assertIsNone(state.get("last_processed"))
            finally:
                calibration_tracker.DATA_DIR = old_data_dir

    def test_live_calibration_cache_is_instrument_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = model_utils.DATA_DIR
            old_cache = dict(model_utils._LIVE_CAL_CACHE)
            model_utils.DATA_DIR = tmp
            model_utils._LIVE_CAL_CACHE.clear()
            try:
                now = datetime.now(timezone.utc).isoformat()
                payload_a = {
                    "last_update": now,
                    "by_model": {"base": {"Weak": {"wins": 1, "total": 3}}},
                }
                payload_b = {
                    "last_update": now,
                    "by_model": {"base": {"Weak": {"wins": 9, "total": 9}}},
                }
                with open(os.path.join(tmp, "calibration_live_A.json"), "w") as f:
                    json.dump(payload_a, f)
                with open(os.path.join(tmp, "calibration_live_B.json"), "w") as f:
                    json.dump(payload_b, f)

                a = model_utils._load_live_calibration("A")
                b = model_utils._load_live_calibration("B")

                self.assertIsNotNone(a)
                self.assertIsNotNone(b)
                self.assertEqual(a["by_model"]["base"]["Weak"]["wins"], 1)
                self.assertEqual(b["by_model"]["base"]["Weak"]["wins"], 9)
            finally:
                model_utils.DATA_DIR = old_data_dir
                model_utils._LIVE_CAL_CACHE.clear()
                model_utils._LIVE_CAL_CACHE.update(old_cache)

    def test_live_calibration_probability_is_smoothed(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = model_utils.DATA_DIR
            old_cache = dict(model_utils._LIVE_CAL_CACHE)
            model_utils.DATA_DIR = tmp
            model_utils._LIVE_CAL_CACHE.clear()
            try:
                now = datetime.now(timezone.utc).isoformat()
                payload = {
                    "last_update": now,
                    "by_model": {"base": {"Weak": {"wins": 1, "total": 1}}},
                }
                with open(os.path.join(tmp, "calibration_live_A.json"), "w") as f:
                    json.dump(payload, f)

                prob, n = model_utils._live_calibration_prob("A", "base", "Weak")
                self.assertEqual(n, 1)
                self.assertAlmostEqual(prob, 2.0 / 3.0, places=6)
            finally:
                model_utils.DATA_DIR = old_data_dir
                model_utils._LIVE_CAL_CACHE.clear()
                model_utils._LIVE_CAL_CACHE.update(old_cache)


if __name__ == "__main__":
    unittest.main()
