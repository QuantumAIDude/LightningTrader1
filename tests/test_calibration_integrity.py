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
                self.assertEqual(int(state.get("pending_unresolved_rows") or 0), 1)
                self.assertTrue(bool(state.get("evidence_stale")))
                self.assertEqual(int((state.get("skip_breakdown") or {}).get("unresolved") or 0), 1)
                hist_path = os.path.join(tmp, f"calibration_history_{instr}.jsonl")
                self.assertTrue(os.path.exists(hist_path))
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

    def test_flat_no_trade_rows_are_skipped_from_calibration(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = calibration_tracker.DATA_DIR
            calibration_tracker.DATA_DIR = tmp
            try:
                instr = "TEST"
                ts = datetime.now(timezone.utc) - timedelta(minutes=10)

                preds_path = os.path.join(tmp, f"predictions_{instr}.csv")
                with open(preds_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "instrument",
                            "last_price",
                            "base_pred_ret",
                            "base_sig_bucket",
                            "forest_pred_ret",
                            "forest_sig_bucket",
                            "consensus_dir",
                            "consensus_pred_ret",
                            "consensus_sig_bucket",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": ts.isoformat(),
                            "instrument": instr,
                            "last_price": 100.0,
                            "base_pred_ret": 0.0,
                            "base_sig_bucket": "Weak",
                            "forest_pred_ret": 0.0,
                            "forest_sig_bucket": "Weak",
                            "consensus_dir": "-",
                            "consensus_pred_ret": "",
                            "consensus_sig_bucket": "",
                        }
                    )

                prices_path = os.path.join(tmp, f"prices_{instr}.csv")
                idx = [pd.Timestamp(ts), pd.Timestamp(ts + timedelta(minutes=4))]
                pd.DataFrame({"close": [100.0, 101.0]}, index=idx).to_csv(prices_path)

                out = calibration_tracker.update_live_calibration(instr, horizon_minutes=4)
                self.assertEqual(out.get("updated"), 0)
                self.assertGreaterEqual(out.get("skipped", 0), 1)

                state = calibration_tracker._load_state(instr)
                self.assertEqual(float(state.get("global", {}).get("total", 0.0)), 0.0)
            finally:
                calibration_tracker.DATA_DIR = old_data_dir

    def test_live_calibration_tracks_price_window_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = calibration_tracker.DATA_DIR
            calibration_tracker.DATA_DIR = tmp
            try:
                instr = "TEST"
                ts = datetime.now(timezone.utc) - timedelta(minutes=10)

                preds_path = os.path.join(tmp, f"predictions_{instr}.csv")
                with open(preds_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "instrument",
                            "last_price",
                            "base_pred_ret",
                            "base_sig_bucket",
                            "consensus_dir",
                            "consensus_pred_ret",
                            "consensus_sig_bucket",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": ts.isoformat(),
                            "instrument": instr,
                            "last_price": 100.0,
                            "base_pred_ret": 0.001,
                            "base_sig_bucket": "Medium",
                            "consensus_dir": "UP",
                            "consensus_pred_ret": 0.001,
                            "consensus_sig_bucket": "Medium",
                        }
                    )

                prices_path = os.path.join(tmp, f"prices_{instr}.csv")
                idx = [pd.Timestamp(ts), pd.Timestamp(ts + timedelta(minutes=4))]
                pd.DataFrame({"close": [100.0, 100.12]}, index=idx).to_csv(prices_path)

                out = calibration_tracker.update_live_calibration(instr, horizon_minutes=4)
                self.assertEqual(out.get("updated"), 1)

                state = calibration_tracker._load_state(instr)
                self.assertEqual(len(state.get("recent_price_abs_errors", [])), 1)
                self.assertEqual(len(state.get("recent_price_window_ratios", [])), 1)
                self.assertEqual(len(state.get("recent_price_window_hits", [])), 1)
                self.assertEqual(len(state.get("recent_price_window_thresholds", [])), 1)
                self.assertIsNotNone(state.get("last_resolved_target_ts"))
                self.assertFalse(bool(state.get("evidence_stale")))
            finally:
                calibration_tracker.DATA_DIR = old_data_dir

    def test_live_calibration_uses_mean_model_price_for_row_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = calibration_tracker.DATA_DIR
            calibration_tracker.DATA_DIR = tmp
            try:
                instr = "TEST"
                ts = datetime.now(timezone.utc) - timedelta(minutes=10)

                preds_path = os.path.join(tmp, f"predictions_{instr}.csv")
                with open(preds_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "instrument",
                            "last_price",
                            "base_pred_ret",
                            "base_sig_bucket",
                            "forest_pred_ret",
                            "forest_sig_bucket",
                            "consensus_dir",
                            "consensus_pred_ret",
                            "consensus_sig_bucket",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": ts.isoformat(),
                            "instrument": instr,
                            "last_price": 100.0,
                            "base_pred_ret": 0.001,
                            "base_sig_bucket": "Medium",
                            "forest_pred_ret": 0.003,
                            "forest_sig_bucket": "Medium",
                            "consensus_dir": "UP",
                            "consensus_pred_ret": 0.020,
                            "consensus_sig_bucket": "Strong",
                        }
                    )

                prices_path = os.path.join(tmp, f"prices_{instr}.csv")
                idx = [pd.Timestamp(ts), pd.Timestamp(ts + timedelta(minutes=4))]
                pd.DataFrame({"close": [100.0, 100.25]}, index=idx).to_csv(prices_path)

                out = calibration_tracker.update_live_calibration(instr, horizon_minutes=4)
                self.assertEqual(out.get("updated"), 1)

                state = calibration_tracker._load_state(instr)
                errs = state.get("recent_price_abs_errors", [])
                self.assertEqual(len(errs), 1)
                self.assertAlmostEqual(float(errs[0]), 0.05, places=6)
            finally:
                calibration_tracker.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()
