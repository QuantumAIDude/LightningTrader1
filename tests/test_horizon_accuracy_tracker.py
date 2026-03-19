import csv
import os
import tempfile
import unittest
from datetime import timedelta

import pandas as pd

import horizon_accuracy_tracker as hat


class HorizonAccuracyTrackerTests(unittest.TestCase):
    def test_resolves_daily_and_weekend_predictions(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = hat.DATA_DIR
            old_instruments = dict(hat.INSTRUMENTS)
            try:
                hat.DATA_DIR = tmp
                instr = "TST"
                hat.INSTRUMENTS[instr] = {
                    "type": "equity",
                    "tz": "America/New_York",
                    "session_open_minute": 9 * 60 + 30,
                    "session_close_minute": 16 * 60,
                    "week_end_day": 4,
                }

                now_local = pd.Timestamp.now(tz="America/New_York")
                target_daily = (now_local - pd.Timedelta(days=2)).date()
                target_week_dt = (now_local - pd.Timedelta(days=10)).date()
                target_week = f"{target_week_dt.isocalendar().year}-W{int(target_week_dt.isocalendar().week):02d}"

                pred_path = os.path.join(tmp, f"predictions_horizon_{instr}.csv")
                with open(pred_path, "w", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=[
                            "timestamp",
                            "instrument",
                            "mode",
                            "target_session_date",
                            "target_week",
                            "base_high_pred_price",
                            "base_low_pred_price",
                            "base_week_pred_price",
                        ],
                    )
                    writer.writeheader()
                    writer.writerow(
                        {
                            "timestamp": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=3)).isoformat(),
                            "instrument": instr,
                            "mode": "daily_high_low",
                            "target_session_date": str(target_daily),
                            "target_week": "",
                            "base_high_pred_price": 105.0,
                            "base_low_pred_price": 95.0,
                            "base_week_pred_price": "",
                        }
                    )
                    writer.writerow(
                        {
                            "timestamp": (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=12)).isoformat(),
                            "instrument": instr,
                            "mode": "weekend",
                            "target_session_date": "",
                            "target_week": target_week,
                            "base_high_pred_price": "",
                            "base_low_pred_price": "",
                            "base_week_pred_price": 102.0,
                        }
                    )

                # Build minimal price history covering target session and target week Friday.
                rows = []
                idx = []
                daily_times_local = [
                    pd.Timestamp(target_daily).tz_localize("America/New_York") + pd.Timedelta(hours=10),
                    pd.Timestamp(target_daily).tz_localize("America/New_York") + pd.Timedelta(hours=15, minutes=30),
                ]
                for t in daily_times_local:
                    idx.append(t.tz_convert("UTC"))
                    rows.append({"open": 100.0, "high": 110.0, "low": 90.0, "close": 101.0})

                monday = pd.Timestamp.fromisocalendar(target_week_dt.isocalendar().year, int(target_week_dt.isocalendar().week), 1).tz_localize("America/New_York")
                friday = monday + pd.Timedelta(days=4, hours=15, minutes=30)
                idx.append(friday.tz_convert("UTC"))
                rows.append({"open": 100.0, "high": 104.0, "low": 99.0, "close": 103.0})

                prices = pd.DataFrame(rows, index=pd.DatetimeIndex(idx))
                prices.to_csv(os.path.join(tmp, f"prices_{instr}.csv"))

                out = hat.update_horizon_accuracy(instr)
                self.assertGreaterEqual(int(out.get("newly_resolved_rows", 0)), 3)
                self.assertIn("by_horizon", out)
                daily_models = out["by_horizon"].get("daily_high", {}).get("models", {})
                self.assertIn("base", daily_models)
                self.assertIn("window_hit_rate", daily_models["base"])
                self.assertIn("window_ratio_mean", daily_models["base"])
                self.assertTrue(os.path.exists(os.path.join(tmp, f"horizon_accuracy_{instr}.json")))
            finally:
                hat.DATA_DIR = old_data_dir
                hat.INSTRUMENTS.clear()
                hat.INSTRUMENTS.update(old_instruments)


if __name__ == "__main__":
    unittest.main()
