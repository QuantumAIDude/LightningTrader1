import json
import os
import tempfile
import unittest

import pandas as pd

import readiness_tracker as rt


class ReadinessTrackerTests(unittest.TestCase):
    def test_compute_pass_streak_counts_consecutive_days(self):
        rows = [
            {"date": "2026-03-01", "passed": True, "ts": "2026-03-01T20:00:00Z"},
            {"date": "2026-03-02", "passed": True, "ts": "2026-03-02T20:00:00Z"},
            {"date": "2026-03-03", "passed": False, "ts": "2026-03-03T20:00:00Z"},
            {"date": "2026-03-04", "passed": True, "ts": "2026-03-04T20:00:00Z"},
            {"date": "2026-03-05", "passed": True, "ts": "2026-03-05T20:00:00Z"},
        ]
        streak = rt._compute_pass_streak(rows)
        self.assertEqual(streak, 2)

    def test_load_readiness_state_marks_stale_as_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            old_data_dir = rt.DATA_DIR
            try:
                rt.DATA_DIR = tmp
                instr = "QQQ"
                path = os.path.join(tmp, f"readiness_state_{instr}.json")
                stale_ts = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=100)).isoformat()
                with open(path, "w") as f:
                    json.dump(
                        {
                            "instrument": instr,
                            "last_update": stale_ts,
                            "live_ready": True,
                            "pass_streak_days": 9,
                            "required_streak_days": 3,
                        },
                        f,
                    )
                state = rt.load_readiness_state(instr)
                self.assertIsNotNone(state)
                self.assertFalse(bool(state.get("fresh", True)))
                self.assertFalse(bool(state.get("live_ready", True)))
            finally:
                rt.DATA_DIR = old_data_dir


if __name__ == "__main__":
    unittest.main()

