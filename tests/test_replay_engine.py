import unittest

import pandas as pd

import replay_engine as re


class ReplayEngineTests(unittest.TestCase):
    def test_build_and_step_replay_session(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1100.0, 1200.0],
            },
            index=pd.date_range("2026-03-24 14:30:00+00:00", periods=3, freq="2min"),
        )
        session = re.build_replay_session(df, "QQQ", starting_cash=50000, profile_name="balanced")
        self.assertEqual(session["status"], "ready")
        self.assertEqual(session["cursor"], 0)
        self.assertEqual(session["current_bar"]["close"], 100.5)
        stepped = re.step_replay_session(session)
        self.assertEqual(stepped["cursor"], 1)
        self.assertEqual(stepped["current_bar"]["close"], 101.5)

    def test_start_step_and_stop_active_replay(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.5, 101.5, 102.5, 103.5],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            },
            index=pd.date_range("2026-03-24 14:30:00+00:00", periods=4, freq="2min"),
        )
        old_path = re._session_path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            try:
                re._session_path = lambda: os.path.join(tmp, "replay.json")
                started = re.start_active_replay(df, "QQQ", profile_name="balanced", warmup_bars=2)
                self.assertTrue(started["active"])
                self.assertEqual(started["status"], "paused")
                stepped = re.step_active_replay()
                self.assertTrue(stepped["active"])
                self.assertIn(stepped["status"], {"paused", "completed"})
                stopped = re.stop_active_replay()
                self.assertFalse(stopped["active"])
            finally:
                re._session_path = old_path

    def test_reset_active_replay_rewinds_cursor_and_clears_replay_paper_state(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 103.0],
                "high": [101.0, 102.0, 103.0, 104.0],
                "low": [99.0, 100.0, 101.0, 102.0],
                "close": [100.0, 101.0, 102.0, 103.0],
                "volume": [1000.0, 1100.0, 1200.0, 1300.0],
            },
            index=pd.date_range("2026-03-24 14:30:00+00:00", periods=4, freq="2min"),
        )
        advisory = {
            "quality_meta": {"score": 0.74, "no_trade": False, "regime_label": "Bullish"},
            "plan_meta": {"bias": "Bullish", "action": "Long setup", "entry": 100.0, "stop": 99.0, "target1": 103.0},
            "liquidity_meta": {"tradability_score": 0.82},
            "pred_state": {"intraday": {"position_size": 0.65, "p_up": 0.64}},
        }
        old_path = re._session_path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            try:
                re._session_path = lambda: os.path.join(tmp, "replay.json")
                started = re.start_active_replay(df, "QQQ", profile_name="balanced", warmup_bars=2)
                self.assertEqual(started["status"], "paused")
                re.place_replay_order("buy", advisory=advisory, qty=2)
                stepped = re.step_active_replay()
                self.assertIn(stepped["status"], {"paused", "completed"})
                reset = re.reset_active_replay()
                self.assertTrue(reset["active"])
                self.assertEqual(reset["status"], "paused")
                self.assertTrue(any(row[0] == "Progress" for row in reset["rows"]))
                session = re._load_active_session()
                self.assertEqual(session["cursor"], session["start_index"])
                self.assertFalse(session["paper_state"].get("positions"))
                self.assertFalse(session["paper_state"].get("closed_trades"))
            finally:
                re._session_path = old_path

    def test_replay_order_lifecycle_records_trade(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.0, 102.0, 103.0],
                "volume": [1000.0, 1100.0, 1200.0],
            },
            index=pd.date_range("2026-03-24 14:30:00+00:00", periods=3, freq="2min"),
        )
        advisory = {
            "quality_meta": {"score": 0.74, "no_trade": False, "regime_label": "Bullish"},
            "plan_meta": {"bias": "Bullish", "action": "Long setup", "entry": 100.0, "stop": 99.0, "target1": 103.0},
            "liquidity_meta": {"tradability_score": 0.82},
            "pred_state": {"intraday": {"position_size": 0.65, "p_up": 0.64}},
        }
        old_path = re._session_path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            try:
                re._session_path = lambda: os.path.join(tmp, "replay.json")
                re.start_active_replay(df, "QQQ", profile_name="balanced", warmup_bars=1)
                buy = re.place_replay_order("buy", advisory=advisory, qty=5)
                self.assertEqual(buy["status"], "filled")
                flat = re.flatten_replay_position(advisory=advisory)
                self.assertEqual(flat["status"], "filled")
                snap = re.get_replay_dashboard_snapshot("QQQ")
                self.assertTrue(snap["performance_rows"])
            finally:
                re._session_path = old_path

    def test_replay_dashboard_snapshot_labels_completion_and_progress(self):
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0],
                "high": [101.0, 102.0, 103.0],
                "low": [99.0, 100.0, 101.0],
                "close": [100.5, 101.5, 102.5],
                "volume": [1000.0, 1100.0, 1200.0],
            },
            index=pd.date_range("2026-03-24 14:30:00+00:00", periods=3, freq="2min"),
        )
        old_path = re._session_path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            try:
                re._session_path = lambda: os.path.join(tmp, "replay.json")
                re.start_active_replay(df, "QQQ", profile_name="balanced", warmup_bars=1)
                re.step_active_replay(steps=5)
                snap = re.get_replay_dashboard_snapshot("QQQ")
                self.assertEqual(snap["status"], "completed")
                self.assertIn("completed", snap["card"].lower())
                self.assertTrue(any(row[0] == "Replay Trades" for row in snap["rows"]))
            finally:
                re._session_path = old_path


if __name__ == "__main__":
    unittest.main()
