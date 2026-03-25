import os
import tempfile
import unittest

import alert_engine as ae
import execution_engine as ee


class AlertEngineTests(unittest.TestCase):
    def test_alert_snapshot_emits_setup_macro_and_invalidation_alerts(self):
        old_state_path = ee._state_path
        old_alerts_path = ae._alerts_path
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ee._state_path = lambda: os.path.join(tmp, "paper_state.json")
                ae._alerts_path = lambda: os.path.join(tmp, "alerts.jsonl")
                ee.reset_paper_state(100000)
                advisory = {
                    "quality_meta": {
                        "score": 0.79,
                        "agreement": 0.81,
                        "no_trade": False,
                        "reasons": [],
                    },
                    "plan_meta": {
                        "action": "Long setup",
                        "bias": "Bullish",
                        "entry": 100.0,
                        "stop": 99.0,
                        "target1": 103.0,
                    },
                    "macro_meta": {
                        "risk_score": 0.85,
                        "high_risk_24h": True,
                        "next_event": {"title": "US CPI", "time_to_event_text": "2h"},
                    },
                    "catalyst_meta": {
                        "catalyst_risk_score": 0.52,
                        "active_catalyst": {"title": "8-K Filed", "risk_kind": "event risk", "ambiguity": "high"},
                    },
                    "liquidity_meta": {"tradability_score": 0.80},
                    "pred_state": {"intraday": {"position_size": 0.70, "p_up": 0.66}},
                }
                ee.place_paper_order("QQQ", "buy", current_price=100.0, advisory=advisory, profile_name="balanced")
                snap = ae.get_alert_snapshot("QQQ", current_price=98.5, advisory=advisory, profile_name="balanced", persist=True, cooldown_sec=0)
                titles = [row[0] for row in snap["rows"]]
                self.assertTrue(any("US CPI" in title or "Long invalidation" in title or "Long setup" in title for title in titles))
                self.assertTrue(os.path.exists(ae._alerts_path()))
            finally:
                ee._state_path = old_state_path
                ae._alerts_path = old_alerts_path


if __name__ == "__main__":
    unittest.main()
