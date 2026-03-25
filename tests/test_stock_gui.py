import unittest

from stock_gui import LightningGUI, _dashboard_prediction_signature


class _FakeVar:
    def __init__(self, value=None):
        self.value = value

    def set(self, value):
        self.value = value

    def get(self):
        return self.value


class _FakeButton:
    def __init__(self):
        self.state = None

    def configure(self, **kwargs):
        if "state" in kwargs:
            self.state = kwargs["state"]


class StockGuiDashboardSignatureTests(unittest.TestCase):
    def test_signature_changes_when_predictions_change_on_same_bar(self):
        preds_a = {
            "intraday": {"consensus": {"direction": "UP", "pred_price": 100.5, "confidence": 61.2}},
            "daily": {
                "high": {
                    "base": {"pred_price": 101.0},
                    "forest": {"pred_price": 101.2},
                    "boost": {"pred_price": 100.9},
                    "extra_trees": {"pred_price": 101.1},
                },
                "low": {
                    "base": {"pred_price": 99.1},
                    "forest": {"pred_price": 99.0},
                    "boost": {"pred_price": 99.2},
                    "extra_trees": {"pred_price": 99.3},
                },
            },
            "weekend": {
                "week": {
                    "base": {"pred_price": 102.0, "pred_high_price": 103.0, "pred_low_price": 100.0},
                    "forest": {"pred_price": 102.2, "pred_high_price": 103.1, "pred_low_price": 100.2},
                }
            },
        }
        preds_b = {
            "intraday": {"consensus": {"direction": "DOWN", "pred_price": 99.8, "confidence": 58.0}},
            "daily": preds_a["daily"],
            "weekend": preds_a["weekend"],
        }
        self.assertNotEqual(
            _dashboard_prediction_signature(preds_a),
            _dashboard_prediction_signature(preds_b),
        )

    def test_signature_is_stable_for_equivalent_prediction_payloads(self):
        preds_a = {
            "daily": {
                "high": {
                    "forest": {"pred_price": 101.2},
                    "base": {"pred_price": 101.0},
                },
                "low": {
                    "boost": {"pred_price": 99.2},
                    "extra_trees": {"pred_price": 99.3},
                },
            }
        }
        preds_b = {
            "daily": {
                "high": {
                    "base": {"pred_price": 101.0},
                    "forest": {"pred_price": 101.2},
                },
                "low": {
                    "extra_trees": {"pred_price": 99.3},
                    "boost": {"pred_price": 99.2},
                },
            }
        }
        self.assertEqual(
            _dashboard_prediction_signature(preds_a),
            _dashboard_prediction_signature(preds_b),
        )

    def test_base_payload_contains_new_trader_workstation_sections(self):
        gui = LightningGUI.__new__(LightningGUI)
        payload = gui._base_payload("4min")
        self.assertIn("alerts_rows", payload)
        self.assertIn("execution_rows", payload)
        self.assertIn("replay_rows", payload)
        self.assertIn("portfolio_rows", payload)
        self.assertIn("performance_rows", payload)
        self.assertIn("opportunity_rows", payload)
        self.assertIn("profile_card", payload)
        self.assertIn("broker_card", payload)
        self.assertIn("replay_card", payload)

    def test_sync_action_buttons_reflects_route_and_replay_state(self):
        gui = LightningGUI.__new__(LightningGUI)
        gui._latest_execution_meta = {
            "route_label": "Alpaca Paper",
            "route_setup_label": "Alpaca Setup",
            "route_refresh_label": "Refresh Broker",
            "route_verify_label": "Verify Alpaca Paper",
            "runtime": {
                "can_place_orders": True,
                "can_cancel": True,
                "route_label": "Alpaca Paper",
            },
        }
        gui._latest_replay_meta = {"active": True, "paused": True, "completed": False}
        gui.route_buy_var = _FakeVar()
        gui.route_sell_var = _FakeVar()
        gui.route_cancel_var = _FakeVar()
        gui.route_setup_var = _FakeVar()
        gui.route_refresh_var = _FakeVar()
        gui.route_verify_var = _FakeVar()
        gui.replay_toggle_var = _FakeVar()
        gui.route_buy_btn = _FakeButton()
        gui.route_sell_btn = _FakeButton()
        gui.route_cancel_btn = _FakeButton()
        gui.route_setup_btn = _FakeButton()
        gui.route_refresh_btn = _FakeButton()
        gui.route_verify_btn = _FakeButton()
        gui.replay_step_btn = _FakeButton()
        gui.replay_toggle_btn = _FakeButton()
        gui.replay_reset_btn = _FakeButton()
        gui.replay_stop_btn = _FakeButton()
        gui.replay_buy_btn = _FakeButton()
        gui.replay_sell_btn = _FakeButton()
        gui.replay_flat_btn = _FakeButton()

        gui._sync_action_buttons()

        self.assertEqual(gui.route_buy_var.get(), "Buy (Alpaca Paper)")
        self.assertEqual(gui.route_setup_var.get(), "Alpaca Setup")
        self.assertEqual(gui.route_refresh_var.get(), "Refresh Broker")
        self.assertEqual(gui.route_verify_var.get(), "Verify Alpaca Paper")
        self.assertEqual(gui.route_buy_btn.state, "normal")
        self.assertEqual(gui.route_setup_btn.state, "normal")
        self.assertEqual(gui.route_cancel_btn.state, "normal")
        self.assertEqual(gui.replay_toggle_var.get(), "Replay Resume")
        self.assertEqual(gui.replay_reset_btn.state, "normal")

    def test_friendly_broker_setup_message_is_beginner_friendly(self):
        gui = LightningGUI.__new__(LightningGUI)
        msg = gui._friendly_broker_setup_message({"config_state": "credentials_missing"})
        self.assertIn("API key", msg)
        msg_invalid = gui._friendly_broker_setup_message({"config_state": "invalid_credentials"})
        self.assertIn("rejected", msg_invalid)

    def test_update_status_helpers_are_honest_and_actionable(self):
        gui = LightningGUI.__new__(LightningGUI)
        self.assertIn("Update available", gui._format_update_status({"status": "update_available", "latest_version": "1.2.3"}))
        self.assertIn("Up to date", gui._format_update_status({"status": "up_to_date"}))
        self.assertEqual(gui._update_status_color({"status": "update_available"}), "#fbbf24")
        self.assertEqual(gui._update_status_color({"status": "error"}), "#fda4af")


if __name__ == "__main__":
    unittest.main()
