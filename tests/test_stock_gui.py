import threading
import unittest

import pandas as pd

import stock_gui as sg
from stock_gui import LightningGUI, _dashboard_prediction_signature, _loading_charge_palette


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


class _ImmediateRoot:
    def after(self, _delay, callback):
        callback()


class _ImmediateExecutor:
    def submit(self, fn):
        fn()


class StockGuiDashboardSignatureTests(unittest.TestCase):
    def test_loading_charge_palette_cycles_consistently(self):
        self.assertEqual(_loading_charge_palette(0), _loading_charge_palette(4))
        core, glow = _loading_charge_palette(1)
        self.assertTrue(core.startswith("#"))
        self.assertTrue(glow.startswith("#"))

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
        self.assertIn("autonomy_card", payload)
        self.assertIn("autonomy_detail", payload)
        self.assertIn("autonomy_meta", payload)
        self.assertIn("profile_card", payload)
        self.assertIn("broker_card", payload)
        self.assertIn("replay_card", payload)
        self.assertIn("options_detail", payload)
        self.assertIn("news_detail", payload)

    def test_waiting_payload_populates_right_side_reasons(self):
        gui = LightningGUI.__new__(LightningGUI)
        payload = gui._build_waiting_payload("4min", "Loading support data…", status="Loading support data…")

        self.assertTrue(payload["market_rows"])
        self.assertTrue(payload["options_rows"])
        self.assertTrue(payload["news_rows"])
        self.assertEqual(payload["options_detail"], "Loading support data…")
        self.assertEqual(payload["news_detail"], "Loading support data…")
        self.assertIn("Loading support data…", payload["status_messages"])

    def test_invalidate_workspace_caches_keeps_other_symbols_warm(self):
        gui = LightningGUI.__new__(LightningGUI)
        gui.instrument_var = _FakeVar("QQQ")
        gui._dashboard_cache = {"QQQ": {"status": "old"}, "AAPL": {"status": "keep"}}
        gui._prediction_payload_cache = {
            ("QQQ", "4min", "bar-1", "live", "balanced"): {"status": "drop"},
            ("AAPL", "4min", "bar-1", "live", "balanced"): {"status": "keep"},
        }

        gui._invalidate_workspace_caches("QQQ")

        self.assertNotIn("QQQ", gui._dashboard_cache)
        self.assertIn("AAPL", gui._dashboard_cache)
        self.assertEqual(len(gui._prediction_payload_cache), 1)
        self.assertIn(("AAPL", "4min", "bar-1", "live", "balanced"), gui._prediction_payload_cache)

    def test_refresh_symbol_runtime_fallback_uses_model_loader_without_data_hub(self):
        gui = LightningGUI.__new__(LightningGUI)
        old_loader = sg.MODEL_STACK_LOADER
        old_update_price_data = sg.update_price_data
        try:
            sg.MODEL_STACK_LOADER = lambda instrument_key=None: {"base": {"instrument": instrument_key}}
            sg.update_price_data = lambda symbol: pd.DataFrame(
                [{"close": 100.0}],
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
            )

            bundle = gui._refresh_symbol_runtime("qqq", force=True, reason="unit", load_models=True)

            self.assertEqual(bundle["symbol"], "QQQ")
            self.assertIn("base", bundle["models"])
            self.assertFalse(bundle["df"].empty)
        finally:
            sg.MODEL_STACK_LOADER = old_loader
            sg.update_price_data = old_update_price_data

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
        gui.route_flat_var = _FakeVar()
        gui.route_cancel_var = _FakeVar()
        gui.route_setup_var = _FakeVar()
        gui.route_refresh_var = _FakeVar()
        gui.route_verify_var = _FakeVar()
        gui.replay_toggle_var = _FakeVar()
        gui.route_buy_btn = _FakeButton()
        gui.route_sell_btn = _FakeButton()
        gui.route_flat_btn = _FakeButton()
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
        self.assertEqual(gui.route_flat_var.get(), "Flat (Alpaca Paper)")
        self.assertEqual(gui.route_setup_var.get(), "Alpaca Setup")
        self.assertEqual(gui.route_refresh_var.get(), "Refresh Broker")
        self.assertEqual(gui.route_verify_var.get(), "Verify Alpaca Paper")
        self.assertEqual(gui.route_buy_btn.state, "normal")
        self.assertEqual(gui.route_setup_btn.state, "normal")
        self.assertEqual(gui.route_cancel_btn.state, "normal")
        self.assertEqual(gui.route_flat_btn.state, "normal")
        self.assertEqual(gui.replay_toggle_var.get(), "Replay Resume")
        self.assertEqual(gui.replay_reset_btn.state, "normal")

    def test_sync_action_buttons_locks_manual_entries_when_actual_autonomy_is_active(self):
        gui = LightningGUI.__new__(LightningGUI)
        gui._latest_execution_meta = {
            "route_label": "Local Paper",
            "runtime": {
                "can_place_orders": True,
                "can_cancel": True,
                "route_label": "Local Paper",
            },
        }
        gui._latest_replay_meta = {"active": False, "paused": True, "completed": False}
        gui._latest_autonomy_meta = {"mode": "paper_auto", "manual_trading_locked": True, "kill_switch": False, "operator_label": "Operator: PAPER AUTO / Local Paper"}
        gui.route_buy_var = _FakeVar()
        gui.route_sell_var = _FakeVar()
        gui.route_flat_var = _FakeVar()
        gui.route_cancel_var = _FakeVar()
        gui.route_setup_var = _FakeVar()
        gui.route_refresh_var = _FakeVar()
        gui.route_verify_var = _FakeVar()
        gui.autonomy_halt_var = _FakeVar()
        gui.autonomy_badge_var = _FakeVar()
        gui.route_buy_btn = _FakeButton()
        gui.route_sell_btn = _FakeButton()
        gui.route_flat_btn = _FakeButton()
        gui.route_cancel_btn = _FakeButton()
        gui.route_setup_btn = _FakeButton()
        gui.route_refresh_btn = _FakeButton()
        gui.route_verify_btn = _FakeButton()
        gui.paper_buy_btn = _FakeButton()
        gui.paper_sell_btn = _FakeButton()
        gui.paper_flatten_btn = _FakeButton()
        gui.replay_toggle_var = _FakeVar()
        gui.replay_step_btn = _FakeButton()
        gui.replay_toggle_btn = _FakeButton()
        gui.replay_reset_btn = _FakeButton()
        gui.replay_stop_btn = _FakeButton()
        gui.replay_buy_btn = _FakeButton()
        gui.replay_sell_btn = _FakeButton()
        gui.replay_flat_btn = _FakeButton()

        gui._sync_action_buttons()

        self.assertEqual(gui.route_buy_btn.state, "disabled")
        self.assertEqual(gui.route_sell_btn.state, "disabled")
        self.assertEqual(gui.paper_buy_btn.state, "disabled")
        self.assertEqual(gui.paper_sell_btn.state, "disabled")
        self.assertEqual(gui.route_flat_btn.state, "normal")
        self.assertEqual(gui.autonomy_halt_var.get(), "Halt Auto")
        self.assertIn("PAPER AUTO", gui.autonomy_badge_var.get())

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

    def test_accuracy_refresh_does_not_overwrite_inactive_workspace(self):
        gui = LightningGUI.__new__(LightningGUI)
        gui._acc_lock = threading.Lock()
        gui._acc_in_flight = False
        gui.instrument_var = _FakeVar("QQQ")
        gui.accuracy_var = _FakeVar("unchanged")
        gui.root = _ImmediateRoot()
        gui.executor = _ImmediateExecutor()
        gui._symbol_key = lambda symbol: str(symbol or "").upper()
        gui._workspace_is_active = lambda symbol: False
        gui._format_accuracy_summary = lambda report, selected: f"Avg Discrepancy ({selected}): updated"

        old_eval = sg.evaluate_readiness_metrics
        try:
            sg.evaluate_readiness_metrics = lambda selected, online=False: {"instrument": selected}
            gui.refresh_accuracy_summary_threaded()
        finally:
            sg.evaluate_readiness_metrics = old_eval

        self.assertEqual(gui.accuracy_var.get(), "unchanged")

    def test_workspace_summary_text_uses_operator_and_evidence_labels(self):
        gui = LightningGUI.__new__(LightningGUI)
        gui.instrument_var = _FakeVar("AAPL")
        gui._workspace_meta = lambda symbol=None: {}
        gui._runtime_bundle = lambda symbol: {"models": {}, "model_summary": {"available_modes": []}, "freshness": {"inputs_fresh": False}}

        old_get_workspace = sg.get_workspace
        old_summary = sg.summarize_instrument_meta
        old_build_lifecycle = sg.build_symbol_lifecycle_snapshot
        try:
            sg.get_workspace = lambda sym: {}
            sg.summarize_instrument_meta = lambda meta, sym: "Test instrument"
            sg.build_symbol_lifecycle_snapshot = lambda symbol, **kwargs: {
                "symbol": "AAPL",
                "lifecycle_label": "Profile resolved",
                "operator_state_label": "Blocked: insufficient symbol maturity",
                "evidence_state_label": "Missing evidence",
                "model_coverage_label": "No model coverage",
                "onboarding_status": "Not started",
                "primary_next_requirement": "Stage canonical price history for this symbol and rerun readiness refresh.",
            }

            text = gui._workspace_summary_text("AAPL")

            self.assertIn("Blocked: insufficient symbol maturity", text)
            self.assertIn("Missing evidence", text)
            self.assertIn("Profile resolved", text)
            self.assertIn("Stage canonical price history", text)
        finally:
            sg.get_workspace = old_get_workspace
            sg.summarize_instrument_meta = old_summary
            sg.build_symbol_lifecycle_snapshot = old_build_lifecycle

    def test_waiting_payload_uses_lifecycle_placeholders_for_symbol(self):
        gui = LightningGUI.__new__(LightningGUI)
        gui.instrument_var = _FakeVar("AAPL")
        gui._symbol_key = lambda symbol: str(symbol or "").upper()
        gui._base_payload = LightningGUI._base_payload.__get__(gui, LightningGUI)
        gui._lifecycle_snapshot = lambda symbol=None, runtime=None: {
            "lifecycle_label": "Profile resolved",
            "lifecycle_detail": "The symbol is registered, but price history staging is still the next gating step.",
            "workspace_summary": "Profile resolved | Blocked: insufficient symbol maturity | Missing evidence",
            "operator_state_label": "Blocked: insufficient symbol maturity",
            "operator_state_detail": "The platform supports this symbol, but symbol-specific market data has not been staged yet.",
            "evidence_state_label": "Missing evidence",
            "evidence_state_detail": "This symbol is still missing some combination of staged data, trained models, or readiness validation evidence.",
            "model_coverage_label": "No model coverage",
            "model_state_detail": "No model artifacts are available for this symbol yet.",
            "stage_label": "Supported / not trained",
            "rollout_label": "Shadow only rollout",
            "symbol_maturity_label": "Supported only",
            "next_requirement_category": "symbol_maturity",
            "primary_next_requirement": "Stage canonical price history for this symbol and rerun readiness refresh.",
            "data_state_label": "No canonical price history",
            "data_state_detail": "This symbol is supported in the registry, but canonical price history has not been staged yet.",
            "feature_state_label": "Feature readiness missing",
            "feature_state_detail": "No feature-readiness artifacts have been captured for this symbol yet.",
            "model_state_label": "No trained models",
            "news_state_label": "No news context",
            "news_state_detail": "No local news history has been staged for this symbol yet.",
            "calibration_state_label": "Calibration blocked",
            "calibration_state_detail": "Calibration is unavailable because the required symbol-specific models and evidence are not ready yet.",
            "onboarding_status": "Not started",
            "onboarding_summary": "No staged onboarding job has run for this symbol yet.",
            "onboarding_progress": "0/9",
        }
        gui._apply_lifecycle_placeholders = LightningGUI._apply_lifecycle_placeholders.__get__(gui, LightningGUI)
        gui._populate_panel_placeholders = LightningGUI._populate_panel_placeholders.__get__(gui, LightningGUI)
        payload = gui._build_waiting_payload("4min", "Waiting for symbol maturity…", symbol="AAPL")

        self.assertEqual(payload["setup_card"], "Lifecycle: Profile resolved")
        self.assertEqual(payload["action_card"], "Action: Blocked: insufficient symbol maturity")
        self.assertTrue(any(row[0] == "Lifecycle" for row in payload["stack_rows"]))
        self.assertTrue(any(row[0] == "Data State" for row in payload["market_rows"]))
        self.assertIn("Stage canonical price history", payload["setup_rows"][0][2])


if __name__ == "__main__":
    unittest.main()
