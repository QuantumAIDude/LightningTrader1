import os
import tempfile
import unittest

import execution_engine as ee


def _runtime_settings(
    *,
    api_key: str = "",
    secret_key: str = "",
    paper: bool = True,
    broker_mode: str = "local_paper",
    provider: str = "alpaca",
    credentials_source: str | None = None,
    broker_mode_source: str = "default",
):
    creds_source = credentials_source
    if creds_source is None:
        creds_source = "env" if api_key and secret_key else "missing"
    return {
        "provider": provider,
        "provider_source": "env",
        "broker_mode": broker_mode,
        "broker_mode_source": broker_mode_source,
        "paper": paper,
        "paper_source": "env",
        "api_key": api_key,
        "secret_key": secret_key,
        "credentials_present": bool(api_key and secret_key),
        "credentials_source": creds_source,
        "has_saved_credentials": creds_source == "local_store",
        "saved_credentials": {},
        "storage_kind": "local_file",
        "storage_path": "/tmp/alpaca_credentials.json",
        "storage_note": "Stored locally on this computer in a private app settings file. This is not OS keychain storage.",
    }


class ExecutionEngineTests(unittest.TestCase):
    def test_paper_order_lifecycle_creates_position_and_closed_trade(self):
        old_state_path = ee._state_path
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ee._state_path = lambda: os.path.join(tmp, "paper_state.json")
                ee.reset_paper_state(100000)
                advisory = {
                    "quality_meta": {"score": 0.74, "no_trade": False, "regime_label": "Bullish"},
                    "plan_meta": {"bias": "Bullish", "action": "Long setup", "entry": 100.0, "stop": 99.0, "target1": 102.0},
                    "liquidity_meta": {"tradability_score": 0.82},
                    "pred_state": {"intraday": {"position_size": 0.65, "p_up": 0.64}},
                }
                rec = ee.recommend_paper_order("QQQ", 100.0, advisory=advisory, profile_name="balanced", side="buy")
                self.assertTrue(rec["allowed"])
                self.assertGreater(rec["qty"], 0)

                order = ee.stage_paper_order("QQQ", "buy", rec["qty"], current_price=100.0, advisory=advisory, profile_name="balanced")
                self.assertEqual(order["status"], "staged")

                submit = ee.submit_paper_order(order["order_id"], fill_price=100.0)
                self.assertEqual(submit["status"], "filled")
                state = ee._load_state()
                self.assertIn("QQQ", state["positions"])
                self.assertGreater(state["positions"]["QQQ"]["qty"], 0)

                flat = ee.flatten_paper_position("QQQ", current_price=102.0, profile_name="balanced")
                self.assertEqual(flat["status"], "filled")
                state = ee._load_state()
                self.assertNotIn("QQQ", state["positions"])
                self.assertEqual(len(state["closed_trades"]), 1)
                self.assertGreater(state["closed_trades"][0]["pnl_dollars"], 0.0)
            finally:
                ee._state_path = old_state_path

    def test_recommendation_blocks_no_trade_or_weak_liquidity(self):
        advisory = {
            "quality_meta": {"score": 0.70, "no_trade": True},
            "plan_meta": {"bias": "Bullish", "entry": 100.0, "stop": 99.0},
            "liquidity_meta": {"tradability_score": 0.40},
            "pred_state": {"intraday": {"position_size": 0.70}},
        }
        rec = ee.recommend_paper_order("QQQ", 100.0, advisory=advisory, profile_name="balanced", side="buy")
        self.assertFalse(rec["allowed"])
        self.assertIn("no-trade", rec["reason"])

    def test_live_broker_snapshot_reports_missing_config_cleanly(self):
        old_avail = ee._ALPACA_TRADING_AVAILABLE
        old_settings = ee.get_runtime_broker_settings
        try:
            ee._ALPACA_TRADING_AVAILABLE = True
            ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(api_key="", secret_key="")
            snap = ee.get_live_broker_snapshot("QQQ")
            self.assertEqual(snap["status"], "credentials_missing")
            self.assertFalse(snap["connected"])
        finally:
            ee._ALPACA_TRADING_AVAILABLE = old_avail
            ee.get_runtime_broker_settings = old_settings

    def test_live_broker_snapshot_reports_connected_when_client_works(self):
        class _FakeAccount:
            cash = "50000"
            buying_power = "100000"
            equity = "51000"
            status = "ACTIVE"

        class _FakePosition:
            symbol = "QQQ"
            side = "long"
            qty = "5"
            avg_entry_price = "100.0"

        class _FakeClient:
            def get_account(self):
                return _FakeAccount()

            def get_orders(self, filter=None):
                return []

            def get_all_positions(self):
                return [_FakePosition()]

        old_avail = ee._ALPACA_TRADING_AVAILABLE
        old_settings = ee.get_runtime_broker_settings
        old_client = ee._alpaca_client
        try:
            ee._ALPACA_TRADING_AVAILABLE = True
            ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(api_key="key", secret_key="secret", broker_mode="alpaca_paper")
            ee._alpaca_client = lambda: _FakeClient()
            snap = ee.get_live_broker_snapshot("QQQ")
            self.assertEqual(snap["status"], "connected")
            self.assertTrue(snap["connected"])
            self.assertTrue(any(row[0] == "Buying Power" for row in snap["rows"]))
        finally:
            ee._ALPACA_TRADING_AVAILABLE = old_avail
            ee.get_runtime_broker_settings = old_settings
            ee._alpaca_client = old_client

    def test_broker_runtime_status_falls_back_to_local_paper_cleanly(self):
        old_avail = ee._ALPACA_TRADING_AVAILABLE
        old_settings = ee.get_runtime_broker_settings
        try:
            ee._ALPACA_TRADING_AVAILABLE = True
            ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(api_key="", secret_key="", broker_mode="alpaca_paper")
            runtime = ee.get_broker_runtime_status("QQQ")
            self.assertEqual(runtime["requested_mode"], "alpaca_paper")
            self.assertEqual(runtime["effective_mode"], "local_paper")
            self.assertEqual(runtime["status"], "fallback_local_paper")
            self.assertTrue(runtime["can_place_orders"])
        finally:
            ee._ALPACA_TRADING_AVAILABLE = old_avail
            ee.get_runtime_broker_settings = old_settings

    def test_execution_dashboard_snapshot_exposes_route_labels(self):
        old_settings = ee.get_runtime_broker_settings
        try:
            ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(broker_mode="local_paper")
            snap = ee.get_execution_dashboard_snapshot("QQQ", current_price=100.0, advisory=None, profile_name="balanced")
            self.assertEqual(snap["runtime"]["effective_mode"], "local_paper")
            self.assertIn("route_label", snap)
            self.assertTrue(any(row[0] == "Route" for row in snap["rows"]))
        finally:
            ee.get_runtime_broker_settings = old_settings

    def test_broker_diagnostics_snapshot_reports_requested_effective_and_capabilities(self):
        old_avail = ee._ALPACA_TRADING_AVAILABLE
        old_settings = ee.get_runtime_broker_settings
        try:
            ee._ALPACA_TRADING_AVAILABLE = False
            ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(api_key="", secret_key="", broker_mode="alpaca_paper")
            snap = ee.get_broker_diagnostics_snapshot("QQQ")
            self.assertEqual(snap["requested_mode"], "alpaca_paper")
            self.assertEqual(snap["effective_mode"], "local_paper")
            self.assertFalse(snap["connected"])
            self.assertFalse(snap["verification_ready"])
            self.assertIn("alpaca-py", snap["operator_hint"])
            self.assertEqual(snap["credentials_source"], "missing")
            self.assertIn("rows", snap)
            self.assertTrue(any(row[0] == "SDK / Creds" for row in snap["rows"]))
            self.assertTrue(any(row[0] == "Verify Ready" for row in snap["rows"]))
        finally:
            ee._ALPACA_TRADING_AVAILABLE = old_avail
            ee.get_runtime_broker_settings = old_settings

    def test_broker_diagnostics_snapshot_flags_paper_mode_mismatch(self):
        class _FakeAccount:
            cash = "50000"
            buying_power = "100000"
            equity = "51000"
            status = "ACTIVE"

        class _FakeClient:
            def get_account(self):
                return _FakeAccount()

            def get_orders(self, filter=None):
                return []

            def get_all_positions(self):
                return []

        old_avail = ee._ALPACA_TRADING_AVAILABLE
        old_settings = ee.get_runtime_broker_settings
        old_client = ee._alpaca_client
        try:
            ee._ALPACA_TRADING_AVAILABLE = True
            ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(
                api_key="key",
                secret_key="secret",
                paper=False,
                broker_mode="alpaca_paper",
            )
            ee._alpaca_client = lambda: _FakeClient()
            snap = ee.get_broker_diagnostics_snapshot("QQQ")
            self.assertEqual(snap["config_state"], "paper_flag_off")
            self.assertIn("Paper Mode", snap["operator_hint"])
        finally:
            ee._ALPACA_TRADING_AVAILABLE = old_avail
            ee.get_runtime_broker_settings = old_settings
            ee._alpaca_client = old_client

    def test_run_broker_verification_blocks_cleanly_when_alpaca_paper_is_not_connected(self):
        old_ver_path = ee._broker_verification_path
        old_settings = ee.get_runtime_broker_settings
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(api_key="", secret_key="", broker_mode="alpaca_paper")
                ee._broker_verification_path = lambda: os.path.join(tmp, "verify.json")
                result = ee.run_broker_verification("QQQ", current_price=100.0, advisory=None, profile_name="balanced")
                self.assertEqual(result["status"], "blocked")
                self.assertTrue(result["detail"])
                saved = ee.get_broker_verification_snapshot("QQQ")
                self.assertEqual(saved["status"], "blocked")
            finally:
                ee.get_runtime_broker_settings = old_settings
                ee._broker_verification_path = old_ver_path

    def test_run_broker_verification_round_trip_on_connected_alpaca_paper(self):
        old_diag = ee.refresh_broker_connection
        old_verify_path = ee._broker_verification_path
        old_live_snapshot = ee.get_live_broker_snapshot
        old_place = ee.place_live_order
        old_cancel = ee.cancel_live_order
        old_settings = ee.get_runtime_broker_settings
        with tempfile.TemporaryDirectory() as tmp:
            try:
                ee.get_runtime_broker_settings = lambda override=None: _runtime_settings(api_key="key", secret_key="secret", broker_mode="alpaca_paper")
                ee._broker_verification_path = lambda: os.path.join(tmp, "verify.json")
                steps = {"phase": "before"}

                class _Pos:
                    symbol = "QQQ"
                    side = "long"
                    qty = "1"
                    avg_entry_price = "100.0"

                def _diag(symbol=None):
                    return {
                        "requested_mode": "alpaca_paper",
                        "effective_mode": "alpaca_paper",
                        "route_label": "Alpaca Paper",
                        "checked_at": "2026-03-24T12:00:00+00:00",
                        "connected": True,
                        "detail": "Connected",
                        "runtime": {
                            "routing_mode": "alpaca_paper",
                            "status": "connected",
                            "capability_summary": "connect Y | place Y | cancel Y | read Y",
                        },
                    }

                def _snap(symbol=None):
                    if steps["phase"] == "before":
                        return {"account": {"equity": 100000.0}, "positions": [], "orders": [], "detail": "Connected"}
                    if steps["phase"] == "submitted":
                        return {"account": {"equity": 100000.0}, "positions": [_Pos()], "orders": [], "detail": "Connected"}
                    return {"account": {"equity": 100000.0}, "positions": [], "orders": [], "detail": "Connected"}

                def _place(symbol, side, qty=None, current_price=None, advisory=None, profile_name=None):
                    if str(side).lower() == "buy":
                        steps["phase"] = "submitted"
                        return {"status": "submitted", "order_id": "ord-1", "qty": qty or 1}
                    steps["phase"] = "after_cleanup"
                    return {"status": "submitted", "order_id": "ord-2", "qty": qty or 1}

                ee.refresh_broker_connection = _diag
                ee.get_live_broker_snapshot = _snap
                ee.place_live_order = _place
                ee.cancel_live_order = lambda order_id: {"status": "canceled", "order_id": order_id}
                result = ee.run_broker_verification("QQQ", current_price=100.0, advisory=None, profile_name="balanced")
                self.assertEqual(result["status"], "verified")
                self.assertEqual(result["cleanup"]["status"], "submitted")
                saved = ee.get_broker_verification_snapshot("QQQ")
                self.assertEqual(saved["status"], "verified")
            finally:
                ee.get_runtime_broker_settings = old_settings
                ee.refresh_broker_connection = old_diag
                ee._broker_verification_path = old_verify_path
                ee.get_live_broker_snapshot = old_live_snapshot
                ee.place_live_order = old_place
                ee.cancel_live_order = old_cancel


if __name__ == "__main__":
    unittest.main()
