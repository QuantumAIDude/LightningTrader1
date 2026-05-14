import unittest

import pandas as pd

import data_hub as dh


class DataHubTests(unittest.TestCase):
    def test_store_runtime_bundle_updates_state_and_notifies_subscribers(self):
        hub = dh.DataHub(refresh_ttl_sec=30, bundle_ttl_sec=300)
        observed = []
        token = hub.subscribe("QQQ", lambda payload: observed.append(payload))
        try:
            old_staleness = dh._staleness_info
            dh._staleness_info = lambda df, instrument_key=None: {"is_stale": False, "age_seconds": 5}
            df = pd.DataFrame(
                [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
            )

            bundle = hub.store_runtime_bundle(
                "QQQ",
                df=df,
                models={"base": {"ensemble_models": {"recent": [object()]}}},
                news_health={"status": "ok", "headline_state": "ok", "history_state": "ok"},
                news_runtime={"summary": "symbol specific", "layer_a": {"summary": "symbol specific"}, "layer_b": {"status": "disabled"}},
                support_refresh={"news": {"status": "ok"}, "options": {"status": "ok", "detail": "chain_cached"}},
                deployment_meta={"stage_label": "Advisory only", "validation_status": "passed_advisory_only", "validation_method": "walk_forward"},
                source="unit",
            )

            self.assertEqual(bundle["symbol"], "QQQ")
            self.assertEqual(hub.get_runtime_state("QQQ")["status"], "ready")
            self.assertFalse(bundle["freshness"]["price"]["is_stale"])
            self.assertEqual(bundle["support_state"]["status"], "ok")
            self.assertIn("options", bundle["support_state"]["components"])
            self.assertIn("lifecycle_meta", bundle)
            self.assertEqual(bundle["lifecycle_meta"]["symbol"], "QQQ")
            self.assertEqual(hub.get_runtime_state("QQQ")["validation_status"], "passed_advisory_only")
            self.assertEqual(hub.get_runtime_state("QQQ")["validation_method"], "walk_forward")
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["symbol"], "QQQ")
        finally:
            dh._staleness_info = old_staleness
            hub.unsubscribe("QQQ", token)

    def test_refresh_symbol_uses_cache_until_ttl_expires(self):
        hub = dh.DataHub(refresh_ttl_sec=60, bundle_ttl_sec=300)
        calls = {"price": 0, "models": 0}
        old_update_price_data = dh.update_price_data
        old_build_model_runtime = dh.build_model_runtime
        old_refresh_support_data = dh.refresh_support_data
        old_get_news_health = dh.get_news_health
        old_build_news_runtime_summary = dh.build_news_runtime_summary
        old_load_readiness_state = dh.load_readiness_state
        old_summarize_deployment_readiness = dh.summarize_deployment_readiness
        try:
            df = pd.DataFrame(
                [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
            )
            dh.update_price_data = lambda symbol: calls.__setitem__("price", calls["price"] + 1) or df.copy()
            dh.build_model_runtime = lambda symbol, include_modes=True: calls.__setitem__("models", calls["models"] + 1) or {"models": {"base": {"ensemble_models": {"recent": [object()]}}}}
            dh.refresh_support_data = lambda symbol, current_price=None, force=False: {"news": {"status": "ok"}}
            dh.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "ok"}
            dh.build_news_runtime_summary = lambda symbol, news_health=None: {"summary": "runtime ok", "layer_a": {"summary": "runtime ok"}, "layer_b": {"status": "disabled"}}
            dh.load_readiness_state = lambda symbol: {}
            dh.summarize_deployment_readiness = lambda symbol, state=None, news_health=None: {"stage_label": "Advisory only"}

            first = hub.refresh_symbol("QQQ", force=False, reason="test")
            second = hub.refresh_symbol("QQQ", force=False, reason="test")

            self.assertFalse(first["cache_hit"])
            self.assertTrue(second["cache_hit"])
            self.assertEqual(calls["price"], 1)
            self.assertEqual(calls["models"], 1)
        finally:
            dh.update_price_data = old_update_price_data
            dh.build_model_runtime = old_build_model_runtime
            dh.refresh_support_data = old_refresh_support_data
            dh.get_news_health = old_get_news_health
            dh.build_news_runtime_summary = old_build_news_runtime_summary
            dh.load_readiness_state = old_load_readiness_state
            dh.summarize_deployment_readiness = old_summarize_deployment_readiness

    def test_snapshot_symbols_prefetches_and_invalidate_clears_bundle(self):
        hub = dh.DataHub(refresh_ttl_sec=60, bundle_ttl_sec=300)
        calls = {"price": 0}
        old_update_price_data = dh.update_price_data
        old_build_model_runtime = dh.build_model_runtime
        old_refresh_support_data = dh.refresh_support_data
        old_get_news_health = dh.get_news_health
        old_build_news_runtime_summary = dh.build_news_runtime_summary
        old_load_readiness_state = dh.load_readiness_state
        old_summarize_deployment_readiness = dh.summarize_deployment_readiness
        try:
            df = pd.DataFrame(
                [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10_000}],
                index=pd.DatetimeIndex([pd.Timestamp("2026-04-10T15:56:00Z")]),
            )
            dh.update_price_data = lambda symbol: calls.__setitem__("price", calls["price"] + 1) or df.copy()
            dh.build_model_runtime = lambda symbol, include_modes=True: {"models": {"base": {"ensemble_models": {"recent": [object()]}}}}
            dh.refresh_support_data = lambda symbol, current_price=None, force=False: {"news": {"status": "ok"}, "options": {"status": "ok"}}
            dh.get_news_health = lambda symbol: {"status": "ok", "headline_state": "ok", "history_state": "ok"}
            dh.build_news_runtime_summary = lambda symbol, news_health=None: {"summary": "runtime ok", "layer_a": {"summary": "runtime ok"}, "layer_b": {"status": "disabled"}}
            dh.load_readiness_state = lambda symbol: {}
            dh.summarize_deployment_readiness = lambda symbol, state=None, news_health=None: {"stage_label": "Advisory only"}

            snapshot = hub.snapshot_symbols(["QQQ"], allow_stale=False, prefetch_missing=True, reason="unit_prefetch")
            self.assertIn("QQQ", snapshot)
            self.assertEqual(calls["price"], 1)
            self.assertIn("state", snapshot["QQQ"])

            invalidated = hub.invalidate_symbol("QQQ", reason="unit_test")
            self.assertEqual(invalidated["status"], "invalidated")
            self.assertIsNone(hub.get_runtime_bundle("QQQ"))
        finally:
            dh.update_price_data = old_update_price_data
            dh.build_model_runtime = old_build_model_runtime
            dh.refresh_support_data = old_refresh_support_data
            dh.get_news_health = old_get_news_health
            dh.build_news_runtime_summary = old_build_news_runtime_summary
            dh.load_readiness_state = old_load_readiness_state
            dh.summarize_deployment_readiness = old_summarize_deployment_readiness


if __name__ == "__main__":
    unittest.main()
