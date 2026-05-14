import unittest

import model_zoo as mz


class ModelZooTests(unittest.TestCase):
    def test_summarize_model_stack_reports_core_and_horizon_modes(self):
        models = {
            "base": {"ensemble_models": {"recent": [object()]}},
            "forest": {"ensemble_models": {"recent": [object()]}},
            "boost": {"ensemble_models": {"recent": [object()]}},
            "extra_trees": {"ensemble_models": {"recent": [object()]}},
            "meta": {"edge": 0.1},
            "regime": object(),
            "modes": {
                "daily_high": {"base": {"ensemble_models": {"recent": [object()]}}},
                "daily_low": {"forest": {"ensemble_models": {"recent": [object()]}}},
                "weekend": {"boost": {"ensemble_models": {"recent": [object()]}}},
            },
        }

        summary = mz.summarize_model_stack(models)

        self.assertEqual(summary["family"], mz.DEFAULT_MODEL_FAMILY)
        self.assertTrue(summary["core_ready"])
        self.assertIn("4min", summary["available_modes"])
        self.assertIn("DailyHigh", summary["available_modes"])
        self.assertIn("WeekEnd", summary["available_modes"])
        self.assertTrue(summary["tradeability_head_present"])
        self.assertTrue(summary["regime_present"])

    def test_predict_mode_normalizes_aliases_and_uses_underlying_predictors(self):
        calls = []
        old_predict_4min = mz.predict_4min
        old_predict_horizon = mz.predict_horizon
        try:
            mz.predict_4min = lambda models, df, instrument_key=None, **kwargs: {"kind": "4min", "instrument": instrument_key, "models": models}
            mz.predict_horizon = lambda models, df, instrument_key=None, mode=None, **kwargs: {"kind": mode, "instrument": instrument_key, "models": models}

            intraday = mz.predict_mode({"models": {"base": {}}}, object(), instrument_key="qqq", mode="intraday")
            weekend = mz.predict_mode({"base": {}}, object(), instrument_key="aapl", mode="week_end")

            self.assertEqual(intraday["kind"], "4min")
            self.assertEqual(intraday["instrument"], "QQQ")
            self.assertIn("base", intraday["models"])
            self.assertEqual(weekend["kind"], "WeekEnd")
            self.assertEqual(weekend["instrument"], "AAPL")
            self.assertIn("base", weekend["models"])
        finally:
            mz.predict_4min = old_predict_4min
            mz.predict_horizon = old_predict_horizon

    def test_mode_specs_and_inference_payload_are_consistent(self):
        old_predict_4min = mz.predict_4min
        try:
            mz.predict_4min = lambda models, df, instrument_key=None, **kwargs: {"consensus": {"direction": "UP", "pred_price": 101.0}}
            payload = mz.build_inference_payload(
                {"models": {"base": {"ensemble_models": {"recent": [object()]}}}},
                object(),
                instrument_key="qqq",
                mode="intraday",
            )

            self.assertEqual(mz.get_mode_spec("week_end")["mode"], "WeekEnd")
            self.assertEqual(payload["instrument"], "QQQ")
            self.assertEqual(payload["mode"], "4min")
            self.assertEqual(payload["prediction_key"], "intraday")
            self.assertIn("prediction", payload)
        finally:
            mz.predict_4min = old_predict_4min

    def test_train_baseline_models_returns_runtime_summary(self):
        old_train_all_models = mz.train_all_models
        old_load_model_stack = mz.load_model_stack
        try:
            mz.train_all_models = lambda df, instrument_key=None, train_horizons=True, modes_filter=None, progress_cb=None, max_runtime_sec=None, persist=True: {"base": {}, "requested": list(modes_filter or [])}
            mz.load_model_stack = lambda instrument_key=None, include_modes=True: {
                "base": {"ensemble_models": {"recent": [object()]}},
                "forest": {"ensemble_models": {"recent": [object()]}},
                "boost": {"ensemble_models": {"recent": [object()]}},
                "extra_trees": {"ensemble_models": {"recent": [object()]}},
                "meta": {"edge": 0.1},
                "regime": object(),
            }

            runtime = mz.train_baseline_models(object(), "qqq", train_horizons=True, modes_filter=["4min"])

            self.assertEqual(runtime["instrument"], "QQQ")
            self.assertEqual(runtime["family"], mz.DEFAULT_MODEL_FAMILY)
            self.assertTrue(runtime["summary"]["core_ready"])
            self.assertEqual(runtime["requested_modes"], ["4min"])
            self.assertEqual(len(mz.list_mode_specs(runtime["models"])), 3)
        finally:
            mz.train_all_models = old_train_all_models
            mz.load_model_stack = old_load_model_stack


if __name__ == "__main__":
    unittest.main()
