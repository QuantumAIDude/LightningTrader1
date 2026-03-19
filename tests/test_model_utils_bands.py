import unittest
from unittest import mock

import pandas as pd

from model_utils import (
    _apply_horizon_overlay_block,
    _effective_model_count,
    _posterior_shrink_return,
    _prediction_band,
    _resolve_predictability_gate,
)


class PredictionBandTests(unittest.TestCase):
    def test_prediction_band_outputs_range_and_bps(self):
        lo, hi, err_bps = _prediction_band(100.0, 0.01, 0.002, instr_meta={"tick_size": 0.01})
        self.assertIsNotNone(lo)
        self.assertIsNotNone(hi)
        self.assertIsNotNone(err_bps)
        self.assertLess(lo, hi)
        self.assertAlmostEqual(float(err_bps), 20.0, places=6)

    def test_prediction_band_handles_bad_inputs(self):
        lo, hi, err_bps = _prediction_band("bad", 0.01, 0.002, instr_meta={})
        self.assertIsNone(lo)
        self.assertIsNone(hi)
        self.assertIsNone(err_bps)

    def test_effective_model_count_matches_weight_concentration(self):
        self.assertAlmostEqual(_effective_model_count([1.0, 1.0, 1.0]), 3.0, places=6)
        self.assertAlmostEqual(_effective_model_count([3.0, 0.0, 0.0]), 1.0, places=6)

    def test_posterior_shrink_return_reduces_weak_signal(self):
        shrunk, shrink = _posterior_shrink_return(0.001, 0.004)
        self.assertGreater(shrink, 0.0)
        self.assertLess(shrink, 1.0)
        self.assertLess(abs(shrunk), 0.001)

    def test_resolve_predictability_gate_prefers_valid_scale(self):
        payloads = {
            "base": {"meta": {"predictability_gate": {"scale": 1.25, "quantile": 0.7}}},
            "forest": {"meta": {"predictability_gate": {"scale": None}}},
        }
        gate = _resolve_predictability_gate(payloads)
        self.assertIsInstance(gate, dict)
        self.assertAlmostEqual(float(gate["scale"]), 1.25, places=6)

    def test_apply_horizon_overlay_block_copies_weekly_high_low(self):
        hist = pd.DataFrame({"close": [100.0]}, index=pd.DatetimeIndex(["2026-03-10T20:58:00Z"]))
        block = {"base": {"pred_price": 101.0, "note": ""}}
        with mock.patch(
            "model_utils.apply_horizon_overlay",
            return_value={
                "pred_price": 101.5,
                "pred_high_price": 103.25,
                "pred_low_price": 98.75,
                "blend_weight": 0.4,
                "features": [],
            },
        ):
            out = _apply_horizon_overlay_block(hist, "QQQ", "weekend", 100.0, block)
        self.assertAlmostEqual(float(out["base"]["pred_price"]), 101.5, places=6)
        self.assertAlmostEqual(float(out["base"]["pred_high_price"]), 103.25, places=6)
        self.assertAlmostEqual(float(out["base"]["pred_low_price"]), 98.75, places=6)


if __name__ == "__main__":
    unittest.main()
