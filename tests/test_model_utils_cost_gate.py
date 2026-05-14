import unittest

import pandas as pd

import model_utils as mu


class ModelUtilsCostGateTests(unittest.TestCase):
    def test_live_edge_floor_inputs_use_configured_cost_and_min_edge(self):
        old_min_edge = mu.LIVE_MIN_EDGE_BPS
        old_unc = mu.LIVE_UNCERTAINTY_EDGE_MULT
        try:
            mu.LIVE_MIN_EDGE_BPS = 1.5
            mu.LIVE_UNCERTAINTY_EDGE_MULT = 0.2
            df = pd.DataFrame({"bid": [99.99], "ask": [100.01]})
            out = mu._live_edge_floor_inputs(100.0, df, instr_meta={"tick_size": 0.01})
            self.assertAlmostEqual(out["min_edge_floor"], 0.00015, places=8)
            self.assertGreater(out["cost_ret_est"], 0.0)
            self.assertEqual(out["uncertainty_mult"], 0.2)
            self.assertGreater(out["spread_frac"], 0.0)
        finally:
            mu.LIVE_MIN_EDGE_BPS = old_min_edge
            mu.LIVE_UNCERTAINTY_EDGE_MULT = old_unc

    def test_posterior_shrink_return_does_not_quadratically_collapse_small_signal(self):
        shrunk_ret, shrink = mu._posterior_shrink_return(0.001, 0.010)
        self.assertAlmostEqual(shrink, 1.0 / 11.0, places=6)
        self.assertAlmostEqual(shrunk_ret, 0.001 / 11.0, places=6)


if __name__ == "__main__":
    unittest.main()
