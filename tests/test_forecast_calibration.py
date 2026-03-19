import unittest

import numpy as np

from forecast_calibration import apply_return_calibration, fit_return_calibration, score_return_forecast


class ForecastCalibrationTests(unittest.TestCase):
    def test_fit_return_calibration_shrinks_oversized_signal(self):
        close = np.full(80, 100.0)
        actual = np.linspace(-0.002, 0.002, 80)
        pred = actual * 2.0

        raw = score_return_forecast(pred, actual, close, horizon="intraday")
        calib = fit_return_calibration(pred, actual, close, horizon="intraday", min_samples=20)
        adj = score_return_forecast(apply_return_calibration(pred, calib), actual, close, horizon="intraday")

        self.assertLess(calib["scale"], 1.0)
        self.assertLessEqual(adj["objective_score"], raw["objective_score"])

    def test_fit_return_calibration_uses_anchor_when_sample_is_too_small(self):
        calib = fit_return_calibration(
            [0.001, -0.001, 0.002, -0.002],
            [0.0, 0.0, 0.0, 0.0],
            [100.0, 100.0, 100.0, 100.0],
            horizon="intraday",
            min_samples=10,
        )
        self.assertEqual(calib["selected"], "anchor_only")
        self.assertAlmostEqual(float(calib["scale"]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
