import unittest

from optimize_readiness import _modes_from_checks


class OptimizeReadinessTests(unittest.TestCase):
    def test_selects_specific_failed_horizons(self):
        checks = {
            "weekend_window_ok": False,
            "daily_window_ok": True,
            "intraday_window_ok": False,
            "horizon_score_ok": True,
            "folds_ok": True,
        }
        self.assertEqual(_modes_from_checks(checks), ["4min", "weekend"])

    def test_daily_failure_trains_both_daily_modes(self):
        checks = {
            "weekend_window_ok": True,
            "daily_window_ok": False,
            "intraday_window_ok": True,
            "horizon_score_ok": True,
            "folds_ok": True,
        }
        self.assertEqual(_modes_from_checks(checks), ["daily_high", "daily_low"])

    def test_low_score_without_specific_failures_trains_all_modes(self):
        checks = {
            "weekend_window_ok": True,
            "daily_window_ok": True,
            "intraday_window_ok": True,
            "horizon_score_ok": False,
            "folds_ok": True,
        }
        self.assertEqual(
            _modes_from_checks(checks),
            ["4min", "daily_high", "daily_low", "weekend"],
        )

    def test_profitability_failure_retrains_intraday(self):
        checks = {
            "weekend_window_ok": True,
            "daily_window_ok": True,
            "intraday_window_ok": True,
            "net_ok": False,
            "sharpe_ok": True,
            "active_ok": True,
            "horizon_score_ok": True,
            "folds_ok": True,
        }
        self.assertEqual(_modes_from_checks(checks), ["4min"])


if __name__ == "__main__":
    unittest.main()
