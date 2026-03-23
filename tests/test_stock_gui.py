import unittest

from stock_gui import _dashboard_prediction_signature


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


if __name__ == "__main__":
    unittest.main()
