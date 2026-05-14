import unittest

from meta_label_builder import (
    base_model_direction_correctness,
    best_base_model,
    build_meta_labels,
    meta_target_action_class,
)


def _record(**overrides):
    base = {
        "average_prediction": 0.004,
        "advisory_action": "buy",
        "economic_edge": {"state_key": "finished_edge"},
        "regime_label": "bullish",
        "base_predictions": {
            "base": {"pred_ret": 0.003},
            "forest": {"pred_ret": -0.002},
            "boost": {"pred_ret": 0.006},
        },
        "future_return_by_horizon": {"1_step": 0.005},
        "direction_correct_by_horizon": {"1_step": True},
        "net_after_costs_by_horizon": {"1_step": 0.003},
        "signal_survived_costs_by_horizon": {"1_step": True},
        "best_base_model_by_horizon": {},
    }
    base.update(overrides)
    return base


class MetaLabelBuilderTests(unittest.TestCase):
    def test_correct_directional_label_when_prediction_agrees(self):
        labels = build_meta_labels(_record(), horizon="1_step")
        self.assertTrue(labels["average_prediction_was_directionally_correct"])
        self.assertTrue(labels["advisory_action_was_directionally_correct"])

    def test_incorrect_directional_label_when_prediction_disagrees(self):
        labels = build_meta_labels(_record(future_return_by_horizon={"1_step": -0.004}), horizon="1_step")
        self.assertFalse(labels["average_prediction_was_directionally_correct"])
        self.assertFalse(labels["advisory_action_was_directionally_correct"])

    def test_signal_survived_costs_true_and_false(self):
        self.assertTrue(build_meta_labels(_record())["signal_survived_costs"])
        failed = _record(
            net_after_costs_by_horizon={"1_step": -0.001},
            signal_survived_costs_by_horizon={"1_step": False},
        )
        self.assertFalse(build_meta_labels(failed)["signal_survived_costs"])

    def test_stand_aside_was_reasonable_when_realized_edge_is_too_small(self):
        labels = build_meta_labels(
            _record(
                advisory_action="stand_aside",
                future_return_by_horizon={"1_step": 0.0004},
                net_after_costs_by_horizon={},
                signal_survived_costs_by_horizon={},
            ),
            cost_floor=0.001,
        )
        self.assertTrue(labels["stand_aside_was_reasonable"])
        self.assertEqual(labels["meta_target_action_class"], "stand_aside")

    def test_best_base_model_from_lowest_error_and_direction_map(self):
        record = _record(future_return_by_horizon={"1_step": 0.004})
        self.assertEqual(best_base_model(record), "base")
        correctness = base_model_direction_correctness(record)
        self.assertTrue(correctness["base"])
        self.assertFalse(correctness["forest"])

    def test_no_label_when_required_inputs_are_missing(self):
        record = _record(future_return_by_horizon={}, net_after_costs_by_horizon={})
        labels = build_meta_labels(record)
        self.assertIsNone(labels["average_prediction_was_directionally_correct"])
        self.assertEqual(meta_target_action_class(record), "no_label")
        self.assertEqual(labels["meta_target_action_class"], "no_label")


if __name__ == "__main__":
    unittest.main()
