import unittest

import symbol_lifecycle as sl


class SymbolLifecycleTests(unittest.TestCase):
    def test_supported_symbol_without_data_stays_honestly_immature(self):
        old_instruments = dict(sl.INSTRUMENTS)
        try:
            sl.INSTRUMENTS.clear()
            sl.INSTRUMENTS.update({"AAPL": {"type": "equity", "sector": "Technology"}})
            snapshot = sl.build_symbol_lifecycle_snapshot(
                "AAPL",
                runtime_bundle={},
                onboarding_summary={"status": "not_started", "status_label": "Not started", "summary": "No staged onboarding job has run for this symbol yet.", "artifacts": {}, "progress_completed": 0, "progress_total": 9},
                deployment_summary={
                    "stage_label": "Supported / not trained",
                    "rollout_label": "Shadow only rollout",
                    "operator_state_label": "Blocked: insufficient symbol maturity",
                    "operator_state_detail": "The platform supports this symbol, but symbol-specific market data has not been staged yet.",
                    "evidence_state_label": "Missing evidence",
                    "evidence_state_detail": "This symbol is still missing some combination of staged data, trained models, or readiness validation evidence.",
                    "symbol_maturity_label": "Supported only",
                    "symbol_maturity_detail": "The platform can host this symbol, but symbol-specific data and evidence have not been staged yet.",
                    "primary_next_requirement": "Stage canonical price history for this symbol and rerun readiness refresh.",
                    "next_requirement_category": "symbol_maturity",
                    "artifact_summary": {"prices": {"rows": 0}, "news": {"rows": 0}},
                    "calibration_summary": {},
                    "status_flags": {"supported": True, "onboarded": False, "trained": False},
                },
                news_health={"status": "missing", "headline_state": "missing", "history_state": "missing"},
                model_summary={"available_modes": [], "core_ready": False},
            )
            self.assertEqual(snapshot["lifecycle_label"], "Profile resolved")
            self.assertEqual(snapshot["data_state_label"], "No canonical price history")
            self.assertEqual(snapshot["operator_state_label"], "Blocked: insufficient symbol maturity")
            self.assertEqual(snapshot["news_state_label"], "No news context")
            rows = sl.lifecycle_rows(snapshot)
            self.assertTrue(any(row[0] == "Next Requirement" for row in rows))
        finally:
            sl.INSTRUMENTS.clear()
            sl.INSTRUMENTS.update(old_instruments)

    def test_partial_symbol_snapshot_surfaces_partial_progress_cleanly(self):
        old_instruments = dict(sl.INSTRUMENTS)
        try:
            sl.INSTRUMENTS.clear()
            sl.INSTRUMENTS.update({"MSFT": {"type": "equity", "sector": "Technology"}})
            snapshot = sl.build_symbol_lifecycle_snapshot(
                "MSFT",
                runtime_bundle={},
                onboarding_summary={
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "Feature readiness built",
                    "artifacts": {
                        "market_data": {"rows": 640},
                        "feature_readiness": {"intraday_rows": 180, "daily_rows": 0, "weekend_rows": 0},
                    },
                    "progress_completed": 4,
                    "progress_total": 9,
                },
                deployment_summary={
                    "stage_label": "Supported / not trained",
                    "rollout_label": "Shadow only rollout",
                    "operator_state_label": "Blocked: missing trained models",
                    "operator_state_detail": "The workspace exists, but the core trained model set is still missing or incomplete for this symbol.",
                    "evidence_state_label": "Missing evidence",
                    "evidence_state_detail": "This symbol is still missing some combination of staged data, trained models, or readiness validation evidence.",
                    "symbol_maturity_label": "Onboarded",
                    "symbol_maturity_detail": "Canonical market data has been staged, but symbol-specific model and readiness maturity is still incomplete.",
                    "primary_next_requirement": "Train the default core 4-minute model stack for this symbol.",
                    "next_requirement_category": "symbol_maturity",
                    "artifact_summary": {"prices": {"rows": 640}, "news": {"rows": 12, "headline_state": "ok", "history_state": "insufficient_history"}},
                    "calibration_summary": {},
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                },
                news_health={"status": "ok", "headline_state": "ok", "history_state": "insufficient_history", "rows": 12},
                model_summary={"available_modes": [], "core_ready": False},
            )
            self.assertEqual(snapshot["feature_state_label"], "Intraday only")
            self.assertEqual(snapshot["lifecycle_label"], "Features partially ready")
            self.assertEqual(snapshot["model_state_label"], "No trained models")
            self.assertEqual(snapshot["operator_state_label"], "Blocked: missing trained models")
        finally:
            sl.INSTRUMENTS.clear()
            sl.INSTRUMENTS.update(old_instruments)

    def test_history_assessment_drives_data_state_for_partial_symbols(self):
        old_instruments = dict(sl.INSTRUMENTS)
        try:
            sl.INSTRUMENTS.clear()
            sl.INSTRUMENTS.update({"NVDA": {"type": "equity", "sector": "Technology"}})
            snapshot = sl.build_symbol_lifecycle_snapshot(
                "NVDA",
                runtime_bundle={},
                onboarding_summary={
                    "status": "completed",
                    "status_label": "Completed",
                    "summary": "Onboarding complete with partial maturity",
                    "artifacts": {
                        "history_assessment": {
                            "status_key": "advisory_core_eligible",
                            "status_label": "Core advisory path eligible",
                            "detail": "The 4-minute path has enough history for training, while longer horizons remain partial or deferred.",
                            "price_rows": 1200,
                            "feature_ready_modes": ["4min", "DailyHigh"],
                            "eligible_modes": ["4min"],
                            "primary_next_requirement": "Extend session-level history until the daily range path reaches 15 usable rows.",
                            "paths": {
                                "4min": {"feature_rows": 180, "feature_ready": True, "train_eligible": True},
                                "DailyHigh": {"feature_rows": 12, "feature_ready": True, "train_eligible": False},
                                "WeekEnd": {"feature_rows": 0, "feature_ready": False, "train_eligible": False},
                            },
                        },
                        "feature_readiness": {
                            "status": "partial",
                            "summary": "Feature coverage exists for 4min, DailyHigh; training-eligible modes: 4min.",
                            "feature_ready_modes": ["4min", "DailyHigh"],
                            "eligible_modes": ["4min"],
                            "intraday_rows": 180,
                            "daily_rows": 12,
                            "weekend_rows": 0,
                        },
                    },
                    "progress_completed": 5,
                    "progress_total": 9,
                    "primary_next_requirement": "Extend session-level history until the daily range path reaches 15 usable rows.",
                },
                deployment_summary={
                    "stage_label": "Supported / not trained",
                    "rollout_label": "Shadow only rollout",
                    "operator_state_label": "Blocked: missing trained models",
                    "evidence_state_label": "Missing evidence",
                    "symbol_maturity_label": "Onboarded, not trained",
                    "primary_next_requirement": "",
                    "artifact_summary": {"prices": {"rows": 1200}, "news": {"rows": 10, "headline_state": "ok", "history_state": "thin"}},
                    "calibration_summary": {},
                    "status_flags": {"supported": True, "onboarded": True, "trained": False},
                },
                news_health={"status": "ok", "headline_state": "ok", "history_state": "thin", "rows": 10},
                model_summary={"available_modes": [], "core_ready": False},
            )
            self.assertEqual(snapshot["data_state_label"], "History staged")
            self.assertEqual(snapshot["feature_state_label"], "Partial horizon coverage")
            self.assertIn("daily range path reaches 15 usable rows", snapshot["workspace_summary"])
        finally:
            sl.INSTRUMENTS.clear()
            sl.INSTRUMENTS.update(old_instruments)


if __name__ == "__main__":
    unittest.main()
