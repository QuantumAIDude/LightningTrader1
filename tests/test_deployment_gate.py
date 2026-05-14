import contextlib
import io
import json
import sys
import unittest
from unittest.mock import patch

import deployment_gate as dg


class DeploymentGateTests(unittest.TestCase):
    def test_report_carries_validation_evidence_from_readiness_state(self):
        readiness = {
            "fresh": True,
            "live_ready": False,
            "stage_key": "advisory_only",
            "stage_label": "Advisory only",
            "validation_evidence": {
                "validation_status": "passed_advisory_only",
                "validation_method": "walk_forward",
                "summary": "Validation supports advisory only",
            },
            "validation_status": "passed_advisory_only",
            "validation_method": "walk_forward",
            "validation_blockers": ["no_paper_trading_record"],
            "evaluation": {},
        }
        stdout = io.StringIO()
        argv = ["deployment_gate.py", "--instrument", "QQQ", "--quick", "--allow-warn"]
        with patch.dict(dg.INSTRUMENTS, {"QQQ": {"type": "equity"}}, clear=True):
            with patch.object(dg, "_model_dir_size_bytes", lambda: 0):
                with patch.object(dg, "update_live_calibration", lambda symbol, horizon_minutes=4: {"status": "updated"}):
                    with patch.object(dg, "update_horizon_accuracy", lambda symbol: {"status": "updated"}):
                        with patch.object(dg, "update_readiness_state", lambda symbol, online=False: readiness):
                            with patch.object(dg, "summarize_deployment_readiness", lambda symbol, state=None: {"stage_label": "Advisory only"}):
                                with patch.object(
                                    dg,
                                    "_check_instrument",
                                    lambda symbol, quick=True, online=False: {
                                        "status": "ok",
                                        "models": {"missing": []},
                                        "predictions": {"4min": {}, "4min_latency_s": 0.01},
                                    },
                                ):
                                    with patch.object(sys, "argv", argv):
                                        with contextlib.redirect_stdout(stdout):
                                            with self.assertRaises(SystemExit) as cm:
                                                dg.main()

        self.assertEqual(cm.exception.code, 0)
        report = json.loads(stdout.getvalue())
        evidence = report["instruments"]["QQQ"]["deployment_evidence"]
        self.assertEqual(evidence["validation_status"], "passed_advisory_only")
        self.assertEqual(evidence["validation_method"], "walk_forward")
        self.assertEqual(evidence["validation_blockers"], ["no_paper_trading_record"])


if __name__ == "__main__":
    unittest.main()
