import tempfile
import unittest

import workspace_state as ws


class WorkspaceStateTests(unittest.TestCase):
    def test_workspace_state_persists_view_and_active_symbol(self):
        old_data_dir = ws.DATA_DIR
        old_instruments = dict(ws.INSTRUMENTS)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ws.DATA_DIR = tmp
                ws.INSTRUMENTS.clear()
                ws.INSTRUMENTS.update(
                    {
                        "QQQ": {"display_name": "QQQ", "type": "etf", "sector": "Technology"},
                        "AAPL": {
                            "display_name": "Apple",
                            "company_name": "Apple Inc",
                            "news_aliases": ["AAPL", "Apple", "Apple Inc"],
                            "sector": "Technology",
                            "type": "equity",
                        },
                    }
                )
                ws.open_workspace("QQQ", activate=True, pinned=True)
                ws.open_workspace("AAPL", activate=True)
                ws.remember_workspace_view("AAPL", mode="DailyHigh", profile_name="swing", note="Ready", status="ready")

                active = ws.get_workspace("AAPL")
                self.assertEqual(active["view_mode"], "DailyHigh")
                self.assertEqual(active["profile_name"], "swing")
                self.assertEqual(ws.list_workspaces()[0]["symbol"], "AAPL")

                state = ws.close_workspace("AAPL")
                self.assertEqual(state["active_symbol"], "QQQ")
                self.assertFalse(ws.get_workspace("AAPL")["opened"])
        finally:
            ws.DATA_DIR = old_data_dir
            ws.INSTRUMENTS.clear()
            ws.INSTRUMENTS.update(old_instruments)
