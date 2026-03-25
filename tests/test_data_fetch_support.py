import unittest

import data_fetch as dfetch


class DataFetchSupportRefreshTests(unittest.TestCase):
    def test_refresh_support_data_collects_feed_statuses(self):
        old_news = dfetch.maybe_refresh_news
        old_context = dfetch.maybe_refresh_context_data
        old_macro = dfetch.maybe_refresh_macro_events
        old_event = dfetch.get_event_snapshot
        old_options = dfetch.get_options_snapshot
        old_breadth = dfetch.get_breadth_snapshot
        old_catalyst = dfetch.get_catalyst_provider_snapshot
        old_instruments = dict(dfetch.INSTRUMENTS)
        try:
            dfetch.INSTRUMENTS.clear()
            dfetch.INSTRUMENTS.update({"QQQ": {"news_enabled": True}})
            dfetch.maybe_refresh_news = lambda symbol="QQQ": {"status": "ok", "rows": 12}
            dfetch.maybe_refresh_context_data = lambda symbol="QQQ": None
            dfetch.maybe_refresh_macro_events = lambda force=False: {"status": "live", "provider_note": "Macro cache warm"}
            dfetch.get_event_snapshot = lambda symbol, max_age_sec=None: {"status": "ok", "source": "cache"}
            dfetch.get_options_snapshot = lambda symbol, spot_price=None, max_age_sec=None: {"status": "ok", "source": "cache"}
            dfetch.get_breadth_snapshot = lambda symbol: {"status": "live_partial", "provider_note": "Proxy breadth live"}
            dfetch.get_catalyst_provider_snapshot = lambda symbol: {"status": "partial", "active_catalyst": {"title": "Fresh filing"}}

            out = dfetch.refresh_support_data("QQQ", current_price=500.0)

            self.assertEqual(out["symbol"], "QQQ")
            self.assertEqual(out["news"]["status"], "ok")
            self.assertEqual(out["context"]["status"], "ok")
            self.assertEqual(out["macro"]["status"], "live")
            self.assertEqual(out["event"]["status"], "ok")
            self.assertEqual(out["options"]["status"], "ok")
            self.assertEqual(out["breadth"]["status"], "live_partial")
            self.assertEqual(out["catalyst"]["status"], "partial")
        finally:
            dfetch.maybe_refresh_news = old_news
            dfetch.maybe_refresh_context_data = old_context
            dfetch.maybe_refresh_macro_events = old_macro
            dfetch.get_event_snapshot = old_event
            dfetch.get_options_snapshot = old_options
            dfetch.get_breadth_snapshot = old_breadth
            dfetch.get_catalyst_provider_snapshot = old_catalyst
            dfetch.INSTRUMENTS.clear()
            dfetch.INSTRUMENTS.update(old_instruments)


if __name__ == "__main__":
    unittest.main()
