import unittest

import update_checker as uc


class UpdateCheckerTests(unittest.TestCase):
    def test_normalize_and_compare_versions(self):
        self.assertEqual(uc.normalize_version("v1.2.3"), "1.2.3")
        self.assertTrue(uc.is_newer_version("1.2.4", "1.2.3"))
        self.assertFalse(uc.is_newer_version("1.2.3", "1.2.3"))

    def test_update_status_reports_available_when_live_release_is_newer(self):
        old_fetch = uc._fetch_latest_release
        old_write = uc._write_cache
        old_read = uc._read_cache
        try:
            uc._fetch_latest_release = lambda: {
                "latest_version": "9.9.9",
                "release_name": "v9.9.9",
                "release_url": "https://example.com/release",
                "published_at": "2026-03-25T00:00:00Z",
                "source_kind": "live",
                "repo": "owner/repo",
            }
            uc._write_cache = lambda payload: None
            uc._read_cache = lambda: {}
            status = uc.get_update_status(force=True)
            self.assertEqual(status["status"], "update_available")
            self.assertEqual(status["latest_version"], "9.9.9")
        finally:
            uc._fetch_latest_release = old_fetch
            uc._write_cache = old_write
            uc._read_cache = old_read

    def test_update_status_uses_cache_when_live_check_fails(self):
        old_fetch = uc._fetch_latest_release
        old_read = uc._read_cache
        try:
            def _boom():
                raise uc.urllib.error.URLError("offline")
            uc._fetch_latest_release = _boom
            uc._read_cache = lambda: {
                "status": "up_to_date",
                "current_version": "0.10.0",
                "latest_version": "0.10.0",
                "checked_at": "2026-03-25T00:00:00+00:00",
                "message": "Cached status",
            }
            status = uc.get_update_status(force=True)
            self.assertEqual(status["status"], "cached")
            self.assertEqual(status["source_kind"], "cache")
        finally:
            uc._fetch_latest_release = old_fetch
            uc._read_cache = old_read

    def test_update_status_reports_unavailable_without_cache(self):
        old_fetch = uc._fetch_latest_release
        old_read = uc._read_cache
        try:
            def _boom():
                raise uc.urllib.error.URLError("offline")
            uc._fetch_latest_release = _boom
            uc._read_cache = lambda: {}
            status = uc.get_update_status(force=True)
            self.assertEqual(status["status"], "unavailable")
            self.assertEqual(status["source_kind"], "unavailable")
        finally:
            uc._fetch_latest_release = old_fetch
            uc._read_cache = old_read


if __name__ == "__main__":
    unittest.main()
