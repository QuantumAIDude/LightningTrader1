import tempfile
import unittest
from pathlib import Path

import build_release as br


class BuildReleaseTests(unittest.TestCase):
    def test_release_archive_name_is_platform_specific(self):
        self.assertEqual(
            br.release_archive_name("darwin"),
            f"{br.APP_DISPLAY_NAME}-macOS-v{br.APP_VERSION}.zip",
        )
        self.assertEqual(
            br.release_archive_name("win32"),
            f"{br.APP_DISPLAY_NAME}-Windows-v{br.APP_VERSION}.zip",
        )

    def test_create_release_archive_uses_platform_bundle_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            win_bundle = dist / br.APP_DISPLAY_NAME
            win_bundle.mkdir()
            (win_bundle / "Lightning Trader Terminal.exe").write_text("binary", encoding="utf-8")

            zip_path = br.create_release_archive(dist, "win32")

            self.assertIsNotNone(zip_path)
            self.assertTrue(Path(zip_path).exists())
            self.assertTrue(str(zip_path).endswith(f"Windows-v{br.APP_VERSION}.zip"))


if __name__ == "__main__":
    unittest.main()
