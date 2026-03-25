import unittest

import app_paths as ap


class AppPathsTests(unittest.TestCase):
    def test_window_icon_path_exists(self):
        icon = ap.get_window_icon_path()
        self.assertIsNotNone(icon)
        self.assertTrue(icon.exists())
        self.assertEqual(icon.name, "icon.png")

    def test_packaging_icon_path_is_optional(self):
        mac_icon = ap.get_packaging_icon_path("darwin")
        win_icon = ap.get_packaging_icon_path("win32")
        self.assertTrue(mac_icon is None or mac_icon.exists())
        self.assertTrue(win_icon is None or win_icon.exists())
        if mac_icon is not None:
            self.assertEqual(mac_icon.suffix.lower(), ".icns")
        if win_icon is not None:
            self.assertEqual(win_icon.suffix.lower(), ".ico")


if __name__ == "__main__":
    unittest.main()
