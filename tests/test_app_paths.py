import unittest

import app_paths as ap


class AppPathsTests(unittest.TestCase):
    def test_window_icon_path_exists(self):
        icon = ap.get_window_icon_path()
        self.assertIsNotNone(icon)
        self.assertTrue(icon.exists())
        self.assertEqual(icon.name, "icon.png")

    def test_packaging_icon_path_is_optional(self):
        icon = ap.get_packaging_icon_path("darwin")
        self.assertTrue(icon is None or icon.exists())


if __name__ == "__main__":
    unittest.main()
