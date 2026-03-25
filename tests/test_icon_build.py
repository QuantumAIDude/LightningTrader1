import tempfile
import unittest
from pathlib import Path

from PIL import Image

import icon_build as ib


class IconBuildTests(unittest.TestCase):
    def test_canonical_icon_source_prefers_icon_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            branding = root / "assets" / "branding"
            branding.mkdir(parents=True)
            Image.new("RGBA", (1024, 1024), (12, 34, 56, 255)).save(branding / "icon.png", format="PNG")

            self.assertEqual(ib.canonical_icon_source(root), branding / "icon.png")

    def test_ensure_generated_icons_creates_ico(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            branding = root / "assets" / "branding"
            branding.mkdir(parents=True)
            Image.new("RGBA", (1024, 1024), (12, 34, 56, 255)).save(branding / "icon.png", format="PNG")

            result = ib.ensure_generated_icons(root, force=True)

            self.assertIn(result["status"], {"ok", "partial"})
            self.assertTrue((branding / "lightning_trader.ico").exists())


if __name__ == "__main__":
    unittest.main()
