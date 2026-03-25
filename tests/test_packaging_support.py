import tempfile
import unittest
from pathlib import Path

import packaging_support as ps


class PackagingSupportTests(unittest.TestCase):
    def test_collect_model_seed_files_prefers_instrument_scoped_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = root / "models"
            models.mkdir()
            (models / "base_model_QQQ.pkl").write_text("x", encoding="utf-8")
            (models / "forest_model_QQQ.pkl").write_text("x", encoding="utf-8")
            (models / "extra_trees_model.pkl").write_text("legacy", encoding="utf-8")

            files = ps.collect_model_seed_files(root, instrument="QQQ")

            names = {p.name for p in files}
            self.assertIn("base_model_QQQ.pkl", names)
            self.assertIn("forest_model_QQQ.pkl", names)
            self.assertNotIn("extra_trees_model.pkl", names)

    def test_packaging_manifest_lists_seed_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "models").mkdir()
            (root / "data").mkdir()
            (root / "assets" / "branding").mkdir(parents=True)
            (root / "models" / "base_model_QQQ.pkl").write_text("x", encoding="utf-8")
            (root / "data" / "prices_QQQ.csv").write_text("x", encoding="utf-8")
            (root / "data" / "macro_calendar_cache.json").write_text("{}", encoding="utf-8")
            (root / "data" / "market_breadth_live.json").write_text("{}", encoding="utf-8")
            (root / "data" / "options_snapshot_history_QQQ.csv").write_text("x", encoding="utf-8")
            (root / "data" / "sec_events_live_QQQ_cache.json").write_text("{}", encoding="utf-8")
            (root / "data" / "instruments.json").write_text("{}", encoding="utf-8")
            (root / "assets" / "branding" / "icon.png").write_bytes(b"x")

            manifest = ps.packaging_manifest(root)

            self.assertEqual(manifest["instrument"], "QQQ")
            self.assertIn("models/base_model_QQQ.pkl", manifest["model_files"])
            self.assertIn("data/prices_QQQ.csv", manifest["data_files"])
            self.assertIn("data/macro_calendar_cache.json", manifest["data_files"])
            self.assertIn("data/market_breadth_live.json", manifest["data_files"])
            self.assertIn("data/options_snapshot_history_QQQ.csv", manifest["data_files"])
            self.assertIn("data/sec_events_live_QQQ_cache.json", manifest["data_files"])
            self.assertIn("data/instruments.json", manifest["data_files"])
            self.assertIn("assets/branding/icon.png", manifest["branding_files"])
            self.assertEqual(manifest["canonical_icon"], str(root / "assets" / "branding" / "icon.png"))


if __name__ == "__main__":
    unittest.main()
