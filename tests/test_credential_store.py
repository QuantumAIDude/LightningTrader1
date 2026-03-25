import os
import tempfile
import unittest

import credential_store as cs


class CredentialStoreTests(unittest.TestCase):
    def test_save_and_load_local_credentials(self):
        old_dir = os.environ.get("LIGHTNINGTRADER_SETTINGS_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = tmp
                saved = cs.save_alpaca_credentials(
                    "key-1234",
                    "secret-5678",
                    paper=True,
                    broker_mode="alpaca_paper",
                    saved_at="2026-03-25T00:00:00+00:00",
                )
                self.assertEqual(saved["api_key"], "key-1234")
                self.assertEqual(saved["secret_key"], "secret-5678")
                self.assertTrue(saved["paper"])
                self.assertEqual(saved["broker_mode"], "alpaca_paper")
                loaded = cs.load_saved_alpaca_credentials()
                self.assertEqual(loaded["api_key"], "key-1234")
                self.assertEqual(loaded["secret_key"], "secret-5678")
                self.assertEqual(loaded["storage_kind"], "local_file")
            finally:
                if old_dir is None:
                    os.environ.pop("LIGHTNINGTRADER_SETTINGS_DIR", None)
                else:
                    os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = old_dir

    def test_runtime_settings_use_local_store_when_env_missing(self):
        old_dir = os.environ.get("LIGHTNINGTRADER_SETTINGS_DIR")
        old_env = {k: os.environ.get(k) for k in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER", "BROKER_MODE_DEFAULT"]}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = tmp
                for key in old_env:
                    os.environ.pop(key, None)
                cs.save_alpaca_credentials("saved-key", "saved-secret", paper=True, broker_mode="alpaca_paper")
                settings = cs.get_runtime_broker_settings()
                self.assertEqual(settings["credentials_source"], "local_store")
                self.assertEqual(settings["broker_mode"], "alpaca_paper")
                self.assertTrue(settings["credentials_present"])
            finally:
                if old_dir is None:
                    os.environ.pop("LIGHTNINGTRADER_SETTINGS_DIR", None)
                else:
                    os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = old_dir
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_runtime_settings_environment_overrides_saved_store(self):
        old_dir = os.environ.get("LIGHTNINGTRADER_SETTINGS_DIR")
        old_env = {k: os.environ.get(k) for k in ["ALPACA_API_KEY", "ALPACA_SECRET_KEY", "ALPACA_PAPER", "BROKER_MODE_DEFAULT"]}
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = tmp
                cs.save_alpaca_credentials("saved-key", "saved-secret", paper=True, broker_mode="alpaca_paper")
                os.environ["ALPACA_API_KEY"] = "env-key"
                os.environ["ALPACA_SECRET_KEY"] = "env-secret"
                os.environ["ALPACA_PAPER"] = "false"
                os.environ["BROKER_MODE_DEFAULT"] = "alpaca_live"
                settings = cs.get_runtime_broker_settings()
                self.assertEqual(settings["api_key"], "env-key")
                self.assertEqual(settings["secret_key"], "env-secret")
                self.assertEqual(settings["credentials_source"], "env")
                self.assertFalse(settings["paper"])
                self.assertEqual(settings["broker_mode"], "alpaca_live")
            finally:
                if old_dir is None:
                    os.environ.pop("LIGHTNINGTRADER_SETTINGS_DIR", None)
                else:
                    os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = old_dir
                for key, value in old_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_clear_saved_credentials(self):
        old_dir = os.environ.get("LIGHTNINGTRADER_SETTINGS_DIR")
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = tmp
                cs.save_alpaca_credentials("saved-key", "saved-secret")
                self.assertTrue(cs.clear_saved_alpaca_credentials())
                self.assertFalse(cs.load_saved_alpaca_credentials())
            finally:
                if old_dir is None:
                    os.environ.pop("LIGHTNINGTRADER_SETTINGS_DIR", None)
                else:
                    os.environ["LIGHTNINGTRADER_SETTINGS_DIR"] = old_dir


if __name__ == "__main__":
    unittest.main()
