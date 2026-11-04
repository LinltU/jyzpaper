import configparser
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# These constructor tests exercise configuration only; no HTTP client is used.
with patch.dict(sys.modules, {"requests": Mock()}):
    from data_fetcher import tolocation

DataFetcher = tolocation.DataFetcher


class DataFetcherConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "config.ini"
        self.settings = {
            "BASE_URL": "https://example.invalid",
            "username": "example-user",
            "password": "example-password",
            "table_id": "example-table",
            "app_id": "example-app",
            "app_secret": "example-secret",
            "wiki_id": "example-wiki",
        }

    def write_config(self):
        config = configparser.ConfigParser()
        config["DEFAULT"] = self.settings
        with self.path.open("w", encoding="utf-8") as handle:
            config.write(handle)

    def test_configured_credentials_are_loaded_without_network_access(self):
        self.write_config()
        with patch.object(tolocation.requests, "Session") as session:
            fetcher = DataFetcher(config_file=str(self.path))
        session.assert_not_called()
        for name, value in self.settings.items():
            self.assertEqual(getattr(fetcher, name), value)
        self.assertEqual(fetcher.LOGIN_DATA, {
            "username": self.settings["username"],
            "password": self.settings["password"],
        })

    def test_missing_or_empty_required_settings_fail_before_network_access(self):
        original = self.settings.copy()
        for name in ("BASE_URL", "username", "password"):
            for value in (None, "", "NOT FOUND"):
                with self.subTest(name=name, value=value):
                    self.settings = original.copy()
                    if value is None:
                        self.settings.pop(name)
                    else:
                        self.settings[name] = value
                    self.write_config()
                    with patch.object(tolocation.requests, "Session") as session:
                        with self.assertRaisesRegex(ValueError, f"DEFAULT.{name}"):
                            DataFetcher(config_file=str(self.path))
                    session.assert_not_called()

    def test_absent_config_does_not_supply_embedded_credentials(self):
        with self.assertRaisesRegex(ValueError, "DEFAULT.BASE_URL"):
            DataFetcher(config_file=str(self.path))


if __name__ == "__main__":
    unittest.main()
