import unittest
from unittest.mock import patch

import app


class AppManagerStartupTests(unittest.TestCase):
    def test_app_manager_init_does_not_load_settings_immediately(self):
        with patch("app.Settings") as settings_cls:
            manager = app.AppManager()

            self.assertIsNone(manager._settings)
            settings_cls.assert_not_called()


if __name__ == "__main__":
    unittest.main()
