import unittest
from unittest.mock import patch

import app


class AppManagerStartupTests(unittest.TestCase):
    def test_app_manager_init_does_not_load_settings_immediately(self):
        with patch("app.Settings") as settings_cls:
            manager = app.AppManager()

            self.assertIsNone(manager._settings)
            settings_cls.assert_not_called()

    def test_initialize_continues_when_cloudwatch_setup_fails(self):
        from llama_index.core.llms.mock import MockLLM
        from llama_index.core.embeddings.mock_embed_model import MockEmbedding
        mock_llm = MockLLM()
        mock_embed = MockEmbedding(embed_dim=384)
        settings = type("Settings", (), {"config": {"HF_EMBED": "mock-model"}, "secret": {}})()

        with patch("app.Settings", return_value=settings), patch(
            "app.LogManager.setup_logging", side_effect=RuntimeError("cloudwatch unavailable")
        ), patch("app.PromptManager.load_prompts", return_value="prompts"), patch(
            "app.LLMManager.init_llm", return_value=mock_llm
        ), patch("llama_index.embeddings.fastembed.FastEmbedEmbedding", return_value=mock_embed):
            manager = app.AppManager()
            manager.initialize()

            self.assertTrue(manager._initialized)
            self.assertEqual(manager._prompts, "prompts")
            self.assertEqual(manager._llm, mock_llm)


if __name__ == "__main__":
    unittest.main()
