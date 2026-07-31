"""Tests for headless configuration from environment variables.

In a container there is nobody to answer the wizard: it blocks on a prompt that
never gets an answer and the container dies. These cover the path that lets
Lumen start unattended.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


class ConfigFromEnvTests(unittest.TestCase):
    """Tests for _config_from_env."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.lumen_dir = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _call(self, **env):
        from lumen.cli.main import _config_from_env

        with patch.dict(os.environ, env, clear=False):
            for key in ("LUMEN_MODEL", "LUMEN_API_KEY", "LUMEN_API_BASE",
                        "LUMEN_LANGUAGE", "LUMEN_PERSONALITY"):
                if key not in env:
                    os.environ.pop(key, None)
            return _config_from_env(lumen_dir=self.lumen_dir)

    def test_returns_none_without_model(self):
        """No LUMEN_MODEL means no opinion: the wizard still runs as before."""
        self.assertIsNone(self._call())

    def test_builds_config_from_model(self):
        config = self._call(LUMEN_MODEL="deepseek/deepseek-chat")

        self.assertIsNotNone(config)
        self.assertEqual(config["model"], "deepseek/deepseek-chat")

    def test_writes_config_file(self):
        self._call(LUMEN_MODEL="openai/gpt-4o-mini")

        saved = yaml.safe_load(
            (self.lumen_dir / "config.yaml").read_text(encoding="utf-8"))
        self.assertEqual(saved["model"], "openai/gpt-4o-mini")

    def test_includes_api_key_when_present(self):
        config = self._call(LUMEN_MODEL="openai/gpt-4o-mini",
                            LUMEN_API_KEY="sk-test")

        self.assertEqual(config["api_key"], "sk-test")

    def test_binds_the_key_to_the_provider_variable(self):
        """Without api_key_env, apply_provider_runtime_env never exports the
        key and the first reply fails with AuthenticationError -- with nothing
        in the message pointing at the config."""
        config = self._call(LUMEN_MODEL="deepseek/deepseek-chat",
                            LUMEN_API_KEY="sk-test")

        self.assertEqual(config["api_key_env"], "DEEPSEEK_API_KEY")

    def test_binding_follows_the_model_provider(self):
        for model, expected in [
            ("openai/gpt-4o-mini", "OPENAI_API_KEY"),
            ("anthropic/claude-sonnet-4-20250514", "ANTHROPIC_API_KEY"),
            ("openrouter/openai/gpt-oss-120b:free", "OPENROUTER_API_KEY"),
            ("gpt-4o-mini", "OPENAI_API_KEY"),
        ]:
            with self.subTest(model=model):
                config = self._call(LUMEN_MODEL=model, LUMEN_API_KEY="k")
                self.assertEqual(config["api_key_env"], expected)

    def test_no_binding_without_a_key(self):
        """Keyless providers such as Ollama need no binding, and an empty one
        would only confuse the provider layer."""
        config = self._call(LUMEN_MODEL="ollama/llama3")

        self.assertNotIn("api_key_env", config)

    def test_omits_api_key_when_absent(self):
        """Local providers such as Ollama need no key; an empty one would only
        confuse the provider layer."""
        config = self._call(LUMEN_MODEL="ollama/llama3")

        self.assertNotIn("api_key", config)

    def test_supports_custom_api_base(self):
        """Self-hosted and proxied endpoints are the common case in production."""
        config = self._call(LUMEN_MODEL="openai/gpt-4o-mini",
                            LUMEN_API_BASE="https://api.example.com/v1")

        self.assertEqual(config["api_base"], "https://api.example.com/v1")

    def test_language_defaults_to_es(self):
        config = self._call(LUMEN_MODEL="ollama/llama3")

        self.assertEqual(config["language"], "es")

    def test_language_can_be_overridden(self):
        config = self._call(LUMEN_MODEL="ollama/llama3", LUMEN_LANGUAGE="en")

        self.assertEqual(config["language"], "en")

    def test_ignores_blank_values(self):
        """An unset variable in a compose file arrives as an empty string, not
        as absent. Treating "" as a model would write a broken config."""
        self.assertIsNone(self._call(LUMEN_MODEL="   "))

    def test_selects_personality_module(self):
        """A deployment shipping its own personality needs to select it in the
        same step, or the module sits there ignored."""
        config = self._call(LUMEN_MODEL="ollama/llama3",
                            LUMEN_PERSONALITY="ambar")

        self.assertEqual(config["active_personality"], "ambar")

    def test_no_personality_key_when_unset(self):
        config = self._call(LUMEN_MODEL="ollama/llama3")

        self.assertNotIn("active_personality", config)

    def test_creates_missing_directory(self):
        nested = self.lumen_dir / "instances" / "elena"
        from lumen.cli.main import _config_from_env

        with patch.dict(os.environ, {"LUMEN_MODEL": "ollama/llama3"}):
            _config_from_env(lumen_dir=nested)

        self.assertTrue((nested / "config.yaml").exists())


if __name__ == "__main__":
    unittest.main()
