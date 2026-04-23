"""Tests for P2 #6 — UI tag configurable by personality."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import yaml
from fastapi.testclient import TestClient

from lumen.channels import web
from lumen.core.connectors import ConnectorRegistry
from lumen.core.personality import Personality
from lumen.core.registry import Registry


class BrainStub:
    def __init__(self, personality):
        self.registry = Registry()
        self.connectors = ConnectorRegistry()
        self.flows = []
        self.memory = MagicMock()
        self.personality = personality
        self.capability_awareness = None


class PersonalityUIConfigTests(unittest.TestCase):
    def test_personality_exposes_ui_config(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "personality.yaml"
            path.write_text(
                yaml.dump(
                    {
                        "identity": {"name": "Kit"},
                        "ui": {
                            "tag": "my-ui",
                            "surfaces": ["chat_right_rail", "briefing"],
                        },
                    }
                ),
                encoding="utf-8",
            )
            personality = Personality(path)
            assert personality.ui == {
                "tag": "my-ui",
                "surfaces": ["chat_right_rail", "briefing"],
            }

    def test_personality_ui_defaults_empty_dict(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "personality.yaml"
            path.write_text(yaml.dump({"identity": {"name": "Kit"}}), encoding="utf-8")
            personality = Personality(path)
            assert personality.ui == {}


class WebStatusPersonalityUITests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.original_brain = web._brain
        self.original_config = web._config
        self.original_locale = web._locale
        self.original_lumen_dir = web.LUMEN_DIR
        self.original_config_path = web.CONFIG_PATH
        web._config = {"model": "test-model", "language": "es"}
        web._locale = {}
        web.LUMEN_DIR = Path(self.temp_dir.name)
        web.CONFIG_PATH = web.LUMEN_DIR / "config.yaml"

        personality_path = Path(self.temp_dir.name) / "personality.yaml"
        personality_path.write_text(
            yaml.dump(
                {
                    "identity": {"name": "Kit"},
                    "ui": {
                        "tag": "my-ui",
                        "surfaces": ["chat_right_rail", "briefing"],
                    },
                }
            ),
            encoding="utf-8",
        )
        web._brain = BrainStub(Personality(personality_path))
        self.client = TestClient(web.app)

    def tearDown(self):
        self.temp_dir.cleanup()
        web._brain = self.original_brain
        web._config = self.original_config
        web._locale = self.original_locale
        web.LUMEN_DIR = self.original_lumen_dir
        web.CONFIG_PATH = self.original_config_path

    def test_api_status_includes_personality_ui(self):
        response = self.client.get("/api/status")
        assert response.status_code == 200
        data = response.json()
        assert data["personality_ui"]["tag"] == "my-ui"
        assert data["personality_ui"]["surfaces"] == ["chat_right_rail", "briefing"]
