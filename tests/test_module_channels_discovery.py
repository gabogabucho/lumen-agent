"""Tests for P1 #5 — external channels as installable modules."""

import tempfile
import unittest
from pathlib import Path

import yaml

from lumen.core.connectors import ConnectorRegistry
from lumen.core.discovery import discover_all
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry


class ModuleChannelDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pkg_dir = Path(self.temp_dir.name) / "pkg"
        (self.pkg_dir / "modules").mkdir(parents=True)
        (self.pkg_dir / "skills").mkdir(parents=True)
        (self.pkg_dir / "connectors").mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_module(self, name: str, manifest: dict, skill: str = None):
        module_dir = self.pkg_dir / "modules" / name
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / "module.yaml").write_text(
            yaml.dump(manifest, default_flow_style=False), encoding="utf-8"
        )
        if skill:
            (module_dir / "SKILL.md").write_text(skill, encoding="utf-8")
        return module_dir

    def test_web_app_channel_module_registers_channel_capability(self):
        manifest = {
            "name": "my-web-channel",
            "description": "External web app channel",
            "provides": ["channel.web-app"],
            "tags": ["x-lumen", "channel"],
            "x-lumen": {
                "channel": {
                    "type": "web-app",
                    "auth": "rest-api",
                    "cors": ["https://shop.example.com"],
                    "response_format": "json",
                }
            },
        }
        self._write_module(
            "my-web-channel",
            manifest,
            skill="---\nname: my-web-channel\ndescription: Channel skill\n---\nUse the channel.",
        )

        registry = Registry()
        discover_all(registry, self.pkg_dir, ConnectorRegistry(), active_channels=["web"], config={})

        channel = registry.get(CapabilityKind.CHANNEL, "my-web-channel")
        assert channel is not None
        assert channel.status == CapabilityStatus.READY
        assert channel.metadata.get("source_module") == "my-web-channel"
        assert channel.metadata.get("channel_type") == "web-app"
        assert channel.metadata.get("auth") == "rest-api"
        assert channel.metadata.get("cors") == ["https://shop.example.com"]

    def test_channel_module_without_skill_is_available(self):
        manifest = {
            "name": "empty-channel",
            "description": "Channel without runtime skill",
            "provides": ["channel.web-app"],
            "tags": ["x-lumen", "channel"],
            "x-lumen": {
                "channel": {
                    "type": "web-app",
                    "auth": "rest-api",
                }
            },
        }
        self._write_module("empty-channel", manifest)

        registry = Registry()
        discover_all(registry, self.pkg_dir, ConnectorRegistry(), active_channels=["web"], config={})

        channel = registry.get(CapabilityKind.CHANNEL, "empty-channel")
        assert channel is not None
        assert channel.status == CapabilityStatus.AVAILABLE

    def test_non_channel_module_does_not_register_channel(self):
        manifest = {
            "name": "normal-module",
            "description": "Normal module",
            "provides": ["docs.answer"],
            "tags": ["x-lumen"],
        }
        self._write_module(
            "normal-module",
            manifest,
            skill="---\nname: normal-module\ndescription: Normal skill\n---\nDo normal things.",
        )

        registry = Registry()
        discover_all(registry, self.pkg_dir, ConnectorRegistry(), active_channels=["web"], config={})

        channel = registry.get(CapabilityKind.CHANNEL, "normal-module")
        assert channel is None


if __name__ == "__main__":
    unittest.main()
