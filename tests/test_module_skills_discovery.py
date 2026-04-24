"""Tests for P1 #4 — skills declared inside modules register automatically."""

import tempfile
import unittest
from pathlib import Path

import yaml

from lumen.core.connectors import ConnectorRegistry
from lumen.core.discovery import discover_all
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry


class ModuleSkillsDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.pkg_dir = Path(self.temp_dir.name) / "pkg"
        (self.pkg_dir / "modules").mkdir(parents=True)
        (self.pkg_dir / "skills").mkdir(parents=True)
        (self.pkg_dir / "connectors").mkdir(parents=True)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _write_module(self, name: str, manifest: dict, files: dict[str, str]):
        module_dir = self.pkg_dir / "modules" / name
        module_dir.mkdir(parents=True, exist_ok=True)
        (module_dir / "module.yaml").write_text(
            yaml.dump(manifest, default_flow_style=False), encoding="utf-8"
        )
        for rel, content in files.items():
            path = module_dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
        return module_dir

    def test_declared_module_skills_register(self):
        manifest = {
            "name": "shop-kit",
            "description": "Shop kit",
            "provides": ["shop.help"],
            "skills": ["skills/ecommerce-ops.md", "skills/pricing-strategy.md"],
            "tags": ["x-lumen"],
        }
        self._write_module(
            "shop-kit",
            manifest,
            {
                "skills/ecommerce-ops.md": "---\nname: ecommerce-ops\ndescription: Operate ecommerce\n---\nDo things.",
                "skills/pricing-strategy.md": "---\nname: pricing-strategy\ndescription: Price products\n---\nPrice things.",
            },
        )

        registry = Registry()
        discover_all(registry, self.pkg_dir, ConnectorRegistry(), active_channels=["web"], config={})

        skill1 = registry.get(CapabilityKind.SKILL, "ecommerce-ops")
        skill2 = registry.get(CapabilityKind.SKILL, "pricing-strategy")
        module = registry.get(CapabilityKind.MODULE, "shop-kit")

        assert skill1 is not None
        assert skill2 is not None
        assert "shop-kit/ecommerce-ops" in skill1.metadata.get("aliases", [])
        assert skill1.metadata.get("path", "").endswith("skills\\ecommerce-ops.md") or skill1.metadata.get("path", "").endswith("skills/ecommerce-ops.md")
        assert module is not None
        assert module.status == CapabilityStatus.READY

    def test_root_skill_still_works(self):
        manifest = {
            "name": "simple-kit",
            "description": "Simple kit",
            "provides": ["simple.help"],
            "tags": ["x-lumen"],
        }
        self._write_module(
            "simple-kit",
            manifest,
            {
                "SKILL.md": "---\nname: simple-kit\ndescription: Root skill\n---\nRoot skill content.",
            },
        )

        registry = Registry()
        discover_all(registry, self.pkg_dir, ConnectorRegistry(), active_channels=["web"], config={})

        skill = registry.get(CapabilityKind.SKILL, "simple-kit")
        module = registry.get(CapabilityKind.MODULE, "simple-kit")
        assert skill is not None
        assert module is not None
        assert module.status == CapabilityStatus.READY

    def test_missing_declared_skills_keep_module_available(self):
        manifest = {
            "name": "broken-kit",
            "description": "Broken kit",
            "provides": ["broken.help"],
            "skills": ["skills/missing.md"],
            "tags": ["x-lumen"],
        }
        self._write_module("broken-kit", manifest, {})

        registry = Registry()
        discover_all(registry, self.pkg_dir, ConnectorRegistry(), active_channels=["web"], config={})

        module = registry.get(CapabilityKind.MODULE, "broken-kit")
        assert module is not None
        assert module.status == CapabilityStatus.AVAILABLE
        assert registry.get(CapabilityKind.SKILL, "missing") is None


if __name__ == "__main__":
    unittest.main()
