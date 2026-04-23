"""Tests for Kits Productivos P0 features.

#1: module install from local path
#2: modules declare system requirements (terminal allowlist, env vars)
#3: personality auto-set when installing module with personality tag
"""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from lumen.core.catalog import Catalog
from lumen.core.connectors import ConnectorRegistry
from lumen.core.installer import Installer


def _make_installer(tmp_dir: Path) -> Installer:
    pkg_dir = tmp_dir / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "modules").mkdir()
    return Installer(
        pkg_dir=pkg_dir,
        connectors=ConnectorRegistry(),
        memory=None,
        catalog=Catalog(),
        lumen_dir=tmp_dir / "lumen",
        config={},
    )


def _create_local_module(parent: Path, name: str = "test-mod", *, tags=None,
                          x_lumen=None, manifest_extra=None) -> Path:
    """Create a local module directory with module.yaml."""
    mod_dir = parent / name
    mod_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "name": name,
        "version": "1.0.0",
        "display_name": name.replace("-", " ").title(),
        "description": f"Test module {name}",
        "provides": [],
        "tags": tags or [],
    }
    if x_lumen:
        manifest["x-lumen"] = x_lumen
    if manifest_extra:
        manifest.update(manifest_extra)
    (mod_dir / "module.yaml").write_text(
        yaml.dump(manifest, default_flow_style=False), encoding="utf-8"
    )
    return mod_dir


# ── #1: Local path install ──────────────────────────────────────────────────


class LocalPathInstallTests(unittest.TestCase):
    """#1: lumen module install /path/to/module"""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.installer = _make_installer(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_install_from_local_path(self):
        """install_from_local_path copies module from local directory."""
        src = _create_local_module(Path(self.temp_dir.name) / "src", "my-kit")
        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"
        assert result["name"] == "my-kit"

        # Files should be copied to installed_dir
        installed = self.installer.installed_dir / "my-kit"
        assert installed.exists()
        assert (installed / "module.yaml").exists()

    def test_install_from_local_path_with_skill_md(self):
        """Local module with SKILL.md copies correctly."""
        src_dir = Path(self.temp_dir.name) / "src"
        src = _create_local_module(src_dir, "skill-mod")
        (src / "SKILL.md").write_text("# My Skill\n\nDoes things.", encoding="utf-8")

        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"

        installed = self.installer.installed_dir / "skill-mod"
        assert (installed / "SKILL.md").exists()

    def test_install_from_local_path_invalid(self):
        """install_from_local_path with non-existent path returns error."""
        result = self.installer.install_from_local_path(Path("/nonexistent/path"))
        assert result["status"] == "error"

    def test_install_from_local_path_no_manifest(self):
        """install_from_local_path with directory but no module.yaml returns error."""
        empty_dir = Path(self.temp_dir.name) / "empty-mod"
        empty_dir.mkdir()
        result = self.installer.install_from_local_path(empty_dir)
        assert result["status"] == "error"
        assert "module.yaml" in result["error"].lower() or "manifest" in result["error"].lower()

    def test_cli_detects_local_path(self):
        """CLI _is_local_path helper detects local paths."""
        from lumen.cli.main import _is_local_path
        assert _is_local_path("/abs/path") is True
        assert _is_local_path("./relative") is True
        assert _is_local_path("../parent") is True
        assert _is_local_path("my-module") is False
        assert _is_local_path("github:owner/repo") is False
        assert _is_local_path("https://github.com/owner/repo") is False


# ── #2: System requirements ─────────────────────────────────────────────────


class SystemRequirementsTests(unittest.TestCase):
    """#2: Modules declare terminal allowlist and env requirements."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.installer = _make_installer(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_install_merges_terminal_allowlist(self):
        """Installer merges x-lumen.requires.terminal.allowlist into config."""
        x_lumen = {
            "requires": {
                "terminal": {
                    "allowlist": ["python3", "git"]
                }
            }
        }
        src = _create_local_module(
            Path(self.temp_dir.name) / "src", "terminal-mod", x_lumen=x_lumen
        )
        self.installer.config = {"terminal": {"allowlist": ["echo"]}}

        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"

        # Config should now have merged allowlist
        terminal_config = self.installer.config.get("terminal", {})
        merged_allowlist = terminal_config.get("allowlist", [])
        assert "python3" in merged_allowlist
        assert "git" in merged_allowlist
        assert "echo" in merged_allowlist  # original preserved

    def test_install_detects_missing_env_vars(self):
        """Installer reports missing env vars in result."""
        x_lumen = {
            "requires": {
                "env": ["MISSING_API_TOKEN", "MISSING_STORE_ID"]
            }
        }
        src = _create_local_module(
            Path(self.temp_dir.name) / "src", "env-mod", x_lumen=x_lumen
        )

        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"
        assert "missing_env" in result
        assert "MISSING_API_TOKEN" in result["missing_env"]

    def test_install_no_missing_env_when_set(self):
        """Installer doesn't report env vars that are set."""
        import os
        os.environ["PRESENT_TOKEN"] = "abc123"

        x_lumen = {
            "requires": {
                "env": ["PRESENT_TOKEN"]
            }
        }
        src = _create_local_module(
            Path(self.temp_dir.name) / "src", "env-mod2", x_lumen=x_lumen
        )

        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"
        missing = result.get("missing_env", [])
        assert "PRESENT_TOKEN" not in missing

        os.environ.pop("PRESENT_TOKEN", None)


# ── #3: Personality auto-set ────────────────────────────────────────────────


class PersonalityAutoSetTests(unittest.TestCase):
    """#3: Installing module with personality tag auto-sets active_personality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.installer = _make_installer(Path(self.temp_dir.name))

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_install_personality_module_auto_sets(self):
        """Installing a module with tag 'personality' auto-sets active_personality."""
        src = _create_local_module(
            Path(self.temp_dir.name) / "src", "my-barber",
            tags=["personality", "x-lumen"],
            manifest_extra={"personality": "personality.yaml"},
        )
        (src / "personality.yaml").write_text(
            yaml.dump({"name": "my-barber", "tone": "friendly"}),
            encoding="utf-8",
        )

        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"
        assert result.get("personality_set") is True
        assert result.get("active_personality") == "my-barber"

    def test_install_non_personality_module_no_auto_set(self):
        """Installing module without personality tag does NOT set active_personality."""
        src = _create_local_module(
            Path(self.temp_dir.name) / "src", "regular-mod",
            tags=["x-lumen"],
        )

        result = self.installer.install_from_local_path(src)
        assert result["status"] == "installed"
        assert "personality_set" not in result or result.get("personality_set") is False

    def test_install_second_personality_warns(self):
        """Installing a second personality module warns about conflict."""
        # First personality
        src1 = _create_local_module(
            Path(self.temp_dir.name) / "src1", "barber-v1",
            tags=["personality", "x-lumen"],
            manifest_extra={"personality": "personality.yaml"},
        )
        (src1 / "personality.yaml").write_text("name: barber-v1\n", encoding="utf-8")
        self.installer.config["active_personality"] = "barber-v1"

        # Second personality — should warn
        src2 = _create_local_module(
            Path(self.temp_dir.name) / "src2", "barber-v2",
            tags=["personality", "x-lumen"],
            manifest_extra={"personality": "personality.yaml"},
        )
        (src2 / "personality.yaml").write_text("name: barber-v2\n", encoding="utf-8")

        result = self.installer.install_from_local_path(src2)
        assert result["status"] == "installed"
        assert result.get("personality_conflict") is True
        assert "barber-v1" in result.get("existing_personality", "")


if __name__ == "__main__":
    unittest.main()
