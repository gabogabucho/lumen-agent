"""Tests for workspace integration in runtime."""

import asyncio
import tempfile
import unittest
from pathlib import Path

import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _make_workspace_dir(with_workspace: bool = True) -> tempfile.TemporaryDirectory:
    """Create a temp directory optionally containing a valid workspace."""
    tmp = tempfile.TemporaryDirectory()
    d = Path(tmp.name)

    if with_workspace:
        _write_yaml(
            d / "workspace.yaml",
            {
                "name": "acme",
                "display_name": "Acme Corp",
                "branding": {
                    "logo": "logo.png",
                    "primary_color": "#000",
                    "app_name": "Acme",
                },
                "admins": [
                    {
                        "email": "admin@acme.com",
                        "display_name": "Admin",
                        "pin_hash": "$2b$12$hash",
                    }
                ],
            },
        )
        _write_yaml(
            d / "teams" / "marketing" / "team.yaml",
            {
                "name": "marketing",
                "display_name": "Marketing",
                "enabled_skills": ["web-search"],
                "users": [
                    {
                        "email": "alice@acme.com",
                        "role": "member",
                        "display_name": "Alice",
                        "pin_hash": "$2b$12$hash",
                    }
                ],
            },
        )
    return tmp


class TestRuntimeWorkspaceIndex(unittest.TestCase):
    """Test that RuntimeBootstrap gains a workspace_index field."""

    def test_runtime_has_workspace_index_attribute(self):
        from lumen.core.runtime import RuntimeBootstrap

        # RuntimeBootstrap should have workspace_index field
        assert hasattr(RuntimeBootstrap, "__dataclass_fields__")
        assert "workspace_index" in RuntimeBootstrap.__dataclass_fields__


class TestWorkspaceDetection(unittest.TestCase):
    """Test workspace detection and loading via the load function."""

    def test_detect_workspace_present(self):
        from lumen.core.workspace import load_workspace, load_teams, build_workspace_index

        tmp = _make_workspace_dir(with_workspace=True)
        try:
            ws = load_workspace(Path(tmp.name))
            assert ws is not None
            teams = load_teams(Path(tmp.name))
            assert "marketing" in teams
            idx = build_workspace_index(ws, teams)
            assert idx.is_workspace_mode() is True
            assert idx.lookup_user("admin@acme.com") is not None
            assert idx.lookup_user("alice@acme.com") is not None
        finally:
            tmp.cleanup()

    def test_no_workspace_returns_none(self):
        from lumen.core.workspace import load_workspace, load_teams, build_workspace_index

        tmp = _make_workspace_dir(with_workspace=False)
        try:
            ws = load_workspace(Path(tmp.name))
            assert ws is None
            teams = load_teams(Path(tmp.name))
            assert teams == {}
            # If ws is None, build_workspace_index should not be called
            # but we test the contract: None workspace means no index
        finally:
            tmp.cleanup()
