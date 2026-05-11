"""Tests for workspace YAML loader."""

import os
import tempfile
import unittest
from pathlib import Path

import yaml


def _write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _workspace_dir() -> tempfile.TemporaryDirectory:
    """Create a temp directory with a valid workspace.yaml."""
    tmp = tempfile.TemporaryDirectory()
    d = Path(tmp.name)

    _write_yaml(
        d / "workspace.yaml",
        {
            "name": "acme",
            "display_name": "Acme Corp",
            "branding": {
                "logo": "https://acme.com/logo.png",
                "primary_color": "#FF5733",
                "app_name": "Acme",
            },
            "admins": [
                {
                    "email": "admin@acme.com",
                    "display_name": "Admin User",
                    "pin_hash": "$2b$12$hashvalue",
                }
            ],
        },
    )
    return tmp


class TestLoadWorkspace(unittest.TestCase):
    def test_valid_workspace_yaml(self):
        from lumen.core.workspace import load_workspace

        tmp = _workspace_dir()
        try:
            result = load_workspace(Path(tmp.name))
            assert result is not None
            assert result.name == "acme"
            assert result.display_name == "Acme Corp"
            assert len(result.admins) == 1
            assert result.admins[0].email == "admin@acme.com"
        finally:
            tmp.cleanup()

    def test_missing_workspace_yaml(self):
        from lumen.core.workspace import load_workspace

        tmp = tempfile.TemporaryDirectory()
        try:
            result = load_workspace(Path(tmp.name))
            assert result is None
        finally:
            tmp.cleanup()

    def test_malformed_yaml_returns_none(self):
        from lumen.core.workspace import load_workspace

        tmp = tempfile.TemporaryDirectory()
        try:
            (Path(tmp.name) / "workspace.yaml").write_text(
                "{invalid yaml: [unclosed", encoding="utf-8"
            )
            result = load_workspace(Path(tmp.name))
            assert result is None
        finally:
            tmp.cleanup()

    def test_invalid_schema_returns_none(self):
        from lumen.core.workspace import load_workspace

        tmp = tempfile.TemporaryDirectory()
        try:
            _write_yaml(
                Path(tmp.name) / "workspace.yaml",
                {"name": "acme"},  # missing required fields
            )
            result = load_workspace(Path(tmp.name))
            assert result is None
        finally:
            tmp.cleanup()


class TestLoadTeams(unittest.TestCase):
    def test_no_teams_dir(self):
        from lumen.core.workspace import load_teams

        tmp = tempfile.TemporaryDirectory()
        try:
            result = load_teams(Path(tmp.name))
            assert result == {}
        finally:
            tmp.cleanup()

    def test_single_team(self):
        from lumen.core.workspace import load_teams

        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            _write_yaml(
                d / "teams" / "marketing" / "team.yaml",
                {
                    "name": "marketing",
                    "display_name": "Marketing Team",
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
            result = load_teams(d)
            assert "marketing" in result
            assert result["marketing"].display_name == "Marketing Team"
            assert len(result["marketing"].users) == 1
        finally:
            tmp.cleanup()

    def test_multiple_teams(self):
        from lumen.core.workspace import load_teams

        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            _write_yaml(
                d / "teams" / "marketing" / "team.yaml",
                {
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": ["web-search"],
                    "users": [],
                },
            )
            _write_yaml(
                d / "teams" / "engineering" / "team.yaml",
                {
                    "name": "engineering",
                    "display_name": "Engineering",
                    "enabled_skills": ["code-review"],
                    "users": [],
                },
            )
            result = load_teams(d)
            assert len(result) == 2
            assert "marketing" in result
            assert "engineering" in result
        finally:
            tmp.cleanup()

    def test_malformed_team_skipped(self):
        from lumen.core.workspace import load_teams

        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            # Good team
            _write_yaml(
                d / "teams" / "marketing" / "team.yaml",
                {
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": [],
                    "users": [],
                },
            )
            # Bad team (malformed YAML)
            bad_path = d / "teams" / "broken" / "team.yaml"
            bad_path.parent.mkdir(parents=True, exist_ok=True)
            bad_path.write_text("{bad yaml", encoding="utf-8")
            # Another good team
            _write_yaml(
                d / "teams" / "sales" / "team.yaml",
                {
                    "name": "sales",
                    "display_name": "Sales",
                    "enabled_skills": [],
                    "users": [],
                },
            )
            result = load_teams(d)
            assert len(result) == 2
            assert "marketing" in result
            assert "sales" in result
            assert "broken" not in result
        finally:
            tmp.cleanup()

    def test_invalid_team_schema_skipped(self):
        from lumen.core.workspace import load_teams

        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            # Good team
            _write_yaml(
                d / "teams" / "marketing" / "team.yaml",
                {
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": [],
                    "users": [],
                },
            )
            # Bad team (invalid schema — missing required fields)
            _write_yaml(
                d / "teams" / "invalid" / "team.yaml",
                {"name": "invalid"},  # missing display_name, enabled_skills, users
            )
            result = load_teams(d)
            assert "marketing" in result
            assert "invalid" not in result
        finally:
            tmp.cleanup()


class TestBuildWorkspaceIndex(unittest.TestCase):
    def test_index_includes_admins(self):
        from lumen.core.workspace import (
            AdminRecord,
            BrandingConfig,
            WorkspaceConfig,
            WorkspaceIndex,
            build_workspace_index,
        )

        ws = WorkspaceConfig(
            name="acme",
            display_name="Acme Corp",
            branding=BrandingConfig(logo="x", primary_color="#000", app_name="Acme"),
            admins=[
                AdminRecord(
                    email="admin@acme.com",
                    display_name="Admin",
                    pin_hash="$2b$12$hash",
                )
            ],
        )
        idx = build_workspace_index(ws, {})
        assert idx.is_workspace_mode() is True
        admin = idx.lookup_user("admin@acme.com")
        assert admin is not None
        assert admin["role"] == "admin"
        assert admin["team"] is None

    def test_index_includes_team_users(self):
        from lumen.core.workspace import (
            BrandingConfig,
            TeamConfig,
            UserRecord,
            WorkspaceConfig,
            build_workspace_index,
        )

        ws = WorkspaceConfig(
            name="acme",
            display_name="Acme",
            branding=BrandingConfig(logo="x", primary_color="#000", app_name="Acme"),
            admins=[],
        )
        teams = {
            "marketing": TeamConfig(
                name="marketing",
                display_name="Marketing",
                enabled_skills=["web-search", "email-draft"],
                users=[
                    UserRecord(
                        email="alice@acme.com",
                        role="member",
                        display_name="Alice",
                        pin_hash="$2b$12$hash",
                    ),
                ],
            ),
        }
        idx = build_workspace_index(ws, teams)
        alice = idx.lookup_user("alice@acme.com")
        assert alice is not None
        assert alice["role"] == "member"
        assert alice["team"] == "marketing"
        assert alice["enabled_skills"] == ["web-search", "email-draft"]

    def test_empty_workspace(self):
        from lumen.core.workspace import (
            BrandingConfig,
            WorkspaceConfig,
            build_workspace_index,
        )

        ws = WorkspaceConfig(
            name="acme",
            display_name="Acme",
            branding=BrandingConfig(logo="x", primary_color="#000", app_name="Acme"),
            admins=[],
        )
        idx = build_workspace_index(ws, {})
        assert idx.is_workspace_mode() is False
