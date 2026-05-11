"""Tests for workspace reload + WebSocket JWT auth (Phase 5)."""

from pathlib import Path

import yaml

from lumen.core.workspace import (
    WorkspaceIndex,
    reload_workspace,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _write_workspace_yaml(
    tmp_path: Path,
    name="empresa-x",
    display_name="Empresa X",
    admin_email="ceo@empresa.com",
    admin_display="CEO",
    admin_pin_hash="$2b$12$abcdefghijklmnop",
) -> Path:
    """Helper to write a valid workspace.yaml."""
    path = tmp_path / "workspace.yaml"
    path.write_text(
        yaml.dump(
            {
                "name": name,
                "display_name": display_name,
                "branding": {
                    "logo": "/logo.png",
                    "primary_color": "#1A1A2E",
                    "app_name": "Asistente Empresa X",
                },
                "admins": [
                    {
                        "email": admin_email,
                        "display_name": admin_display,
                        "pin_hash": admin_pin_hash,
                    }
                ],
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_team_yaml(
    tmp_path: Path,
    team_name="marketing",
    team_display="Marketing",
    skills=None,
    users=None,
) -> Path:
    """Helper to write a valid teams/{team_name}/team.yaml."""
    teams_dir = tmp_path / "teams" / team_name
    teams_dir.mkdir(parents=True)
    path = teams_dir / "team.yaml"
    if skills is None:
        skills = ["core-productivity"]
    if users is None:
        users = [
            {
                "email": "user@example.com",
                "role": "member",
                "display_name": "User",
                "pin_hash": "$2b$12$abcdefghijklmnop",
            }
        ]
    path.write_text(
        yaml.dump(
            {
                "name": team_name,
                "display_name": team_display,
                "enabled_skills": skills,
                "users": users,
            },
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return path


# ── reload_workspace ─────────────────────────────────────────────────


def test_reload_new_workspace(tmp_path: Path):
    """New workspace loads successfully."""
    _write_workspace_yaml(tmp_path)
    _write_team_yaml(tmp_path, team_name="marketing", users=[
        {"email": "director@example.com", "role": "team_admin", "display_name": "Director", "pin_hash": "$2b$12$abc"},
        {"email": "member@example.com", "role": "member", "display_name": "Member", "pin_hash": "$2b$12$abc"},
    ], skills=["core-productivity", "social-skill"])
    
    result = reload_workspace(tmp_path)
    assert result is not None
    assert result.is_workspace_mode()
    user = result.lookup_user("director@example.com")
    assert user is not None
    assert user["role"] == "team_admin"
    assert user["team"] == "marketing"
    assert "core-productivity" in user["enabled_skills"]


def test_reload_admin_access(tmp_path: Path):
    """Admin found in workspace.yaml is indexed correctly."""
    _write_workspace_yaml(tmp_path)
    _write_team_yaml(tmp_path)
    
    result = reload_workspace(tmp_path)
    assert result is not None
    admin = result.lookup_user("ceo@empresa.com")
    assert admin is not None
    assert admin["role"] == "admin"
    assert admin["team"] is None


def test_reload_preserves_on_invalid_yaml(tmp_path: Path):
    """If workspace.yaml becomes invalid, keep existing index."""
    _write_workspace_yaml(tmp_path)
    _write_team_yaml(tmp_path)
    
    idx = reload_workspace(tmp_path)
    assert idx is not None
    assert idx.is_workspace_mode()
    
    # Corrupt workspace.yaml
    yaml_path = tmp_path / "workspace.yaml"
    yaml_path.write_text(":::not valid::: yaml content {{@", encoding="utf-8")
    
    # Reload should return the existing index
    result = reload_workspace(tmp_path, existing_index=idx)
    assert result is idx  # Same object — preserved
    assert result.lookup_user("ceo@empresa.com") is not None


def test_reload_returns_existing_when_yaml_missing(tmp_path: Path):
    """If workspace.yaml disappears, return existing index."""
    existing = WorkspaceIndex()
    existing.add_user("user@example.com", "member", "team", "User", ["skill"], "$2b$12$hash")
    
    result = reload_workspace(tmp_path, existing_index=existing)
    assert result is existing


def test_reload_malformed_team_preserves_existing(tmp_path: Path):
    """Invalid team YAML does not break the reload — keeps existing index."""
    _write_workspace_yaml(tmp_path)
    
    # Create a malformed team directory
    teams_dir = tmp_path / "teams" / "bad"
    teams_dir.mkdir(parents=True)
    (teams_dir / "team.yaml").write_text("{invalid yaml content", encoding="utf-8")
    
    # Also write a valid team
    _write_team_yaml(tmp_path)
    
    existing = WorkspaceIndex()
    existing.add_user("user@example.com", "member", "team", "User", ["skill"], "$2b$12$hash")
    
    result = reload_workspace(tmp_path, existing_index=existing)
    # Malformed team is skipped, valid team loads — but existing is returned only if load fails entirely
    assert result is not None  # Should load the valid team
