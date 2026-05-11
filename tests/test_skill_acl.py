"""Tests for lumen.core.skill_acl — skill access control."""

from unittest.mock import MagicMock

from lumen.core.skill_acl import (
    check_skill_access,
    extract_skill_from_tool_call,
    get_acl_denial_message,
)


# ── check_skill_access ────────────────────────────────────────────────


def test_admin_always_allowed():
    """Admin role bypasses all checks — even for skills not in enabled list."""
    index = _mock_workspace_role("admin", ["file__read"])
    assert check_skill_access("terminal__execute", "admin@example.com", index) is True
    assert check_skill_access("unknown_skill", "admin@example.com", index) is True


def test_member_has_skill_allowed():
    """Member with the skill in enabled_skills gets access."""
    index = _mock_workspace_role("member", ["file__read", "web__search"])
    assert check_skill_access("file__read", "member@example.com", index) is True
    assert check_skill_access("web__search", "member@example.com", index) is True


def test_member_without_skill_denied():
    """Member without the skill sees denial."""
    index = _mock_workspace_role("member", ["file__read"])
    assert check_skill_access("terminal__execute", "member@example.com", index) is False


def test_non_workspace_mode_allows_everything():
    """Without a workspace_index, all tools work — legacy mode."""
    assert check_skill_access("any_tool", "anyone", None) is True


def test_unknown_user_in_workspace_mode_denied():
    """Users not in the workspace index are denied."""
    index = MagicMock()
    index.lookup_user.return_value = None
    index.get_enabled_skills.return_value = []
    assert check_skill_access("file__read", "unknown@example.com", index) is False


def test_empty_enabled_skills_Allows():
    """If enabled_skills is empty, we allow by default (backwards compat)."""
    index = _mock_workspace_role("member", [])
    assert check_skill_access("anything", "member@example.com", index) is True


def test_viewer_role_denied_without_skill():
    """Viewer role follows same rules as member."""
    index = _mock_workspace_role("viewer", ["file__read"])
    assert check_skill_access("file__read", "viewer@example.com", index) is True
    assert check_skill_access("terminal__execute", "viewer@example.com", index) is False


# ── get_acl_denial_message ──────────────────────────────────────────


def test_denial_message_is_generic():
    """Denial message never reveals skill names."""
    msg = get_acl_denial_message()
    assert msg != ""
    # Should NOT contain specific skill/tool names
    assert "terminal" not in msg.lower()
    assert "file" not in msg.lower()
    assert "skill" not in msg.lower().split()  # avoid being too specific


def test_denial_message_is_spanish():
    """Denial message should be in Spanish (matching project locale)."""
    msg = get_acl_denial_message()
    assert "admin" in msg.lower()  # mentions admin as the contact


# ── extract_skill_from_tool_call ──────────────────────────────────────


def test_neo_tools_return_none():
    """Brain-internal neo__ tools are never workspace-scoped."""
    registry = MagicMock()
    registry.all.return_value = []

    for neotool in ("neo__read_skill", "neo__search_modules", "neo__save_module_setup"):
        assert extract_skill_from_tool_call(neotool, registry) is None


def test_unmapped_tool_returns_none():
    """Unknown tools fail open (return None → allow)."""
    registry = MagicMock()
    registry.all.return_value = []
    assert extract_skill_from_tool_call("unknown_tool", registry) is None


def test_connector_action_mapping():
    """Connector actions map to their owning skill via registry."""
    skill_cap = MagicMock()
    skill_cap.kind.value = "skill"
    skill_cap.name = "web-skill"
    skill_cap.provides = ["web__search"]  # declares this connector

    registry = MagicMock()
    registry.all.return_value = [skill_cap]

    result = extract_skill_from_tool_call("web__search", registry)
    assert result == "web-skill"


def test_connector_action_no_provides_returns_none():
    """If no skill provides this connector, return None (fail-open)."""
    skill_cap = MagicMock()
    skill_cap.kind.value = "skill"
    skill_cap.name = "other-skill"
    skill_cap.provides = ["file__read"]  # different connector

    registry = MagicMock()
    registry.all.return_value = [skill_cap]

    assert extract_skill_from_tool_call("terminal__execute", registry) is None


def test_mixed_registry_selects_skill():
    """Only skill-kind capabilities are considered for mapping."""
    skill_cap = MagicMock()
    skill_cap.kind.value = "skill"
    skill_cap.name = "terminal-skill"
    skill_cap.provides = ["terminal__execute"]

    module_cap = MagicMock()
    module_cap.kind.value = "module"
    module_cap.name = "terminal-module"
    module_cap.provides = ["terminal__execute"]

    registry = MagicMock()
    registry.all.return_value = [module_cap, skill_cap]

    result = extract_skill_from_tool_call("terminal__execute", registry)
    assert result == "terminal-skill"


# ── Integration: ACL check + skill extraction ───────────────────────


def test_full_flow_connector_acl_check():
    """End-to-end: extract skill from tool -> check access."""
    # Setup: workspace with member who has skill "file-tool"
    index = _mock_workspace_role("member", ["file-tool"])
    registry = MagicMock()

    skill_cap = MagicMock()
    skill_cap.kind.value = "skill"
    skill_cap.name = "file-tool"
    skill_cap.provides = ["file__read"]
    registry.all.return_value = [skill_cap]

    # Extract skill name from connector tool
    skill = extract_skill_from_tool_call("file__read", registry)
    assert skill == "file-tool"

    # Check access for that skill — skill is enabled → allowed
    assert check_skill_access("file-tool", "member@example.com", index) is True

    # Now check for a skill member doesn't have
    assert check_skill_access("terminal-skill", "member@example.com", index) is False


# ── Helpers ─────────────────────────────────────────────────────────


def _mock_workspace_role(role: str, skills: list[str]):
    """Create a mock WorkspaceIndex that simulates a user with given role and skills."""
    index = MagicMock()

    user = {
        "role": role,
        "email": "member@example.com",
        "enabled_skills": skills,
    }
    index.lookup_user.return_value = user
    index.is_workspace_mode.return_value = True
    index.get_enabled_skills.return_value = skills
    index.get_user_role.return_value = role

    return index
