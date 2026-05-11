"""Tests for WorkspaceIndex lookup methods."""

import unittest


class TestLookupUser(unittest.TestCase):
    def _make_index(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx.add_user(
            email="admin@acme.com",
            role="admin",
            team=None,
            display_name="Admin",
            enabled_skills=[],
            pin_hash="$2b$12$adminhash",
        )
        idx.add_user(
            email="alice@acme.com",
            role="member",
            team="marketing",
            display_name="Alice",
            enabled_skills=["web-search", "email-draft"],
            pin_hash="$2b$12$alicehash",
        )
        idx.add_user(
            email="bob@acme.com",
            role="viewer",
            team="sales",
            display_name="Bob",
            enabled_skills=["crm-read"],
            pin_hash="$2b$12$bobhash",
        )
        idx.add_user(
            email="carol@acme.com",
            role="team_admin",
            team="engineering",
            display_name="Carol",
            enabled_skills=["code-review", "deploy", "incident-response"],
            pin_hash="$2b$12$carolhash",
        )
        return idx

    def test_lookup_admin(self):
        idx = self._make_index()
        user = idx.lookup_user("admin@acme.com")
        assert user is not None
        assert user["role"] == "admin"
        assert user["team"] is None
        assert user["display_name"] == "Admin"
        assert user["pin_hash"] == "$2b$12$adminhash"

    def test_lookup_member(self):
        idx = self._make_index()
        user = idx.lookup_user("alice@acme.com")
        assert user is not None
        assert user["role"] == "member"
        assert user["team"] == "marketing"

    def test_lookup_viewer(self):
        idx = self._make_index()
        user = idx.lookup_user("bob@acme.com")
        assert user is not None
        assert user["role"] == "viewer"

    def test_lookup_team_admin(self):
        idx = self._make_index()
        user = idx.lookup_user("carol@acme.com")
        assert user is not None
        assert user["role"] == "team_admin"
        assert user["team"] == "engineering"

    def test_lookup_nonexistent(self):
        idx = self._make_index()
        assert idx.lookup_user("nobody@acme.com") is None

    def test_lookup_empty_string(self):
        idx = self._make_index()
        assert idx.lookup_user("") is None


class TestGetEnabledSkills(unittest.TestCase):
    def _make_index(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx.add_user(
            email="admin@acme.com",
            role="admin",
            team=None,
            display_name="Admin",
            enabled_skills=[],
            pin_hash="$2b$12$hash",
        )
        idx.add_user(
            email="alice@acme.com",
            role="member",
            team="marketing",
            display_name="Alice",
            enabled_skills=["web-search", "email-draft"],
            pin_hash="$2b$12$hash",
        )
        return idx

    def test_admin_has_empty_skills(self):
        idx = self._make_index()
        assert idx.get_enabled_skills("admin@acme.com") == []

    def test_member_has_team_skills(self):
        idx = self._make_index()
        skills = idx.get_enabled_skills("alice@acme.com")
        assert skills == ["web-search", "email-draft"]

    def test_nonexistent_user_empty(self):
        idx = self._make_index()
        assert idx.get_enabled_skills("nobody@acme.com") == []

    def test_returns_copy_not_reference(self):
        """Modifying the returned list should not affect the index."""
        idx = self._make_index()
        skills = idx.get_enabled_skills("alice@acme.com")
        skills.append("new-skill")
        assert idx.get_enabled_skills("alice@acme.com") == ["web-search", "email-draft"]


class TestGetUserRole(unittest.TestCase):
    def _make_index(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx.add_user("admin@acme.com", "admin", None, "Admin", [], "$2b$12$")
        idx.add_user("alice@acme.com", "member", "marketing", "Alice", [], "$2b$12$")
        idx.add_user("bob@acme.com", "viewer", "sales", "Bob", [], "$2b$12$")
        idx.add_user("carol@acme.com", "team_admin", "eng", "Carol", [], "$2b$12$")
        return idx

    def test_admin_role(self):
        assert self._make_index().get_user_role("admin@acme.com") == "admin"

    def test_member_role(self):
        assert self._make_index().get_user_role("alice@acme.com") == "member"

    def test_viewer_role(self):
        assert self._make_index().get_user_role("bob@acme.com") == "viewer"

    def test_team_admin_role(self):
        assert self._make_index().get_user_role("carol@acme.com") == "team_admin"

    def test_nonexistent_returns_none(self):
        assert self._make_index().get_user_role("nobody@acme.com") is None


class TestGetUserTeam(unittest.TestCase):
    def _make_index(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx.add_user("admin@acme.com", "admin", None, "Admin", [], "$2b$12$")
        idx.add_user("alice@acme.com", "member", "marketing", "Alice", [], "$2b$12$")
        return idx

    def test_admin_team_is_none(self):
        assert self._make_index().get_user_team("admin@acme.com") is None

    def test_member_has_team(self):
        assert self._make_index().get_user_team("alice@acme.com") == "marketing"

    def test_nonexistent_returns_none(self):
        assert self._make_index().get_user_team("nobody@acme.com") is None


class TestIsWorkspaceMode(unittest.TestCase):
    def test_empty_index_not_workspace(self):
        from lumen.core.workspace import WorkspaceIndex

        assert WorkspaceIndex().is_workspace_mode() is False

    def test_index_with_users_is_workspace(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx.add_user("a@b.com", "admin", None, "A", [], "$2b$12$")
        assert idx.is_workspace_mode() is True

    def test_index_after_clearing_users(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx.add_user("a@b.com", "admin", None, "A", [], "$2b$12$")
        assert idx.is_workspace_mode() is True
        # Clear users
        idx._users.clear()
        assert idx.is_workspace_mode() is False
