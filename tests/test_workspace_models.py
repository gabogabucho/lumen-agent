"""Tests for workspace Pydantic models."""

import unittest

from pydantic import ValidationError


class TestBrandingConfig(unittest.TestCase):
    def test_valid_branding(self):
        from lumen.core.workspace import BrandingConfig

        b = BrandingConfig(
            logo="https://example.com/logo.png",
            primary_color="#FF5733",
            app_name="Acme Corp",
        )
        assert b.logo == "https://example.com/logo.png"
        assert b.primary_color == "#FF5733"
        assert b.app_name == "Acme Corp"

    def test_all_fields_required(self):
        from lumen.core.workspace import BrandingConfig

        with self.assertRaises(ValidationError):
            BrandingConfig()  # missing all fields

    def test_missing_one_field(self):
        from lumen.core.workspace import BrandingConfig

        with self.assertRaises(ValidationError):
            BrandingConfig(logo="x", primary_color="#000")  # missing app_name


class TestAdminRecord(unittest.TestCase):
    def test_valid_admin(self):
        from lumen.core.workspace import AdminRecord

        a = AdminRecord(
            email="admin@acme.com",
            display_name="Admin User",
            pin_hash="$2b$12$hashvalue",
        )
        assert a.email == "admin@acme.com"
        assert a.display_name == "Admin User"
        assert a.pin_hash == "$2b$12$hashvalue"

    def test_invalid_email_rejected(self):
        from lumen.core.workspace import AdminRecord

        with self.assertRaises(ValidationError):
            AdminRecord(
                email="not-an-email",
                display_name="Bad",
                pin_hash="$2b$12$hashvalue",
            )

    def test_missing_required_field(self):
        from lumen.core.workspace import AdminRecord

        with self.assertRaises(ValidationError):
            AdminRecord(email="admin@acme.com", display_name="Admin")


class TestUserRecord(unittest.TestCase):
    def test_valid_member(self):
        from lumen.core.workspace import UserRecord

        u = UserRecord(
            email="alice@acme.com",
            role="member",
            display_name="Alice",
            pin_hash="$2b$12$hashvalue",
        )
        assert u.role == "member"
        assert u.email == "alice@acme.com"

    def test_valid_viewer(self):
        from lumen.core.workspace import UserRecord

        u = UserRecord(
            email="bob@acme.com",
            role="viewer",
            display_name="Bob",
            pin_hash="$2b$12$hashvalue",
        )
        assert u.role == "viewer"

    def test_valid_team_admin(self):
        from lumen.core.workspace import UserRecord

        u = UserRecord(
            email="carol@acme.com",
            role="team_admin",
            display_name="Carol",
            pin_hash="$2b$12$hashvalue",
        )
        assert u.role == "team_admin"

    def test_invalid_role_rejected(self):
        from lumen.core.workspace import UserRecord

        with self.assertRaises(ValidationError):
            UserRecord(
                email="bad@acme.com",
                role="superadmin",
                display_name="Bad",
                pin_hash="$2b$12$hashvalue",
            )

    def test_invalid_email_rejected(self):
        from lumen.core.workspace import UserRecord

        with self.assertRaises(ValidationError):
            UserRecord(
                email="invalid",
                role="member",
                display_name="Bad",
                pin_hash="$2b$12$hashvalue",
            )


class TestTeamConfig(unittest.TestCase):
    def test_valid_team(self):
        from lumen.core.workspace import TeamConfig, UserRecord

        t = TeamConfig(
            name="marketing",
            display_name="Marketing Team",
            enabled_skills=["web-search", "email-draft"],
            users=[
                UserRecord(
                    email="alice@acme.com",
                    role="team_admin",
                    display_name="Alice",
                    pin_hash="$2b$12$hash",
                ),
            ],
        )
        assert t.name == "marketing"
        assert len(t.enabled_skills) == 2
        assert len(t.users) == 1

    def test_empty_skills_allowed(self):
        from lumen.core.workspace import TeamConfig

        t = TeamConfig(
            name="empty-team",
            display_name="Empty",
            enabled_skills=[],
            users=[],
        )
        assert t.enabled_skills == []
        assert t.users == []

    def test_empty_users_allowed(self):
        from lumen.core.workspace import TeamConfig

        t = TeamConfig(
            name="no-users",
            display_name="No Users",
            enabled_skills=["skill-1"],
            users=[],
        )
        assert t.users == []


class TestWorkspaceConfig(unittest.TestCase):
    def test_valid_workspace(self):
        from lumen.core.workspace import (
            AdminRecord,
            BrandingConfig,
            WorkspaceConfig,
        )

        w = WorkspaceConfig(
            name="acme",
            display_name="Acme Corp",
            branding=BrandingConfig(
                logo="logo.png", primary_color="#000", app_name="Acme"
            ),
            admins=[
                AdminRecord(
                    email="admin@acme.com",
                    display_name="Admin",
                    pin_hash="$2b$12$hash",
                )
            ],
        )
        assert w.name == "acme"
        assert len(w.admins) == 1

    def test_empty_admins_allowed(self):
        from lumen.core.workspace import (
            BrandingConfig,
            WorkspaceConfig,
        )

        w = WorkspaceConfig(
            name="acme",
            display_name="Acme Corp",
            branding=BrandingConfig(
                logo="logo.png", primary_color="#000", app_name="Acme"
            ),
            admins=[],
        )
        assert w.admins == []


class TestWorkspaceIndex(unittest.TestCase):
    def _make_index(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        idx._users = {
            "admin@acme.com": {
                "role": "admin",
                "team": None,
                "display_name": "Admin",
                "enabled_skills": [],
                "pin_hash": "$2b$12$hash",
            },
            "alice@acme.com": {
                "role": "member",
                "team": "marketing",
                "display_name": "Alice",
                "enabled_skills": ["web-search"],
                "pin_hash": "$2b$12$hash",
            },
        }
        return idx

    def test_lookup_user_found(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        result = idx.lookup_user("alice@acme.com")
        assert result is not None
        assert result["role"] == "member"
        assert result["team"] == "marketing"

    def test_lookup_user_not_found(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        assert idx.lookup_user("nobody@acme.com") is None

    def test_get_enabled_skills(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        assert idx.get_enabled_skills("alice@acme.com") == ["web-search"]
        assert idx.get_enabled_skills("admin@acme.com") == []

    def test_get_enabled_skills_unknown_user(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        assert idx.get_enabled_skills("nobody@acme.com") == []

    def test_get_user_role(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        assert idx.get_user_role("admin@acme.com") == "admin"
        assert idx.get_user_role("alice@acme.com") == "member"
        assert idx.get_user_role("nobody@acme.com") is None

    def test_get_user_team(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        assert idx.get_user_team("admin@acme.com") is None
        assert idx.get_user_team("alice@acme.com") == "marketing"
        assert idx.get_user_team("nobody@acme.com") is None

    def test_is_workspace_mode_with_users(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = self._make_index()
        assert idx.is_workspace_mode() is True

    def test_is_workspace_mode_empty(self):
        from lumen.core.workspace import WorkspaceIndex

        idx = WorkspaceIndex()
        assert idx.is_workspace_mode() is False


class TestWorkspaceValidationError(unittest.TestCase):
    def test_exception_exists(self):
        from lumen.core.workspace import WorkspaceValidationError

        with self.assertRaises(WorkspaceValidationError):
            raise WorkspaceValidationError("test error")
