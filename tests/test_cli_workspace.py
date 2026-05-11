"""Tests for CLI workspace wizard — Phase 4.

Tests follow TDD: RED → GREEN → TRIANGULATE → REFACTOR.
"""

import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml


class TestWorkspaceAppImport(unittest.TestCase):
    """Verify workspace_app can be imported and has correct structure."""

    def test_import_workspace_app(self):
        from lumen.cli.workspace import workspace_app

        assert workspace_app is not None

    def test_workspace_app_has_init_command(self):
        from lumen.cli.workspace import workspace_app

        # Typer app stores registered commands
        assert any(
            hasattr(cmd, "callback") and "init" in (cmd.callback.__name__ if hasattr(cmd, "callback") else "")
            for cmd in workspace_app.registered_commands
        )

    def test_workspace_app_registered_in_main(self):
        """Verify workspace_app is registered as a sub-command in main app."""
        from lumen.cli.main import app

        # Check that 'workspace' is a registered sub-typer
        group_names = [g.name for g in app.registered_groups]
        assert "workspace" in group_names


# ── Prompt responses helper ────────────────────────────────────────

# Maps exact prompt message → response key
PROMPT_KEY_MAP = {
    "Nombre del workspace (slug, ej: mi-empresa)": "company_slug",
    "Nombre para mostrar (ej: Mi Empresa)": "company_display",
    "Email del administrador": "admin_email",
    "Nombre del administrador": "admin_name",
    "PIN del administrador (mínimo 4 caracteres)": "admin_pin",
    "Confirmar PIN": "admin_pin_confirm",
    "¿Cuántos equipos querés crear ahora?": "team_count",
}

# Team prompt patterns (partial match) → key suffix
TEAM_PROMPT_PATTERNS = [
    ("Slug del equipo", "_slug"),
    ("Nombre del equipo", "_display"),
    ("Email del administrador del equipo", "_admin_email"),
    ("Nombre del admin del equipo", "_admin_name"),
    ("PIN del admin del equipo (mínimo 4 caracteres)", "_admin_pin"),
    ("Confirmar PIN del admin del equipo", "_admin_pin_confirm"),
]


def _make_prompt_side_effect(responses: dict):
    """Create a side_effect function for _prompt that maps messages to responses.

    Uses an ordered queue approach: company prompts first, then team prompts
    in order, matching by exact message or partial pattern.
    """
    team_queue = []  # list of remaining team response values

    def build_team_queue():
        """Pre-build ordered list of team responses."""
        nonlocal team_queue
        team_count = int(responses.get("team_count", "0"))
        team_queue = []
        for i in range(team_count):
            for _, suffix in TEAM_PROMPT_PATTERNS:
                key = f"team_{i}{suffix}"
                if key in responses:
                    team_queue.append(responses[key])

    build_team_queue()

    def side_effect(message, **kwargs):
        # Check company prompts by exact match
        for msg, key in PROMPT_KEY_MAP.items():
            if msg == message and key in responses:
                # If this is the team_count prompt, build team queue
                if key == "team_count":
                    build_team_queue()
                return responses[key]

        # Check team prompts by partial pattern
        for pattern, _ in TEAM_PROMPT_PATTERNS:
            if pattern in message and team_queue:
                return team_queue.pop(0)

        return None

    return side_effect


class TestWorkspaceInitCompanySetup(unittest.TestCase):
    """Tests for the company setup portion of workspace init."""

    def _make_tmp(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory()

    def test_init_creates_workspace_yaml(self):
        """workspace init creates workspace.yaml with correct structure."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin User",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            ws_path = Path(tmp.name) / "workspace.yaml"
            assert ws_path.exists()

            data = yaml.safe_load(ws_path.read_text(encoding="utf-8"))
            assert data["name"] == "acme"
            assert data["display_name"] == "Acme Corp"
            assert len(data["admins"]) == 1
            assert data["admins"][0]["email"] == "admin@acme.com"
            assert data["admins"][0]["display_name"] == "Admin User"
        finally:
            tmp.cleanup()

    def test_init_pin_is_hashed(self):
        """PIN stored in workspace.yaml must be bcrypt-hashed, not plaintext."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "mysecret",
                "admin_pin_confirm": "mysecret",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            ws_path = Path(tmp.name) / "workspace.yaml"
            data = yaml.safe_load(ws_path.read_text(encoding="utf-8"))
            pin_hash = data["admins"][0]["pin_hash"]

            # Must NOT be plaintext
            assert pin_hash != "mysecret"
            # Must be bcrypt hash (starts with $2b$)
            assert pin_hash.startswith("$2b$")
        finally:
            tmp.cleanup()

    def test_init_creates_workspace_secret(self):
        """workspace init creates workspace_secret file."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            secret_path = Path(tmp.name) / "workspace_secret"
            assert secret_path.exists()
            secret = secret_path.read_text(encoding="utf-8").strip()
            # Should be a base64-encoded 256-bit key (32 bytes = 44 chars base64)
            assert len(secret) >= 32
        finally:
            tmp.cleanup()

    def test_init_creates_teams_directory(self):
        """workspace init creates teams/ directory."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            teams_dir = Path(tmp.name) / "teams"
            assert teams_dir.is_dir()
        finally:
            tmp.cleanup()

    def test_init_validates_email_format(self):
        """Invalid email format should raise ValueError."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "not-an-email",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                with self.assertRaises(ValueError):
                    run_workspace_init(Path(tmp.name))
        finally:
            tmp.cleanup()

    def test_init_validates_pin_minimum_length(self):
        """PIN shorter than 4 chars should raise ValueError."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "12",
                "admin_pin_confirm": "12",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                with self.assertRaises(ValueError):
                    run_workspace_init(Path(tmp.name))
        finally:
            tmp.cleanup()

    def test_init_validates_pin_confirmation_match(self):
        """PIN confirmation mismatch should raise ValueError."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "4321",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                with self.assertRaises(ValueError):
                    run_workspace_init(Path(tmp.name))
        finally:
            tmp.cleanup()


class TestWorkspaceInitTeamSetup(unittest.TestCase):
    """Tests for the team setup portion of workspace init."""

    def _make_tmp(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory()

    def _base_responses(self) -> dict:
        return {
            "company_slug": "acme",
            "company_display": "Acme Corp",
            "admin_email": "admin@acme.com",
            "admin_name": "Admin User",
            "admin_pin": "1234",
            "admin_pin_confirm": "1234",
            "team_count": "0",
        }

    def test_zero_teams(self):
        """Zero teams → no team directories created."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = self._base_responses()
            responses["team_count"] = "0"

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            teams_dir = Path(tmp.name) / "teams"
            assert teams_dir.is_dir()
            # No subdirectories
            assert len(list(teams_dir.iterdir())) == 0
        finally:
            tmp.cleanup()

    def test_single_team_creates_team_yaml(self):
        """Single team creates teams/{slug}/team.yaml."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "1",
                "team_0_slug": "marketing",
                "team_0_display": "Marketing Team",
                "team_0_admin_email": "marketo@acme.com",
                "team_0_admin_name": "Marketo",
                "team_0_admin_pin": "1234",
                "team_0_admin_pin_confirm": "1234",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            team_yaml = Path(tmp.name) / "teams" / "marketing" / "team.yaml"
            assert team_yaml.exists()

            data = yaml.safe_load(team_yaml.read_text(encoding="utf-8"))
            assert data["name"] == "marketing"
            assert data["display_name"] == "Marketing Team"
            assert len(data["users"]) == 1
            assert data["users"][0]["email"] == "marketo@acme.com"
            assert data["users"][0]["role"] == "team_admin"
            # PIN should be hashed
            assert data["users"][0]["pin_hash"].startswith("$2b$")
        finally:
            tmp.cleanup()

    def test_multiple_teams(self):
        """Two teams create two team.yaml files."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "2",
                "team_0_slug": "marketing",
                "team_0_display": "Marketing",
                "team_0_admin_email": "mark@acme.com",
                "team_0_admin_name": "Mark",
                "team_0_admin_pin": "1234",
                "team_0_admin_pin_confirm": "1234",
                "team_1_slug": "engineering",
                "team_1_display": "Engineering",
                "team_1_admin_email": "eng@acme.com",
                "team_1_admin_name": "Engineer",
                "team_1_admin_pin": "1234",
                "team_1_admin_pin_confirm": "1234",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name))

            assert (Path(tmp.name) / "teams" / "marketing" / "team.yaml").exists()
            assert (Path(tmp.name) / "teams" / "engineering" / "team.yaml").exists()
        finally:
            tmp.cleanup()

    def test_duplicate_email_across_teams_rejected(self):
        """Duplicate email between admin and team admin raises ValueError."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            # Team admin uses same email as workspace admin
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "1",
                "team_0_slug": "marketing",
                "team_0_display": "Marketing",
                "team_0_admin_email": "admin@acme.com",  # duplicate!
                "team_0_admin_name": "Dup",
                "team_0_admin_pin": "1234",
                "team_0_admin_pin_confirm": "1234",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                with self.assertRaises(ValueError) as ctx:
                    run_workspace_init(Path(tmp.name))
                assert "duplicado" in str(ctx.exception).lower() or "duplicate" in str(ctx.exception).lower()
        finally:
            tmp.cleanup()

    def test_invalid_slug_rejected(self):
        """Invalid slug (spaces, uppercase) raises ValueError."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "ACME Corp",  # invalid slug
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                with self.assertRaises(ValueError):
                    run_workspace_init(Path(tmp.name))
        finally:
            tmp.cleanup()


class TestWorkspaceInitGuard(unittest.TestCase):
    """Tests for existing workspace guard (--force flag)."""

    def _make_tmp(self) -> tempfile.TemporaryDirectory:
        return tempfile.TemporaryDirectory()

    def test_existing_workspace_without_force_raises(self):
        """If workspace.yaml exists and no --force, raises FileExistsError."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            # Pre-create workspace.yaml
            (Path(tmp.name) / "workspace.yaml").write_text("name: old\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                run_workspace_init(Path(tmp.name), force=False)
        finally:
            tmp.cleanup()

    def test_existing_workspace_with_force_backs_up(self):
        """With --force, existing workspace.yaml is backed up before overwrite."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            # Pre-create workspace.yaml
            (Path(tmp.name) / "workspace.yaml").write_text("name: old\n", encoding="utf-8")

            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                run_workspace_init(Path(tmp.name), force=True)

            # Backup should exist
            backups = list(Path(tmp.name).glob("workspace.yaml.bak*"))
            assert len(backups) >= 1

            # workspace.yaml should have new content
            data = yaml.safe_load((Path(tmp.name) / "workspace.yaml").read_text(encoding="utf-8"))
            assert data["name"] == "acme"
        finally:
            tmp.cleanup()

    def test_no_existing_workspace_allows_init(self):
        """No workspace.yaml → init proceeds normally."""
        from lumen.cli.workspace import run_workspace_init

        tmp = self._make_tmp()
        try:
            responses = {
                "company_slug": "acme",
                "company_display": "Acme Corp",
                "admin_email": "admin@acme.com",
                "admin_name": "Admin",
                "admin_pin": "1234",
                "admin_pin_confirm": "1234",
                "team_count": "0",
            }

            with patch("lumen.cli.workspace._prompt", side_effect=_make_prompt_side_effect(responses)):
                result = run_workspace_init(Path(tmp.name), force=False)
                assert result is not None
        finally:
            tmp.cleanup()


class TestHashPin(unittest.TestCase):
    """Tests for hash_pin utility."""

    def test_hash_pin_returns_bcrypt_hash(self):
        from lumen.core.workspace_auth import hash_pin

        hashed = hash_pin("1234")
        assert hashed.startswith("$2b$")
        assert hashed != "1234"

    def test_hash_pin_different_inputs_different_hashes(self):
        from lumen.core.workspace_auth import hash_pin

        h1 = hash_pin("1234")
        h2 = hash_pin("5678")
        assert h1 != h2


class TestGenerateAndSaveWorkspaceSecret(unittest.TestCase):
    """Tests for workspace secret generation and saving."""

    def test_generate_workspace_secret_returns_string(self):
        from lumen.core.workspace_auth import generate_workspace_secret

        secret = generate_workspace_secret()
        assert isinstance(secret, str)
        assert len(secret) == 64  # 256 bits hex-encoded

    def test_save_workspace_secret_creates_file(self):
        from lumen.core.workspace_auth import generate_workspace_secret, save_workspace_secret

        tmp = tempfile.TemporaryDirectory()
        try:
            d = Path(tmp.name)
            secret = generate_workspace_secret()
            save_workspace_secret(d, secret)

            secret_path = d / "workspace_secret"
            assert secret_path.exists()
            content = secret_path.read_text(encoding="utf-8").strip()
            assert content == secret
        finally:
            tmp.cleanup()

    def test_generate_workspace_secret_is_unique(self):
        from lumen.core.workspace_auth import generate_workspace_secret

        s1 = generate_workspace_secret()
        s2 = generate_workspace_secret()
        assert s1 != s2
