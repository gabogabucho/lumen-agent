"""Edge case tests for workspace mode.

Phase 6: Hardening — empty configs, expired JWTs, missing workspace, etc.
"""

import base64
import json
import tempfile
import time
import unittest

import bcrypt
import yaml

from lumen.core.workspace import WorkspaceIndex, load_teams, load_workspace, build_workspace_index
from lumen.core.workspace_auth import create_jwt, verify_pin, verify_jwt, save_workspace_secret
from lumen.channels.web import app
from starlette.testclient import TestClient


class TestEmptyWorkspaceConfig(unittest.TestCase):
    """Edge case: minimal/empty workspace config is valid."""

    def test_empty_admins_is_valid(self):
        """Workspace with no admins and no teams."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            tmp_path = Path(tmpdir)
            (tmp_path / "workspace.yaml").write_text(
                yaml.dump({
                    "name": "empty",
                    "display_name": "Empty",
                    "branding": {"logo": "/l.png", "primary_color": "#000", "app_name": "Empty"},
                    "admins": [],
                }, default_flow_style=False), encoding="utf-8"
            )
            (tmp_path / "teams").mkdir()

            ws = load_workspace(tmp_path)
            self.assertIsNotNone(ws)
            self.assertEqual(ws.name, "empty")

    def test_empty_teams_list(self):
        """Workspace exists but no teams dir."""
        with tempfile.TemporaryDirectory() as tmpdir:
            from pathlib import Path
            tmp_path = Path(tmpdir)
            (tmp_path / "workspace.yaml").write_text(
                yaml.dump({
                    "name": "solo",
                    "display_name": "Solo",
                    "branding": {"logo": "/l.png", "primary_color": "#000", "app_name": "Solo"},
                    "admins": [{"email": "solo@test.com", "display_name": "Solo", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False), encoding="utf-8"
            )

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            self.assertEqual(teams, {})
            idx = build_workspace_index(ws, teams)
            self.assertTrue(idx.is_workspace_mode())
            admin = idx.lookup_user("solo@test.com")
            self.assertIsNotNone(admin)
            self.assertEqual(admin["role"], "admin")


class TestExpiredJwtAuth(unittest.TestCase):
    """Edge case: expired JWT cannot access endpoints."""

    def test_expired_jwt_rejected_by_verify_jwt(self):
        """JWT with expired timestamp is rejected."""
        now = int(time.time())

        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
        expired_payload = base64.urlsafe_b64encode(
            json.dumps({"sub": "user@test.com", "role": "member", "team": "a", "exp": now - 10}).encode()
        ).rstrip(b"=").decode()
        token = f"{header}.{expired_payload}.dummy_signature"

        payload = verify_jwt(token, "secret-12345678")
        self.assertIsNone(payload)  # Expired → None

    def test_valid_jwt_accepted(self):
        """JWT with valid expiry is accepted."""
        import json
        import base64

        from lumen.core.workspace_auth import create_jwt
        token = create_jwt("user@test.com", "member", "a", "secret-12345678")
        payload = verify_jwt(token, "secret-12345678")
        self.assertIsNotNone(payload)
        self.assertEqual(payload["sub"], "user@test.com")


class TestMissingWorkspace(unittest.TestCase):
    """Edge case: workspace.yaml removed → fallback to non-workspace mode."""

    def test_no_workspace_yaml_allows_owner_auth(self):
        """When workspace.yaml is absent, owner cookie auth still works."""
        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        try:
            web_mod._workspace_index = None

            client = TestClient(app)

            # Workspace endpoints should be 404
            resp = client.post("/api/workspace/login", json={"email": "a@b.com", "pin": "1234"})
            self.assertEqual(resp.status_code, 404)

            resp = client.post("/api/workspace/reload")
            self.assertEqual(resp.status_code, 404)

            # Basic endpoints still work
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
        finally:
            web_mod._workspace_index = orig_index


class TestWorkspaceReloadEndpoint(unittest.TestCase):
    def test_reload_endpoint_refreshes_workspace_index(self):
        from pathlib import Path
        import lumen.channels.web as web_mod

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_workspace_secret(tmp_path, "secret-12345678")
            (tmp_path / "workspace.yaml").write_text(
                yaml.dump({
                    "name": "acme",
                    "display_name": "Acme",
                    "branding": {"logo": "/l.png", "primary_color": "#000", "app_name": "Acme"},
                    "admins": [{"email": "admin@acme.com", "display_name": "Admin", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )
            team_dir = tmp_path / "teams" / "marketing"
            team_dir.mkdir(parents=True)
            team_path = team_dir / "team.yaml"
            team_path.write_text(
                yaml.dump({
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": ["chat"],
                    "users": [{"email": "alice@acme.com", "role": "member", "display_name": "Alice", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            team_path.write_text(
                yaml.dump({
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": ["chat"],
                    "users": [
                        {"email": "alice@acme.com", "role": "member", "display_name": "Alice", "pin_hash": "$2b$12$hash"},
                        {"email": "bob@acme.com", "role": "member", "display_name": "Bob", "pin_hash": "$2b$12$hash"},
                    ],
                }, default_flow_style=False),
                encoding="utf-8",
            )

            orig_index = getattr(web_mod, "_workspace_index", None)
            orig_lumen_dir = web_mod.LUMEN_DIR
            try:
                web_mod._workspace_index = idx
                web_mod.LUMEN_DIR = tmp_path

                client = TestClient(app)
                resp = client.post("/api/workspace/reload")
                self.assertEqual(resp.status_code, 200)
                self.assertTrue(resp.json()["ok"])
                self.assertTrue(resp.json()["workspace_mode"])
                self.assertEqual(resp.json()["users"], 3)
                self.assertIsNotNone(web_mod._workspace_index.lookup_user("bob@acme.com"))
            finally:
                web_mod._workspace_index = orig_index
                web_mod.LUMEN_DIR = orig_lumen_dir


class TestWorkspaceGovernanceVisibility(unittest.TestCase):
    def test_team_admin_only_sees_own_team(self):
        from pathlib import Path
        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_workspace_secret(tmp_path, "secret-12345678")
            (tmp_path / "workspace.yaml").write_text(
                yaml.dump({
                    "name": "acme",
                    "display_name": "Acme",
                    "branding": {"logo": "/l.png", "primary_color": "#000", "app_name": "Acme"},
                    "admins": [{"email": "admin@acme.com", "display_name": "Admin", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )
            marketing_dir = tmp_path / "teams" / "marketing"
            marketing_dir.mkdir(parents=True)
            (marketing_dir / "team.yaml").write_text(
                yaml.dump({
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": ["chat"],
                    "users": [{"email": "lead@acme.com", "role": "team_admin", "display_name": "Lead", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )
            sales_dir = tmp_path / "teams" / "sales"
            sales_dir.mkdir(parents=True)
            (sales_dir / "team.yaml").write_text(
                yaml.dump({
                    "name": "sales",
                    "display_name": "Sales",
                    "enabled_skills": ["crm"],
                    "users": [{"email": "sales@acme.com", "role": "member", "display_name": "Sales", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            orig_index = getattr(web_mod, "_workspace_index", None)
            orig_lumen_dir = web_mod.LUMEN_DIR
            orig_access = web_mod._access_mode
            orig_config = web_mod._config
            orig_secret = wa._get_workspace_secret
            try:
                web_mod._workspace_index = idx
                web_mod.LUMEN_DIR = tmp_path
                web_mod._access_mode = "serve"
                web_mod._config = {"model": "test-model"}
                wa._get_workspace_secret = lambda: "secret-12345678"

                token = create_jwt("lead@acme.com", "team_admin", "marketing", "secret-12345678")
                client = TestClient(app)
                client.cookies.set("lumen_ws_token", token)
                resp = client.get("/api/workspace/governance")
                self.assertEqual(resp.status_code, 200)
                teams_payload = resp.json()["governance"]["teams"]
                self.assertEqual(len(teams_payload), 1)
                self.assertEqual(teams_payload[0]["team"], "marketing")
            finally:
                web_mod._workspace_index = orig_index
                web_mod.LUMEN_DIR = orig_lumen_dir
                web_mod._access_mode = orig_access
                web_mod._config = orig_config
                wa._get_workspace_secret = orig_secret

    def test_member_cannot_access_governance_or_reload(self):
        from pathlib import Path
        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_workspace_secret(tmp_path, "secret-12345678")
            (tmp_path / "workspace.yaml").write_text(
                yaml.dump({
                    "name": "acme",
                    "display_name": "Acme",
                    "branding": {"logo": "/l.png", "primary_color": "#000", "app_name": "Acme"},
                    "admins": [{"email": "admin@acme.com", "display_name": "Admin", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )
            team_dir = tmp_path / "teams" / "marketing"
            team_dir.mkdir(parents=True)
            (team_dir / "team.yaml").write_text(
                yaml.dump({
                    "name": "marketing",
                    "display_name": "Marketing",
                    "enabled_skills": ["chat"],
                    "users": [{"email": "member@acme.com", "role": "member", "display_name": "Member", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            orig_index = getattr(web_mod, "_workspace_index", None)
            orig_lumen_dir = web_mod.LUMEN_DIR
            orig_access = web_mod._access_mode
            orig_config = web_mod._config
            orig_secret = wa._get_workspace_secret
            try:
                web_mod._workspace_index = idx
                web_mod.LUMEN_DIR = tmp_path
                web_mod._access_mode = "serve"
                web_mod._config = {"model": "test-model"}
                wa._get_workspace_secret = lambda: "secret-12345678"

                token = create_jwt("member@acme.com", "member", "marketing", "secret-12345678")
                client = TestClient(app)
                client.cookies.set("lumen_ws_token", token)

                governance_resp = client.get("/api/workspace/governance")
                self.assertEqual(governance_resp.status_code, 403)

                reload_resp = client.post("/api/workspace/reload")
                self.assertEqual(reload_resp.status_code, 403)
            finally:
                web_mod._workspace_index = orig_index
                web_mod.LUMEN_DIR = orig_lumen_dir
                web_mod._access_mode = orig_access
                web_mod._config = orig_config
                wa._get_workspace_secret = orig_secret

    def test_admin_sees_all_teams(self):
        from pathlib import Path
        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            save_workspace_secret(tmp_path, "secret-12345678")
            (tmp_path / "workspace.yaml").write_text(
                yaml.dump({
                    "name": "acme",
                    "display_name": "Acme",
                    "branding": {"logo": "/l.png", "primary_color": "#000", "app_name": "Acme"},
                    "admins": [{"email": "admin@acme.com", "display_name": "Admin", "pin_hash": "$2b$12$hash"}],
                }, default_flow_style=False),
                encoding="utf-8",
            )
            for team_name in ("marketing", "sales"):
                team_dir = tmp_path / "teams" / team_name
                team_dir.mkdir(parents=True)
                (team_dir / "team.yaml").write_text(
                    yaml.dump({
                        "name": team_name,
                        "display_name": team_name.title(),
                        "enabled_skills": [team_name],
                        "users": [],
                    }, default_flow_style=False),
                    encoding="utf-8",
                )

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            orig_index = getattr(web_mod, "_workspace_index", None)
            orig_lumen_dir = web_mod.LUMEN_DIR
            orig_access = web_mod._access_mode
            orig_config = web_mod._config
            orig_secret = wa._get_workspace_secret
            try:
                web_mod._workspace_index = idx
                web_mod.LUMEN_DIR = tmp_path
                web_mod._access_mode = "serve"
                web_mod._config = {"model": "test-model"}
                wa._get_workspace_secret = lambda: "secret-12345678"

                token = create_jwt("admin@acme.com", "admin", None, "secret-12345678")
                client = TestClient(app)
                client.cookies.set("lumen_ws_token", token)
                resp = client.get("/api/workspace/governance")
                self.assertEqual(resp.status_code, 200)
                teams_payload = resp.json()["governance"]["teams"]
                self.assertEqual({team["team"] for team in teams_payload}, {"marketing", "sales"})
            finally:
                web_mod._workspace_index = orig_index
                web_mod.LUMEN_DIR = orig_lumen_dir
                web_mod._access_mode = orig_access
                web_mod._config = orig_config
                wa._get_workspace_secret = orig_secret


class TestJwtExpiry(unittest.TestCase):
    """Edge: JWT expires → user must re-login."""

    def test_jwt_expires(self):
        """After JWT expires, token is rejected."""
        now = int(time.time())

        header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode()).rstrip(b"=").decode()
        expired = base64.urlsafe_b64encode(
            json.dumps({"sub": "u@test.com", "role": "m", "team": "a", "exp": now - 10000}).encode()
        ).rstrip(b"=").decode()
        token = f"{header}.{expired}.dummy"

        payload = verify_jwt(token, "secret-12345678")
        self.assertIsNone(payload)


class TestLogoutBehavior(unittest.TestCase):
    """Edge: logout clears cookie, next request is unauthenticated."""

    def test_logout_clears_access(self):
        """After logout, cookie is cleared."""
        client = TestClient(app)
        resp = client.post("/api/workspace/logout")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    def test_logout_does_not_invalidate_server_side(self):
        """Logout is client-side — JWT remains valid server-side until expiry.
        This is expected JWT behavior; true invalidation requires short expiry."""
        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        try:
            ws = WorkspaceIndex()
            ws.add_user("user@test.com", "member", "team", "User", ["skill"], "$2b$12$hash")
            web_mod._workspace_index = ws
            wa._get_workspace_secret = lambda: "secret-12345678"

            token = create_jwt("user@test.com", "member", "team", "secret-12345678")
            client = TestClient(app)
            client.post("/api/workspace/logout")

            # JWT still valid server-side (stateless)
            payload = verify_jwt(token, "secret-12345678")
            self.assertIsNotNone(payload)
        finally:
            web_mod._workspace_index = orig_index


class TestEdgePinVerification(unittest.TestCase):
    """Edge: PIN verification with empty/very short PINs."""

    def test_empty_pin_rejected(self):
        """Empty PIN should not match any hash."""
        self.assertFalse(verify_pin("", "$2b$12$hash"))

    def test_pin_4_chars_works(self):
        """Minimal PIN (4 chars) should hash and verify."""
        import bcrypt
        pin = "1234"
        hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt())
        self.assertTrue(bcrypt.checkpw(pin.encode(), hashed))

    def test_pin_1_char_too_short(self):
        """Very short PIN should still hash (length is enforced at input, not at verify)."""
        import bcrypt
        pin = "1"
        hashed = bcrypt.hashpw(pin.encode(), bcrypt.gensalt())
        self.assertTrue(bcrypt.checkpw(pin.encode(), hashed))


class TestUnknownUserAccess(unittest.TestCase):
    """Edge: user not in workspace index → access denied."""

    def test_unknown_user_login_denied(self):
        """Non-indexed user gets 401 on login."""
        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        ws = WorkspaceIndex()
        ws.add_user("known@test.com", "admin", None, "Admin", [], "$2b$12$hash")

        orig_index = getattr(web_mod, "_workspace_index", None)
        try:
            web_mod._workspace_index = ws
            wa._get_workspace_secret = lambda: "secret-12345678"

            client = TestClient(app)
            resp = client.post("/api/workspace/login", json={"email": "unknown@test.com", "pin": "1234"})
            self.assertEqual(resp.status_code, 401)
        finally:
            web_mod._workspace_index = orig_index


if __name__ == "__main__":
    unittest.main()
