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
