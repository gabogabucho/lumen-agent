"""Tests for workspace auth: JWT secrets, tokens, PIN verification, middleware, endpoints."""

import unittest
from pathlib import Path


# ── Task 2.1: JWT secret management ─────────────────────────────────────


class TestGenerateWorkspaceSecret(unittest.TestCase):
    def test_returns_string(self):
        from lumen.core.workspace_auth import generate_workspace_secret

        secret = generate_workspace_secret()
        assert isinstance(secret, str)
        assert len(secret) > 0

    def test_generates_different_secrets(self):
        from lumen.core.workspace_auth import generate_workspace_secret

        s1 = generate_workspace_secret()
        s2 = generate_workspace_secret()
        assert s1 != s2

    def test_secret_is_long_enough(self):
        from lumen.core.workspace_auth import generate_workspace_secret

        secret = generate_workspace_secret()
        # 256-bit → 64 hex chars
        assert len(secret) >= 32


class TestSaveAndLoadWorkspaceSecret(unittest.TestCase):
    def test_save_and_load_roundtrip(self):
        import tempfile

        from lumen.core.workspace_auth import (
            load_workspace_secret,
            save_workspace_secret,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            save_workspace_secret(workspace_dir, "my-secret-123")
            loaded = load_workspace_secret(workspace_dir)
            assert loaded == "my-secret-123"

    def test_load_missing_returns_none(self):
        import tempfile

        from lumen.core.workspace_auth import load_workspace_secret

        with tempfile.TemporaryDirectory() as tmpdir:
            loaded = load_workspace_secret(Path(tmpdir))
            assert loaded is None

    def test_save_creates_file(self):
        import tempfile

        from lumen.core.workspace_auth import save_workspace_secret

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            save_workspace_secret(workspace_dir, "test-secret")
            secret_file = workspace_dir / "workspace_secret"
            assert secret_file.exists()

    def test_save_overwrites_existing(self):
        import tempfile

        from lumen.core.workspace_auth import (
            load_workspace_secret,
            save_workspace_secret,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            save_workspace_secret(workspace_dir, "first")
            save_workspace_secret(workspace_dir, "second")
            assert load_workspace_secret(workspace_dir) == "second"


class TestGetOrCreateWorkspaceSecret(unittest.TestCase):
    def test_creates_on_first_call(self):
        import tempfile

        from lumen.core.workspace_auth import get_or_create_workspace_secret

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            secret = get_or_create_workspace_secret(workspace_dir)
            assert isinstance(secret, str)
            assert len(secret) > 0

    def test_reuses_existing(self):
        import tempfile

        from lumen.core.workspace_auth import get_or_create_workspace_secret

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            s1 = get_or_create_workspace_secret(workspace_dir)
            s2 = get_or_create_workspace_secret(workspace_dir)
            assert s1 == s2

    def test_file_created_after_call(self):
        import tempfile

        from lumen.core.workspace_auth import get_or_create_workspace_secret

        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_dir = Path(tmpdir)
            get_or_create_workspace_secret(workspace_dir)
            secret_file = workspace_dir / "workspace_secret"
            assert secret_file.exists()


# ── Task 2.2: JWT token operations ──────────────────────────────────────


class TestCreateAndVerifyJWT(unittest.TestCase):
    def test_roundtrip(self):
        from lumen.core.workspace_auth import create_jwt, verify_jwt

        secret = "test-secret-key-123"
        token = create_jwt("alice@example.com", "admin", None, secret)
        payload = verify_jwt(token, secret)
        assert payload is not None
        assert payload["sub"] == "alice@example.com"
        assert payload["role"] == "admin"
        assert payload["team"] is None

    def test_payload_contains_team(self):
        from lumen.core.workspace_auth import create_jwt, verify_jwt

        secret = "test-secret-key-123"
        token = create_jwt("bob@example.com", "member", "marketing", secret)
        payload = verify_jwt(token, secret)
        assert payload["team"] == "marketing"

    def test_expired_token_returns_none(self):
        from lumen.core.workspace_auth import create_jwt, verify_jwt

        secret = "test-secret-key-123"
        token = create_jwt(
            "alice@example.com", "admin", None, secret, expires_hours=-1
        )
        payload = verify_jwt(token, secret)
        assert payload is None

    def test_tampered_token_returns_none(self):
        from lumen.core.workspace_auth import create_jwt, verify_jwt

        secret = "test-secret-key-123"
        token = create_jwt("alice@example.com", "admin", None, secret)
        tampered = token[:-5] + "XXXXX"
        payload = verify_jwt(tampered, secret)
        assert payload is None

    def test_wrong_secret_returns_none(self):
        from lumen.core.workspace_auth import create_jwt, verify_jwt

        token = create_jwt("alice@example.com", "admin", None, "secret-a")
        payload = verify_jwt(token, "secret-b")
        assert payload is None

    def test_empty_token_returns_none(self):
        from lumen.core.workspace_auth import verify_jwt

        payload = verify_jwt("", "some-secret")
        assert payload is None

    def test_gibberish_token_returns_none(self):
        from lumen.core.workspace_auth import verify_jwt

        payload = verify_jwt("not.a.valid.token", "some-secret")
        assert payload is None


# ── Task 2.3: PIN verification ──────────────────────────────────────────


class TestHashAndVerifyPin(unittest.TestCase):
    def test_correct_pin(self):
        from lumen.core.workspace_auth import hash_pin, verify_pin

        hashed = hash_pin("1234")
        assert verify_pin("1234", hashed) is True

    def test_wrong_pin(self):
        from lumen.core.workspace_auth import hash_pin, verify_pin

        hashed = hash_pin("1234")
        assert verify_pin("9999", hashed) is False

    def test_empty_pin(self):
        from lumen.core.workspace_auth import hash_pin, verify_pin

        hashed = hash_pin("1234")
        assert verify_pin("", hashed) is False

    def test_different_pins_different_hashes(self):
        from lumen.core.workspace_auth import hash_pin

        h1 = hash_pin("1234")
        h2 = hash_pin("5678")
        assert h1 != h2

    def test_hash_format_is_bcrypt(self):
        from lumen.core.workspace_auth import hash_pin

        hashed = hash_pin("test")
        assert hashed.startswith("$2")


# ── Task 2.4: Auth middleware (FastAPI dependencies) ────────────────────


class TestWorkspaceUserModel(unittest.TestCase):
    def test_model_creation(self):
        from lumen.core.workspace_auth import WorkspaceUser

        user = WorkspaceUser(
            email="alice@example.com",
            role="admin",
            team=None,
            display_name="Alice",
        )
        assert user.email == "alice@example.com"
        assert user.role == "admin"
        assert user.team is None
        assert user.display_name == "Alice"

    def test_model_with_team(self):
        from lumen.core.workspace_auth import WorkspaceUser

        user = WorkspaceUser(
            email="bob@example.com",
            role="member",
            team="marketing",
            display_name="Bob",
        )
        assert user.team == "marketing"


def _make_index():
    """Create a test WorkspaceIndex with sample users."""
    from lumen.core.workspace import WorkspaceIndex

    idx = WorkspaceIndex()
    idx.add_user("alice@example.com", "member", "marketing", "Alice", [], "$2b$12$hash")
    return idx


class TestGetWorkspaceUser(unittest.TestCase):
    """Tests for the get_workspace_user FastAPI dependency."""

    def test_no_workspace_mode_returns_none(self):
        """In non-workspace mode, get_workspace_user returns None."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import get_workspace_user

        import lumen.core.workspace_auth as wa

        app = FastAPI()
        orig_index = wa._get_workspace_index
        wa._get_workspace_index = lambda: None

        @app.get("/test")
        async def test_endpoint(user=Depends(get_workspace_user)):
            if user is None:
                return {"auth": False}
            return {"auth": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json() == {"auth": False}

        wa._get_workspace_index = orig_index

    def test_valid_jwt_returns_user(self):
        """Valid JWT cookie returns WorkspaceUser."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import (
            create_jwt,
            get_workspace_user,
        )

        idx = _make_index()
        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        orig_secret = wa._get_workspace_secret
        wa._get_workspace_index = lambda: idx
        wa._get_workspace_secret = lambda: "test-secret"

        @app.get("/test")
        async def test_endpoint(user=Depends(get_workspace_user)):
            if user is None:
                return {"auth": False}
            return {"auth": True, "email": user.email, "role": user.role, "team": user.team}

        token = create_jwt("alice@example.com", "member", "marketing", "test-secret")

        client = TestClient(app)
        client.cookies.set("lumen_ws_token", token)
        resp = client.get("/test")
        assert resp.status_code == 200
        data = resp.json()
        assert data["auth"] is True
        assert data["email"] == "alice@example.com"
        assert data["role"] == "member"
        assert data["team"] == "marketing"

        wa._get_workspace_index = orig_index
        wa._get_workspace_secret = orig_secret

    def test_no_token_returns_none(self):
        """No cookie in workspace mode returns None."""
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import get_workspace_user

        idx = _make_index()
        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        orig_secret = wa._get_workspace_secret
        wa._get_workspace_index = lambda: idx
        wa._get_workspace_secret = lambda: "test-secret"

        @app.get("/test")
        async def test_endpoint(user=Depends(get_workspace_user)):
            if user is None:
                return {"auth": False}
            return {"auth": True}

        client = TestClient(app)
        resp = client.get("/test")
        assert resp.status_code == 200
        assert resp.json() == {"auth": False}

        wa._get_workspace_index = orig_index
        wa._get_workspace_secret = orig_secret


class TestRequireWorkspaceAuth(unittest.TestCase):
    def test_valid_token_passes(self):
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import (
            create_jwt,
            require_workspace_auth,
        )

        idx = _make_index()
        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        orig_secret = wa._get_workspace_secret
        wa._get_workspace_index = lambda: idx
        wa._get_workspace_secret = lambda: "test-secret"

        @app.get("/protected")
        async def protected(user=Depends(require_workspace_auth)):
            return {"email": user.email}

        token = create_jwt("alice@example.com", "member", "marketing", "test-secret")
        client = TestClient(app)
        client.cookies.set("lumen_ws_token", token)
        resp = client.get("/protected")
        assert resp.status_code == 200
        assert resp.json()["email"] == "alice@example.com"

        wa._get_workspace_index = orig_index
        wa._get_workspace_secret = orig_secret

    def test_no_token_returns_401(self):
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import require_workspace_auth

        idx = _make_index()
        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        orig_secret = wa._get_workspace_secret
        wa._get_workspace_index = lambda: idx
        wa._get_workspace_secret = lambda: "test-secret"

        @app.get("/protected")
        async def protected(user=Depends(require_workspace_auth)):
            return {"email": user.email}

        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 401

        wa._get_workspace_index = orig_index
        wa._get_workspace_secret = orig_secret

    def test_non_workspace_mode_passes(self):
        """In non-workspace mode, require_workspace_auth returns None (pass-through)."""
        from typing import Optional

        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import (
            WorkspaceUser,
            require_workspace_auth,
        )

        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        wa._get_workspace_index = lambda: None

        @app.get("/protected")
        async def protected(user: Optional[WorkspaceUser] = Depends(require_workspace_auth)):
            if user is None:
                return {"auth": "none"}
            return {"email": user.email}

        client = TestClient(app)
        resp = client.get("/protected")
        assert resp.status_code == 200
        assert resp.json()["auth"] == "none"

        wa._get_workspace_index = orig_index


class TestRequireWorkspaceAdmin(unittest.TestCase):
    def test_admin_passes(self):
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace import WorkspaceIndex
        from lumen.core.workspace_auth import (
            create_jwt,
            require_workspace_admin,
        )

        idx = WorkspaceIndex()
        idx.add_user("admin@example.com", "admin", None, "Admin", [], "$2b$12$hash")

        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        orig_secret = wa._get_workspace_secret
        wa._get_workspace_index = lambda: idx
        wa._get_workspace_secret = lambda: "test-secret"

        @app.get("/admin-only")
        async def admin_only(user=Depends(require_workspace_admin)):
            return {"email": user.email}

        token = create_jwt("admin@example.com", "admin", None, "test-secret")
        client = TestClient(app)
        client.cookies.set("lumen_ws_token", token)
        resp = client.get("/admin-only")
        assert resp.status_code == 200
        assert resp.json()["email"] == "admin@example.com"

        wa._get_workspace_index = orig_index
        wa._get_workspace_secret = orig_secret

    def test_member_gets_403(self):
        from fastapi import Depends, FastAPI
        from fastapi.testclient import TestClient
        from lumen.core.workspace_auth import (
            create_jwt,
            require_workspace_admin,
        )

        idx = _make_index()
        app = FastAPI()

        import lumen.core.workspace_auth as wa

        orig_index = wa._get_workspace_index
        orig_secret = wa._get_workspace_secret
        wa._get_workspace_index = lambda: idx
        wa._get_workspace_secret = lambda: "test-secret"

        @app.get("/admin-only")
        async def admin_only(user=Depends(require_workspace_admin)):
            return {"email": user.email}

        token = create_jwt("alice@example.com", "member", "marketing", "test-secret")
        client = TestClient(app)
        client.cookies.set("lumen_ws_token", token)
        resp = client.get("/admin-only")
        assert resp.status_code == 403

        wa._get_workspace_index = orig_index
        wa._get_workspace_secret = orig_secret


# ── Task 2.5: Login endpoint ────────────────────────────────────────────


class TestWorkspaceLoginEndpoint(unittest.TestCase):
    """Tests for POST /api/workspace/login."""

    def test_valid_login_returns_jwt(self):
        from lumen.core.workspace_auth import hash_pin
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        pin_hash = hash_pin("1234")
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], pin_hash)

        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_secret_loader = wa._get_workspace_secret

        web_mod._workspace_index = idx
        wa._get_workspace_secret = lambda: "test-secret"

        client = TestClient(app)
        resp = client.post(
            "/api/workspace/login",
            json={"email": "alice@example.com", "pin": "1234"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ok"] is True
        assert data["user"]["email"] == "alice@example.com"
        assert data["user"]["role"] == "member"
        assert data["user"]["team"] == "marketing"
        assert data["user"]["display_name"] == "Alice"
        assert "lumen_ws_token" in resp.cookies

        web_mod._workspace_index = orig_index
        wa._get_workspace_secret = orig_secret_loader

    def test_valid_login_sets_owner_cookie_when_server_secret_exists(self):
        from lumen.core.workspace_auth import hash_pin
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        pin_hash = hash_pin("1234")
        idx.add_user("alice@example.com", "team_admin", "marketing", "Alice", [], pin_hash)

        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_secret_loader = wa._get_workspace_secret
        orig_config = web_mod._config

        web_mod._workspace_index = idx
        web_mod._config = {"server_secret": "server-secret", "model": "test-model"}
        wa._get_workspace_secret = lambda: "test-secret"

        client = TestClient(app)
        resp = client.post(
            "/api/workspace/login",
            json={"email": "alice@example.com", "pin": "1234"},
        )
        assert resp.status_code == 200
        assert "lumen_ws_token" in resp.cookies
        assert "lumen_owner" in resp.cookies

        web_mod._workspace_index = orig_index
        web_mod._config = orig_config
        wa._get_workspace_secret = orig_secret_loader

    def test_wrong_pin_returns_401(self):
        from lumen.core.workspace_auth import hash_pin
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], hash_pin("1234"))

        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        web_mod._workspace_index = idx

        client = TestClient(app)
        resp = client.post(
            "/api/workspace/login",
            json={"email": "alice@example.com", "pin": "wrong"},
        )
        assert resp.status_code == 401

        web_mod._workspace_index = orig_index

    def test_unknown_email_returns_401(self):
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], "$2b$12$hash")

        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        web_mod._workspace_index = idx

        client = TestClient(app)
        resp = client.post(
            "/api/workspace/login",
            json={"email": "unknown@example.com", "pin": "1234"},
        )
        assert resp.status_code == 401

        web_mod._workspace_index = orig_index

    def test_no_workspace_returns_404(self):
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        web_mod._workspace_index = None

        client = TestClient(app)
        resp = client.post(
            "/api/workspace/login",
            json={"email": "alice@example.com", "pin": "1234"},
        )
        assert resp.status_code == 404

        web_mod._workspace_index = orig_index

    def test_401_does_not_reveal_email_existence(self):
        """Wrong PIN and unknown email should return same error message."""
        from lumen.core.workspace_auth import hash_pin
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], hash_pin("1234"))

        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        web_mod._workspace_index = idx

        client = TestClient(app)

        resp1 = client.post(
            "/api/workspace/login",
            json={"email": "alice@example.com", "pin": "wrong"},
        )
        resp2 = client.post(
            "/api/workspace/login",
            json={"email": "nonexistent@example.com", "pin": "1234"},
        )

        assert resp1.json() == resp2.json()

        web_mod._workspace_index = orig_index


# ── Task 2.7: Logout endpoint ───────────────────────────────────────────


class TestWorkspaceLogoutEndpoint(unittest.TestCase):
    def test_logout_clears_cookie(self):
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        client = TestClient(app)
        resp = client.post("/api/workspace/logout")
        assert resp.status_code == 200
        assert resp.json()["ok"] is True

    def test_logout_clears_both_workspace_cookies(self):
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        client = TestClient(app)
        client.cookies.set("lumen_ws_token", "token")
        client.cookies.set("lumen_owner", "owner-cookie")

        resp = client.post("/api/workspace/logout")
        assert resp.status_code == 200
        set_cookie = "\n".join(resp.headers.get_list("set-cookie"))
        assert "lumen_ws_token=" in set_cookie
        assert "lumen_owner=" in set_cookie


# ── Task 2.6: Wire middleware into existing endpoints ───────────────────


class TestWorkspaceAuthOnExistingEndpoints(unittest.TestCase):
    """Verify that workspace JWT auth works on existing protected endpoints."""

    def test_protected_endpoint_with_workspace_jwt(self):
        """A workspace JWT should grant access to owner-guarded endpoints."""
        from lumen.core.workspace import WorkspaceIndex
        from lumen.core.workspace_auth import create_jwt
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], "$2b$12$hash")

        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_access_mode = web_mod._access_mode
        orig_config = web_mod._config
        orig_secret = wa._get_workspace_secret

        web_mod._workspace_index = idx
        web_mod._access_mode = "serve"
        web_mod._config = {"model": "test-model"}
        wa._get_workspace_secret = lambda: "test-secret"

        token = create_jwt("alice@example.com", "member", "marketing", "test-secret")

        client = TestClient(app)
        client.cookies.set("lumen_ws_token", token)
        resp = client.get("/api/status")

        # Should not be 401 with "authentication_required"
        if resp.status_code == 401:
            error = resp.json().get("error", "")
            assert error != "authentication_required", (
                "Workspace JWT should bypass owner auth"
            )

        web_mod._workspace_index = orig_index
        web_mod._access_mode = orig_access_mode
        web_mod._config = orig_config
        wa._get_workspace_secret = orig_secret

    def test_protected_endpoint_with_workspace_jwt_bearer_header(self):
        """A workspace JWT in Authorization header should grant access too."""
        from lumen.core.workspace import WorkspaceIndex
        from lumen.core.workspace_auth import create_jwt
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], "$2b$12$hash")

        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_access_mode = web_mod._access_mode
        orig_config = web_mod._config
        orig_secret = wa._get_workspace_secret

        web_mod._workspace_index = idx
        web_mod._access_mode = "serve"
        web_mod._config = {"model": "test-model"}
        wa._get_workspace_secret = lambda: "test-secret"

        token = create_jwt("alice@example.com", "member", "marketing", "test-secret")

        client = TestClient(app)
        resp = client.get(
            "/api/status",
            headers={"Authorization": f"Bearer {token}"},
        )

        if resp.status_code == 401:
            error = resp.json().get("error", "")
            assert error != "authentication_required", (
                "Workspace JWT bearer header should bypass owner auth"
            )

        web_mod._workspace_index = orig_index
        web_mod._access_mode = orig_access_mode
        web_mod._config = orig_config
        wa._get_workspace_secret = orig_secret

    def test_denied_without_auth_in_workspace_mode(self):
        """No auth in workspace serve mode should return 401."""
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("alice@example.com", "member", "marketing", "Alice", [], "$2b$12$hash")

        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_access_mode = web_mod._access_mode
        orig_config = web_mod._config

        web_mod._workspace_index = idx
        web_mod._access_mode = "serve"
        web_mod._config = {"model": "test-model"}

        client = TestClient(app)
        resp = client.get("/api/status")

        assert resp.status_code == 401

        web_mod._workspace_index = orig_index
        web_mod._access_mode = orig_access_mode
        web_mod._config = orig_config

    def test_legacy_auth_still_works_without_workspace(self):
        """Without workspace_index, existing auth should be unchanged."""
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        import lumen.channels.web as web_mod

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_access_mode = web_mod._access_mode
        orig_config = web_mod._config

        web_mod._workspace_index = None
        web_mod._access_mode = "run"
        web_mod._config = {"model": "test-model"}

        client = TestClient(app)
        resp = client.get("/api/status")

        # In local mode, no auth required
        assert resp.status_code != 401

        web_mod._workspace_index = orig_index
        web_mod._access_mode = orig_access_mode
        web_mod._config = orig_config

    def test_workspace_login_allows_workspace_settings_page(self):
        from lumen.core.workspace_auth import hash_pin
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("lead@example.com", "team_admin", "marketing", "Lead", [], hash_pin("1234"))

        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_access_mode = web_mod._access_mode
        orig_config = web_mod._config
        orig_secret = wa._get_workspace_secret

        web_mod._workspace_index = idx
        web_mod._access_mode = "serve"
        web_mod._config = {"model": "test-model", "server_secret": "server-secret"}
        wa._get_workspace_secret = lambda: "test-secret"

        client = TestClient(app)
        login_resp = client.post(
            "/api/workspace/login",
            json={"email": "lead@example.com", "pin": "1234"},
        )
        assert login_resp.status_code == 200

        page_resp = client.get("/settings/workspace", follow_redirects=False)
        assert page_resp.status_code == 200

        web_mod._workspace_index = orig_index
        web_mod._access_mode = orig_access_mode
        web_mod._config = orig_config
        wa._get_workspace_secret = orig_secret

    def test_workspace_login_member_cannot_open_workspace_settings_page(self):
        from lumen.core.workspace_auth import hash_pin
        from lumen.core.workspace import WorkspaceIndex
        from lumen.channels.web import app
        from starlette.testclient import TestClient

        idx = WorkspaceIndex()
        idx.add_user("member@example.com", "member", "marketing", "Member", [], hash_pin("1234"))

        import lumen.channels.web as web_mod
        import lumen.core.workspace_auth as wa

        orig_index = getattr(web_mod, "_workspace_index", None)
        orig_access_mode = web_mod._access_mode
        orig_config = web_mod._config
        orig_secret = wa._get_workspace_secret

        web_mod._workspace_index = idx
        web_mod._access_mode = "serve"
        web_mod._config = {"model": "test-model", "server_secret": "server-secret"}
        wa._get_workspace_secret = lambda: "test-secret"

        client = TestClient(app)
        login_resp = client.post(
            "/api/workspace/login",
            json={"email": "member@example.com", "pin": "1234"},
        )
        assert login_resp.status_code == 200

        page_resp = client.get("/settings/workspace", follow_redirects=False)
        assert page_resp.status_code == 403

        web_mod._workspace_index = orig_index
        web_mod._access_mode = orig_access_mode
        web_mod._config = orig_config
        wa._get_workspace_secret = orig_secret
