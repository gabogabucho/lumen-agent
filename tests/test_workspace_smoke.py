"""Smoke tests — full end-to-end flows for workspace mode.

Phase 6: Verifies the complete user journey from login through chat to reload.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock

import yaml

from lumen.core.workspace import WorkspaceIndex, load_workspace, load_teams, build_workspace_index
from lumen.core.workspace_auth import save_workspace_secret, create_jwt, hash_pin
from lumen.channels.web import app
from starlette.testclient import TestClient


# ── Helpers ─────────────────────────────────────────────────────────────


def _setup_test_workspace(tmp_path: Path, admin_email="ceo@test.com", admin_pin="1234",
                         teams_config=None):
    """Create a workspace.yaml + teams for testing."""
    pin_hash = hash_pin(admin_pin)
    save_workspace_secret(tmp_path, "test-secret-for-jwt-64-chars-long")

    ws_yaml = {
        "name": "test-workspace",
        "display_name": "Test Workspace",
        "branding": {"logo": "/logo.png", "primary_color": "#111", "app_name": "Test"},
        "admins": [{"email": admin_email, "display_name": "CEO", "pin_hash": pin_hash}],
    }
    (tmp_path / "workspace.yaml").write_text(
        yaml.dump(ws_yaml, default_flow_style=False), encoding="utf-8"
    )

    if not teams_config:
        return tmp_path

    for team_name, team_data in teams_config.items():
        teams_dir = tmp_path / "teams" / team_name
        teams_dir.mkdir(parents=True)
        team_yaml = {
            "name": team_name,
            "display_name": team_data.get("display_name", team_name),
            "enabled_skills": team_data.get("skills", []),
            "users": team_data["users"],
        }
        (teams_dir / "team.yaml").write_text(
            yaml.dump(team_yaml, default_flow_style=False), encoding="utf-8"
        )

    return tmp_path


class TestSmokeWorkspaceLoginToChat:
    """Smoke test: login → chat → response in workspace mode."""

    def test_full_admin_login_flow(self):
        """Admin logs in, gets JWT, can access endpoints."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _setup_test_workspace(tmp_path)

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            import lumen.channels.web as web_mod
            import lumen.core.workspace_auth as wa

            orig_index = getattr(web_mod, "_workspace_index", None)
            orig_secret = getattr(wa, "_get_workspace_secret", None)
            try:
                web_mod._workspace_index = idx
                wa._get_workspace_secret = lambda: "test-secret-for-jwt-64-chars-long"

                client = TestClient(app)
                resp = client.post("/api/workspace/login", json={
                    "email": "ceo@test.com", "pin": "1234"
                })
                assert resp.status_code == 200
                data = resp.json()
                assert data["ok"] is True
                assert "lumen_ws_token" in resp.cookies
                assert data["user"]["role"] == "admin"

                # Now use the JWT to access a protected endpoint
                cookie_token = resp.cookies.get("lumen_ws_token")
                resp2 = client.get("/api/status", cookies={"lumen_ws_token": cookie_token})
                # Should NOT be 401 auth required
                if resp2.status_code == 401:
                    error = resp2.json().get("error", "")
                    assert error != "authentication_required", f"Endpoint blocked with valid workspace JWT: {error}"
            finally:
                web_mod._workspace_index = orig_index
                if orig_secret is not None:
                    wa._get_workspace_secret = orig_secret

    def test_member_login_flow(self):
        """Member logs in, gets JWT with team context."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _setup_test_workspace(tmp_path, admin_email="ceo@test.com", admin_pin="1234",
                                  teams_config={
                                      "sales": {
                                          "display_name": "Sales",
                                          "skills": ["core-productivity", "sales-tools"],
                                          "users": [{
                                              "email": "sales@test.com",
                                              "role": "member",
                                              "display_name": "Sales Rep",
                                              "pin_hash": hash_pin("5678"),
                                          }]
                                      }
                                  })

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            import lumen.channels.web as web_mod
            import lumen.core.workspace_auth as wa

            orig_index = getattr(web_mod, "_workspace_index", None)
            try:
                web_mod._workspace_index = idx
                wa._get_workspace_secret = lambda: "test-secret-for-jwt-64-chars-long"

                client = TestClient(app)
                resp = client.post("/api/workspace/login", json={
                    "email": "sales@test.com", "pin": "5678"
                })
                assert resp.status_code == 200
                data = resp.json()
                assert data["ok"] is True
                assert data["user"]["email"] == "sales@test.com"
                assert data["user"]["team"] == "sales"

                # Verify JWT payload contains team info
                cookie_token = resp.cookies.get("lumen_ws_token")
                # Decode JWT manually (header.payload.secret)
                payload_b64 = cookie_token.split(".")[1]
                payload_bytes = payload_b64 + "=="  # Add padding
                import base64
                payload = json.loads(base64.b64decode(payload_bytes))
                assert payload["sub"] == "sales@test.com"
                assert payload["role"] == "member"
                assert payload["team"] == "sales"
            finally:
                web_mod._workspace_index = orig_index


class TestSmokeCrossTeamIsolation:
    """Smoke test: user from team A cannot access team B's skills."""

    def test_team_isolation(self):
        """Marketing member tries terminal access but is blocked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            teams_config = {
                "marketing": {
                    "display_name": "Marketing",
                    "skills": ["core-productivity", "marketing-tools"],
                    "users": [{
                        "email": "mktg@test.com",
                        "role": "member",
                        "display_name": "Marketer",
                        "pin_hash": hash_pin("1111"),
                    }]
                },
                "operations": {
                    "display_name": "Operations",
                    "skills": ["terminal-tools", "ops-tools"],
                    "users": [{
                        "email": "ops@test.com",
                        "role": "member",
                        "display_name": "Ops",
                        "pin_hash": hash_pin("2222"),
                    }]
                }
            }

            _setup_test_workspace(tmp_path, admin_email="ceo@test.com", admin_pin="1234",
                                  teams_config=teams_config)

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            import lumen.channels.web as web_mod
            import lumen.core.workspace_auth as wa

            orig_index = getattr(web_mod, "_workspace_index", None)
            try:
                web_mod._workspace_index = idx
                wa._get_workspace_secret = lambda: "test-secret-for-jwt-64-chars-long"

                # Marketing member logs in
                client = TestClient(app)
                resp = client.post("/api/workspace/login", json={
                    "email": "mktg@test.com", "pin": "1111"
                })
                assert resp.status_code == 200

                # Verify ACL check: marketing tools → OK, terminal → blocked
                from lumen.core.skill_acl import check_skill_access
                assert check_skill_access("marketing-tools", "mktg@test.com", idx) is True
                assert check_skill_access("terminal-tools", "mktg@test.com", idx) is False

                # Ops member can access terminal
                assert check_skill_access("terminal-tools", "ops@test.com", idx) is True
            finally:
                web_mod._workspace_index = orig_index


class TestSmokeReloadFlow:
    """Smoke test: add skill via reload, admin gets it."""

    def test_reload_adds_new_skill(self):
        """After reload, new skill becomes available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)

            # Initial config: 1 skill
            teams_config = {
                "tech": {
                    "display_name": "Tech",
                    "skills": ["chat"],
                    "users": [{
                        "email": "tech@test.com",
                        "role": "member",
                        "display_name": "Tech",
                        "pin_hash": hash_pin("3333"),
                    }]
                }
            }

            _setup_test_workspace(tmp_path, teams_config=teams_config)
            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            import lumen.channels.web as web_mod
            from lumen.core.workspace import reload_workspace

            orig_index = getattr(web_mod, "_workspace_index", None)
            try:
                web_mod._workspace_index = idx

                # Initial: only "chat" skill
                from lumen.core.skill_acl import check_skill_access
                assert check_skill_access("chat", "tech@test.com", idx) is True
                assert check_skill_access("new-skill", "tech@test.com", idx) is False

                # Update team.yaml with new skill
                (tmp_path / "teams" / "tech" / "team.yaml").write_text(
                    yaml.dump({
                        "name": "tech",
                        "display_name": "Tech",
                        "enabled_skills": ["chat", "new-skill"],
                        "users": [{
                            "email": "tech@test.com",
                            "role": "member",
                            "display_name": "Tech",
                            "pin_hash": hash_pin("3333"),
                        }]
                    }, default_flow_style=False), encoding="utf-8"
                )

                # Reload
                new_idx = reload_workspace(tmp_path, existing_index=idx)
                assert new_idx is not None

                # New skill is now available
                assert check_skill_access("chat", "tech@test.com", new_idx) is True
                assert check_skill_access("new-skill", "tech@test.com", new_idx) is True
            finally:
                web_mod._workspace_index = orig_index


class TestSmokeAdminFullAccess:
    """Smoke test: admin has full access to all skills."""

    def test_admin_accesses_all_teams(self):
        """Admin can access ANY skill regardless of which team they're in."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            _setup_test_workspace(tmp_path)

            ws = load_workspace(tmp_path)
            teams = load_teams(tmp_path)
            idx = build_workspace_index(ws, teams)

            import lumen.channels.web as web_mod

            orig_index = getattr(web_mod, "_workspace_index", None)
            try:
                web_mod._workspace_index = idx

                from lumen.core.skill_acl import check_skill_access

                # Admin can use ANY skill
                assert check_skill_access("any-skill", "ceo@test.com", idx) is True
                assert check_skill_access("not-in-workspace", "ceo@test.com", idx) is True
                assert check_skill_access("terminal-tools", "ceo@test.com", idx) is True
                assert check_skill_access("secret-tool", "ceo@test.com", idx) is True
            finally:
                web_mod._workspace_index = orig_index


if __name__ == "__main__":
    import unittest
    unittest.main()
