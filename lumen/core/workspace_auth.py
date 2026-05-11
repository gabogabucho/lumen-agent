"""Workspace authentication: JWT secrets, tokens, PIN hashing, FastAPI middleware.

Phase 2 adds:
- JWT secret management (generate, save, load, auto-create)
- JWT token operations (create, verify)
- PIN verification (bcrypt)
- FastAPI dependencies (get_workspace_user, require_workspace_auth, require_workspace_admin)
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import bcrypt
from fastapi import Request
from jose import JWTError, jwt
from pydantic import BaseModel


# ── PIN hashing ──────────────────────────────────────────────────────────


def hash_pin(pin: str) -> str:
    """Hash a PIN using bcrypt.

    Args:
        pin: Plain-text PIN (min 4 chars, validated by caller).

    Returns:
        bcrypt hash string starting with $2b$.
    """
    return bcrypt.hashpw(pin.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_pin(pin: str, pin_hash: str) -> bool:
    """Verify a PIN against a stored bcrypt hash.

    Args:
        pin: Plain-text PIN attempt.
        pin_hash: Stored bcrypt hash.

    Returns:
        True if PIN matches, False otherwise.
    """
    if not pin or not pin_hash:
        return False
    try:
        return bcrypt.checkpw(
            pin.encode("utf-8"), pin_hash.encode("utf-8")
        )
    except Exception:
        return False


# ── JWT secret management ────────────────────────────────────────────────

_SECRET_FILE = "workspace_secret"


def generate_workspace_secret() -> str:
    """Generate a random 256-bit secret for JWT signing.

    Returns:
        Hex-encoded 64-character string (256-bit entropy).
    """
    return os.urandom(32).hex()


def save_workspace_secret(workspace_dir: Path, secret: str) -> None:
    """Save the workspace secret to a file in the workspace directory.

    Args:
        workspace_dir: Directory containing workspace.yaml.
        secret: Secret string to save.
    """
    path = workspace_dir / _SECRET_FILE
    path.write_text(secret, encoding="utf-8")


def load_workspace_secret(workspace_dir: Path) -> str | None:
    """Load an existing workspace secret.

    Args:
        workspace_dir: Directory containing workspace.yaml.

    Returns:
        Secret string, or None if the file does not exist.
    """
    path = workspace_dir / _SECRET_FILE
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8").strip()


def get_or_create_workspace_secret(workspace_dir: Path) -> str:
    """Load existing secret or generate and save a new one.

    Args:
        workspace_dir: Directory containing workspace.yaml.

    Returns:
        Secret string (existing or newly generated).
    """
    existing = load_workspace_secret(workspace_dir)
    if existing:
        return existing
    secret = generate_workspace_secret()
    save_workspace_secret(workspace_dir, secret)
    return secret


# ── JWT token operations ─────────────────────────────────────────────────

JWT_ALGORITHM = "HS256"


def create_jwt(
    user_email: str,
    role: str,
    team: str | None,
    secret: str,
    expires_hours: int = 24,
) -> str:
    """Create a JWT token for a workspace user.

    Args:
        user_email: User's email address (sub claim).
        role: User role (admin, member, viewer, team_admin).
        team: Team slug, or None for admins.
        secret: Signing secret.
        expires_hours: Token TTL in hours.

    Returns:
        Encoded JWT string.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_email,
        "role": role,
        "team": team,
        "exp": int(now.timestamp()) + (expires_hours * 3600),
        "iat": int(now.timestamp()),
    }
    return jwt.encode(payload, secret, algorithm=JWT_ALGORITHM)


def verify_jwt(token: str, secret: str) -> dict | None:
    """Verify and decode a JWT token.

    Args:
        token: Encoded JWT string.
        secret: Signing secret.

    Returns:
        Payload dict if valid, None if invalid or expired.
    """
    if not token:
        return None
    try:
        payload = jwt.decode(token, secret, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


# ── FastAPI dependencies ─────────────────────────────────────────────────

_WS_COOKIE_NAME = "lumen_ws_token"


class WorkspaceUser(BaseModel):
    """Authenticated workspace user extracted from JWT."""

    email: str
    role: str
    team: str | None
    display_name: str


# Module-level functions that can be monkey-patched by web.py at startup
# to wire in the live index and secret. This avoids circular imports.


def _get_workspace_index():
    """Return the current WorkspaceIndex or None."""
    return None


def _get_workspace_secret():
    """Return the current workspace JWT secret string, or None."""
    return None


def get_workspace_user(request: Request):
    """FastAPI dependency: extract WorkspaceUser from JWT cookie.

    Returns WorkspaceUser if in workspace mode and valid token present.
    Returns None if not in workspace mode or no valid token.
    """
    idx = _get_workspace_index()
    if idx is None:
        return None

    token = request.cookies.get(_WS_COOKIE_NAME)
    if not token:
        return None

    secret = _get_workspace_secret()
    if not secret:
        return None

    payload = verify_jwt(token, secret)
    if payload is None:
        return None

    email = payload.get("sub", "")
    user_data = idx.lookup_user(email)
    if user_data is None:
        return None

    return WorkspaceUser(
        email=email,
        role=payload.get("role", user_data.get("role", "member")),
        team=payload.get("team", user_data.get("team")),
        display_name=user_data.get("display_name", email),
    )


def require_workspace_auth(request: Request):
    """FastAPI dependency: require valid workspace JWT or raise 401.

    Returns WorkspaceUser if authenticated.
    Raises HTTPException(401) if workspace mode is active but no valid token.
    Returns None if not in workspace mode (pass-through for legacy auth).
    """
    from fastapi import HTTPException

    idx = _get_workspace_index()
    if idx is None:
        return None

    user = get_workspace_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="authentication_required")
    return user


def require_workspace_admin(request: Request):
    """FastAPI dependency: require workspace admin role or raise 403.

    Returns WorkspaceUser if user is admin.
    Raises HTTPException(403) if user is not admin.
    Raises HTTPException(401) if not authenticated.
    """
    from fastapi import HTTPException

    idx = _get_workspace_index()
    if idx is None:
        return None

    user = get_workspace_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="authentication_required")
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="admin_required")
    return user
