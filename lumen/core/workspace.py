"""Workspace loader and index helpers for enterprise company/team/user hierarchy."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import hmac
from pathlib import Path
import secrets

from pydantic import BaseModel, EmailStr, ValidationError, field_validator
import yaml


class WorkspaceValidationError(ValueError):
    """Raised when workspace configuration is invalid."""


class BrandingConfig(BaseModel):
    logo: str
    primary_color: str
    app_name: str


class AdminRecord(BaseModel):
    email: EmailStr
    display_name: str
    pin_hash: str


class UserRecord(BaseModel):
    email: EmailStr
    role: str
    display_name: str
    pin_hash: str

    @field_validator("role")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        if value not in {"member", "viewer", "team_admin", "admin"}:
            raise ValueError("invalid role")
        return value


class TeamConfig(BaseModel):
    name: str
    display_name: str
    enabled_skills: list[str]
    users: list[UserRecord]


class WorkspaceConfig(BaseModel):
    name: str
    display_name: str
    branding: BrandingConfig
    admins: list[AdminRecord]


@dataclass
class WorkspaceUserRecord:
    email: str
    display_name: str
    role: str
    team: str | None
    enabled_skills: list[str]
    pin_hash: str | None = None


@dataclass
class WorkspaceSnapshot:
    workspace_path: Path
    workspace: WorkspaceConfig | None
    teams: dict[str, TeamConfig]
    users_by_email: dict[str, WorkspaceUserRecord]
    errors: list[str]

    @property
    def enabled(self) -> bool:
        return self.workspace_path.exists()

    @property
    def valid(self) -> bool:
        return self.workspace is not None and not self.errors

    @property
    def name(self) -> str | None:
        return self.workspace.name if self.workspace else None

    @property
    def display_name(self) -> str | None:
        return self.workspace.display_name if self.workspace else None

    @property
    def admins(self) -> list[AdminRecord]:
        return self.workspace.admins if self.workspace else []


class WorkspaceIndex:
    def __init__(self):
        self._users: dict[str, dict] = {}

    def add_user(
        self,
        email: str,
        role: str,
        team: str | None,
        display_name: str,
        enabled_skills: list[str],
        pin_hash: str | None,
    ) -> None:
        normalized = str(email or "").strip().lower()
        if not normalized:
            return
        self._users[normalized] = {
            "email": normalized,
            "role": role,
            "team": team,
            "display_name": display_name,
            "enabled_skills": list(enabled_skills or []),
            "pin_hash": pin_hash,
        }

    def lookup_user(self, email: str) -> dict | None:
        return self._users.get(str(email or "").strip().lower())

    def get_enabled_skills(self, email: str) -> list[str]:
        user = self.lookup_user(email)
        if user is None:
            return []
        return list(user.get("enabled_skills") or [])

    def get_user_role(self, email: str) -> str | None:
        user = self.lookup_user(email)
        return user.get("role") if user else None

    def get_user_team(self, email: str) -> str | None:
        user = self.lookup_user(email)
        return user.get("team") if user else None

    def is_workspace_mode(self) -> bool:
        return bool(self._users)


def _read_yaml_file(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _write_yaml_file(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _normalize_slug(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in (value or "").strip())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


def _team_dirs(workspace_root: Path) -> list[Path]:
    teams_root = workspace_root / "teams"
    if not teams_root.exists():
        return []
    return sorted(path for path in teams_root.iterdir() if path.is_dir())


def hash_secret(value: str, *, salt: str | None = None) -> str:
    used_salt = salt or secrets.token_urlsafe(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        value.encode("utf-8"),
        used_salt.encode("utf-8"),
        260000,
    )
    encoded = base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")
    return f"pbkdf2_sha256$260000${used_salt}${encoded}"


def verify_secret(value: str, stored_hash: str | None) -> bool:
    if not value or not stored_hash:
        return False
    try:
        algorithm, iterations, salt, digest = str(stored_hash).split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        computed = hashlib.pbkdf2_hmac(
            "sha256",
            value.encode("utf-8"),
            salt.encode("utf-8"),
            int(iterations),
        )
    except (TypeError, ValueError):
        return False
    encoded = base64.urlsafe_b64encode(computed).decode("utf-8").rstrip("=")
    return hmac.compare_digest(encoded, digest)


def load_workspace(lumen_dir: Path, *, snapshot: bool = False):
    workspace_path = lumen_dir / "workspace.yaml"
    workspace_doc = _read_yaml_file(workspace_path)
    if workspace_doc is None:
        if snapshot:
            return WorkspaceSnapshot(workspace_path, None, {}, {}, ["workspace.yaml missing or invalid"] if workspace_path.exists() else [])
        return None

    try:
        workspace = WorkspaceConfig.model_validate(workspace_doc)
    except ValidationError:
        if snapshot:
            return WorkspaceSnapshot(workspace_path, None, {}, {}, ["workspace.yaml invalid"])
        return None

    if snapshot:
        teams = load_teams(lumen_dir)
        users_by_email: dict[str, WorkspaceUserRecord] = {}
        for admin in workspace.admins:
            users_by_email[str(admin.email).lower()] = WorkspaceUserRecord(
                email=str(admin.email).lower(),
                display_name=admin.display_name,
                role="admin",
                team=None,
                enabled_skills=[],
                pin_hash=admin.pin_hash,
            )
        for team_slug, team in teams.items():
            for user in team.users:
                users_by_email[str(user.email).lower()] = WorkspaceUserRecord(
                    email=str(user.email).lower(),
                    display_name=user.display_name,
                    role=user.role,
                    team=team_slug,
                    enabled_skills=list(team.enabled_skills),
                    pin_hash=user.pin_hash,
                )
        return WorkspaceSnapshot(workspace_path, workspace, teams, users_by_email, [])
    return workspace


def load_teams(lumen_dir: Path) -> dict[str, TeamConfig]:
    teams: dict[str, TeamConfig] = {}
    for team_dir in _team_dirs(lumen_dir):
        team_doc = _read_yaml_file(team_dir / "team.yaml")
        if team_doc is None:
            continue
        try:
            team = TeamConfig.model_validate(team_doc)
        except ValidationError:
            continue
        teams[_normalize_slug(team.name) or team_dir.name] = team
    return teams


def build_workspace_index(workspace, teams: dict[str, TeamConfig]) -> WorkspaceIndex:
    idx = WorkspaceIndex()

    admins = []
    if isinstance(workspace, WorkspaceSnapshot):
        admins = workspace.admins
        workspace_name = workspace.name
    else:
        admins = getattr(workspace, "admins", []) or []
        workspace_name = getattr(workspace, "name", None)

    for admin in admins:
        idx.add_user(
            str(admin.email),
            "admin",
            None,
            admin.display_name,
            ["*"],
            admin.pin_hash,
        )

    for team_slug, team in (teams or {}).items():
        for user in team.users:
            idx.add_user(
                str(user.email),
                user.role,
                team_slug,
                user.display_name,
                list(team.enabled_skills),
                user.pin_hash,
            )

    return idx


def reload_workspace(lumen_dir: Path, *, existing_index: WorkspaceIndex | None = None) -> WorkspaceIndex | None:
    workspace = load_workspace(lumen_dir)
    if workspace is None:
        return existing_index
    teams = load_teams(lumen_dir)
    idx = build_workspace_index(workspace, teams)
    if not idx.is_workspace_mode() and existing_index is not None:
        return existing_index
    return idx


def workspace_branding(snapshot: WorkspaceSnapshot) -> dict:
    branding = snapshot.workspace.branding if snapshot.workspace else None
    return {
        "workspace": snapshot.name,
        "display_name": (branding.app_name if branding else None) or snapshot.display_name or snapshot.name or "Lumen",
        "logo": branding.logo if branding else None,
        "primary_color": branding.primary_color if branding else None,
        "enabled": snapshot.enabled,
        "valid": snapshot.valid,
    }


def team_file_path(*, lumen_dir: Path, team_slug: str) -> Path:
    return lumen_dir / "teams" / team_slug / "team.yaml"


def load_team(*, lumen_dir: Path, team_slug: str) -> dict:
    team = load_teams(lumen_dir).get(team_slug)
    if team is None:
        return {}
    return team.model_dump()


def save_team(*, lumen_dir: Path, team_slug: str, team_doc: dict) -> None:
    _write_yaml_file(team_file_path(lumen_dir=lumen_dir, team_slug=team_slug), team_doc)


def list_governance(*, lumen_dir: Path) -> dict:
    snapshot = load_workspace(lumen_dir, snapshot=True)
    teams = []
    for slug, team in sorted(snapshot.teams.items()):
        teams.append(
            {
                "team": slug,
                "display_name": team.display_name,
                "enabled_skills": list(team.enabled_skills),
                "users": [user.model_dump() for user in team.users],
            }
        )
    return {
        "workspace": snapshot.name,
        "display_name": snapshot.display_name,
        "enabled": snapshot.enabled,
        "valid": snapshot.valid,
        "errors": snapshot.errors,
        "teams": teams,
    }


def resolve_memory_domain(*, workspace: str, role: str, team: str | None, email: str | None) -> str:
    """Contract helper for external adapters (Honcho/Obsidian/etc.)."""
    workspace_slug = (workspace or "default").strip() or "default"
    role_slug = (role or "").strip().lower()
    team_slug = (team or "no-team").strip() or "no-team"
    email_slug = (email or "anon").strip().lower() or "anon"
    if role_slug == "admin":
        return f"global:{workspace_slug}"
    if role_slug == "team_admin":
        return f"team:{workspace_slug}:{team_slug}"
    return f"user:{workspace_slug}:{team_slug}:{email_slug}"
