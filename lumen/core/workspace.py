"""Workspace loader for enterprise company/team/user hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
from pathlib import Path
import secrets

import yaml


@dataclass
class WorkspaceUserRecord:
    email: str
    display_name: str
    role: str
    team: str
    enabled_skills: list[str]
    pin_hash: str | None = None


@dataclass
class WorkspaceSnapshot:
    workspace_path: Path
    workspace: dict
    teams: dict[str, dict]
    users_by_email: dict[str, WorkspaceUserRecord]
    errors: list[str]

    @property
    def enabled(self) -> bool:
        return self.workspace_path.exists()

    @property
    def valid(self) -> bool:
        return self.enabled and not self.errors


def _read_yaml_file(path: Path) -> dict:
    if not path.exists():
        return {}
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return loaded if isinstance(loaded, dict) else {}


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


def load_workspace(*, lumen_dir: Path) -> WorkspaceSnapshot:
    workspace_path = lumen_dir / "workspace.yaml"
    workspace_root = workspace_path.parent
    workspace = _read_yaml_file(workspace_path)
    teams: dict[str, dict] = {}
    users_by_email: dict[str, WorkspaceUserRecord] = {}
    errors: list[str] = []

    if not workspace_path.exists():
        return WorkspaceSnapshot(workspace_path, workspace, teams, users_by_email, errors)

    if not workspace.get("name"):
        errors.append("workspace.yaml missing required field: name")

    team_names_seen: set[str] = set()
    for team_dir in _team_dirs(workspace_root):
        team_path = team_dir / "team.yaml"
        team_doc = _read_yaml_file(team_path)
        if not team_doc:
            errors.append(f"team file missing or invalid: {team_path}")
            continue

        raw_team_name = str(team_doc.get("name") or team_dir.name)
        team_slug = _normalize_slug(raw_team_name) or _normalize_slug(team_dir.name) or team_dir.name
        if team_slug in team_names_seen:
            errors.append(f"duplicate team name detected: {team_slug}")
            continue
        team_names_seen.add(team_slug)

        enabled_skills = team_doc.get("enabled_skills") or []
        if not isinstance(enabled_skills, list):
            errors.append(f"team {team_slug} has invalid enabled_skills (must be list)")
            enabled_skills = []
        normalized_skills = [str(skill).strip() for skill in enabled_skills if str(skill).strip()]

        team_users = team_doc.get("users") or []
        if not isinstance(team_users, list):
            errors.append(f"team {team_slug} has invalid users (must be list)")
            team_users = []

        teams[team_slug] = {
            **team_doc,
            "name": team_slug,
            "enabled_skills": normalized_skills,
            "users": team_users,
        }

        for user in team_users:
            if not isinstance(user, dict):
                errors.append(f"team {team_slug} has non-object user entry")
                continue
            email = str(user.get("email") or "").strip().lower()
            if not email:
                errors.append(f"team {team_slug} has user without email")
                continue
            if email in users_by_email:
                errors.append(f"duplicate user email across teams: {email}")
                continue
            users_by_email[email] = WorkspaceUserRecord(
                email=email,
                display_name=str(user.get("display_name") or email),
                role=str(user.get("role") or "member"),
                team=team_slug,
                enabled_skills=normalized_skills,
                pin_hash=str(user.get("pin_hash")) if user.get("pin_hash") else None,
            )

    return WorkspaceSnapshot(workspace_path, workspace, teams, users_by_email, errors)


def workspace_branding(snapshot: WorkspaceSnapshot) -> dict:
    branding = snapshot.workspace.get("branding") if isinstance(snapshot.workspace, dict) else {}
    if not isinstance(branding, dict):
        branding = {}
    return {
        "workspace": snapshot.workspace.get("name") if isinstance(snapshot.workspace, dict) else None,
        "display_name": branding.get("app_name")
        or snapshot.workspace.get("display_name")
        or snapshot.workspace.get("name")
        or "Lumen",
        "logo": branding.get("logo"),
        "primary_color": branding.get("primary_color"),
        "enabled": snapshot.enabled,
        "valid": snapshot.valid,
    }


def team_file_path(*, lumen_dir: Path, team_slug: str) -> Path:
    return lumen_dir / "teams" / team_slug / "team.yaml"


def load_team(*, lumen_dir: Path, team_slug: str) -> dict:
    return _read_yaml_file(team_file_path(lumen_dir=lumen_dir, team_slug=team_slug))


def save_team(*, lumen_dir: Path, team_slug: str, team_doc: dict) -> None:
    _write_yaml_file(team_file_path(lumen_dir=lumen_dir, team_slug=team_slug), team_doc)


def list_governance(*, lumen_dir: Path) -> dict:
    snapshot = load_workspace(lumen_dir=lumen_dir)
    teams = []
    for slug, team_doc in sorted(snapshot.teams.items()):
        users = team_doc.get("users") if isinstance(team_doc, dict) else []
        skills = team_doc.get("enabled_skills") if isinstance(team_doc, dict) else []
        teams.append(
            {
                "team": slug,
                "display_name": team_doc.get("display_name") if isinstance(team_doc, dict) else slug,
                "enabled_skills": skills if isinstance(skills, list) else [],
                "users": users if isinstance(users, list) else [],
            }
        )
    return {
        "workspace": snapshot.workspace.get("name") if isinstance(snapshot.workspace, dict) else None,
        "display_name": snapshot.workspace.get("display_name") if isinstance(snapshot.workspace, dict) else None,
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
