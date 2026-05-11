"""Skill ACL — control which users can use which tools.

Phase 3: When a workspace is active, only permitted users can call tools
that map to skills not in their enabled_skills list.

Rules:
  - Admins always bypass ACL checks.
  - Non-workspace mode (no workspace_index) always allows.
  - Unknown tool→skill mappings fail open (allow by default).
  - Denial messages never reveal the skill name.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Generic, non-revealing denial message
_DENIAL_MESSAGE = (
    "No está permitido usar esa herramienta. Contactá a tu administrador."
)


def check_skill_access(
    skill_name: str,
    user_email: str,
    workspace_index: Any | None,
) -> bool:
    """Return ``True`` if the user may use the given skill.

    Rules:
    - Admins always get access.
    - Non-workspace mode (``workspace_index`` is ``None``) allows everything.
    - If the user's enabled_skills list is empty, we allow by default
      (empty-list means skill assignment hasn't been enforced yet).
    - Otherwise the skill must be in the user's enabled_skills.
    """

    # No workspace → legacy mode → allow all
    if workspace_index is None:
        return True

    user = workspace_index.lookup_user(user_email)
    if user is None:
        # Unknown user in workspace mode → deny
        return False

    # Admins bypass ACL entirely
    if user.get("role") == "admin":
        return True

    # Check if skill is in the user's enabled skills
    enabled = workspace_index.get_enabled_skills(user_email)
    if not enabled:
        # No skills configured yet → allow (backwards compat)
        return True

    return skill_name in enabled


def get_acl_denial_message() -> str:
    """Return a generic denial message.

    Never reveals the skill name in the message.
    """
    return _DENIAL_MESSAGE


# ── Skill extraction ────────────────────────────────────────────────────


def extract_skill_from_tool_call(
    tool_name: str,
    registry: Any,
) -> str | None:
    """Map a tool_name → skill_name via the capability registry.

    Mapping strategy:
    1. **Connector actions** (``connector__action``): look up the connector's
       registered skill capabilities.  The skill that *declares* this connector
       as a tool is the owning skill.  Return that skill's name.
    2. **Top-level tools** (e.g. ``neo__read_skill``): these are brain-internal
       tools — return ``None`` → caller should ALLOW.
    3. **No match** → ``None`` → allow by default (fail-open).
    """

    # Neo brain-internal introspection tools → allow (not workspace-scoped)
    if tool_name.startswith("neo__"):
        return None

    # Parse connector__action pattern
    connector_name = None
    if "__" in tool_name:
        # "connector__action" → the ownsig skill provides "connector__action"
        connector_name = tool_name
    else:
        connector_name = tool_name

    if not connector_name:
        return None

    # Find capabilities that declare this connector as a provided skill.
    # A connector tool becomes available when a skill's ``provides`` lists the
    # connector name, and the registry has it registered as a capability.
    for cap in registry.all():
        if cap.kind.value != "skill":
            continue
        if not cap.provides:
            continue
        if connector_name in cap.provides:
            return cap.name

    # Could not map → fail-open
    return None
