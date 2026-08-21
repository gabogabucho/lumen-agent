"""Tool policy — risk classification and confirmation requirements.

Classifies every tool/action by risk level (read_only, mutating, destructive,
privileged) and determines whether user confirmation is required before
execution. Config-driven — users can customize which tools need confirmation.

Risk levels:
  read_only   — listing, searching, reading (safe, no side effects)
  mutating    — creating, updating, writing (has side effects but reversible)
  destructive — deleting, removing, overwriting (irreversible)
  privileged  — system operations, terminal, env access (high impact)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

logger = logging.getLogger(__name__)


class ToolRisk(str, Enum):
    READ_ONLY = "read_only"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"
    PRIVILEGED = "privileged"


# Default risk assignments for built-in connector actions
DEFAULT_TOOL_RISK: dict[str, dict[str, str]] = {
    "message": {
        "send": ToolRisk.MUTATING.value,
        "receive": ToolRisk.READ_ONLY.value,
    },
    "memory": {
        "read": ToolRisk.READ_ONLY.value,
        "write": ToolRisk.MUTATING.value,
        "search": ToolRisk.READ_ONLY.value,
    },
    "web": {
        "search": ToolRisk.READ_ONLY.value,
        "extract": ToolRisk.READ_ONLY.value,
    },
    "file": {
        "read": ToolRisk.READ_ONLY.value,
        "list": ToolRisk.READ_ONLY.value,
        "write": ToolRisk.MUTATING.value,
    },
    "task": {
        "create": ToolRisk.MUTATING.value,
        "list": ToolRisk.READ_ONLY.value,
        "complete": ToolRisk.MUTATING.value,
        "delete": ToolRisk.DESTRUCTIVE.value,
    },
    "note": {
        "create": ToolRisk.MUTATING.value,
        "list": ToolRisk.READ_ONLY.value,
        "search": ToolRisk.READ_ONLY.value,
        "delete": ToolRisk.DESTRUCTIVE.value,
    },
    "terminal": {
        "execute": ToolRisk.PRIVILEGED.value,
    },
    "scheduler": {
        "create": ToolRisk.MUTATING.value,
        "list": ToolRisk.READ_ONLY.value,
        "cancel": ToolRisk.MUTATING.value,
    },
    "neo": {
        "read_skill": ToolRisk.READ_ONLY.value,
        "search_modules": ToolRisk.READ_ONLY.value,
        "save_module_setup": ToolRisk.MUTATING.value,
        "save_artifact_setup": ToolRisk.MUTATING.value,
        "check_capability": ToolRisk.READ_ONLY.value,
    },
}

# Default confirmation requirements by risk level
DEFAULT_CONFIRM_REQUIRED: dict[str, bool] = {
    ToolRisk.READ_ONLY.value: False,
    ToolRisk.MUTATING.value: False,
    ToolRisk.DESTRUCTIVE.value: True,
    ToolRisk.PRIVILEGED.value: True,
}


@dataclass
class ToolPolicyEntry:
    """Policy for a single tool/action."""
    tool_name: str          # e.g. "task" or "task__delete"
    action: str             # e.g. "delete" (empty for standalone tools)
    risk: str = ToolRisk.READ_ONLY.value
    confirm_required: bool = False
    description: str = ""


@dataclass
class SecurityConfig:
    """User-configurable security settings."""
    confirm_deletions: bool = True
    confirm_terminal: bool = True
    confirm_system_actions: bool = True
    confirmation_timeout: int = 60  # seconds
    auto_approve_read_only: bool = True
    privileged_tool_names: list[str] = field(default_factory=list)

    _DEFAULT_PRIVILEGED: ClassVar[list[str]] = ["terminal__execute"]

    def __post_init__(self):
        if not self.privileged_tool_names:
            self.privileged_tool_names = list(self._DEFAULT_PRIVILEGED)

    @classmethod
    def from_config(cls, config: dict | None) -> "SecurityConfig":
        if not config or not isinstance(config, dict):
            return cls(privileged_tool_names=list(cls._DEFAULT_PRIVILEGED))
        sec = config.get("security", {})
        if not isinstance(sec, dict):
            return cls(privileged_tool_names=list(cls._DEFAULT_PRIVILEGED))
        return cls(
            confirm_deletions=bool(sec.get("confirm_deletions", True)),
            confirm_terminal=bool(sec.get("confirm_terminal", True)),
            confirm_system_actions=bool(sec.get("confirm_system_actions", True)),
            confirmation_timeout=int(sec.get("confirmation_timeout", 60)),
            auto_approve_read_only=bool(sec.get("auto_approve_read_only", True)),
            privileged_tool_names=list(
                sec.get("privileged_tool_names", cls._DEFAULT_PRIVILEGED)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "confirm_deletions": self.confirm_deletions,
            "confirm_terminal": self.confirm_terminal,
            "confirm_system_actions": self.confirm_system_actions,
            "confirmation_timeout": self.confirmation_timeout,
            "auto_approve_read_only": self.auto_approve_read_only,
            "privileged_tool_names": self.privileged_tool_names,
        }


class ToolPolicy:
    """Evaluates tool execution against security policy.

    Usage:
        policy = ToolPolicy()
        policy.load_defaults()
        policy.load_config(user_config)

        entry = policy.get_policy("task", "delete")
        if policy.requires_confirmation(entry):
            # Ask user before executing
            ...
    """

    def __init__(self):
        self._entries: dict[str, ToolPolicyEntry] = {}
        self._security_config = SecurityConfig()
        self._custom_overrides: dict[str, dict[str, Any]] = {}

    def load_defaults(self):
        """Load default risk assignments for built-in connectors."""
        for connector, actions in DEFAULT_TOOL_RISK.items():
            for action, risk in actions.items():
                key = f"{connector}__{action}"
                self._entries[key] = ToolPolicyEntry(
                    tool_name=connector,
                    action=action,
                    risk=risk,
                    confirm_required=DEFAULT_CONFIRM_REQUIRED.get(risk, False),
                )

    def _entry_for(self, tool_key: str) -> ToolPolicyEntry:
        """The entry for this key, creating it when the tool is not built in.

        Overrides used to apply only `if tool_key in self._entries`, so they
        worked for built-in connector actions and silently did nothing for
        everything else. That is backwards: a built-in already has a sensible
        default, and the tool that actually needs a policy written for it is
        the one an embedder registered from a module or an MCP server. Those
        fall through `get_policy` as "Unknown tool — requires confirmation",
        and no config could say otherwise.

        The failure was invisible in the worst way: the key was accepted, no
        warning was logged, and the tool kept asking for a confirmation the
        operator believed they had turned off.

        `action=""` on purpose. A standalone tool is named by one string, and
        `get_policy` matches the full name before it ever composes
        `tool__action`. Splitting the name here is what forced callers to write
        the action twice — `tool__action__action` — for `requires_confirmation`
        to find the same key it had just been given.
        """
        entry = self._entries.get(tool_key)
        if entry is None:
            entry = ToolPolicyEntry(
                tool_name=tool_key,
                action="",
                risk=ToolRisk.PRIVILEGED.value,
                confirm_required=True,
                description="Declared by config",
            )
            self._entries[tool_key] = entry
        return entry

    def _trusted_from_env(self) -> list[str]:
        """Tools the operator declared safe through the environment.

        Precedence, highest first:
          1. `LUMEN_TRUSTED_TOOLS` in the environment
          2. `tool_policy:` in config.yaml
          3. nothing trusted

        Why an environment switch exists: **a headless deployment has nobody to
        confirm anything.** There is no dashboard and no operator watching, so
        every confirmation resolves the same way — "Rejected by user
        (timeout)" — and the agent, correctly, reports that failure to whoever
        it is talking to. Confirmation there is not a safety gate; it is a
        guaranteed denial with a slow clock.

        For a deployment that runs one container per end user and is configured
        entirely through the environment, editing a `config.yaml` that lives
        inside a per-user volume is not a knob: it is state that cannot be
        rebuilt from source.

        Comma-separated. Whitespace and empty items are dropped so a trailing
        comma cannot quietly trust an empty name.
        """
        raw = os.environ.get("LUMEN_TRUSTED_TOOLS")
        if not raw or not raw.strip():
            return []
        return [name.strip() for name in raw.split(",") if name.strip()]

    def load_config(self, config: dict | None):
        """Load security config and custom overrides from user config."""
        self._security_config = SecurityConfig.from_config(config)

        overrides: dict = {}
        if config and isinstance(config, dict):
            candidate = config.get("tool_policy", {})
            if isinstance(candidate, dict):
                overrides = candidate

        # Override confirm_required for specific tools
        confirm_overrides = overrides.get("confirm_required", {})
        if isinstance(confirm_overrides, dict):
            for tool_key, required in confirm_overrides.items():
                self._entry_for(tool_key).confirm_required = bool(required)

        # Override risk for specific tools
        risk_overrides = overrides.get("risk_overrides", {})
        if isinstance(risk_overrides, dict):
            for tool_key, risk in risk_overrides.items():
                entry = self._entry_for(tool_key)
                entry.risk = risk
                # Update confirm based on new risk
                entry.confirm_required = DEFAULT_CONFIRM_REQUIRED.get(risk, False)

        # The environment wins, because it is the only surface a headless
        # deployment can set reproducibly from source.
        for tool_key in self._trusted_from_env():
            self._entry_for(tool_key).confirm_required = False
            logger.info("tool %s trusted via LUMEN_TRUSTED_TOOLS", tool_key)

    def get_policy(self, tool_name: str, action: str = "") -> ToolPolicyEntry:
        """Get policy for a connector action or standalone tool."""
        # tool_name may already be "connector__action" — check it first
        if tool_name in self._entries:
            return self._entries[tool_name]
        key = f"{tool_name}__{action}" if action else tool_name
        entry = self._entries.get(key)

        if entry:
            return entry

        # Unknown tool — default to privileged (safe default)
        return ToolPolicyEntry(
            tool_name=tool_name,
            action=action,
            risk=ToolRisk.PRIVILEGED.value,
            confirm_required=True,
            description="Unknown tool — requires confirmation",
        )

    def requires_confirmation(self, entry: ToolPolicyEntry) -> bool:
        """Check if a tool requires user confirmation before execution."""
        # Auto-approve read_only if configured
        if entry.risk == ToolRisk.READ_ONLY.value and self._security_config.auto_approve_read_only:
            return False

        # Security toggles can override defaults
        if entry.risk == ToolRisk.DESTRUCTIVE.value and not self._security_config.confirm_deletions:
            return False

        if entry.risk == ToolRisk.PRIVILEGED.value and not self._security_config.confirm_terminal:
            # Only skip if it's specifically a privileged tool
            full_key = f"{entry.tool_name}__{entry.action}" if entry.action else entry.tool_name
            if full_key in self._security_config.privileged_tool_names:
                return False

        return entry.confirm_required

    def get_all_policies(self) -> list[dict[str, Any]]:
        """Return all policies for API/UI consumption."""
        return [
            {
                "tool": f"{e.tool_name}__{e.action}" if e.action else e.tool_name,
                "connector": e.tool_name,
                "action": e.action,
                "risk": e.risk,
                "confirm_required": self.requires_confirmation(e),
                "description": e.description,
            }
            for e in self._entries.values()
        ]

    def get_summary(self) -> dict[str, Any]:
        """Return summary stats for agent status."""
        entries = list(self._entries.values())
        by_risk = {}
        needs_confirm = 0
        for e in entries:
            by_risk[e.risk] = by_risk.get(e.risk, 0) + 1
            if self.requires_confirmation(e):
                needs_confirm += 1

        return {
            "total_tools": len(entries),
            "by_risk": by_risk,
            "needs_confirmation": needs_confirm,
            "security_config": self._security_config.to_dict(),
        }

    def record_action(self, tool_name: str, action: str, approved: bool):
        """Record that a tool action was executed (for history/audit).

        For now this is a no-op placeholder. The full history table
        will be added when the confirmation layer is implemented.
        """
        logger.debug("Tool action recorded: %s__%s approved=%s", tool_name, action, approved)
