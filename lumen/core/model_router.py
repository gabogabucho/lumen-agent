"""Model router — role-based model selection with fallback.

Resolves which LLM model to use for a given task role (planner, executor,
summarizer, responder). Supports per-role overrides, a default model,
and a guaranteed fallback. Config-driven — all changes via config.yaml,
no code edits needed.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Valid roles for model routing
VALID_ROLES = ("planner", "executor", "summarizer", "responder", "main")


@dataclass
class ModelRouterConfig:
    """Configuration for model routing."""
    default: str = "deepseek/deepseek-chat"
    roles: dict[str, str] = field(default_factory=dict)
    fallback: str = "google/gemini-2.0-flash"
    use_default_for_all: bool = True  # toggle: use default model for everything

    @classmethod
    def from_config(cls, config: dict | None) -> "ModelRouterConfig":
        """Build from the full Lumen config dict.

        Looks for:
          config["models"]["default"]
          config["models"]["roles"]["planner"]
          config["models"]["roles"]["executor"]
          config["models"]["fallback"]
          config["models"]["use_default_for_all"]

        Falls back to config["model"] (legacy single-model) if models section missing.
        """
        if not config or not isinstance(config, dict):
            return cls()

        models_cfg = config.get("models")
        if not isinstance(models_cfg, dict):
            # Legacy: single model in config root
            return cls(default=config.get("model", cls.default))

        return cls(
            default=models_cfg.get("default", config.get("model", cls.default)),
            roles={
                k: v for k, v in models_cfg.get("roles", {}).items()
                if k in VALID_ROLES and isinstance(v, str)
            },
            fallback=models_cfg.get("fallback", cls.fallback),
            use_default_for_all=models_cfg.get("use_default_for_all", True),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize back to config-compatible dict."""
        result = {
            "default": self.default,
            "roles": dict(self.roles),
            "fallback": self.fallback,
            "use_default_for_all": self.use_default_for_all,
        }
        return result


class ModelRouter:
    """Routes model selection by task role with fallback chain.

    Resolution order:
      1. If use_default_for_all → default model (skip roles)
      2. Role-specific model if configured
      3. Default model
      4. Fallback model (guaranteed)
    """

    def __init__(self, config: ModelRouterConfig | None = None):
        self._config = config or ModelRouterConfig()

    @property
    def config(self) -> ModelRouterConfig:
        return self._config

    def update_config(self, config: ModelRouterConfig) -> None:
        """Live-update the router config (e.g., from API/CLI)."""
        self._config = config

    def update_from_dict(self, config: dict) -> None:
        """Live-update from raw config dict."""
        self._config = ModelRouterConfig.from_config({"models": config})

    def get_model(self, role: str = "main") -> str:
        """Resolve the model for a given role.

        Args:
            role: Task role (planner, executor, summarizer, responder, main).
                  "main" is the default general-purpose role.

        Returns:
            Model string suitable for litellm (e.g., "deepseek/deepseek-chat").
        """
        cfg = self._config

        # Toggle: use default for everything
        if cfg.use_default_for_all:
            return cfg.default

        # Role-specific
        if role != "main" and role in cfg.roles:
            return cfg.roles[role]

        # Default
        return cfg.default

    def get_fallback(self) -> str:
        """Return the guaranteed fallback model."""
        return self._config.fallback

    def resolve_with_fallback(self, role: str = "main") -> tuple[str, str]:
        """Return (primary_model, fallback_model) for a role.

        Useful for callers that want to implement their own fallback logic.
        """
        primary = self.get_model(role)
        fallback = self._config.fallback
        if primary != fallback:
            return primary, fallback
        return primary, ""  # no fallback if primary == fallback

    def list_roles(self) -> dict[str, str]:
        """Return all configured roles and their models.

        Always includes 'default' and 'fallback' keys.
        """
        result = {
            "default": self._config.default,
            "fallback": self._config.fallback,
        }
        result.update(self._config.roles)
        return result

    def set_role_model(self, role: str, model: str) -> bool:
        """Set a model for a specific role. Returns True if valid."""
        if role not in VALID_ROLES:
            return False
        if not model or not isinstance(model, str):
            return False
        self._config.roles[role] = model
        return True

    def set_default(self, model: str) -> bool:
        """Set the default model. Returns True if valid."""
        if not model or not isinstance(model, str):
            return False
        self._config.default = model
        return True

    def set_fallback(self, model: str) -> bool:
        """Set the fallback model. Returns True if valid."""
        if not model or not isinstance(model, str):
            return False
        self._config.fallback = model
        return True

    def set_use_default_for_all(self, value: bool) -> None:
        """Toggle 'use default for all' mode."""
        self._config.use_default_for_all = bool(value)
