"""Personality — who Lumen is in this context. Loaded from YAML, swappable by modules."""

import logging
from pathlib import Path
import re
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class Personality:
    """Defines Lumen's identity, tone, rules, and domain knowledge for a context.

    Loaded from a YAML file. Modules can replace the personality to transform
    Lumen into a different assistant (e.g. barbershop, restaurant, support).
    """

    def __init__(self, path: Path | str):
        with open(path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f) or {}
            
        KNOWN_KEYS = {"identity", "tone", "rules", "knowledge", "ui", "context_prompt", "system_prompt_override"}
        unknown = set(self._config.keys()) - KNOWN_KEYS
        if unknown:
            logger.warning("Personality %s has unrecognized fields: %s", path, unknown)

    @property
    def identity(self) -> dict:
        return self._config.get("identity", {})

    @property
    def tone(self) -> dict:
        return self._config.get("tone", {})

    @property
    def rules(self) -> list[str]:
        return self._config.get("rules", [])

    @property
    def knowledge(self) -> dict:
        return self._config.get("knowledge", {})

    @property
    def ui(self) -> dict:
        return self._config.get("ui", {})

    @property
    def context_prompt(self) -> str:
        return self._config.get("context_prompt", "")

    @property
    def system_prompt_override(self) -> str | None:
        return self._config.get("system_prompt_override")

    def current(self) -> dict:
        return self._config

    def as_context(self, slot_values: dict | None = None) -> str:
        """Format personality for LLM system prompt.

        When system_prompt_override is present and non-empty, returns only the
        interpolated override text. Otherwise falls through to the legacy
        concatenation (identity + tone + rules + knowledge + context_prompt).
        Slot interpolation via re.sub applies in both paths.
        """
        if self.system_prompt_override:
            result = self.system_prompt_override.strip()
        else:
            identity = self.identity
            lines = [
                f"Your name is {identity.get('name', 'Lumen')}.",
                f"Your role: {identity.get('role', 'AI Assistant')}.",
            ]

            if identity.get("description"):
                lines.append(identity["description"])

            if self.tone:
                lines.append(f"\nTone: {self.tone.get('style', 'friendly, direct')}")

            if self.rules:
                lines.append("\nRules you MUST follow:")
                for rule in self.rules:
                    lines.append(f"- {rule}")

            if self.knowledge:
                lines.append("\nDomain knowledge:")
                for key, value in self.knowledge.items():
                    lines.append(self._format_knowledge(key, value))

            if self.context_prompt:
                lines.append("\nContext Prompt:")
                lines.append(self.context_prompt.strip())

            result = "\n".join(lines)

        if slot_values:
            # Replace {{key}} with actual values
            result = re.sub(
                r'\{\{(\w+)\}\}',
                lambda m: str(slot_values.get(m.group(1), m.group(0))),
                result
            )

        return result

    def _format_knowledge(self, key: str, value: Any, indent: int = 2) -> str:
        prefix = " " * indent
        if isinstance(value, list):
            items = "\n".join(f"{prefix}  - {item}" for item in value)
            return f"{prefix}{key}:\n{items}"
        if isinstance(value, dict):
            items = "\n".join(
                self._format_knowledge(k, v, indent + 2) for k, v in value.items()
            )
            return f"{prefix}{key}:\n{items}"
        return f"{prefix}{key}: {value}"
