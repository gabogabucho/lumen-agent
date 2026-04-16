"""Immutable nucleus + self-awareness — Neo's BIOS.

The consciousness has TWO parts:
1. IDENTITY (immutable) — what Neo IS. Loaded from consciousness.yaml. Never changes.
2. AWARENESS (dynamic) — what Neo HAS. Populated by discovery from the registry.

Together they make Neo self-aware: "I am a modular agent. I have 3 skills,
6 connectors (3 ready, 3 missing handlers), 0 modules, and 1 active channel."
"""

from __future__ import annotations

from pathlib import Path

import yaml

from neo.core.registry import Registry


class Consciousness:
    """Neo's self-awareness. Immutable identity + dynamic capability awareness.

    Identity cannot be changed by modules or plugins.
    Awareness is populated by the discovery system at startup.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent / "consciousness.yaml"
        with open(config_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

        # Registry is set after discovery runs
        self._registry: Registry | None = None

    @property
    def identity(self) -> dict:
        return self._config["identity"]

    @property
    def nature(self) -> list[str]:
        return self._config["nature"]

    @property
    def name(self) -> str:
        return self.identity["name"]

    @property
    def registry(self) -> Registry | None:
        return self._registry

    def become_aware(self, registry: Registry):
        """Connect consciousness to the capability registry.

        Called after discovery completes. This is what makes Neo
        self-aware — not just knowing WHAT it is, but WHAT it has.
        """
        self._registry = registry

    def as_context(self) -> str:
        """Format full self-awareness for the LLM system prompt.

        Includes both immutable identity AND dynamic capabilities.
        """
        # Part 1: Who I am (immutable)
        lines = [
            f"I am {self.identity['name']}, a {self.identity['type']}.",
            "",
            "My nature:",
        ]
        for trait in self.nature:
            lines.append(f"- {trait}")

        # Part 2: What I have (dynamic, from registry)
        if self._registry:
            lines.append("")
            lines.append(self._registry.as_context())

        return "\n".join(lines)
