"""Immutable nucleus — Neo's BIOS. Always knows what it is."""

from pathlib import Path

import yaml


class Consciousness:
    """Neo's immutable core identity. Cannot be changed by modules or plugins.

    The consciousness defines WHAT Neo IS, not what it does.
    Specifics come from personality and modules. Universals come from here.
    """

    def __init__(self, config_path: Path | None = None):
        if config_path is None:
            config_path = Path(__file__).parent / "consciousness.yaml"
        with open(config_path, encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    @property
    def identity(self) -> dict:
        return self._config["identity"]

    @property
    def nature(self) -> list[str]:
        return self._config["nature"]

    @property
    def name(self) -> str:
        return self.identity["name"]

    def as_context(self) -> str:
        """Format consciousness for LLM system prompt."""
        lines = [
            f"I am {self.identity['name']}, a {self.identity['type']}.",
            "",
            "My nature:",
        ]
        for trait in self.nature:
            lines.append(f"- {trait}")
        return "\n".join(lines)
