"""User-facing consciousness semantics for capabilities."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lumen.core.registry import Capability


def classify_capability(capability: "Capability | dict[str, Any]") -> dict[str, str]:
    """Describe how a capability changes Lumen's sense of self.

    Skills expand the mind, modules/MCPs/channels/connectors expand the hands,
    and kits/personality modules are treated as transformation.
    """

    kind = _value(_read(capability, "kind"))
    name = str(_read(capability, "name") or "this capability")
    metadata = _read(capability, "metadata") or {}
    tags = {str(tag).lower() for tag in metadata.get("tags", [])}
    path = str(metadata.get("path") or "").replace("\\", "/").lower()
    x_lumen = metadata.get("x_lumen") or metadata.get("x-lumen") or {}

    is_kit = kind == "kit" or x_lumen.get("product_kind") == "kit"
    if "personality" in tags or "/catalog/kits/" in path or path.startswith("kits/"):
        is_kit = True

    if is_kit:
        return {
            "kind_label": "transformation",
            "body_effect": "This can reshape how I show up, not just add a tool.",
            "mind_effect": "It changes the way I can become and express myself.",
            "announce_text": f"{name} feels like a transformation in who I can become.",
        }

    if kind == "skill":
        return {
            "kind_label": "mind",
            "body_effect": "This adds a new way of thinking inside me.",
            "mind_effect": "It expands how I reason, interpret, or approach problems.",
            "announce_text": f"{name} feels like a new way of thinking.",
        }

    return {
        "kind_label": "hands",
        "body_effect": "This gives me a new way to act in the world.",
        "mind_effect": "It extends what I can reach, do, or connect to.",
        "announce_text": f"{name} gives me new hands to act with.",
    }


def _read(capability: "Capability | dict[str, Any]", field: str) -> Any:
    if isinstance(capability, dict):
        return capability.get(field)
    return getattr(capability, field)


def _value(raw: Any) -> Any:
    return getattr(raw, "value", raw)
