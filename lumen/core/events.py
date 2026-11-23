"""Capability Events — the pulse of Lumen's body.

When the Registry changes (something added, removed, or changed status),
it emits a CapabilityEvent. Subscribers receive it and react.

This is Level 1 of the Capability Awareness system.
Consciousness feels. Brain orchestrates. Body changes.
Events are how the body tells the rest.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

from lumen.core.capability_consciousness import classify_capability
from lumen.core.interoperability import awareness_interoperability_note

if TYPE_CHECKING:
    from lumen.core.registry import Capability


@dataclass
class CapabilityEvent:
    """A single change in Lumen's body."""

    kind: str
    capability: "Capability"
    timestamp: float = field(default_factory=time.time)
    details: dict[str, Any] = field(default_factory=dict)

    def is_addition(self) -> bool:
        return self.kind in {
            "capability_discovered",
            "capability_connected",
            "capability_integrated",
        }

    def is_removal(self) -> bool:
        return self.kind == "capability_removed"

    def is_status_change(self) -> bool:
        return self.kind in {
            "capability_connected",
            "capability_integrated",
            "capability_degraded",
        }

    def classification(self) -> dict[str, str]:
        return classify_capability(self.capability)

    def announce_text(self) -> str | None:
        pending_setup = self.details.get("pending_setup") or self.capability.metadata.get(
            "pending_setup"
        )
        if pending_setup:
            capability_label = (
                self.capability.metadata.get("display_name") or self.capability.name
            )
            missing = _pending_setup_names(pending_setup)
            if missing:
                return (
                    f"{capability_label} is installed, but it still needs configuration "
                    f"before it can work (missing: {missing})."
                )
            count = len(pending_setup.get("env_specs") or [])
            if count == 1:
                return f"{capability_label} is installed, but it still needs one setup value before it can work."
            return f"{capability_label} is installed, but it still needs {count} setup values before it can work."
        return self.details.get("announce_text") or self.classification().get("announce_text")

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.kind,
            "timestamp": self.timestamp,
            "summary": self.summary(),
            "announce_text": self.announce_text(),
            "capability": self.capability.to_dict(),
            "classification": self.classification(),
            "details": self.details,
        }

    def summary(self) -> str:
        cap = self.capability
        lens = _display_lens(self.classification()["kind_label"])
        note = awareness_interoperability_note(cap)
        if cap.kind.value == "surface":
            return _append_interoperability_summary(
                _surface_summary(self.kind, cap, self.details), note
            )
        if cap.kind.value == "agent":
            return _append_interoperability_summary(
                _agent_summary(self.kind, cap, self.details), note
            )
        if self.kind == "capability_discovered":
            summary = f"+ discovered {cap.name} ({cap.kind.value}, {lens}): {cap.description}"
            pending_setup = self.details.get("pending_setup") or cap.metadata.get("pending_setup")
            if pending_setup:
                missing = _pending_setup_names(pending_setup)
                if missing:
                    summary = f"{summary} [installed but not ready: missing {missing}]"
                else:
                    count = len(pending_setup.get("env_specs") or [])
                    summary = f"{summary} [setup required: {count} env]"
            return _append_interoperability_summary(summary, note)
        if self.kind == "capability_connected":
            return _append_interoperability_summary(
                f"+ connected {cap.name} ({cap.kind.value}, {lens})", note
            )
        if self.kind == "capability_integrated":
            return _append_interoperability_summary(
                f"+ integrated {cap.name} ({cap.kind.value}, {lens})", note
            )
        if self.is_removal():
            return _append_interoperability_summary(
                f"- removed {cap.name} ({cap.kind.value}, {lens})", note
            )
        if self.kind == "capability_degraded":
            frm = self.details.get("from", "?")
            to = self.details.get("to", "?")
            return _append_interoperability_summary(
                f"~ degraded {cap.name} ({cap.kind.value}, {lens}): {frm} → {to}",
                note,
            )
        return f"? {cap.name}: {self.kind}"


# Type alias for clarity
EventCallback = Callable[[CapabilityEvent], None]


_LENS_DISPLAY = {
    "hands": "connection",
    "mind": "thinking",
    "transformation": "transformation",
    "surface": "surface",
    "agent": "agent",
}


def _display_lens(kind_label: str) -> str:
    """Map internal classifier labels to user-facing lenses.

    'hands' is a valid internal category but carries anatomical baggage in
    user-facing text. Present it as 'connection' without changing the label
    itself, so tests and telemetry stay stable.
    """
    return _LENS_DISPLAY.get(kind_label, kind_label)


def _append_interoperability_summary(
    base: str, note: dict[str, str] | None
) -> str:
    if not note:
        return base
    return f"{base} — {note['summary']}"


def _pending_setup_names(pending_setup: dict[str, Any] | None) -> str:
    specs = (pending_setup or {}).get("env_specs") or []
    names = [str(spec.get("name") or "").strip() for spec in specs if spec.get("name")]
    return ", ".join(names)


def _surface_summary(kind: str, cap: Any, details: dict[str, Any]) -> str:
    if kind == "capability_discovered":
        if cap.status.value == "no_renderer":
            return (
                f"~ superficie {cap.name}: definida, sin renderer confirmado; "
                "disponible como artifact si hace falta"
            )
        return f"+ superficie {cap.name}: lista para mostrarse — {_surface_hint(cap)}"
    if kind in {"capability_connected", "capability_integrated"}:
        return f"+ superficie {cap.name}: lista para mostrarse — {_surface_hint(cap)}"
    if kind == "capability_degraded":
        frm = details.get("from", "?")
        to = details.get("to", "?")
        return f"~ superficie {cap.name} (degraded): {frm} → {to}"
    if kind == "capability_removed":
        return f"- superficie {cap.name}: ya no está"
    return f"? superficie {cap.name}: {kind}"


def _surface_hint(cap: Any) -> str:
    metadata = cap.metadata or {}
    hint = str(metadata.get("when_to_use") or cap.description or "").strip()
    return " ".join(hint.split()) or "lista para mostrarse"


def _agent_summary(kind: str, cap: Any, details: dict[str, Any]) -> str:
    metadata = cap.metadata or {}
    if kind in {"capability_discovered", "capability_integrated", "capability_connected"}:
        skills = ", ".join(metadata.get("skills") or []) or "ninguna"
        surfaces = ", ".join(metadata.get("surfaces") or []) or "ninguna"
        min_capability = metadata.get("min_capability") or "tier-1"
        return (
            f"+ especialista {cap.name}: {cap.description} "
            f"[skills: {skills}; superficies: {surfaces}; modelo mínimo: {min_capability}]"
        )
    if kind == "capability_degraded":
        frm = details.get("from", "?")
        to = details.get("to", "?")
        return f"~ especialista {cap.name} (degraded): {frm} → {to}"
    if kind == "capability_removed":
        return f"- especialista {cap.name}: ya no está"
    return f"? especialista {cap.name}: {kind}"
