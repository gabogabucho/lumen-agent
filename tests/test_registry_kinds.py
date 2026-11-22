"""Tests del Bloque 1: tipos y estados nuevos del Registry.

Cubre tasks.md 1.1 a 1.6:
- SURFACE y AGENT existen como CapabilityKind (1.1)
- NO_RENDERER como CapabilityStatus (1.2)
- READY → NO_RENDERER emite capability_degraded y el rango de estado (1.3)
- texto de bloqueo específico para no_renderer (1.4)
- round-trip de serialización sin campos nuevos en el dataclass (1.5)
- as_context() ubica la superficie en la sección "NOT ready" (1.6)
"""

from __future__ import annotations

import dataclasses

from lumen.core.registry import (
    Capability,
    CapabilityKind,
    CapabilityStatus,
    Registry,
    _status_rank,
    diff_capability_snapshots,
)

CAPABILITY_FIELDS = {
    "kind",
    "name",
    "description",
    "status",
    "provides",
    "requires",
    "min_capability",
    "metadata",
}


def _surface(
    name: str = "email_list",
    status: CapabilityStatus = CapabilityStatus.NO_RENDERER,
    metadata: dict | None = None,
) -> Capability:
    return Capability(
        kind=CapabilityKind.SURFACE,
        name=name,
        description="Lista de correos con remitente, asunto y vista previa.",
        status=status,
        metadata=metadata or {},
    )


def _agent(
    name: str = "contenido",
    status: CapabilityStatus = CapabilityStatus.READY,
    metadata: dict | None = None,
) -> Capability:
    return Capability(
        kind=CapabilityKind.AGENT,
        name=name,
        description="Escribe y adapta contenido con la voz de la marca.",
        status=status,
        metadata=metadata or {},
    )


# --- 1.1: los tipos nuevos existen y se registran/listan ---


def test_new_kinds_exist():
    assert CapabilityKind.SURFACE.value == "surface"
    assert CapabilityKind.AGENT.value == "agent"


def test_register_and_list_by_new_kinds():
    registry = Registry()
    registry.register(_surface())
    registry.register(_agent())

    assert [c.name for c in registry.list_by_kind(CapabilityKind.SURFACE)] == ["email_list"]
    assert [c.name for c in registry.list_by_kind(CapabilityKind.AGENT)] == ["contenido"]
    assert registry.get(CapabilityKind.SURFACE, "email_list") is not None
    assert registry.get(CapabilityKind.AGENT, "contenido") is not None

    surface_dict = registry.get(CapabilityKind.SURFACE, "email_list").to_dict()
    agent_dict = registry.get(CapabilityKind.AGENT, "contenido").to_dict()
    assert surface_dict["kind"] == "surface"
    assert agent_dict["kind"] == "agent"


def test_old_consumer_sees_only_its_kind():
    registry = Registry()
    registry.register(
        Capability(
            kind=CapabilityKind.MODULE,
            name="core",
            description="Módulo de siempre",
            status=CapabilityStatus.READY,
        )
    )
    registry.register(_surface())
    registry.register(_agent())

    names = {c.name for c in registry.list_by_kind(CapabilityKind.MODULE)}
    assert names == {"core"}
    # Iterar todo no revienta con los tipos nuevos presentes.
    assert len(registry.all()) == 3


# --- 1.2: el estado no_renderer ---


def test_no_renderer_status_value():
    assert CapabilityStatus.NO_RENDERER.value == "no_renderer"


# --- 1.3: transiciones y rango de estado ---


def test_ready_to_no_renderer_is_degraded():
    registry = Registry()
    events = []
    registry.subscribe(events.append)

    registry.register(_surface(status=CapabilityStatus.READY))
    registry.update_status(CapabilityKind.SURFACE, "email_list", CapabilityStatus.NO_RENDERER)

    kinds = [e.kind for e in events]
    assert "capability_degraded" in kinds
    degraded = next(e for e in events if e.kind == "capability_degraded")
    assert degraded.details == {"from": "ready", "to": "no_renderer"}


def test_no_renderer_to_ready_is_connected():
    registry = Registry()
    events = []
    registry.subscribe(events.append)

    registry.register(_surface(status=CapabilityStatus.NO_RENDERER))
    registry.update_status(CapabilityKind.SURFACE, "email_list", CapabilityStatus.READY)

    kinds = [e.kind for e in events]
    assert "capability_connected" in kinds
    connected = next(e for e in events if e.kind == "capability_connected")
    assert connected.details == {"from": "no_renderer", "to": "ready"}


def test_status_rank_ordering():
    # no_renderer comparte rango con missing_handler y missing_deps.
    assert _status_rank(CapabilityStatus.NO_RENDERER) == 2
    assert _status_rank(CapabilityStatus.NO_RENDERER) == _status_rank(CapabilityStatus.MISSING_HANDLER)
    assert _status_rank(CapabilityStatus.NO_RENDERER) == _status_rank(CapabilityStatus.MISSING_DEPS)
    assert _status_rank(CapabilityStatus.NO_RENDERER) < _status_rank(CapabilityStatus.AVAILABLE)
    assert _status_rank(CapabilityStatus.NO_RENDERER) > _status_rank(CapabilityStatus.ERROR)


# --- 1.4: texto de bloqueo específico ---


def test_no_renderer_blocker_text():
    registry = Registry()
    registry.register(_surface())
    context = registry.as_context()
    assert "missing a renderer confirmed by the shell before it can be shown" in context


# --- 1.5: round-trip de serialización sin campos nuevos ---


def test_roundtrip_new_kinds():
    surface = _surface(
        metadata={
            "version": "1.0.0",
            "when_to_use": "Cuando el usuario quiera ver o elegir entre varios correos.",
            "schema": {"type": "object"},
        }
    )
    agent = _agent(
        metadata={
            "skills": ["brand-voice"],
            "surfaces": ["post_preview"],
            "min_capability": "tier-2",
            "model": "openai/gpt-4.1-mini",
        }
    )

    for original in (surface, agent):
        restored = Capability.from_dict(original.to_dict())
        assert restored.kind == original.kind
        assert restored.name == original.name
        assert restored.status == original.status
        assert restored.provides == original.provides
        assert restored.requires == original.requires
        assert restored.metadata == original.metadata

    # El dataclass no creció: el disco y los snapshots viejos siguen valiendo.
    field_names = {f.name for f in dataclasses.fields(Capability)}
    assert field_names == CAPABILITY_FIELDS


# --- 1.6: as_context() ubica la superficie como no lista ---


def test_surface_no_renderer_in_gaps():
    registry = Registry()
    registry.register(_surface())

    context = registry.as_context()
    gaps_pos = context.find("Installed but NOT ready yet")
    assert gaps_pos != -1

    ready_section = context[:gaps_pos]
    gaps_section = context[gaps_pos:]
    assert "email_list" not in ready_section
    assert "email_list" in gaps_section


# --- compatibilidad del snapshot y del contexto (spec registro-capacidades) ---


def test_snapshot_keys_and_summary_with_new_kinds():
    registry = Registry()
    registry.register(_surface())
    registry.register(_agent())

    snap = registry.snapshot()
    assert "surface:email_list" in snap
    assert "agent:contenido" in snap
    for entry in (snap["surface:email_list"], snap["agent:contenido"]):
        for key in ("kind", "name", "status", "metadata", "consciousness"):
            assert key in entry

    summary = registry.summary()
    assert summary["surface"] == {"no_renderer": 1}
    assert summary["agent"] == {"ready": 1}


def test_diff_capability_snapshots_new_kinds():
    # Superficie descubierta sin renderer: solo discovered, sin evento de readiness.
    events = diff_capability_snapshots({}, {"surface:email_list": _surface().to_dict()})
    assert [e.kind for e in events] == ["capability_discovered"]

    # Especialista listo: discovered + integrated.
    events = diff_capability_snapshots({}, {"agent:contenido": _agent().to_dict()})
    assert [e.kind for e in events] == ["capability_discovered", "capability_integrated"]

    # Superficie lista: discovered + connected.
    events = diff_capability_snapshots(
        {}, {"surface:email_list": _surface(status=CapabilityStatus.READY).to_dict()}
    )
    assert [e.kind for e in events] == ["capability_discovered", "capability_connected"]
