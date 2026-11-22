"""Tests del Bloque 2: clasificación, eventos y textos de awareness.

Cubre tasks.md 2.1, 2.2, 2.4 a 2.7 y 2.9 (los escenarios de MACHINE quedan
fuera del alcance MVP de este frente):
- el especialista gana a la heurística de kit (2.1)
- anuncios de superficie por estado (2.2)
- evento de readiness por tipo (2.4)
- plantillas de summary() por tipo (2.5)
- lo descubierto no listo no se anuncia como usable (2.6)
- tope de 5 pensamientos proactivos (2.7)
- tope del prompt y nota de recorte (2.9)
"""

from __future__ import annotations

from lumen.core.awareness import CapabilityAwareness
from lumen.core.capability_consciousness import classify_capability
from lumen.core.events import CapabilityEvent
from lumen.core.registry import (
    Capability,
    CapabilityKind,
    CapabilityStatus,
    Registry,
    diff_capability_snapshots,
)


def _surface(
    name: str = "email_list",
    status: CapabilityStatus = CapabilityStatus.NO_RENDERER,
    metadata: dict | None = None,
) -> Capability:
    return Capability(
        kind=CapabilityKind.SURFACE,
        name=name,
        description=f"{name} surface",
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
        metadata=metadata or {"skills": [], "surfaces": []},
    )


# --- 2.1: la clasificación de especialista gana a la de kit ---


def test_agent_classification_beats_kit_tag():
    cap = _agent(
        metadata={
            "tags": ["personality"],
            "x_lumen": {"product_kind": "kit"},
            "skills": [],
            "surfaces": [],
        }
    )
    result = classify_capability(cap)
    assert result["kind_label"] == "agent"
    assert result["announce_text"] == "Tengo un especialista nuevo: contenido. Puedo delegarle trabajo."


def test_agent_not_ready_announcements_mention_blocker():
    cap = _agent(status=CapabilityStatus.MISSING_DEPS, metadata={"skills": [], "surfaces": []})
    result = classify_capability(cap)
    assert result["kind_label"] == "agent"
    assert (
        result["announce_text"]
        == "contenido está instalado, pero todavía le falta missing deps."
    )


# --- 2.2: anuncios de superficie por estado ---


def test_surface_announce_by_status():
    not_ready = classify_capability(_surface(name="timeline"))
    assert not_ready["kind_label"] == "surface"
    assert (
        not_ready["announce_text"]
        == "Guardé la superficie timeline, pero la interfaz todavía no me confirmó cómo dibujarla."
    )

    ready = classify_capability(_surface(status=CapabilityStatus.READY))
    assert ready["kind_label"] == "surface"
    assert ready["announce_text"] == "Ahora puedo mostrarte email_list."


# --- 2.4: evento de readiness por tipo ---


def test_readiness_event_kind_per_type():
    # Especialista listo: el evento de readiness es capability_integrated.
    events = diff_capability_snapshots(
        {}, {"agent:contenido": _agent(metadata={"skills": [], "surfaces": []}).to_dict()}
    )
    assert [e.kind for e in events] == ["capability_discovered", "capability_integrated"]

    # Superficie: el evento de readiness es capability_connected.
    events = diff_capability_snapshots(
        {}, {"surface:email_list": _surface(status=CapabilityStatus.READY).to_dict()}
    )
    assert [e.kind for e in events] == ["capability_discovered", "capability_connected"]


# --- 2.5: plantillas de summary() por tipo ---


def test_summary_templates_per_type():
    discovered = CapabilityEvent(
        kind="capability_discovered",
        capability=_surface(name="timeline"),
    )
    assert (
        discovered.summary()
        == "~ superficie timeline: definida, sin renderer confirmado; disponible como artifact si hace falta"
    )

    connected = CapabilityEvent(
        kind="capability_connected",
        capability=_surface(
            status=CapabilityStatus.READY,
            metadata={"when_to_use": "Cuando el usuario quiera ver o elegir entre varios correos."},
        ),
        details={"from": "no_renderer", "to": "ready"},
    )
    assert (
        connected.summary()
        == "+ superficie email_list: lista para mostrarse — Cuando el usuario quiera ver o elegir entre varios correos."
    )

    degraded = CapabilityEvent(
        kind="capability_degraded",
        capability=_surface(status=CapabilityStatus.NO_RENDERER),
        details={"from": "ready", "to": "no_renderer"},
    )
    assert degraded.summary() == "~ superficie email_list (degraded): ready → no_renderer"

    removed = CapabilityEvent(kind="capability_removed", capability=_surface())
    assert removed.summary() == "- superficie email_list: ya no está"


def test_summary_template_for_agent():
    agent = _agent(
        metadata={
            "skills": ["brand-voice", "copy-instagram"],
            "surfaces": ["post_preview"],
            "min_capability": "tier-2",
        }
    )
    event = CapabilityEvent(kind="capability_discovered", capability=agent)
    assert event.summary() == (
        "+ especialista contenido: Escribe y adapta contenido con la voz de la marca. "
        "[skills: brand-voice, copy-instagram; superficies: post_preview; modelo mínimo: tier-2]"
    )

    degraded = CapabilityEvent(
        kind="capability_degraded",
        capability=_agent(status=CapabilityStatus.MISSING_DEPS, metadata={"skills": [], "surfaces": []}),
        details={"from": "ready", "to": "missing_deps"},
    )
    assert degraded.summary() == "~ especialista contenido (degraded): ready → missing_deps"

    removed = CapabilityEvent(kind="capability_removed", capability=_agent())
    assert removed.summary() == "- especialista contenido: ya no está"


# --- 2.6: lo descubierto no listo no se anuncia como usable ---


def test_discovered_not_ready_does_not_claim_usable():
    awareness = CapabilityAwareness(Registry())
    event = CapabilityEvent(
        kind="capability_discovered",
        capability=Capability(
            kind=CapabilityKind.CHANNEL,
            name="telegram",
            description="Un canal que todavía no anda",
            status=CapabilityStatus.AVAILABLE,
        ),
    )
    awareness.ingest_event(event)
    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert "telegram" in prompt
    assert "I know I can work with it now" not in prompt


def test_discovered_ready_claims_usable_and_mentions_type():
    awareness = CapabilityAwareness(Registry())
    event = CapabilityEvent(kind="capability_discovered", capability=_agent())
    awareness.ingest_event(event)
    proactive = awareness.format_for_proactive()
    assert proactive is not None
    assert "Tengo un especialista nuevo: contenido. Puedo delegarle trabajo." in proactive


# --- pensamientos proactivos para los tipos nuevos ---


def test_surface_no_renderer_has_no_proactive_thought_but_appears_in_prompt():
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    registry.register(_surface(name="timeline"))

    assert awareness.format_for_proactive() is None
    assert awareness.has_pending()

    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert "superficie timeline" in prompt


def test_agent_ready_generates_proactive_thought():
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    registry.register(_agent())

    proactive = awareness.format_for_proactive()
    assert proactive is not None
    assert "Tengo un especialista nuevo: contenido. Puedo delegarle trabajo." in proactive

    # El evento sigue disponible para el prompt: los drenajes son independientes.
    assert awareness.has_pending()
    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert "especialista contenido" in prompt


def test_surface_degraded_thought_is_not_the_generic_connection_text():
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    registry.register(_surface(status=CapabilityStatus.READY))
    awareness.format_for_prompt()

    registry.update_status(CapabilityKind.SURFACE, "email_list", CapabilityStatus.NO_RENDERER)
    proactive = awareness.format_for_proactive()
    assert proactive is not None
    assert "My connection with" not in proactive
    assert "email_list" in proactive


# --- 2.7: tope de pensamientos proactivos ---


def test_proactive_cap_five_thoughts():
    awareness = CapabilityAwareness(Registry())
    for i in range(8):
        awareness.ingest_event(
            CapabilityEvent(
                kind="capability_connected",
                capability=_agent(name=f"especialista-{i}"),
            )
        )

    proactive = awareness.format_for_proactive()
    assert proactive is not None
    bullets = [line for line in proactive.splitlines() if line.startswith("- ")]
    assert len(bullets) == 4  # tres pensamientos + el aviso
    assert "- y 5 cambios más que puedo contarte si preguntás." in bullets


# --- 2.9: tope del prompt y nota de recorte ---


def test_prompt_cap_and_truncation_note_by_lines():
    awareness = CapabilityAwareness(Registry())
    for i in range(50):
        awareness.ingest_event(
            CapabilityEvent(
                kind="capability_connected",
                capability=_agent(name=f"especialista-{i:02d}"),
            )
        )

    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert len(prompt) <= 4000
    assert "(+10 cambios más)" in prompt or "(+25 cambios más)" in prompt
    # Cuando recorta, la última línea es el aviso de recorte.
    assert prompt.rstrip().endswith("cambios más)")
    assert prompt.startswith("## Something shifted in what I sense and know")


def test_prompt_stable_header_and_footer_without_truncation():
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    registry.register(_surface(name="timeline"))

    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert prompt.startswith("## Something shifted in what I sense and know")
    assert prompt.rstrip().endswith("Use your own words — not technical jargon.")


def test_prompt_cap_by_characters():
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    for i in range(2):
        long_hint = "c" * 2500
        registry.register(
            _surface(
                name=f"grande_{i}",
                status=CapabilityStatus.READY,
                metadata={"when_to_use": long_hint},
            )
        )

    prompt = awareness.format_for_prompt()
    assert prompt is not None
    assert len(prompt) <= 4000
    assert "(+3 cambios más)" in prompt


# --- remoción e idempotencia (spec awareness-capacidades) ---


def test_unregister_new_kinds_emits_removed_event():
    registry = Registry()
    awareness = CapabilityAwareness(registry)
    registry.register(_surface())
    registry.register(_agent())
    awareness.format_for_prompt()

    registry.unregister(CapabilityKind.SURFACE, "email_list")
    registry.unregister(CapabilityKind.AGENT, "contenido")

    counts = awareness.peek_summary()["counts"]
    assert counts.get("capability_removed") == 2


def test_re_register_same_capability_does_not_duplicate_events():
    registry = Registry()
    awareness = CapabilityAwareness(registry)

    registry.register(_surface())
    registry.register(_surface())  # el refresh del poller vuelve a registrar igual

    counts = awareness.peek_summary()["counts"]
    assert counts.get("capability_discovered") == 1
