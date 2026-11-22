"""Tests del Bloque 6: handshake y readiness de superficies.

Cubre tasks.md 6.1 a 6.7 (el puente al WebSocket es F4 y queda fuera):
- apply_hello promueve a READY una superficie con schema y renderer confirmado (6.1)
- contract con major distinto de 3 se rechaza (6.2)
- renderers desconocidos no crean capacidades (6.3)
- el hello es un snapshot por conexión (6.4)
- idempotencia del hello (6.5)
- drop_connection recalcula con múltiples conexiones (6.6)
- un schema inválido nunca alcanza READY (6.7)
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from lumen.core import discovery
from lumen.core.awareness import CapabilityAwareness
from lumen.core.registry import CapabilityKind, CapabilityStatus, Registry
from lumen.core.surfaces import (
    RendererConfirmation,
    apply_hello,
    apply_renderer_added,
    drop_connection,
)

CONTRACT_V3 = "x-lumen-desktop/3"


def _discovered_registry_with(tmp_path: Path, name: str) -> Registry:
    (tmp_path / "surfaces" / name).mkdir(parents=True)
    (tmp_path / "surfaces" / name / "schema.json").write_text(
        json.dumps({"type": "object", "properties": {"items": {"type": "array"}}}),
        encoding="utf-8",
    )
    (tmp_path / "surfaces" / name / "surface.yaml").write_text(
        yaml.safe_dump(
            {
                "manifest_version": 1,
                "name": name,
                "description": f"Superficie {name}.",
                "schema": "./schema.json",
            }
        ),
        encoding="utf-8",
    )
    registry = Registry()
    discovery._discover_surfaces(registry, [tmp_path / "surfaces"])
    return registry


# --- 6.1: el hello confirma el renderer ---


def test_hello_promotes_to_ready(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")

    events = apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
        host={"kind": "electron", "interactive": True, "screens": 1},
    )

    cap = registry.get(CapabilityKind.SURFACE, "email_list")
    assert cap.status == CapabilityStatus.READY
    assert cap.metadata["renderer_confirmed"] is True
    assert cap.metadata["renderer_version"] == "1.0.0"
    assert cap.metadata["host_kind"] == "electron"
    assert cap.metadata["confirmed_at"]

    assert [e.kind for e in events] == ["capability_connected"]
    assert events[0].details == {"from": "no_renderer", "to": "ready"}

    # Con schema + renderer, la superficie es usable para el Registry.
    assert [c.name for c in registry.ready()] == ["email_list"]


def test_renderer_confirmation_shape():
    confirmation = RendererConfirmation(
        name="email_list", version="1.0.0", host_kind="electron", connection_id="shell-1"
    )
    assert confirmation.name == "email_list"
    assert confirmation.connection_id == "shell-1"


# --- 6.2: contract incompatible ---


def test_hello_rejects_wrong_major(tmp_path, caplog):
    registry = _discovered_registry_with(tmp_path, "email_list")

    events = apply_hello(
        registry,
        connection_id="shell-1",
        contract="x-lumen-desktop/2",
        renderers=[{"name": "email_list", "version": "1.0.0"}],
    )

    assert events == []
    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.NO_RENDERER
    assert any("contract" in r.getMessage().lower() for r in caplog.records)


# --- 6.3: el hello no crea capacidades ---


def test_hello_ignores_unknown_renderer(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")

    apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "ghost_surface", "version": "1.0.0"}],
    )

    # El renderer fantasma no registra nada nuevo.
    kinds = {cap.kind for cap in registry.all()}
    assert kinds == {CapabilityKind.SURFACE}
    assert {c.name for c in registry.all()} == {"email_list"}
    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.NO_RENDERER

    # El resto del hello se procesa igual: lo conocido sí promociona.
    apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[
            {"name": "ghost_surface", "version": "1.0.0"},
            {"name": "email_list", "version": "1.0.0"},
        ],
    )
    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.READY


# --- 6.4: snapshot por conexión ---


def test_hello_is_per_connection_snapshot(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    # Segunda superficie para probar el retiro por snapshot.
    (tmp_path / "surfaces" / "timeline").mkdir()
    (tmp_path / "surfaces" / "timeline" / "schema.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (tmp_path / "surfaces" / "timeline" / "surface.yaml").write_text(
        yaml.safe_dump(
            {"manifest_version": 1, "name": "timeline", "description": "x", "schema": "./schema.json"}
        ),
        encoding="utf-8",
    )
    discovery._discover_surfaces(registry, [tmp_path / "surfaces"])

    apply_hello(
        registry,
        connection_id="a",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
    )
    apply_hello(
        registry,
        connection_id="b",
        contract=CONTRACT_V3,
        renderers=[{"name": "timeline", "version": "1.0.0"}],
    )

    # El snapshot de b llega sin timeline: la pierde b, pero email_list queda por a.
    apply_hello(
        registry,
        connection_id="b",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
    )

    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.READY
    assert registry.get(CapabilityKind.SURFACE, "timeline").status == CapabilityStatus.NO_RENDERER


# --- 6.5: idempotencia ---


def test_hello_idempotent(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    hello = dict(
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
        host={"kind": "web"},
    )

    first = apply_hello(registry, **hello)
    second = apply_hello(registry, **hello)

    assert [e.kind for e in first] == ["capability_connected"]
    assert second == []


def test_re_register_with_awareness_does_not_duplicate_events(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    awareness = CapabilityAwareness(registry)

    first = apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
    )
    assert first

    counts = awareness.peek_summary()["counts"]
    assert counts.get("capability_connected") == 1

    # Un segundo hello idéntico no le llega a la awareness.
    apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
    )
    assert awareness.peek_summary()["counts"].get("capability_connected") == 1


# --- 6.6: múltiples conexiones y desconexión ---


def test_drop_connection_recomputes(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    for connection_id in ("a", "b"):
        apply_hello(
            registry,
            connection_id=connection_id,
            contract=CONTRACT_V3,
            renderers=[{"name": "email_list", "version": "1.0.0"}],
        )
    assert registry.get(CapabilityKind.SURFACE, "email_list").to_dict() is not None

    # Se va a: email_list sigue READY porque b la sigue confirmando.
    events_a = drop_connection(registry, connection_id="a")
    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.READY
    assert events_a == []

    # Se va b: nadie la confirma más → no_renderer con capability_degraded.
    events_b = drop_connection(registry, connection_id="b")
    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.NO_RENDERER
    assert [e.kind for e in events_b] == ["capability_degraded"]
    assert registry.get(CapabilityKind.SURFACE, "email_list").metadata["renderer_confirmed"] is False


def test_drop_unknown_connection_is_noop(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    assert drop_connection(registry, connection_id="fantasma") == []


# --- 6.7: schema inválido no promociona ---


def test_invalid_schema_never_ready(tmp_path):
    surface_dir = tmp_path / "surfaces" / "rota"
    surface_dir.mkdir(parents=True)
    (surface_dir / "surface.yaml").write_text(
        yaml.safe_dump(
            {
                "manifest_version": 1,
                "name": "rota",
                "description": "x",
                "schema": "./schema.json",
            }
        ),
        encoding="utf-8",
    )
    (surface_dir / "schema.json").write_text("{ roto", encoding="utf-8")

    registry = Registry()
    discovery._discover_surfaces(registry, [tmp_path / "surfaces"])
    assert registry.get(CapabilityKind.SURFACE, "rota").status == CapabilityStatus.MISSING_DEPS

    apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "rota", "version": "1.0.0"}],
    )

    cap = registry.get(CapabilityKind.SURFACE, "rota")
    assert cap.status == CapabilityStatus.MISSING_DEPS
    assert cap.metadata["error"]


# --- renderer_added en caliente ---


def test_renderer_added_confirms_single_renderer(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")

    event = apply_renderer_added(
        registry,
        connection_id="shell-1",
        renderer={"name": "email_list", "version": "1.1.0"},
    )

    assert event is not None
    assert event.kind == "capability_connected"
    cap = registry.get(CapabilityKind.SURFACE, "email_list")
    assert cap.status == CapabilityStatus.READY
    assert cap.metadata["renderer_version"] == "1.1.0"


def test_renderer_added_is_additive_not_snapshot(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    # Segunda superficie.
    (tmp_path / "surfaces" / "timeline").mkdir()
    (tmp_path / "surfaces" / "timeline" / "schema.json").write_text(
        json.dumps({"type": "object"}), encoding="utf-8"
    )
    (tmp_path / "surfaces" / "timeline" / "surface.yaml").write_text(
        yaml.safe_dump(
            {"manifest_version": 1, "name": "timeline", "description": "x", "schema": "./schema.json"}
        ),
        encoding="utf-8",
    )
    discovery._discover_surfaces(registry, [tmp_path / "surfaces"])

    apply_hello(
        registry,
        connection_id="shell-1",
        contract=CONTRACT_V3,
        renderers=[{"name": "email_list", "version": "1.0.0"}],
    )

    # El hot-add no es un snapshot: no baja las demás confirmaciones de la conexión.
    apply_renderer_added(
        registry,
        connection_id="shell-1",
        renderer={"name": "timeline", "version": "1.0.0"},
    )

    assert registry.get(CapabilityKind.SURFACE, "timeline").status == CapabilityStatus.READY
    assert registry.get(CapabilityKind.SURFACE, "email_list").status == CapabilityStatus.READY


def test_renderer_added_unknown_surface_is_ignored(tmp_path):
    registry = _discovered_registry_with(tmp_path, "email_list")
    event = apply_renderer_added(
        registry, connection_id="shell-1", renderer={"name": "fantasma", "version": "1.0.0"}
    )
    assert event is None
    assert {c.name for c in registry.all()} == {"email_list"}
