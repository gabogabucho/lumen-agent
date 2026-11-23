"""Tests del Bloque 7: tool render_surface dinámica.

Cubre tasks.md 7.1 a 7.6:
- build_render_surface_tool: enum, descripción y recortes (7.1)
- sin superficies READY devuelve None (7.2)
- reconstrucción por turno (7.3)
- enganche en _prepare_think_context sin tocar _tool_schemas (7.4)
- ejecución contra una superficie que dejó de estar READY (7.5)
- no_renderer no aparece ni en el enum ni en la descripción (7.6)
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from lumen.core.brain import Brain
from lumen.core.connectors import Connector, ConnectorRegistry
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry
from lumen.core.session import Session
from lumen.core.surfaces import build_render_surface_tool, execute_render_surface

pytest.importorskip("litellm")  # el Brain importa litellm al construirse


def _registry_with_surfaces(*specs: tuple[str, CapabilityStatus, dict]) -> Registry:
    """Registra superficies listas a mano.

    El handshake (apply_hello) es lo que deja READY en producción; acá
    pasamos el estado directo porque estas pruebas miran la tool, no el ciclo.
    """
    registry = Registry()
    for name, status, metadata in specs:
        registry.register(
            Capability(
                kind=CapabilityKind.SURFACE,
                name=name,
                description=f"Superficie {name}.",
                status=status,
                metadata=metadata,
            )
        )
    return registry


def _ready(name: str, when_to_use: str | None = None):
    metadata = {"schema": {"type": "object"}}
    if when_to_use is not None:
        metadata["when_to_use"] = when_to_use
    return (name, CapabilityStatus.READY, metadata)


def _surface_unready(name: str) -> Capability:
    return Capability(
        kind=CapabilityKind.SURFACE,
        name=name,
        description=f"Superficie {name}.",
        status=CapabilityStatus.NO_RENDERER,
        metadata={"schema": {"type": "object"}},
    )


# --- 7.1: enum y descripción ---


def test_enum_and_description():
    registry = _registry_with_surfaces(
        _ready("post_preview", "Cuando tengan que previsualizar un posteo."),
        _ready("email_list", "Cuando el usuario quiera ver o elegir entre varios correos."),
    )
    # Una superficie no lista no entra.
    registry.register(_surface_unready("table"))

    tool = build_render_surface_tool(registry)
    assert tool is not None
    assert tool["type"] == "function"

    function = tool["function"]
    assert function["name"] == "render_surface"
    parameters = function["parameters"]
    assert parameters["type"] == "object"
    assert parameters["required"] == ["surface", "data"]
    surface_property = parameters["properties"]["surface"]
    assert surface_property["type"] == "string"
    # Orden alfabético, solo las READY.
    assert surface_property["enum"] == ["email_list", "post_preview"]
    assert parameters["properties"]["data"]["type"] == "object"
    assert "id" in parameters["properties"]

    description = function["description"]
    assert "- email_list: Cuando el usuario quiera ver o elegir entre varios correos." in description
    assert "- post_preview: Cuando tengan que previsualizar un posteo." in description
    assert "table" not in description


def test_when_to_use_collapsed_and_trimmed():
    long_hint = "palabra " * 40  # 280 caracteres
    registry = _registry_with_surfaces(_ready("email_list", long_hint))

    description = build_render_surface_tool(registry)["function"]["description"]
    line = next(l for l in description.splitlines() if l.startswith("- email_list:"))
    body = line.removeprefix("- email_list:").strip()
    # Colapsados a una línea y recortados a 200 caracteres.
    assert "\n" not in body
    assert len(body) <= 200


def test_description_total_cap_and_surface_cap():
    spec = [(f"surface_{i:03d}", "Cada una con su cuando usar.") for i in range(150)]
    registry = _registry_with_surfaces(*[_ready(name, hint) for name, hint in spec])

    tool = build_render_surface_tool(registry)
    description = tool["function"]["description"]
    assert len(description) <= 4000
    # El aviso queda siempre visible aunque el recorte baje líneas.
    assert "(+" in description and "superficies más)" in description
    # A lo sumo 100 superficies: la 101 no aparece.
    assert "surface_100" not in description
    assert "surface_000" in description
    enum = tool["function"]["parameters"]["properties"]["surface"]["enum"]
    assert len(enum) <= 100
    assert enum == [f"surface_{i:03d}" for i in range(len(enum))]


# --- 7.2: sin READY no hay tool ---


def test_no_ready_surfaces_returns_none():
    registry = _registry_with_surfaces(
        ("timeline", CapabilityStatus.NO_RENDERER, {"schema": {"type": "object"}})
    )
    assert build_render_surface_tool(registry) is None

    assert build_render_surface_tool(Registry()) is None


# --- 7.3: se reconstruye en cada turno ---


def test_rebuilt_per_turn():
    registry = _registry_with_surfaces(_ready("email_list", "para los correos"))

    first = build_render_surface_tool(registry)
    assert first["function"]["parameters"]["properties"]["surface"]["enum"] == ["email_list"]

    # Entre dos turnos se confirma timeline: la tool del turno siguiente la muestra.
    registry.register(
        Capability(
            kind=CapabilityKind.SURFACE,
            name="timeline",
            description="Superficie timeline.",
            status=CapabilityStatus.READY,
            metadata={"schema": {"type": "object"}, "when_to_use": "para el hilo"},
        )
    )
    second = build_render_surface_tool(registry)
    assert second["function"]["parameters"]["properties"]["surface"]["enum"] == [
        "email_list",
        "timeline",
    ]

    # Y si timeline pierde el renderer, el turno siguiente la saca.
    registry.update_status(CapabilityKind.SURFACE, "timeline", CapabilityStatus.NO_RENDERER)
    third = build_render_surface_tool(registry)
    assert third["function"]["parameters"]["properties"]["surface"]["enum"] == ["email_list"]


# --- 7.4: enganche en el Brain ---


def _make_brain(registry: Registry, surface_renderer=None) -> Brain:
    from lumen.core.catalog import Catalog
    from lumen.core.memory import Memory
    from unittest.mock import MagicMock

    import tempfile

    from lumen.core.consciousness import Consciousness
    from lumen.core.personality import Personality

    with tempfile.TemporaryDirectory() as tmp:
        identity_path = Path(tmp) / "consciousness.yaml"
        identity_path.write_text(
            yaml.dump({"identity": {"name": "TestLumen", "type": "agent"}, "nature": ["x"]}),
            encoding="utf-8",
        )
        consciousness = Consciousness(config_path=identity_path)

        personality_path = Path(tmp) / "personality.yaml"
        personality_path.write_text(
            yaml.dump({"identity": {"name": "TestLumen", "role": "x"}, "tone": {}, "rules": []}),
            encoding="utf-8",
        )
        personality = Personality(path=personality_path)

        catalog_path = Path(tmp) / "index.yaml"
        catalog_path.write_text(yaml.dump({"modules": []}), encoding="utf-8")
        catalog = Catalog(catalog_path)

        connectors = ConnectorRegistry()
        connectors.register(Connector("test_conn", "Test connector", ["run"]))

        return Brain(
            consciousness=consciousness,
            personality=personality,
            memory=MagicMock(spec=Memory),
            connectors=connectors,
            registry=registry,
            catalog=catalog,
            model="test-model",
            flows=[],
            surface_renderer=surface_renderer,
        )


def test_brain_injects_surface_tool():
    registry = _registry_with_surfaces(_ready("email_list", "para los correos"))
    brain = _make_brain(registry)

    context, _messages, tools = asyncio.run(brain._prepare_think_context("hola", Session()))

    names = [tool["function"]["name"] for tool in (tools or [])]
    assert "render_surface" in names

    surface_tool = next(t for t in tools if t["function"]["name"] == "render_surface")
    assert surface_tool["function"]["parameters"]["properties"]["surface"]["enum"] == ["email_list"]


def test_brain_omits_surface_tool_without_ready_surfaces():
    registry = Registry()
    brain = _make_brain(registry)

    _context, _messages, tools = asyncio.run(brain._prepare_think_context("hola", Session()))
    names = [tool["function"]["name"] for tool in (tools or [])]
    assert "render_surface" not in names


def test_brain_does_not_pollute_connector_tool_schemas():
    registry = _registry_with_surfaces(_ready("email_list", "para los correos"))
    brain = _make_brain(registry)

    asyncio.run(brain._prepare_think_context("hola", Session()))

    # La tool de superficie no se cachea en el ConnectorRegistry.
    cached_names = [
        schema["function"]["name"] for schema in getattr(brain.connectors, "_tool_schemas", []) or []
    ]
    assert "render_surface" not in cached_names


# --- 7.5: ejecución ---


def test_stale_surface_rejected():
    registry = _registry_with_surfaces(
        ("email_list", CapabilityStatus.NO_RENDERER, {"schema": {"type": "object"}})
    )

    calls = []

    async def _render(surface, data, render_id):
        calls.append(surface)
        return {"ok": True, "id": "x", "surface": surface}

    result = asyncio.run(
        execute_render_surface(registry, surface="email_list", data={"items": []}, render=_render)
    )

    assert result == {
        "ok": False,
        "error": "surface_not_ready",
        "surface": "email_list",
        "status": "no_renderer",
    }
    assert calls == []


def test_unknown_surface_rejected():
    registry = Registry()
    result = asyncio.run(
        execute_render_surface(registry, surface="fantasma", data={}, render=lambda s, d, i: {})
    )
    assert result == {"ok": False, "error": "surface_unknown", "surface": "fantasma"}


def test_ready_surface_delegates_to_renderer():
    registry = _registry_with_surfaces(_ready("email_list", "para los correos"))

    async def _render(surface, data, render_id):
        return {"ok": True, "id": render_id or "s1", "surface": surface}

    result = asyncio.run(
        execute_render_surface(
            registry, surface="email_list", data={"items": [1]}, render=_render, id="s1"
        )
    )
    assert result == {"ok": True, "id": "s1", "surface": "email_list"}


def test_brain_executes_render_surface_with_stale_surface():
    registry = _registry_with_surfaces(
        ("email_list", CapabilityStatus.NO_RENDERER, {"schema": {"type": "object"}})
    )
    brain = _make_brain(registry)

    result = asyncio.run(brain._render_surface(json.dumps({"surface": "email_list", "data": {}})))

    assert result == {
        "ok": False,
        "error": "surface_not_ready",
        "surface": "email_list",
        "status": "no_renderer",
    }


def test_brain_executes_render_surface_with_wired_renderer():
    registry = _registry_with_surfaces(_ready("email_list", "para los correos"))

    async def _renderer(surface, data, render_id):
        return {"ok": True, "id": render_id or "gen", "surface": surface, "delivered": True}

    brain = _make_brain(registry, surface_renderer=_renderer)
    result = asyncio.run(brain._render_surface(json.dumps({"surface": "email_list", "data": {"x": 1}})))

    assert result["ok"] is True
    assert result["surface"] == "email_list"
    assert result["delivered"] is True


def test_brain_executes_render_surface_without_transport_is_honest():
    registry = _registry_with_surfaces(_ready("email_list", "para los correos"))
    brain = _make_brain(registry)

    result = asyncio.run(brain._render_surface(json.dumps({"surface": "email_list", "data": {}})))

    assert result == {"ok": False, "error": "surface_unavailable", "surface": "email_list"}


# --- 7.6: no_renderer excluido ---


def test_no_renderer_excluded():
    registry = _registry_with_surfaces(
        ("timeline", CapabilityStatus.NO_RENDERER, {"schema": {}, "when_to_use": "pronto"}),
        _ready("email_list", "para los correos"),
    )

    tool = build_render_surface_tool(registry)
    enum = tool["function"]["parameters"]["properties"]["surface"]["enum"]
    description = tool["function"]["description"]
    assert enum == ["email_list"]
    assert "timeline" not in description
