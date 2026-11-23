"""Tests for ``Brain.set_surface_renderer`` (upstream plug E4).

The render transport must be connectable at any point of the Brain lifecycle,
without reaching into private attributes. Covers the scenarios of the spec
``enchufes-upstream`` (requirement E4).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from lumen.core.brain import Brain
from lumen.core.catalog import Catalog
from lumen.core.connectors import Connector, ConnectorRegistry
from lumen.core.memory import Memory
from lumen.core.registry import Capability, CapabilityKind, CapabilityStatus, Registry


def _make_consciousness():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(
        {"identity": {"name": "TestLumen", "type": "test"}, "nature": ["modular"]},
        tmp,
    )
    tmp.flush()
    from lumen.core.consciousness import Consciousness

    return Consciousness(config_path=Path(tmp.name))


def _make_personality():
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    yaml.dump(
        {"identity": {"name": "TestLumen", "role": "Test"}, "rules": ["Be helpful"]},
        tmp,
    )
    tmp.flush()
    from lumen.core.personality import Personality

    return Personality(path=Path(tmp.name))


def _make_registry():
    registry = Registry()
    registry.register(
        Capability(
            kind=CapabilityKind.SURFACE,
            name="notification",
            description="Test surface",
            status=CapabilityStatus.READY,
        )
    )
    return registry


def _make_brain(surface_renderer=None):
    connectors = ConnectorRegistry()
    connectors.register(Connector("test_conn", "Test connector", ["run"]))
    with tempfile.TemporaryDirectory() as tmp:
        catalog_path = Path(tmp) / "index.yaml"
        catalog_path.write_text(yaml.dump({"modules": []}), encoding="utf-8")
        catalog = Catalog(catalog_path)
    return Brain(
        consciousness=_make_consciousness(),
        personality=_make_personality(),
        memory=MagicMock(spec=Memory),
        connectors=connectors,
        registry=_make_registry(),
        catalog=catalog,
        model="test-model",
        surface_renderer=surface_renderer,
    )


def _payload(surface="notification"):
    return json.dumps({"surface": surface, "data": {"title": "hola"}})


class TestSetSurfaceRenderer:
    async def test_without_transport_surface_is_unavailable(self):
        brain = _make_brain()
        result = await brain._render_surface(_payload())
        assert result["ok"] is False
        assert result["error"] == "surface_unavailable"

    async def test_set_surface_renderer_connects_transport(self):
        """Spec scenario: a kit connects its shell and the next render uses it."""
        brain = _make_brain()
        calls = []

        async def transport(surface, data, render_id):
            calls.append((surface, data, render_id))
            return {"ok": True, "surface": surface}

        assert brain.set_surface_renderer(transport) is None

        result = await brain._render_surface(_payload())
        assert result == {"ok": True, "surface": "notification"}
        assert calls == [("notification", {"title": "hola"}, None)]

    async def test_set_surface_renderer_accepts_sync_transport(self):
        brain = _make_brain()
        seen = {}

        def transport(surface, data, render_id):
            seen["surface"] = surface
            return {"ok": True}

        brain.set_surface_renderer(transport)
        result = await brain._render_surface(_payload())
        assert result == {"ok": True}
        assert seen["surface"] == "notification"

    async def test_constructor_transport_still_works(self):
        async def transport(surface, data, render_id):
            return {"ok": True, "via": "constructor"}

        brain = _make_brain(surface_renderer=transport)
        result = await brain._render_surface(_payload())
        assert result == {"ok": True, "via": "constructor"}

    async def test_set_surface_renderer_replaces_transport(self):
        async def first(surface, data, render_id):
            return {"ok": True, "via": "first"}

        async def second(surface, data, render_id):
            return {"ok": True, "via": "second"}

        brain = _make_brain(surface_renderer=first)
        brain.set_surface_renderer(second)
        result = await brain._render_surface(_payload())
        assert result == {"ok": True, "via": "second"}

    async def test_set_surface_renderer_none_disconnects(self):
        async def transport(surface, data, render_id):
            return {"ok": True}

        brain = _make_brain(surface_renderer=transport)
        brain.set_surface_renderer(None)
        result = await brain._render_surface(_payload())
        assert result["error"] == "surface_unavailable"
